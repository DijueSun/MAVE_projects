"""SGE screen mechanistic simulator + synthetic-data surrogate modelling.

This module extends a mean-coverage toy model by:
- modelling *library skew* (uneven library element abundances)
- simulating the pipeline stochastically (cells -> edited -> PCR -> reads)
- generating synthetic datasets and fitting a simple interpretable surrogate model
- suggesting candidate experimental parameter sets for target outcomes

Intended as a scaffold: replace parameter ranges and/or link functions to real calibration data when available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


def gini_coefficient(x: np.ndarray) -> float:
    """Gini coefficient (0 = perfectly even, 1 = maximally skewed)."""
    x = np.asarray(x, dtype=float)
    x = x[x >= 0]
    if x.size == 0:
        return 0.0
    s = float(x.sum())
    if s == 0.0:
        return 0.0
    x = np.sort(x)
    n = x.size
    i = np.arange(1, n + 1)
    return float((2.0 * np.sum(i * x) / (n * s)) - (n + 1) / n)


def summarize_counts(counts: np.ndarray, name: str = "counts") -> Dict[str, float]:
    """Summary stats for an array of nonnegative counts."""
    counts = np.asarray(counts, dtype=float)
    mean = float(np.mean(counts))
    out = {
        f"{name}_mean": mean,
        f"{name}_median": float(np.median(counts)),
        f"{name}_p10": float(np.quantile(counts, 0.10)),
        f"{name}_p01": float(np.quantile(counts, 0.01)),
        f"{name}_min": float(np.min(counts)),
        f"{name}_max": float(np.max(counts)),
        f"{name}_cv": float(np.std(counts) / mean) if mean > 0 else float("nan"),
        f"{name}_gini": gini_coefficient(counts),
        f"{name}_dropout_frac": float(np.mean(counts <= 0)),
    }
    return out


def library_fractions(
    library_size: int,
    skew_model: str = "lognormal",
    skew_param: float = 0.7,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Return a probability vector p_i (sums to 1) describing library element abundances.

    Parameters
    ----------
    skew_model:
      - "lognormal": skew_param = sigma of log-abundance (0 => uniform)
      - "dirichlet": skew_param = concentration alpha (small => skewed, large => uniform)
    """
    if rng is None:
        rng = np.random.default_rng()
    L = int(library_size)
    if L <= 0:
        raise ValueError("library_size must be > 0")

    skew_model = skew_model.lower().strip()
    if skew_model == "lognormal":
        sigma = float(skew_param)
        if sigma < 0:
            raise ValueError("lognormal sigma must be >= 0")
        log_a = rng.normal(loc=0.0, scale=sigma, size=L)
        a = np.exp(log_a)
        return a / a.sum()

    if skew_model == "dirichlet":
        alpha = float(skew_param)
        if alpha <= 0:
            raise ValueError("dirichlet alpha must be > 0")
        return rng.dirichlet(alpha * np.ones(L))

    raise ValueError("skew_model must be one of: 'lognormal', 'dirichlet'")




def estimate_skew_from_counts(
    counts: np.ndarray,
    skew_model: str = "lognormal",
    n_mc: int = 200,
    random_state: int = 0,
) -> float:
    """Estimate the skew parameter from an observed per-design count vector.

    This is a *rough* calibration helper, useful for:
      - comparing plasmid / library distributions between timepoints
      - setting a prior range for the skew parameter in simulations

    Method:
      1) compute the observed Gini of the counts
      2) find the skew parameter whose *expected* Gini (under the chosen skew model)
         matches the observed Gini (via binary search + Monte Carlo).

    Notes:
      - If counts come from sequencing reads, they include sampling noise; you may want to use
        a high-depth timepoint (or plasmid sequencing) for this step.
      - For Dirichlet, skew_param is concentration alpha (small => more skew).
      - For Lognormal, skew_param is sigma of log-abundance (0 => uniform).

    Returns
    -------
    skew_param_est : float
    """
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 1:
        raise ValueError("counts must be a 1D array")
    if np.any(counts < 0):
        raise ValueError("counts must be nonnegative")

    L = int(counts.size)
    target = gini_coefficient(counts)

    rng = np.random.default_rng(random_state)

    def expected_gini(param: float) -> float:
        gs = []
        for _ in range(int(n_mc)):
            p = library_fractions(L, skew_model=skew_model, skew_param=param, rng=rng)
            gs.append(gini_coefficient(p))
        return float(np.mean(gs))

    skew_model_l = skew_model.lower().strip()

    if skew_model_l == "lognormal":
        lo, hi = 0.0, 3.0
        # Ensure bracket: g(lo) <= target <= g(hi)
        g_lo = expected_gini(lo)
        g_hi = expected_gini(hi)
        if target <= g_lo:
            return lo
        if target >= g_hi:
            return hi
        for _ in range(25):
            mid = 0.5 * (lo + hi)
            g_mid = expected_gini(mid)
            if g_mid < target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    if skew_model_l == "dirichlet":
        # For Dirichlet: alpha up => more uniform => smaller gini.
        lo, hi = 1e-3, 200.0
        g_lo = expected_gini(lo)
        g_hi = expected_gini(hi)
        # g_lo is very skewed (high gini), g_hi near uniform (low gini)
        if target >= g_lo:
            return lo
        if target <= g_hi:
            return hi
        for _ in range(25):
            mid = 0.5 * (lo + hi)
            g_mid = expected_gini(mid)
            if g_mid > target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    raise ValueError("skew_model must be one of: 'lognormal', 'dirichlet'")


def simulate_once(
    *,
    library_size: int = 2500,
    cells_transfected: int = int(20e6),
    hdr_rate: float = 0.4,
    cells_retained: int = int(5e6),
    cells_pelleted: int = int(3e6),
    genomes_input: int = int(1.52e6),
    reads_per_replicate: int = int(10e6),
    skew_model: str = "lognormal",
    skew_param: float = 0.7,
    design_edit_kappa: float = 80.0,
    assignment_model: str = "poisson",
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """Simulate one replicate of the screen (per-design level)."""
    if rng is None:
        rng = np.random.default_rng()

    L = int(library_size)
    N = int(cells_transfected)
    N_ret = int(cells_retained)
    N_pel = int(cells_pelleted)
    G_in = int(genomes_input)
    R = int(reads_per_replicate)

    if not (0 <= hdr_rate <= 1):
        raise ValueError("hdr_rate must be in [0, 1]")
    if N <= 0 or L <= 0:
        raise ValueError("cells_transfected and library_size must be > 0")

    # 1) Library abundance / skew
    p = library_fractions(L, skew_model=skew_model, skew_param=skew_param, rng=rng)

    # 2) Allocate cells to designs
    assignment_model = assignment_model.lower().strip()
    if assignment_model == "multinomial":
        cell_counts = rng.multinomial(N, pvals=p)
    elif assignment_model == "poisson":
        cell_counts = rng.poisson(lam=N * p)
    else:
        raise ValueError("assignment_model must be 'poisson' or 'multinomial'")

    # 3) Design-to-design editing variability
    kappa = max(float(design_edit_kappa), 1e-6)
    a = max(hdr_rate * kappa, 1e-6)
    b = max((1 - hdr_rate) * kappa, 1e-6)
    p_edit_i = rng.beta(a, b, size=L)
    edited_counts = rng.binomial(cell_counts, p=p_edit_i)

    # 4) Retained pool (binomial thinning approximation)
    if N_ret <= 0:
        retained_edited = np.zeros(L, dtype=int)
    else:
        retain_frac = min(max(N_ret / max(N, 1), 0.0), 1.0)
        retained_edited = rng.binomial(edited_counts, p=retain_frac)

    # 5) Pelleted pool
    if N_pel <= 0:
        pelleted_counts = np.zeros(L, dtype=int)
        pelleted_edited = np.zeros(L, dtype=int)
    else:
        pel_frac = min(max(N_pel / max(N, 1), 0.0), 1.0)
        pelleted_counts = rng.binomial(cell_counts, p=pel_frac)
        pelleted_edited = rng.binomial(edited_counts, p=pel_frac)

    # 6) Genomes into PCR
    if G_in <= 0 or pelleted_counts.sum() <= 0:
        pcr_counts = np.zeros(L, dtype=int)
        pcr_edited = np.zeros(L, dtype=int)
    else:
        probs = pelleted_counts / pelleted_counts.sum()
        pcr_counts = rng.multinomial(G_in, pvals=probs)

        frac_edited = np.divide(
            pelleted_edited,
            pelleted_counts,
            out=np.zeros_like(pelleted_edited, dtype=float),
            where=pelleted_counts > 0,
        )
        pcr_edited = rng.binomial(pcr_counts, p=frac_edited)

    # 7) Sequencing reads
    if R <= 0 or pcr_edited.sum() <= 0:
        reads = np.zeros(L, dtype=int)
    else:
        probs = pcr_edited / pcr_edited.sum()
        reads = rng.multinomial(R, pvals=probs)

    metrics: Dict[str, Any] = {}
    metrics.update(summarize_counts(cell_counts, "cells"))
    metrics.update(summarize_counts(edited_counts, "edited"))
    metrics.update(summarize_counts(retained_edited, "edited_retained"))
    metrics.update(summarize_counts(pcr_edited, "edited_pcr"))
    metrics.update(summarize_counts(reads, "reads"))

    metrics["library_size"] = L
    metrics["cells_transfected_total"] = int(cell_counts.sum())
    metrics["reads_total"] = int(reads.sum())
    metrics["skew_model"] = skew_model
    metrics["skew_param"] = float(skew_param)
    metrics["plasmid_gini"] = gini_coefficient(p)
    metrics["plasmid_p01"] = float(np.quantile(p, 0.01))
    metrics["plasmid_p10"] = float(np.quantile(p, 0.10))
    metrics["plasmid_max"] = float(np.max(p))

    return {
        "p": p,
        "cell_counts": cell_counts,
        "edited_counts": edited_counts,
        "retained_edited": retained_edited,
        "pcr_edited": pcr_edited,
        "reads": reads,
        "metrics": metrics,
    }


def run_monte_carlo(n_reps: int = 50, random_state: int = 0, **params: Any) -> pd.DataFrame:
    """Run many stochastic replicates and return per-replicate summary metrics."""
    rng = np.random.default_rng(random_state)
    rows = []
    for _ in range(int(n_reps)):
        sim = simulate_once(rng=rng, **params)
        rows.append(sim["metrics"])
    return pd.DataFrame(rows)


def sample_parameters(n: int, rng: Optional[np.random.Generator] = None) -> pd.DataFrame:
    """Sample a plausible parameter space. Adjust ranges as you learn more."""
    if rng is None:
        rng = np.random.default_rng()

    library_size = rng.integers(500, 5001, size=n)
    coverage_mean = rng.uniform(100, 1000, size=n)  # 100x–1000x
    cells_transfected = np.round(coverage_mean * library_size).astype(int)

    hdr_rate = rng.uniform(0.05, 0.8, size=n)
    skew_param = rng.uniform(0.0, 1.5, size=n)  # lognormal sigma

    cells_retained = np.round(cells_transfected * rng.uniform(0.15, 0.40, size=n)).astype(int)
    cells_pelleted = np.round(cells_transfected * rng.uniform(0.10, 0.30, size=n)).astype(int)

    genomes_input = rng.integers(int(0.5e6), int(5e6) + 1, size=n)
    reads_per_replicate = rng.integers(int(1e6), int(20e6) + 1, size=n)

    design_edit_kappa = rng.uniform(30, 200, size=n)
    gc_mean = rng.uniform(0.35, 0.65, size=n)

    return pd.DataFrame(
        {
            "library_size": library_size,
            "coverage_mean": coverage_mean,
            "cells_transfected": cells_transfected,
            "cells_retained": cells_retained,
            "cells_pelleted": cells_pelleted,
            "hdr_rate": hdr_rate,
            "skew_param": skew_param,
            "genomes_input": genomes_input,
            "reads_per_replicate": reads_per_replicate,
            "design_edit_kappa": design_edit_kappa,
            "gc_mean": gc_mean,
        }
    )


def build_synthetic_dataset(
    n_samples: int = 1000,
    n_reps_per_sample: int = 10,
    random_state: int = 0,
    skew_model: str = "lognormal",
    assignment_model: str = "poisson",
) -> pd.DataFrame:
    """Generate synthetic data by sampling parameters and simulating outcomes."""
    rng = np.random.default_rng(random_state)
    params_df = sample_parameters(n_samples, rng=rng)

    rows = []
    for _, row in params_df.iterrows():
        mc = run_monte_carlo(
            n_reps=n_reps_per_sample,
            random_state=int(rng.integers(0, 1_000_000)),
            library_size=int(row.library_size),
            cells_transfected=int(row.cells_transfected),
            hdr_rate=float(row.hdr_rate),
            cells_retained=int(row.cells_retained),
            cells_pelleted=int(row.cells_pelleted),
            genomes_input=int(row.genomes_input),
            reads_per_replicate=int(row.reads_per_replicate),
            skew_model=skew_model,
            skew_param=float(row.skew_param),
            design_edit_kappa=float(row.design_edit_kappa),
            assignment_model=assignment_model,
        )
        out = row.to_dict()
        for col in ["reads_dropout_frac", "edited_retained_p10", "reads_p10", "reads_gini", "plasmid_gini"]:
            out[col] = float(mc[col].mean())
        rows.append(out)

    return pd.DataFrame(rows)


def fit_interpretable_equation(df: pd.DataFrame, target_col: str) -> Tuple[Pipeline, list[str], str]:
    """Fit a ridge regression surrogate and return a readable equation string."""
    feature_cols = [
        "library_size",
        "coverage_mean",
        "hdr_rate",
        "skew_param",
        "genomes_input",
        "reads_per_replicate",
        "design_edit_kappa",
        "gc_mean",
    ]
    X = df[feature_cols].copy()

    y = df[target_col].values.astype(float)
    y_is_logit = False
    if target_col.endswith("_dropout_frac"):
        eps = 1e-6
        y = np.log((y + eps) / (1 - y + eps))
        y_is_logit = True

    model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    model.fit(X, y)

    scaler: StandardScaler = model.named_steps["scaler"]
    ridge: Ridge = model.named_steps["ridge"]
    coefs = ridge.coef_
    intercept = ridge.intercept_

    terms = []
    for name, coef, mean, scale in zip(feature_cols, coefs, scaler.mean_, scaler.scale_):
        terms.append(f"({coef:+.4f}) * (({name} - {mean:.4g}) / {scale:.4g})")
    equation = "y_hat = " + f"{intercept:.4f} " + " ".join(terms)

    if y_is_logit:
        equation += "\n(note: y_hat is logit(dropout); convert via sigmoid: p = 1/(1+exp(-y_hat)))"

    return model, feature_cols, equation


def suggest_experiments(
    surrogate_model: Pipeline,
    feature_cols: list[str],
    target_constraints: Dict[str, float],
    n_candidates: int = 5000,
    top_k: int = 15,
    random_state: int = 0,
    n_reps_rescore: int = 25,
) -> pd.DataFrame:
    """Suggest candidate parameter sets for target constraints."""
    rng = np.random.default_rng(random_state)
    cand = sample_parameters(n_candidates, rng=rng)

    y_hat = surrogate_model.predict(cand[feature_cols])
    cand = cand.copy()
    cand["surrogate_score"] = y_hat
    top = cand.nsmallest(top_k, "surrogate_score").reset_index(drop=True)

    rescored_rows = []
    for _, row in top.iterrows():
        mc = run_monte_carlo(
            n_reps=n_reps_rescore,
            random_state=int(rng.integers(0, 1_000_000)),
            library_size=int(row.library_size),
            cells_transfected=int(row.cells_transfected),
            hdr_rate=float(row.hdr_rate),
            cells_retained=int(row.cells_retained),
            cells_pelleted=int(row.cells_pelleted),
            genomes_input=int(row.genomes_input),
            reads_per_replicate=int(row.reads_per_replicate),
            skew_model="lognormal",
            skew_param=float(row.skew_param),
            design_edit_kappa=float(row.design_edit_kappa),
            assignment_model="poisson",
        )
        out = row.to_dict()
        out["reads_dropout_frac_mean"] = float(mc["reads_dropout_frac"].mean())
        out["edited_retained_p10_mean"] = float(mc["edited_retained_p10"].mean())
        out["reads_p10_mean"] = float(mc["reads_p10"].mean())
        out["reads_gini_mean"] = float(mc["reads_gini"].mean())
        rescored_rows.append(out)

    res = pd.DataFrame(rescored_rows)

    mask = np.ones(len(res), dtype=bool)
    if "reads_dropout_frac_max" in target_constraints:
        mask &= res["reads_dropout_frac_mean"] <= target_constraints["reads_dropout_frac_max"]
    if "edited_retained_p10_min" in target_constraints:
        mask &= res["edited_retained_p10_mean"] >= target_constraints["edited_retained_p10_min"]
    if "reads_p10_min" in target_constraints:
        mask &= res["reads_p10_mean"] >= target_constraints["reads_p10_min"]

    filtered = res[mask].copy()
    if len(filtered) == 0:
        return res.sort_values("reads_dropout_frac_mean").reset_index(drop=True)
    return filtered.sort_values("reads_dropout_frac_mean").reset_index(drop=True)
