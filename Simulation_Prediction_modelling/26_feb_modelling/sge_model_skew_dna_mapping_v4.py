
import math
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# -----------------------------
# Utilities
# -----------------------------

def hill(x: float, K: float, n: float = 2.0) -> float:
    x = float(max(x, 0.0))
    K = float(max(K, 1e-12))
    return (x**n) / (K**n + x**n)

def effective_transfected_cells(cells_plated: int, transfection_rate: float) -> int:
    """Convert plated cells -> effectively transfected cells."""
    return int(max(0, math.floor(int(cells_plated) * float(np.clip(transfection_rate, 0.0, 1.0)))))

def precise_hdr_fraction_population(
    hdr_rate: float,
    transfection_rate: float,
    precise_hdr_given_hdr: float = 1.0,
) -> float:
    """Population-level precise HDR fraction under a simple decomposition.

    Assumptions:
    - `hdr_rate` is the HDR fraction among effectively transfected cells.
    - `precise_hdr_given_hdr` is the fraction of HDR edits that are the exact intended allele.
    - Untransfected cells contribute ~0 precise HDR.
    """
    hdr_rate = float(np.clip(hdr_rate, 0.0, 1.0))
    transfection_rate = float(np.clip(transfection_rate, 0.0, 1.0))
    precise_hdr_given_hdr = float(np.clip(precise_hdr_given_hdr, 0.0, 1.0))
    return float(np.clip(hdr_rate * transfection_rate * precise_hdr_given_hdr, 0.0, 1.0))

def precise_hdr_events_haploid(
    n_cells: int,
    precise_hdr_fraction: float,
) -> float:
    """Expected number of precise HDR events in a haploid system."""
    n_cells = int(max(0, int(n_cells)))
    precise_hdr_fraction = float(np.clip(precise_hdr_fraction, 0.0, 1.0))
    return float(n_cells * precise_hdr_fraction)

def required_cells_for_precise_hdr_events(
    target_precise_hdr_events: float,
    precise_hdr_fraction: float,
) -> float:
    """Cells required to achieve a target number of precise HDR events."""
    target_precise_hdr_events = float(max(target_precise_hdr_events, 0.0))
    precise_hdr_fraction = float(np.clip(precise_hdr_fraction, 0.0, 1.0))
    if precise_hdr_fraction <= 0.0:
        return float(np.inf)
    return float(target_precise_hdr_events / precise_hdr_fraction)

def conditional_precise_hdr_fraction_given_transfection(
    precise_hdr_fraction_population: float,
    transfection_rate: float,
) -> float:
    """Back-calculate precise HDR fraction among transfected cells."""
    precise_hdr_fraction_population = float(np.clip(precise_hdr_fraction_population, 0.0, 1.0))
    transfection_rate = float(np.clip(transfection_rate, 0.0, 1.0))
    if transfection_rate <= 0.0:
        return 0.0
    return float(np.clip(precise_hdr_fraction_population / transfection_rate, 0.0, 1.0))

def _beta_params_from_mean_kappa(mean: float, kappa: float):
    mean = float(np.clip(mean, 1e-6, 1 - 1e-6))
    kappa = float(max(kappa, 2.0))
    a = mean * kappa
    b = (1 - mean) * kappa
    return a, b

# -----------------------------
# 1) DNA -> HDR rate mapping
# -----------------------------

def dna_to_hdr_rate(
    hdr_ng: float,
    sgrna_ng: float,
    r_min: float = 0.01,
    r_max: float = 0.60,
    K_hdr: float = 500.0,
    K_sg: float = 300.0,
    ratio_opt: float = 2.0,
    ratio_sigma: float = 1.0,
    hill_n: float = 2.0,
) -> float:
    """Predicted HDR/editing rate (0..1) from DNA inputs.

    NOTE: prior/hypothesis until calibrated with real data.
    ratio term is a Gaussian bump centered at ratio_opt.
    """
    hdr_ng = float(max(hdr_ng, 0.0))
    sgrna_ng = float(max(sgrna_ng, 1e-12))
    ratio = hdr_ng / sgrna_ng

    ratio_sigma = float(max(ratio_sigma, 1e-6))
    ratio_term = float(np.exp(-((ratio - ratio_opt) ** 2) / (2.0 * ratio_sigma ** 2)))

    hdr_component = float(hill(hdr_ng, K_hdr, hill_n))
    sgrna_component = float(hill(sgrna_ng, K_sg, hill_n))

    rate = r_min + (r_max - r_min) * hdr_component * sgrna_component * ratio_term
    return float(np.clip(rate, 0.0, 1.0))

# -----------------------------
# 2) Mapping rate model (Beta variability)
# -----------------------------

def sample_mapping_rate(
    mapping_mean: float = 0.55,
    mapping_kappa: float = 60.0,
    mapping_min: float = 0.0,
    rng: np.random.Generator | None = None,
) -> float:
    """Sample mapping rate with optional lower bound (QC threshold)."""
    rng = rng or np.random.default_rng()
    a, b = _beta_params_from_mean_kappa(mapping_mean, mapping_kappa)

    # Simple rejection sampling to enforce mapping_min (fast for reasonable thresholds).
    mapping_min = float(np.clip(mapping_min, 0.0, 1.0))
    for _ in range(2000):
        r = float(rng.beta(a, b))
        if r >= mapping_min:
            return r
    # Fallback if threshold is too strict for the distribution:
    return float(max(mapping_min, mapping_mean))

# -----------------------------
# 3) Library skew model
# -----------------------------

def generate_library_distribution(library_size: int, skew_sigma: float = 0.5, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    library_size = int(library_size)
    skew_sigma = float(max(skew_sigma, 0.0))
    weights = rng.lognormal(mean=0.0, sigma=skew_sigma, size=library_size)
    return weights / weights.sum()

# -----------------------------
# 4) Full stochastic simulation
# -----------------------------

def simulate_once(
    hdr_ng: float,
    sgrna_ng: float,
    ratio_opt: float = 2.0,
    cells_transfected: int = 1_000_000,
    library_size: int = 2_000,
    skew_sigma: float = 0.5,
    reads_total: int = 5_000_000,
    mapping_rate: float | None = None,
    mapping_mean: float = 0.55,
    mapping_kappa: float = 60.0,
    mapping_min: float = 0.0,
    rng: np.random.Generator | None = None,
    return_vectors: bool = False,
) -> dict:
    """One stochastic replicate.

    - If mapping_rate is None, sample mapping_rate ~ Beta(mean,kappa) subject to mapping_min.
    - Usable reads = floor(reads_total * mapping_rate).
    """
    rng = rng or np.random.default_rng()

    hdr_rate = dna_to_hdr_rate(hdr_ng, sgrna_ng, ratio_opt=ratio_opt)

    if mapping_rate is None:
        mapping_rate = sample_mapping_rate(mapping_mean=mapping_mean, mapping_kappa=mapping_kappa, mapping_min=mapping_min, rng=rng)
    else:
        mapping_rate = float(np.clip(mapping_rate, 0.0, 1.0))

    reads_total = int(reads_total)
    reads_usable = int(max(0, math.floor(reads_total * mapping_rate)))

    plasmid_dist = generate_library_distribution(library_size, skew_sigma=skew_sigma, rng=rng)

    cells_transfected = int(cells_transfected)
    cell_alloc = rng.multinomial(cells_transfected, plasmid_dist)

    edited = rng.binomial(cell_alloc, hdr_rate)

    if edited.sum() == 0 or reads_usable == 0:
        out = {
            "hdr_rate": float(hdr_rate),
            "mapping_rate": float(mapping_rate),
            "reads_total": reads_total,
            "reads_usable": reads_usable,
            "dropout_frac": 1.0,
            "p10_reads": 0.0,
        }
        if return_vectors:
            out["read_counts"] = np.zeros(int(library_size), dtype=int)
            out["edited_counts"] = edited.astype(int)
            out["cell_alloc"] = cell_alloc.astype(int)
        return out

    read_dist = edited / edited.sum()
    read_counts = rng.multinomial(reads_usable, read_dist)

    dropout_frac = float(np.mean(read_counts == 0))
    p10_reads = float(np.percentile(read_counts, 10))

    out = {
        "hdr_rate": float(hdr_rate),
        "mapping_rate": float(mapping_rate),
        "reads_total": reads_total,
        "reads_usable": reads_usable,
        "dropout_frac": dropout_frac,
        "p10_reads": p10_reads,
    }
    if return_vectors:
        out["read_counts"] = read_counts.astype(int)
        out["edited_counts"] = edited.astype(int)
        out["cell_alloc"] = cell_alloc.astype(int)
    return out

# -----------------------------
# 5) Monte Carlo wrapper (means + quantiles)
# -----------------------------

def run_monte_carlo(n_reps: int = 50, q=(0.05, 0.5, 0.95), **kwargs) -> dict:
    results = [simulate_once(**kwargs) for _ in range(int(n_reps))]

    dropout = np.array([r["dropout_frac"] for r in results], dtype=float)
    p10 = np.array([r["p10_reads"] for r in results], dtype=float)
    mapping = np.array([r["mapping_rate"] for r in results], dtype=float)
    hdr_rate_vals = np.array([r["hdr_rate"] for r in results], dtype=float)
    reads_usable = np.array([r["reads_usable"] for r in results], dtype=float)

    out = {
        "dropout_mean": float(dropout.mean()),
        "p10_reads_mean": float(p10.mean()),
        "mapping_rate_mean": float(mapping.mean()),
        "hdr_rate_mean": float(hdr_rate_vals.mean()),
        "reads_usable_mean": float(reads_usable.mean()),
        "dropout_q": {str(qq): float(np.quantile(dropout, qq)) for qq in q},
        "p10_reads_q": {str(qq): float(np.quantile(p10, qq)) for qq in q},
        "mapping_rate_q": {str(qq): float(np.quantile(mapping, qq)) for qq in q},
        "reads_usable_q": {str(qq): float(np.quantile(reads_usable, qq)) for qq in q},
    }
    return out

# -----------------------------
# 6) Synthetic dataset + surrogates
# -----------------------------

def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def _inv_logit(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))

def build_synthetic_dataset(
    n_samples: int = 500,
    n_reps_per_sample: int = 15,
    hdr_range=(100.0, 2000.0),
    sgrna_range=(50.0, 1500.0),
    skew_range=(0.1, 1.5),
    mapping_range=(0.30, 0.70),
    reads_total: int = 3_000_000,
    cells_transfected: int = 5_600_000,
    library_size: int = 2_000,
    library_size_range=None,
    ratio_opt: float = 2.0,
    rng: np.random.Generator | None = None,
):
    """Synthetic training rows.

    Features: HDR_ng, sgRNA_ng, ratio, skew_sigma, mapping_rate, reads_total, cells_transfected, library_size
    Targets: dropout_mean, p10_reads_mean, hdr_rate_mean, reads_usable_mean
    """
    rng = rng or np.random.default_rng()
    rows = []
    for _ in range(int(n_samples)):
        hdr = float(rng.uniform(*hdr_range))
        sgrna = float(rng.uniform(*sgrna_range))
        ratio = hdr / max(sgrna, 1e-12)
        skew = float(rng.uniform(*skew_range))
        mapping_rate = float(rng.uniform(*mapping_range))
        if library_size_range is not None:
            lib_low, lib_high = library_size_range
            if lib_high < lib_low:
                lib_high = lib_low
            library_size_sample = int(rng.integers(int(lib_low), int(lib_high) + 1))
        else:
            library_size_sample = int(library_size)

        mc = run_monte_carlo(
            n_reps=n_reps_per_sample,
            hdr_ng=hdr,
            sgrna_ng=sgrna,
            ratio_opt=ratio_opt,
            skew_sigma=skew,
            reads_total=int(reads_total),
            mapping_rate=mapping_rate,  # fixed per sample
            cells_transfected=int(cells_transfected),
            library_size=int(library_size_sample),
        )

        rows.append({
            "HDR_ng": hdr,
            "sgRNA_ng": sgrna,
            "ratio": ratio,
            "skew_sigma": skew,
            "mapping_rate": mapping_rate,
            "reads_total": int(reads_total),
            "cells_transfected": int(cells_transfected),
            "library_size": int(library_size_sample),
            "dropout_mean": mc["dropout_mean"],
            "p10_reads_mean": mc["p10_reads_mean"],
            "hdr_rate_mean": mc["hdr_rate_mean"],
            "reads_usable_mean": mc["reads_usable_mean"],
        })
    return rows

def fit_surrogate_models(rows: list[dict], alpha: float = 1.0):
    """Two Ridge surrogates:
      - dropout_model predicts logit(dropout_mean)
      - p10_model predicts log1p(p10_reads_mean)
    """
    import pandas as pd
    df = pd.DataFrame(rows)

    feature_names = ["HDR_ng", "sgRNA_ng", "ratio", "skew_sigma", "mapping_rate", "reads_total", "cells_transfected", "library_size"]
    X = df[feature_names].to_numpy(dtype=float)

    y_dropout = _logit(df["dropout_mean"].to_numpy(dtype=float))
    y_p10 = np.log1p(df["p10_reads_mean"].to_numpy(dtype=float))

    dropout_model = make_pipeline(StandardScaler(), Ridge(alpha=float(alpha)))
    p10_model = make_pipeline(StandardScaler(), Ridge(alpha=float(alpha)))

    dropout_model.fit(X, y_dropout)
    p10_model.fit(X, y_p10)

    return dropout_model, p10_model, feature_names

def predict_with_surrogates(dropout_model, p10_model, feature_names, rows: list[dict]):
    import pandas as pd
    df = pd.DataFrame(rows)
    X = df[feature_names].to_numpy(dtype=float)

    df["dropout_pred"] = _inv_logit(dropout_model.predict(X))
    df["p10_reads_pred"] = np.expm1(p10_model.predict(X))
    return df

# -----------------------------
# 7) Option B: surrogate search + MC verification with constraints
# -----------------------------

def suggest_experiments_surrogate_verified(
    dropout_model,
    p10_model,
    feature_names,
    n_candidates: int = 20_000,
    top_k_to_verify: int = 50,
    target_dropout: float = 0.02,
    min_p10_reads: float = 100.0,
    # HARD constraints:
    min_mapping_rate: float = 0.30,
    min_reads_total: int = 3_000_000,
    hdr_range=(100.0, 2000.0),
    sgrna_range=(50.0, 1500.0),
    skew_range=(0.1, 1.5),
    mapping_range=(0.30, 0.70),
    reads_total: int = 3_000_000,
    cells_transfected: int = 5_600_000,
    library_size: int = 2_000,
    library_size_range=None,
    ratio_opt: float = 2.0,
    n_reps_verify: int = 40,
    rng: np.random.Generator | None = None,
):
    """Generate candidates with surrogates, enforce QC constraints, then verify by Monte Carlo."""
    import pandas as pd
    rng = rng or np.random.default_rng()

    reads_total = int(max(int(reads_total), int(min_reads_total)))
    min_mapping_rate = float(min_mapping_rate)
    mapping_low = max(float(mapping_range[0]), min_mapping_rate)
    mapping_high = float(mapping_range[1])
    if mapping_high < mapping_low:
        mapping_high = mapping_low

    cand_rows = []
    for _ in range(int(n_candidates)):
        hdr = float(rng.uniform(*hdr_range))
        sgrna = float(rng.uniform(*sgrna_range))
        ratio = hdr / max(sgrna, 1e-12)
        skew = float(rng.uniform(*skew_range))
        mapping_rate = float(rng.uniform(mapping_low, mapping_high))
        if library_size_range is not None:
            lib_low, lib_high = library_size_range
            if lib_high < lib_low:
                lib_high = lib_low
            library_size_sample = int(rng.integers(int(lib_low), int(lib_high) + 1))
        else:
            library_size_sample = int(library_size)

        cand_rows.append({
            "HDR_ng": hdr,
            "sgRNA_ng": sgrna,
            "ratio": ratio,
            "skew_sigma": skew,
            "mapping_rate": mapping_rate,
            "reads_total": reads_total,
            "cells_transfected": int(cells_transfected),
            "library_size": int(library_size_sample),
        })

    df_pred = predict_with_surrogates(dropout_model, p10_model, feature_names, cand_rows)

    # Apply predicted constraints
    df_ok = df_pred[(df_pred["dropout_pred"] <= float(target_dropout)) & (df_pred["p10_reads_pred"] >= float(min_p10_reads))].copy()
    if len(df_ok) == 0:
        df_ok = df_pred.copy()

    # Objective: minimize dropout_pred, maximize p10_reads_pred
    df_ok.sort_values(["dropout_pred", "p10_reads_pred"], ascending=[True, False], inplace=True)
    df_top = df_ok.head(int(top_k_to_verify)).copy()

    # Verify with Monte Carlo
    sim_rows = []
    for _, row in df_top.iterrows():
        mc = run_monte_carlo(
            n_reps=n_reps_verify,
            hdr_ng=float(row["HDR_ng"]),
            sgrna_ng=float(row["sgRNA_ng"]),
            ratio_opt=ratio_opt,
            skew_sigma=float(row["skew_sigma"]),
            reads_total=int(row["reads_total"]),
            mapping_rate=float(row["mapping_rate"]),
            cells_transfected=int(row["cells_transfected"]),
            library_size=int(row["library_size"]),
        )
        sim_rows.append({
            **row.to_dict(),
            "hdr_rate_pred": dna_to_hdr_rate(float(row["HDR_ng"]), float(row["sgRNA_ng"]), ratio_opt=ratio_opt),
            "mapping_rate_assumed": float(row["mapping_rate"]),
            "reads_usable_expected": float(int(row["reads_total"]) * float(row["mapping_rate"])),
            "dropout_sim_mean": mc["dropout_mean"],
            "p10_reads_sim_mean": mc["p10_reads_mean"],
            "mapping_rate_mean": mc["mapping_rate_mean"],
            "reads_usable_mean": mc["reads_usable_mean"],
            "dropout_sim_q05": mc["dropout_q"]["0.05"],
            "dropout_sim_q95": mc["dropout_q"]["0.95"],
            "p10_sim_q05": mc["p10_reads_q"]["0.05"],
            "p10_sim_q95": mc["p10_reads_q"]["0.95"],
        })

    df_verified = pd.DataFrame(sim_rows)
    df_verified.sort_values(["dropout_sim_mean", "p10_reads_sim_mean"], ascending=[True, False], inplace=True)
    return df_verified
