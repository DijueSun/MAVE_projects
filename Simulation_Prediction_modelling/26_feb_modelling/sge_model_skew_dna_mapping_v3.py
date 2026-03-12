
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

def _beta_params_from_mean_kappa(mean: float, kappa: float):
    mean = float(np.clip(mean, 1e-6, 1 - 1e-6))
    kappa = float(max(kappa, 2.0))
    a = mean * kappa
    b = (1 - mean) * kappa
    return a, b

# -----------------------------
# 1) DNA -> HDR rate mapping (Hill-like with ratio term)
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
    """Return a predicted HDR/editing rate (0..1) from DNA inputs.

    Notes:
      - This is a prior (hypothesis) until calibrated with real data.
      - ratio_term is a Gaussian bump centered at ratio_opt.
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

def sample_mapping_rate(mapping_mean: float = 0.55, mapping_kappa: float = 60.0, rng: np.random.Generator | None = None) -> float:
    """Sample a mapping rate (0..1) around mapping_mean.
    mapping_kappa controls variability: higher = tighter around the mean.
    """
    rng = rng or np.random.default_rng()
    a, b = _beta_params_from_mean_kappa(mapping_mean, mapping_kappa)
    return float(rng.beta(a, b))

# -----------------------------
# 3) Library skew model
# -----------------------------

def generate_library_distribution(library_size: int, skew_sigma: float = 0.5, rng: np.random.Generator | None = None) -> np.ndarray:
    """Lognormal weights -> normalized probabilities."""
    rng = rng or np.random.default_rng()
    library_size = int(library_size)
    skew_sigma = float(max(skew_sigma, 0.0))
    weights = rng.lognormal(mean=0.0, sigma=skew_sigma, size=library_size)
    probs = weights / weights.sum()
    return probs

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
    mapping_mean: float = 0.55,
    mapping_kappa: float = 60.0,
    mapping_rate: float | None = None,
    rng: np.random.Generator | None = None,
    return_vectors: bool = False,
) -> dict:
    """One stochastic replicate.

    If mapping_rate is None, samples mapping_rate ~ Beta(mapping_mean, mapping_kappa).
    Uses reads_usable = floor(reads_total * mapping_rate).
    """
    rng = rng or np.random.default_rng()

    # Predicted HDR/editing rate from DNA inputs
    hdr_rate = dna_to_hdr_rate(hdr_ng, sgrna_ng, ratio_opt=ratio_opt)

    # Mapping rate (predicted/assumed per replicate)
    if mapping_rate is None:
        mapping_rate = sample_mapping_rate(mapping_mean=mapping_mean, mapping_kappa=mapping_kappa, rng=rng)
    else:
        mapping_rate = float(np.clip(mapping_rate, 0.0, 1.0))

    reads_usable = int(max(0, math.floor(int(reads_total) * mapping_rate)))

    # Skewed plasmid distribution across library elements
    plasmid_dist = generate_library_distribution(library_size, skew_sigma=skew_sigma, rng=rng)

    # Allocate transfected cells across designs
    cells_transfected = int(cells_transfected)
    cell_alloc = rng.multinomial(cells_transfected, plasmid_dist)

    # Editing per design (binomial)
    edited = rng.binomial(cell_alloc, hdr_rate)

    if edited.sum() == 0 or reads_usable == 0:
        out = {
            "hdr_rate": hdr_rate,
            "mapping_rate": mapping_rate,
            "reads_total": int(reads_total),
            "reads_usable": reads_usable,
            "dropout_frac": 1.0,
            "p10_reads": 0.0,
        }
        if return_vectors:
            out["read_counts"] = np.zeros(library_size, dtype=int)
            out["edited_counts"] = edited.astype(int)
            out["cell_alloc"] = cell_alloc.astype(int)
        return out

    # Allocate usable reads proportional to edited counts
    read_dist = edited / edited.sum()
    read_counts = rng.multinomial(reads_usable, read_dist)

    dropout_frac = float(np.mean(read_counts == 0))
    p10_reads = float(np.percentile(read_counts, 10))

    out = {
        "hdr_rate": hdr_rate,
        "mapping_rate": mapping_rate,
        "reads_total": int(reads_total),
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
# 5) Monte Carlo wrapper (returns means + quantiles)
# -----------------------------

def run_monte_carlo(n_reps: int = 50, q=(0.05, 0.5, 0.95), **kwargs) -> dict:
    results = [simulate_once(**kwargs) for _ in range(int(n_reps))]

    dropout = np.array([r["dropout_frac"] for r in results], dtype=float)
    p10 = np.array([r["p10_reads"] for r in results], dtype=float)
    mapping = np.array([r["mapping_rate"] for r in results], dtype=float)

    # hdr_rate is deterministic given hdr_ng/sgrna_ng in current implementation,
    # but keep the same interface.
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
    }
    return out

# -----------------------------
# 6) Synthetic dataset + surrogate models
# -----------------------------

def build_synthetic_dataset(
    n_samples: int = 500,
    n_reps_per_sample: int = 15,
    hdr_range=(100.0, 2000.0),
    sgrna_range=(50.0, 1500.0),
    skew_range=(0.1, 1.5),
    mapping_range=(0.40, 0.70),
    reads_total: int = 1_000_000,
    cells_transfected: int = 200_000,
    library_size: int = 2_000,
    ratio_opt: float = 2.0,
    rng: np.random.Generator | None = None,
) -> "np.ndarray":
    """Return a DataFrame-like dict of features and targets.

    Features: HDR_ng, sgRNA_ng, ratio, skew_sigma, mapping_rate, reads_total, cells_transfected, library_size
    Targets (from Monte Carlo): dropout_mean, p10_reads_mean, hdr_rate_mean, reads_usable_mean
    """
    rng = rng or np.random.default_rng()

    rows = []
    for _ in range(int(n_samples)):
        hdr = float(rng.uniform(*hdr_range))
        sgrna = float(rng.uniform(*sgrna_range))
        skew = float(rng.uniform(*skew_range))
        mapping_rate = float(rng.uniform(*mapping_range))

        mc = run_monte_carlo(
            n_reps=n_reps_per_sample,
            hdr_ng=hdr,
            sgrna_ng=sgrna,
            ratio_opt=ratio_opt,
            skew_sigma=skew,
            reads_total=int(reads_total),
            mapping_rate=mapping_rate,  # fixed per sample
            cells_transfected=int(cells_transfected),
            library_size=int(library_size),
        )

        ratio = hdr / max(sgrna, 1e-12)

        rows.append({
            "HDR_ng": hdr,
            "sgRNA_ng": sgrna,
            "ratio": ratio,
            "skew_sigma": skew,
            "mapping_rate": mapping_rate,
            "reads_total": int(reads_total),
            "cells_transfected": int(cells_transfected),
            "library_size": int(library_size),
            "dropout_mean": mc["dropout_mean"],
            "p10_reads_mean": mc["p10_reads_mean"],
            "hdr_rate_mean": mc["hdr_rate_mean"],
            "reads_usable_mean": mc["reads_usable_mean"],
        })

    return rows

def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def _inv_logit(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))

def fit_surrogate_models(rows: list[dict], alpha: float = 1.0):
    """Fits two Ridge models:
      - dropout_model predicts logit(dropout_mean)
      - p10_model predicts log1p(p10_reads_mean)

    Returns (dropout_model, p10_model, feature_names).
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

    dropout_pred = _inv_logit(dropout_model.predict(X))
    p10_pred = np.expm1(p10_model.predict(X))

    df["dropout_pred"] = dropout_pred
    df["p10_reads_pred"] = p10_pred
    return df

# -----------------------------
# 7) Option B: Surrogate search + simulation verification
# -----------------------------

def suggest_experiments_surrogate_verified(
    dropout_model,
    p10_model,
    feature_names,
    n_candidates: int = 20_000,
    top_k_to_verify: int = 50,
    target_dropout: float = 0.02,
    min_p10_reads: float = 100.0,
    hdr_range=(100.0, 2000.0),
    sgrna_range=(50.0, 1500.0),
    skew_range=(0.1, 1.5),
    mapping_range=(0.40, 0.70),
    reads_total: int = 1_000_000,
    cells_transfected: int = 200_000,
    library_size: int = 2_000,
    ratio_opt: float = 2.0,
    n_reps_verify: int = 40,
    rng: np.random.Generator | None = None,
):
    """Fast candidate generation with surrogate predictions, then Monte Carlo verification."""
    import pandas as pd
    rng = rng or np.random.default_rng()

    cand_rows = []
    for _ in range(int(n_candidates)):
        hdr = float(rng.uniform(*hdr_range))
        sgrna = float(rng.uniform(*sgrna_range))
        ratio = hdr / max(sgrna, 1e-12)
        skew = float(rng.uniform(*skew_range))
        mapping_rate = float(rng.uniform(*mapping_range))
        cand_rows.append({
            "HDR_ng": hdr,
            "sgRNA_ng": sgrna,
            "ratio": ratio,
            "skew_sigma": skew,
            "mapping_rate": mapping_rate,
            "reads_total": int(reads_total),
            "cells_transfected": int(cells_transfected),
            "library_size": int(library_size),
        })

    df_pred = predict_with_surrogates(dropout_model, p10_model, feature_names, cand_rows)

    # Filter by predicted constraints
    df_ok = df_pred[(df_pred["dropout_pred"] <= float(target_dropout)) & (df_pred["p10_reads_pred"] >= float(min_p10_reads))].copy()
    if len(df_ok) == 0:
        # If too strict, return best-by-objective without hard filtering
        df_ok = df_pred.copy()

    # Objective: minimize predicted dropout, maximize predicted p10 reads
    # We'll sort by dropout_pred then by -p10_reads_pred
    df_ok.sort_values(["dropout_pred", "p10_reads_pred"], ascending=[True, False], inplace=True)

    df_top = df_ok.head(int(top_k_to_verify)).copy()

    # Verify via Monte Carlo simulation
    sim_rows = []
    for _, row in df_top.iterrows():
        mc = run_monte_carlo(
            n_reps=n_reps_verify,
            hdr_ng=float(row["HDR_ng"]),
            sgrna_ng=float(row["sgRNA_ng"]),
            ratio_opt=ratio_opt,
            skew_sigma=float(row["skew_sigma"]),
            reads_total=int(row["reads_total"]),
            mapping_rate=float(row["mapping_rate"]),  # fixed per candidate
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
            "dropout_sim_q05": mc["dropout_q"]["0.05"],
            "dropout_sim_q95": mc["dropout_q"]["0.95"],
            "p10_sim_q05": mc["p10_reads_q"]["0.05"],
            "p10_sim_q95": mc["p10_reads_q"]["0.95"],
        })

    df_verified = pd.DataFrame(sim_rows)
    # Sort by simulated performance
    df_verified.sort_values(["dropout_sim_mean", "p10_reads_sim_mean"], ascending=[True, False], inplace=True)
    return df_verified
