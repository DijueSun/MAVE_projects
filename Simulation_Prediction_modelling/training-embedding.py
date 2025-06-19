# train_target_only_fixed.py
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.impute      import SimpleImputer
from sklearn.linear_model     import LogisticRegression
from sklearn.pipeline    import make_pipeline
from sklearn.compose     import ColumnTransformer
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.base        import BaseEstimator, TransformerMixin

# ── A) load your CSV ──────────────────────────────────────────────────────────────
df = pd.read_csv(
    "/Users/ds39/Documents/Sunny/MAVE/RD_projects/Embedding/experimental_with_embeddings.csv"
)

# binary target: ≥40% mapped?
y = (df["Average_mapped_reads"] >= 40).astype(int)

# ── B) drop everything you do NOT want as raw inputs ─────────────────────────────
to_drop = [
    "Hdr Vector Lot","Pass Fail",
    "D4R1 Mapped Reads","D4R2 Mapped Reads","D4R3 Mapped Reads",
    "HDR_nanodrop (ng/ul)","gRNA_nanodrop (ng/ul)",
    "Average_mapped_reads","Targeton","Targeton Name"
]
df2 = df.drop(columns=to_drop, errors="ignore")

# now *also* drop ALL of your precomputed embeddings
emb_cols = [
    c for c in df2.columns
    if c.startswith("gRNA_emb_")
    or c.startswith("target_region_emb_")
    or c.startswith("primer_bases_emb_")
]
df2 = df2.drop(columns=emb_cols, errors="ignore")

# ── C) separate out the raw sequence vs. the assays ──────────────────────────────
seqs      = df2[["target_region"]].copy()
assay_cols = (
    df2
    .select_dtypes(include="number")
    .dropna(axis=1, how="all")
    .columns
    .tolist()
)

# ── D) toy embedder (must match app) ────────────────────────────────────────────
class DummyTargetEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, n_dim=128):
        self.n_dim = n_dim
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        M = np.zeros((len(X), self.n_dim))
        for i, s in enumerate(X["target_region"]):
            h = (hash(s) % 10000)/10000
            M[i] = h * np.linspace(1,2,self.n_dim)
        return M

# ── E) build the pipeline ───────────────────────────────────────────────────────
pre = ColumnTransformer([
    ("embed_t", make_pipeline(DummyTargetEmbedder(n_dim=128)),
                  ["target_region"]),        # raw string in
    ("impute" , SimpleImputer(strategy="median"),
                  assay_cols),             # only your numeric assays
])

clf  = LogisticRegression(max_iter=2000, random_state=0)
pipe = make_pipeline(pre, clf)

# ── F) grouped CV on target_region to guard against leakage ─────────────────────
df["tg_id"] = df["target_region"].astype("category").cat.codes
gkf = GroupKFold(n_splits=5)
scores = cross_val_score(
    pipe,
    pd.concat([seqs, df2[assay_cols]], axis=1),
    y,
    groups=df["tg_id"],
    cv=gkf,
    scoring="accuracy"
)
print(f"▶  Leave-one-target-region-out CV = {scores.mean():.3f} ± {scores.std():.3f}")

# ── G) fit on ALL data and save ────────────────────────────────────────────────
pipe.fit(pd.concat([seqs, df2[assay_cols]], axis=1), y)
os.makedirs("../model_outputs", exist_ok=True)
joblib.dump(pipe, "model_outputs/final_target_only.pkl")
print("✅ saved model_outputs/final_target_only.pkl")


#%%
import joblib
joblib.load("model_outputs/final_target_only.pkl")
