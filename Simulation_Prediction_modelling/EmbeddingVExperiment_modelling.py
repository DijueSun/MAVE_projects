# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator, TransformerMixin

# ── 1) DummyTargetEmbedder ──────────────────────────────────────────────
class DummyTargetEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, n_dim=128):
        self.n_dim = n_dim
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        M = np.zeros((len(X), self.n_dim))
        for i, s in enumerate(X["target_region"]):
            h = (hash(s) % 10000)/10000
            M[i] = h * np.linspace(1, 2, self.n_dim)
        return M

# ── 2) Load pipeline + data ─────────────────────────────────────────────
@st.cache(allow_output_mutation=True)
def load_all():
    # a) load your saved pipeline
    pipe = joblib.load("model_outputs/final_target_only.pkl")

    # b) raw CSV + target
    df = pd.read_csv(
        "/Users/ds39/Documents/Sunny/MAVE/RD_projects/"
        "Embedding/experimental_with_embeddings.csv"
    )
    df["y"] = (df["Average_mapped_reads"] >= 40).astype(int)

    # c) drop columns you never want in UI
    skip = [
      "Hdr Vector Lot","Pass Fail",
      "D4R1 Mapped Reads","D4R2 Mapped Reads","D4R3 Mapped Reads",
      "HDR_nanodrop (ng/ul)","gRNA_nanodrop (ng/ul)",
      "Average_mapped_reads","Targeton Name"
    ]
    df2 = df.drop(columns=[c for c in skip if c in df], errors="ignore")

    # d) pick only the numeric assay columns (exclude any _emb_ dims!)
    all_nums = df2.select_dtypes(include="number").columns.tolist()
    emb_prefixes = ("gRNA_emb_", "primer_bases_emb_", "target_region_emb_")
    assay_cols = [c for c in all_nums if not c.startswith(emb_prefixes)]

    # e) train‐set positives
    pos = df2[df["y"]==1]
    assays_pos = pos[assay_cols].copy()

    # f) build embeddings + kNN
    embedder = DummyTargetEmbedder(n_dim=128)
    E_pos = embedder.fit_transform(pos[["target_region"]])
    nn   = NearestNeighbors(n_neighbors=5).fit(E_pos)

    return pipe, embedder, nn, assays_pos, assay_cols

pipe, embedder, nn, assays_pos, assay_cols = load_all()


# ── 3) Streamlit UI ──────────────────────────────────────────────────────
st.title("🔬 Recipe Recommender for ≥40% Mapping Reads")

st.markdown(
    "Paste your **new target_region** sequence below, then click **Recommend me a recipe**:\n"
    "- We’ll find the 5 nearest ≥40% successes in embedding‐space  \n"
    "- Show you the **median** experimental‐parameter values  \n"
    "- And compute the model’s P(≥40% mapping) under that recipe"
)

seq = st.text_area("Target Region Sequence", height=150)

if st.button("Recommend me a recipe"):
    if not seq.strip():
        st.error("Please enter a sequence.")
    else:
        # 1) embed your new seq
        emb = embedder.transform(pd.DataFrame({"target_region":[seq]}))

        # 2) find 5 nearest successful neighbors
        _, idxs = nn.kneighbors(emb, return_distance=True)
        rec = assays_pos.iloc[idxs[0]].median()

        st.subheader("🧪 Recommended Experimental Settings")
        for feat in assay_cols:
            st.write(f"• **{feat}** = {rec[feat]:.3f}")

        # 3) forward‐predict P(≥40%) with that recipe
        X_in    = pd.DataFrame([rec[assay_cols]])
        X_full  = pd.concat([pd.DataFrame({"target_region":[seq]}), X_in], axis=1)
        p       = pipe.predict_proba(X_full)[0,1]

        st.markdown(f"**Predicted P(≥40% mapping)** = **{100*p:.1f}%**")

