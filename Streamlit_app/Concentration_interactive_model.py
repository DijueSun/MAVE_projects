import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ── 1. Load and preprocess ──
df = pd.read_csv("final_combined_plasmid_data_cleaned.csv")
df.columns = df.columns.str.strip()

# Drop known leak columns
leak_cols = ['Pass Fail', 'Hdr Vector Lot', 'HDR_nanodrop (ng/ul)', 'gRNA_nanodrop (ng/ul)']
df.drop(columns=[c for c in leak_cols if c in df.columns], inplace=True, errors='ignore')

# Flatten and convert
df = df.applymap(lambda v: v[0] if isinstance(v, (list, np.ndarray)) and len(v) > 0 else v)
df = df.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)

# Target and feature set
target_col = 'Average_mapped_reads'
y = df[target_col]
X_full = df.drop(columns=[target_col])

# ── 2. SHAP feature selection ──
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, test_size=0.2, random_state=0)
model_initial = RandomForestRegressor(n_estimators=100, random_state=0)
model_initial.fit(X_train_full, y_train)

explainer = shap.Explainer(model_initial, X_valid_full)
shap_values = explainer(X_valid_full).values
mean_abs_shap = np.abs(shap_values).mean(axis=0)

feature_importance = pd.DataFrame({
    'feature': X_full.columns,
    'importance': mean_abs_shap
}).sort_values(by='importance', ascending=False)

top_n = 12
top_features = feature_importance['feature'].head(top_n).tolist()

# ── 3. Final model ──
X_train = X_train_full[top_features]
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X_train, y_train)

# ─────────────────────────────────────
# STREAMLIT APP UI
# ─────────────────────────────────────
st.title("🧪 Mapped Read Simulation App")
st.markdown("Adjust top SHAP features and predict **Average Mapped Reads**.\nRed line = 40-read threshold.")

# Sliders for top features
user_input = {}
for feat in top_features:
    min_val = float(np.percentile(X_train[feat], 1))
    max_val = float(np.percentile(X_train[feat], 99))
    median_val = float(np.median(X_train[feat]))
    user_input[feat] = st.slider(
        label=feat,
        min_value=round(min_val, 2),
        max_value=round(max_val, 2),
        value=round(median_val, 2),
        step=(max_val - min_val) / 100
    )

# Predict
input_df = pd.DataFrame([user_input])
pred = model.predict(input_df)[0]

# ── 4. Show bar plot for prediction ──
st.markdown(f"### 📈 Predicted Average Mapped Reads: **{pred:.2f}**")

fig, ax = plt.subplots(figsize=(6, 1.5))
sns.barplot(x=[pred], y=["Prediction"], color='skyblue', ax=ax)
ax.axvline(x=40, color='red', linestyle='--', label='Cutoff (40 Reads)')
ax.set_xlim(0, max(100, y.max() * 1.1))
ax.set_xlabel("Mapped Reads")
ax.set_ylabel("")
ax.legend()
sns.despine(left=True, bottom=True)
plt.tight_layout()
st.pyplot(fig)

# ── 5. SHAP Bar Plot ──
st.subheader("📊 SHAP Feature Importance (Bar)")
fig_bar, ax_bar = plt.subplots(figsize=(8, 6))
ax_bar.barh(
    feature_importance['feature'].head(top_n)[::-1],
    feature_importance['importance'].head(top_n)[::-1],
    color='cornflowerblue'
)
ax_bar.set_xlabel("Mean |SHAP value|")
ax_bar.set_title("Top SHAP Features")
plt.tight_layout()
st.pyplot(fig_bar)

# ── 6. SHAP Beeswarm ──
st.subheader("🐝 SHAP Beeswarm Plot")
fig_swarm, ax_swarm = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, X_valid_full, feature_names=X_full.columns, plot_type="dot", show=False)
plt.subplots_adjust(left=0.3)
st.pyplot(fig_swarm)


