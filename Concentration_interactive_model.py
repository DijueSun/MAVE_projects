import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ── 1. Load dataset ──
path = "AllPlasmidsQCResults.csv"
df = pd.read_csv(path)

# ── 2. Preprocessing ──
cols_to_drop = ['Targeton Name', 'Gene', 'Screen ID', 'Hdr Vector Lot', 'Comment',
                'gRNA Vector Lot', 'D4R1 Mapped Reads', 'D4R2 Mapped Reads', 'D4R3 Mapped Reads']
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
df.fillna(0, inplace=True)

target_col = 'Average_mapped_reads'
X_full = df.drop(columns=[target_col])
y = df[target_col]

# ── 3. SHAP Feature Selection ──
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, test_size=0.2, random_state=0)
model_initial = RandomForestRegressor(n_estimators=100, random_state=0)
model_initial.fit(X_train_full, y_train)

explainer = shap.TreeExplainer(model_initial)
shap_values = explainer.shap_values(X_valid_full)
mean_abs_shap = np.abs(shap_values).mean(axis=0)

feature_importance = pd.DataFrame({
    'feature': X_full.columns,
    'importance': mean_abs_shap
}).sort_values(by='importance', ascending=False)

top_n = 12
top_features = feature_importance['feature'].head(top_n).tolist()

# ── 4. Train Final Model ──
X_train = X_train_full[top_features]
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X_train, y_train)

# ── 5. Streamlit UI ──
st.title("🧪 Concentration Interactive Model")
st.markdown("Adjust sliders to simulate **feature values** and predict the **Average Mapped Reads**.")

# Create sliders dynamically
user_inputs = {}
for feat in top_features:
    min_val = float(X_train[feat].min())
    max_val = float(X_train[feat].max())
    median_val = float(X_train[feat].median())

    user_inputs[feat] = st.slider(
        label=feat,
        min_value=min_val,
        max_value=max_val,
        value=median_val,
        step=(max_val - min_val) / 100
    )

# Predict based on input
input_df = pd.DataFrame([user_inputs])
pred = model.predict(input_df)[0]

# Show prediction and bar
st.markdown(f"### 📈 Predicted Average Mapped Reads: **{pred:.2f}**")

fig, ax = plt.subplots(figsize=(5, 1.5))
sns.barplot(x=[pred], y=["Prediction"], color='skyblue', ax=ax)

# Add 40-read threshold
ax.axvline(x=40, color='red', linestyle='--', label='Cutoff (40)')
ax.set_xlim(0, max(y) * 1.2)
ax.set_xlabel("Mapped Reads")
ax.set_ylabel("")
ax.legend()
sns.despine(left=True, bottom=True)
plt.tight_layout()

st.pyplot(fig)


