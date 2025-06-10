import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ── 1. Load Cleaned Dataset ──
df = pd.read_csv("/Users/ds39/PycharmProjects/MAVE_Project/final_combined_plasmid_data_cleaned.csv")

# ── 2. Drop identifiers and non-numeric ──
df = df.drop(columns=['Hdr Vector Lot', 'Pass Fail'], errors='ignore')
df = df.applymap(lambda val: val[0] if isinstance(val, (list, np.ndarray)) else val)
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna().reset_index(drop=True)

# ── 3. Set target and features ──
target_col = 'Average_mapped_reads'
X_full = df.drop(columns=[target_col])
y = df[target_col]

# ── 4. SHAP feature selection ──
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

# ── 5. Train final model ──
X_train = X_train_full[top_features]
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X_train, y_train)

# ── 6. Streamlit UI ──
st.title("🔬 Interactive MAVE Model")
st.markdown("Adjust sliders to simulate feature inputs and predict **Average Mapped Reads**.")

# Sliders
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

# Predict
input_df = pd.DataFrame([user_inputs])
pred = model.predict(input_df)[0]

# Display prediction
st.markdown(f"### 📈 Predicted Average Mapped Reads: **{pred:.2f}**")

# Plot
fig, ax = plt.subplots(figsize=(6, 1.8))
sns.barplot(x=[pred], y=["Prediction"], color='lightblue', ax=ax)
ax.axvline(x=40, color='red', linestyle='--', label='Cutoff (40)')
ax.set_xlim(0, max(y) * 1.2)
ax.set_xlabel("Mapped Reads")
ax.set_ylabel("")
ax.legend()
sns.despine(left=True, bottom=True)
plt.tight_layout()
st.pyplot(fig)
