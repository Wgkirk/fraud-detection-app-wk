"""
Streamlit Web App — IEEE-CIS Fraud Detection
=============================================

Loads the saved Random Forest pipeline + SHAP TreeExplainer produced by the
notebook, lets a user enter a transaction (or load a sample), and shows:
  1. The model's fraud probability
  2. A SHAP waterfall / bar plot explaining the prediction
  3. Top global feature importances

Run locally:
    streamlit run streamlit_app.py

Required files in ./model_artifacts/ :
    - final_random_forest_pipeline.joblib
    - model_feature_names.joblib
    - shap_tree_explainer.joblib
"""

import os
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ---- Page config ----------------------------------------------------------
st.set_page_config(
    page_title="IEEE-CIS Fraud Detection",
    page_icon="💳",
    layout="wide",
)

ART_DIR = "model_artifacts"
PIPELINE_PATH = os.path.join(ART_DIR, "final_random_forest_pipeline.joblib")
FEATURES_PATH = os.path.join(ART_DIR, "model_feature_names.joblib")
SHAP_PATH = os.path.join(ART_DIR, "shap_tree_explainer.joblib")

# ---- Cached loaders -------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    pipe = joblib.load(PIPELINE_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    shap_explainer = joblib.load(SHAP_PATH)
    return pipe, feature_names, shap_explainer


# ---- Sidebar --------------------------------------------------------------
st.sidebar.title("Fraud Detection App")
st.sidebar.write(
    "This app serves the tuned Random Forest pipeline trained on the "
    "IEEE-CIS Fraud Detection dataset. It returns a fraud probability and "
    "a SHAP-based explanation for every transaction."
)
mode = st.sidebar.radio(
    "Input mode",
    ["Sample transaction (recommended)", "Upload CSV", "Manual entry"],
    index=0,
)
threshold = st.sidebar.slider(
    "Decision threshold (probability of fraud)",
    min_value=0.05, max_value=0.95, value=0.50, step=0.05,
    help="Lower = more recall (catch more fraud, more false alerts). "
         "Higher = more precision (fewer alerts, may miss fraud).",
)

# ---- Main header ----------------------------------------------------------
st.title("💳 IEEE-CIS Fraud Detection")
st.markdown(
    "Score a transaction in real time and see **why** the model flagged it. "
    "Tune the decision threshold on the left to balance recall and precision."
)

# Try to load the artifacts up front and fail loudly if missing
try:
    pipeline, feature_names, shap_explainer = load_artifacts()
except FileNotFoundError as e:
    st.error(
        "Could not load model artifacts. Make sure `model_artifacts/` is in "
        "the working directory and contains the three .joblib files produced "
        f"by the notebook. Error: {e}"
    )
    st.stop()

st.success(
    f"Loaded pipeline with **{len(feature_names)} features** "
    f"and SHAP explainer."
)


# ---- Input collection -----------------------------------------------------
def get_sample_transaction(seed: int = 0) -> pd.DataFrame:
    """Build a synthetic transaction filled with median-ish values + a few
    realistic numbers for the most important columns."""
    rng = np.random.default_rng(seed)
    row = {col: 0.0 for col in feature_names}
    # Fill in plausible values for the columns the model relies on most
    plausible = {
        "TransactionAmt": float(rng.choice([20, 67.5, 125.0, 450.0, 1499.99])),
        "TransactionAmt_log": np.log1p(125.0),
        "TX_hour": int(rng.integers(0, 24)),
        "TX_day": int(rng.integers(0, 7)),
        "TX_week": int(rng.integers(0, 4)),
        "card1": float(rng.integers(1000, 18000)),
        "card2": float(rng.integers(100, 600)),
        "ProductCD": int(rng.integers(0, 5)),
        "addr1": float(rng.integers(100, 500)),
        "addr2": float(rng.integers(10, 90)),
        "C1": float(rng.integers(0, 50)),
        "C2": float(rng.integers(0, 50)),
        "C4": float(rng.integers(0, 5)),
        "C5": float(rng.integers(0, 20)),
        "C6": float(rng.integers(0, 50)),
    }
    for k, v in plausible.items():
        if k in row:
            row[k] = v
    df = pd.DataFrame([row], columns=feature_names)
    return df


input_df = None

if mode.startswith("Sample"):
    seed = st.number_input("Sample seed", min_value=0, max_value=9999, value=42)
    input_df = get_sample_transaction(seed)
    with st.expander("View sample transaction (15 most important fields)"):
        important_cols = [c for c in [
            "TransactionAmt", "TransactionAmt_log",
            "TX_hour", "TX_day", "TX_week",
            "card1", "card2", "ProductCD", "addr1", "addr2",
            "C1", "C2", "C4", "C5", "C6",
        ] if c in input_df.columns]
        st.dataframe(input_df[important_cols].T.rename(columns={0: "value"}))

elif mode == "Upload CSV":
    file = st.file_uploader(
        "Upload a CSV with one or more transactions. "
        "Columns should match the model's feature names.",
        type=["csv"],
    )
    if file is not None:
        df = pd.read_csv(file)
        # Add any missing columns as 0; reorder
        for c in feature_names:
            if c not in df.columns:
                df[c] = 0
        df = df[feature_names]
        input_df = df
        st.write(f"Loaded {len(df)} rows.")
        st.dataframe(df.head(10))

else:  # Manual entry
    st.markdown("### Enter values for the most influential fields")
    cols = st.columns(3)
    manual = {col: 0.0 for col in feature_names}
    field_specs = [
        ("TransactionAmt", 125.00, "USD amount"),
        ("TX_hour", 14, "Hour of day (0-23)"),
        ("TX_day", 3, "Day of week (0-6)"),
        ("card1", 12000, "Card1 id"),
        ("card2", 321, "Card2 id"),
        ("ProductCD", 1, "Product code (encoded)"),
        ("addr1", 250, "Billing address bucket"),
        ("addr2", 87, "Region code"),
        ("C1", 2, "Count feature C1"),
        ("C2", 1, "Count feature C2"),
        ("C4", 0, "Count feature C4"),
        ("C5", 0, "Count feature C5"),
    ]
    for i, (name, default, help_text) in enumerate(field_specs):
        with cols[i % 3]:
            if name in manual:
                manual[name] = st.number_input(
                    name, value=float(default), help=help_text
                )
    # Derived
    manual["TransactionAmt_log"] = float(np.log1p(manual["TransactionAmt"]))
    input_df = pd.DataFrame([manual], columns=feature_names)


# ---- Score + explain ------------------------------------------------------
if input_df is not None and st.button("Score Transaction(s)", type="primary"):
    proba = pipeline.predict_proba(input_df)[:, 1]
    pred = (proba >= threshold).astype(int)

    st.markdown("## Prediction")
    if len(input_df) == 1:
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Fraud probability", f"{proba[0]:.1%}")
        col_b.metric("Decision threshold", f"{threshold:.0%}")
        col_c.metric(
            "Verdict",
            "🚨 FRAUD" if pred[0] == 1 else "✅ Not Fraud",
        )
    else:
        results = input_df.copy()
        results["fraud_probability"] = proba
        results["flagged_as_fraud"] = pred
        st.dataframe(
            results[["fraud_probability", "flagged_as_fraud"]].head(50)
        )
        st.write(f"Flagged {pred.sum()} of {len(pred)} as fraud.")

    # ---- SHAP explanation for the first row -----------------------------
    st.markdown("## SHAP Explanation — why the model decided this")
    first = input_df.iloc[[0]]
    fitted_imputer = pipeline.named_steps["imputer"]
    first_imp = pd.DataFrame(
        fitted_imputer.transform(first), columns=feature_names
    )

    shap_raw = shap_explainer.shap_values(first_imp)
    # Normalize to 1-D positive-class SHAP values
    if isinstance(shap_raw, list):
        shap_pos = shap_raw[1][0]
    else:
        arr = np.asarray(shap_raw)
        if arr.ndim == 3:
            shap_pos = arr[0, :, 1]
        elif arr.ndim == 2:
            shap_pos = arr[0]
        else:
            shap_pos = arr

    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "Value": first_imp.iloc[0].values,
        "SHAP impact": shap_pos,
    })
    shap_df["abs_impact"] = shap_df["SHAP impact"].abs()
    top = shap_df.sort_values("abs_impact", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in top["SHAP impact"]]
    ax.barh(top["Feature"][::-1], top["SHAP impact"][::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("SHAP value (push toward fraud →)")
    ax.set_title("Top 15 features influencing this prediction")
    plt.tight_layout()
    st.pyplot(fig)

    with st.expander("See full SHAP table for this transaction"):
        st.dataframe(top.drop(columns="abs_impact"))


# ---- Sidebar footer with global importances -------------------------------
with st.sidebar.expander("Global feature importance (top 15)"):
    rf_model = pipeline.named_steps["model"]
    imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": rf_model.feature_importances_,
    }).sort_values("Importance", ascending=False).head(15)
    st.dataframe(imp_df.reset_index(drop=True))

st.sidebar.markdown("---")
st.sidebar.caption(
    "Model: tuned Random Forest pipeline (SMOTE + RF) "
    "trained on IEEE-CIS Fraud Detection."
)
