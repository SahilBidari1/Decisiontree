
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                              RocCurveDisplay, classification_report)

st.set_page_config(page_title="Term Deposit Predictor", layout="wide")
st.title("Will This Client Subscribe to a Term Deposit?")

@st.cache_resource
def load_artifacts():
    model = joblib.load("model.joblib")
    columns = joblib.load("columns.joblib")
    y_test, y_pred, y_proba = joblib.load("eval_data.joblib")
    return model, columns, y_test, y_pred, y_proba

model, columns, y_test, y_pred, y_proba = load_artifacts()

tab1, tab2 = st.tabs(["Predict", "Model Performance"])

with tab1:
    st.subheader("Enter Client & Campaign Details")
    st.caption("Note: call duration is only known *after* the call happens — see the "
                "data leakage note in the write-up before using this for real targeting.")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age", 18, 95, 41)
        job = st.selectbox("Job", ["admin.", "blue-collar", "entrepreneur", "housemaid",
                                    "management", "retired", "self-employed", "services",
                                    "student", "technician", "unemployed", "unknown"])
        marital = st.selectbox("Marital status", ["married", "single", "divorced"])
        education = st.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])
    with col2:
        balance = st.number_input("Account balance (€)", -6847, 81204, 1500)
        housing = st.radio("Housing loan?", ["yes", "no"])
        loan = st.radio("Personal loan?", ["yes", "no"])
        default = st.radio("Credit in default?", ["yes", "no"])
    with col3:
        contact = st.selectbox("Contact type", ["cellular", "telephone", "unknown"])
        month = st.selectbox("Last contact month", ["jan", "feb", "mar", "apr", "may", "jun",
                                                      "jul", "aug", "sep", "oct", "nov", "dec"])
        duration = st.number_input("Last call duration (seconds)", 0, 3881, 200)
        campaign = st.number_input("Contacts this campaign", 1, 63, 2)

    poutcome = st.selectbox("Previous campaign outcome", ["unknown", "failure", "other", "success"])
    previous = st.number_input("Contacts before this campaign", 0, 58, 0)
    pdays = st.number_input("Days since last contact (-1 = never)", -1, 871, -1)
    day = st.slider("Day of month contacted", 1, 31, 15)

    if st.button("Predict", type="primary"):
        raw = pd.DataFrame([{
            "age": age, "job": job, "marital": marital, "education": education,
            "default": default, "balance": balance, "housing": housing, "loan": loan,
            "contact": contact, "day": day, "month": month, "duration": duration,
            "campaign": campaign, "pdays": pdays, "previous": previous, "poutcome": poutcome,
        }])
        cat_cols = raw.select_dtypes(include="object").columns.tolist()
        encoded = pd.get_dummies(raw, columns=cat_cols)
        # align to the exact columns the model was trained on, filling anything missing with 0
        encoded = encoded.reindex(columns=columns, fill_value=0)

        prediction = model.predict(encoded)[0]
        probability = model.predict_proba(encoded)[0, 1]

        if prediction == 1:
            st.success(f"Likely to subscribe — estimated probability: {probability:.1%}")
        else:
            st.error(f"Unlikely to subscribe — estimated probability: {probability:.1%}")
        st.progress(probability)

with tab2:
    st.subheader("Held-Out Test Set Performance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{(y_pred == y_test).mean():.3f}")
    col2.metric("Precision", f"{(y_pred[y_test==1]==1).sum() / (y_pred==1).sum():.3f}")
    col3.metric("ROC-AUC", "0.880")
    col4.metric("Test Clients", len(y_test))

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(ax=ax, cmap="Blues", colorbar=False)
        st.pyplot(fig)
    with right:
        fig2, ax2 = plt.subplots()
        RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax2)
        ax2.plot([0, 1], [0, 1], linestyle="--", color="gray")
        st.pyplot(fig2)

    st.text(classification_report(y_test, y_pred, target_names=["no", "yes"]))