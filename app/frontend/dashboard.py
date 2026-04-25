import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from app.backend.data_loader import load_data, get_basic_info
from app.backend.preprocess import split_data, scale_time_amount
from app.backend.train import compare_models, save_best_model, load_best_model, get_sampling_preview
from app.backend.evaluate import evaluate_model, get_precision_recall_data, get_roc_data
from app.backend.threshold import evaluate_threshold

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("Detección de fraude con tarjetas de crédito")
st.write("Sistema analítico con Python, FastAPI y Streamlit")

menu = st.sidebar.radio(
    "Menú",
    [
        "Inicio",
        "EDA",
        "Modelado",
        "Evaluación",
        "Umbral",
        "Simulación"
    ]
)


@st.cache_data
def get_data():
    return load_data()


df = get_data()

if menu == "Inicio":
    st.subheader("Contexto del problema")
    st.write("""
    El fraude financiero es un problema de alto impacto económico. En este dataset,
    la clase fraude es extremadamente minoritaria, por lo que la exactitud (accuracy)
    puede ser engañosa. El objetivo del sistema es detectar más fraudes sin generar
    demasiadas falsas alarmas.
    """)

    info = get_basic_info(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total transacciones", info["rows"])
    c2.metric("Fraudes", info["fraud_count"])
    c3.metric("Legítimas", info["legit_count"])
    c4.metric("% Fraude", f"{info['fraud_ratio'] * 100:.4f}%")

    st.subheader("Objetivo analítico")
    st.write("""
    Comparar modelos y estrategias de manejo del desbalance, analizar la relación
    entre Precisión y Recall, y apoyar una decisión de negocio basada en el umbral de clasificación.
    """)

elif menu == "EDA":
    st.subheader("Exploración de datos")

    st.write("Primeras filas del dataset")
    st.dataframe(df.head(10))

    st.subheader("Distribución de clases")
    fig, ax = plt.subplots()
    df["Class"].value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Legítima", "Fraude"], rotation=0)
    ax.set_ylabel("Cantidad")
    ax.set_title("Desbalance de clases")
    st.pyplot(fig)

    st.subheader("Boxplot de Amount por clase")
    fig2, ax2 = plt.subplots()
    df.boxplot(column="Amount", by="Class", ax=ax2)
    ax2.set_title("Amount por clase")
    ax2.set_xlabel("Class")
    ax2.set_ylabel("Amount")
    st.pyplot(fig2)

    st.subheader("Boxplot de Time por clase")
    fig3, ax3 = plt.subplots()
    df.boxplot(column="Time", by="Class", ax=ax3)
    ax3.set_title("Time por clase")
    ax3.set_xlabel("Class")
    ax3.set_ylabel("Time")
    st.pyplot(fig3)

elif menu == "Modelado":
    st.subheader("Comparación de modelos")

    X_train, X_test, y_train, y_test = split_data(df)
    X_train_scaled, X_test_scaled, scaler = scale_time_amount(X_train, X_test)

    st.write("Distribución de clases antes y después del reesampling")
    sampling_preview = get_sampling_preview(X_train_scaled, y_train)
    st.dataframe(sampling_preview)

    fig_sampling, ax_sampling = plt.subplots()
    sampling_preview.plot(kind="bar", ax=ax_sampling)
    ax_sampling.set_title("Comparación de clases: original vs reesampling")
    ax_sampling.set_ylabel("Cantidad")
    ax_sampling.set_xlabel("Clase")
    st.pyplot(fig_sampling)

    if st.button("Entrenar y comparar modelos"):
        results_df, trained_models = compare_models(X_train_scaled, y_train, X_test_scaled, y_test)

        st.write("Resultados comparativos")
        st.dataframe(results_df)

        best_row = results_df.iloc[0]
        best_key = f"{best_row['model_name']}_{best_row['strategy']}"
        best_model = trained_models[best_key]

        metadata = {
            "best_model_name": best_row["model_name"],
            "strategy": best_row["strategy"],
            "selected_threshold": 0.5,
            "pr_auc": float(best_row["pr_auc"]),
            "roc_auc": float(best_row["roc_auc"]),
            "precision": float(best_row["precision"]),
            "recall": float(best_row["recall"]),
            "f1": float(best_row["f1"]),
        }

        save_best_model(best_model, scaler, metadata)

        st.success(f"Mejor modelo guardado: {best_row['model_name']} + {best_row['strategy']}")

elif menu == "Evaluación":
    st.subheader("Evaluación del mejor modelo")

    if not os.path.exists("models/best_model.pkl"):
        st.warning("Primero debes entrenar el modelo en la sección Modelado.")
    else:
        model, scaler, metadata = load_best_model()

        X_train, X_test, y_train, y_test = split_data(df)
        X_train_scaled, X_test_scaled, scaler = scale_time_amount(X_train, X_test)

        results = evaluate_model(model, X_test_scaled, y_test, threshold=metadata.get("selected_threshold", 0.5))

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{results['accuracy']:.4f}")
        c2.metric("Precision", f"{results['precision']:.4f}")
        c3.metric("Recall", f"{results['recall']:.4f}")
        c4.metric("F1", f"{results['f1']:.4f}")
        c5.metric("PR-AUC", f"{results['pr_auc']:.4f}")

        st.subheader("Matriz de confusión")
        cm = results["confusion_matrix"]
        cm_df = pd.DataFrame(
            cm,
            index=["Real No Fraude", "Real Fraude"],
            columns=["Pred No Fraude", "Pred Fraude"]
        )
        st.dataframe(cm_df)

        st.subheader("Curva Precision-Recall")
        precision, recall, thresholds = get_precision_recall_data(y_test, results["y_probs"])
        fig_pr, ax_pr = plt.subplots()
        ax_pr.plot(recall, precision)
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title("Curva Precision-Recall")
        st.pyplot(fig_pr)

        st.subheader("Curva ROC")
        fpr, tpr, roc_thresholds = get_roc_data(y_test, results["y_probs"])
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr)
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("Curva ROC")
        st.pyplot(fig_roc)

        st.subheader("Reporte de clasificación")
        st.text(results["classification_report"])

elif menu == "Umbral":
    st.subheader("Análisis de umbral de decisión")

    if not os.path.exists("models/best_model.pkl"):
        st.warning("Primero debes entrenar el modelo en la sección Modelado.")
    else:
        model, scaler, metadata = load_best_model()

        X_train, X_test, y_train, y_test = split_data(df)
        X_train_scaled, X_test_scaled, scaler = scale_time_amount(X_train, X_test)

        y_probs = model.predict_proba(X_test_scaled)[:, 1]

        threshold = st.slider("Selecciona el umbral", 0.01, 0.99, 0.50, 0.01)
        threshold_metrics = evaluate_threshold(y_test, y_probs, threshold)

        c1, c2, c3 = st.columns(3)
        c1.metric("Precision", f"{threshold_metrics['precision']:.4f}")
        c2.metric("Recall", f"{threshold_metrics['recall']:.4f}")
        c3.metric("F1", f"{threshold_metrics['f1']:.4f}")

        c4, c5, c6 = st.columns(3)
        c4.metric("Fraudes capturados", threshold_metrics["frauds_captured"])
        c5.metric("Falsas alarmas", threshold_metrics["fp"])
        c6.metric("Fraudes perdidos", threshold_metrics["missed_frauds"])

        st.write("Interpretación:")
        st.write("""
        Al bajar el umbral, normalmente sube el recall y se detectan más fraudes,
        pero también aumentan las falsas alarmas. Esta decisión debe tomarse con criterio de negocio.
        """)

elif menu == "Simulación":
    st.subheader("Simulación de flujo de transacciones")

    if not os.path.exists("models/best_model.pkl"):
        st.warning("Primero debes entrenar el modelo en la sección Modelado.")
    else:
        model, scaler, metadata = load_best_model()

        sample_df = df.sample(15, random_state=42).copy()
        X_sample = sample_df.drop(columns=["Class"]).copy()
        X_sample[["Time", "Amount"]] = scaler.transform(X_sample[["Time", "Amount"]])

        probs = model.predict_proba(X_sample)[:, 1]
        threshold = metadata.get("selected_threshold", 0.5)
        preds = (probs >= threshold).astype(int)

        sample_df["Prob_Fraude"] = probs
        sample_df["Predicción"] = preds
        sample_df["Estado"] = sample_df["Predicción"].map({0: "🟢 Legítima", 1: "🔴 Sospechosa"})

        st.dataframe(
            sample_df[["Time", "Amount", "Class", "Prob_Fraude", "Predicción", "Estado"]]
            .sort_values(by="Prob_Fraude", ascending=False)
        )