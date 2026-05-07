import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import os
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from app.backend.data_loader import load_data, get_basic_info
from app.backend.evaluate import evaluate_model, get_precision_recall_data, get_roc_data
from app.backend.preprocess import (
    split_data,
    build_manual_input_dataframe,
)
from app.backend.threshold import (
    evaluate_threshold,
    generate_threshold_table,
    get_threshold_recommendations,
)
from app.backend.train import (
    compare_models,
    get_sampling_preview,
    load_best_model,
    save_best_model,
)

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("Detección de fraude financiero")
st.write("Sistema analítico con Python, FastAPI y Streamlit")

st.markdown("""
<style>
.result-ok {
    padding: 16px;
    border-radius: 14px;
    background-color: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.5);
    color: #d1fae5;
    font-weight: 600;
}
.result-alert {
    padding: 16px;
    border-radius: 14px;
    background-color: rgba(239, 68, 68, 0.15);
    border: 1px solid rgba(239, 68, 68, 0.5);
    color: #fee2e2;
    font-weight: 600;
}
.note-box {
    padding: 14px;
    border-radius: 12px;
    background-color: rgba(59, 130, 246, 0.10);
    border: 1px solid rgba(59, 130, 246, 0.35);
    margin-bottom: 18px;
}
</style>
""", unsafe_allow_html=True)


def show_section_note(text: str):
    st.markdown(f'<div class="note-box">{text}</div>', unsafe_allow_html=True)


menu = st.sidebar.radio(
    "Menú",
    [
        "Inicio",
        "EDA",
        "Modelado",
        "Evaluación",
        "Umbral",
        "Simulación",
        "Predicción manual"
    ]
)


@st.cache_data
def get_data():
    return load_data()


df = get_data()

if menu == "Inicio":
    st.subheader("Contexto del problema")
    show_section_note(
        "Esta sección presenta el contexto del nuevo caso. "
        "Ahora trabajamos con dos datasets: transacciones e identidad."
    )

    st.write("""
    El fraude financiero es un problema de alto impacto económico. En este nuevo dataset,
    la variable objetivo es **isFraud**, y el reto principal sigue siendo el desbalance
    de clases: hay muchas transacciones legítimas y pocas fraudulentas.
    """)

    info = get_basic_info(df)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total transacciones", info["rows"])
    c2.metric("Fraudes", info["fraud_count"])
    c3.metric("Legítimas", info["legit_count"])
    c4.metric("% Fraude", f"{info['fraud_ratio'] * 100:.4f}%")
    c5.metric("% Nulos", f"{info['missing_ratio'] * 100:.2f}%")

    st.subheader("Objetivo analítico")
    st.write("""
    Unir `train_transaction.csv` con `train_identity.csv`, preparar los datos,
    comparar modelos y estrategias de balanceo, y apoyar una decisión de negocio
    basada en el umbral de clasificación.
    """)

elif menu == "EDA":
    st.subheader("Exploración de datos")
    show_section_note(
        "Aquí se analiza el nuevo dataset usando variables más interpretables como "
        "TransactionAmt, TransactionDT y algunas categóricas."
    )

    st.write("Primeras filas del dataset combinado")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Distribución de clases (isFraud)")
    fig, ax = plt.subplots(figsize=(6, 4))
    df["isFraud"].value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Legítima", "Fraude"], rotation=0)
    ax.set_ylabel("Cantidad")
    ax.set_title("Desbalance de clases")
    st.pyplot(fig)

    st.subheader("Boxplot de TransactionAmt por isFraud")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    df.boxplot(column="TransactionAmt", by="isFraud", ax=ax2)
    ax2.set_title("TransactionAmt por clase")
    ax2.set_xlabel("isFraud")
    ax2.set_ylabel("TransactionAmt")
    st.pyplot(fig2)

    st.subheader("Boxplot de TransactionDT por isFraud")
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    df.boxplot(column="TransactionDT", by="isFraud", ax=ax3)
    ax3.set_title("TransactionDT por clase")
    ax3.set_xlabel("isFraud")
    ax3.set_ylabel("TransactionDT")
    st.pyplot(fig3)

    if "ProductCD" in df.columns:
        st.subheader("Distribución de ProductCD")
        product_counts = df["ProductCD"].fillna("Missing").value_counts().head(10)
        fig4, ax4 = plt.subplots(figsize=(8, 4))
        product_counts.plot(kind="bar", ax=ax4)
        ax4.set_title("Frecuencia de ProductCD")
        ax4.set_ylabel("Cantidad")
        st.pyplot(fig4)

    if "card4" in df.columns:
        st.subheader("Distribución de card4")
        card4_counts = df["card4"].fillna("Missing").value_counts().head(10)
        fig5, ax5 = plt.subplots(figsize=(8, 4))
        card4_counts.plot(kind="bar", ax=ax5)
        ax5.set_title("Frecuencia de card4")
        ax5.set_ylabel("Cantidad")
        st.pyplot(fig5)

    st.subheader("Top columnas con más valores nulos")
    missing_df = (
        df.isna().mean()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )
    missing_df.columns = ["Columna", "Porcentaje_nulos"]

    fig6, ax6 = plt.subplots(figsize=(10, 5))
    ax6.barh(missing_df["Columna"], missing_df["Porcentaje_nulos"])
    ax6.set_title("Porcentaje de nulos por columna")
    ax6.set_xlabel("Porcentaje")
    ax6.invert_yaxis()
    st.pyplot(fig6)

elif menu == "Modelado":
    st.subheader("Comparación de modelos")
    show_section_note(
    "En esta sección se comparan Logistic Regression y Random Forest con "
    "class_weight y undersampling. En este dataset grande se quitó SMOTE "
    "para evitar errores de memoria."
)
    X_train, X_test, y_train, y_test = split_data(df)

    st.write("Vista del efecto del balanceo")
    sampling_preview = get_sampling_preview(y_train)
    st.dataframe(sampling_preview, use_container_width=True)

    fig_sampling, ax_sampling = plt.subplots(figsize=(7, 4))
    sampling_preview.plot(kind="bar", ax=ax_sampling)
    ax_sampling.set_title("Original vs balanceo")
    ax_sampling.set_ylabel("Cantidad")
    ax_sampling.set_xlabel("Clase")
    st.pyplot(fig_sampling)

    if st.button("Entrenar y comparar modelos"):
        results_df, trained_models = compare_models(X_train, y_train, X_test, y_test)

        st.write("Resultados comparativos")
        st.dataframe(results_df, use_container_width=True)

        best_row = results_df.iloc[0]
        best_key = f"{best_row['model_name']}_{best_row['strategy']}"
        best_model = trained_models[best_key]

        metadata = {
            "best_model_name": best_row["model_name"],
            "strategy": best_row["strategy"],
            "selected_threshold": 0.5,
            "cv_precision_mean": float(best_row["cv_precision_mean"]),
            "cv_recall_mean": float(best_row["cv_recall_mean"]),
            "cv_f1_mean": float(best_row["cv_f1_mean"]),
            "cv_roc_auc_mean": float(best_row["cv_roc_auc_mean"]),
            "cv_pr_auc_mean": float(best_row["cv_pr_auc_mean"]),
            "test_accuracy": float(best_row["test_accuracy"]),
            "test_precision": float(best_row["test_precision"]),
            "test_recall": float(best_row["test_recall"]),
            "test_f1": float(best_row["test_f1"]),
            "test_roc_auc": float(best_row["test_roc_auc"]),
            "test_pr_auc": float(best_row["test_pr_auc"]),
        }

        save_best_model(best_model, metadata)
        st.success(f"Mejor modelo guardado: {best_row['model_name']} + {best_row['strategy']}")

elif menu == "Evaluación":
    st.subheader("Evaluación del mejor modelo")
    show_section_note(
        "Aquí se evalúa el mejor modelo usando matriz de confusión, "
        "Precision-Recall, ROC y reporte de clasificación."
    )

    if not os.path.exists("models/best_model.pkl"):
        st.warning("Primero debes entrenar el modelo en la sección Modelado.")
    else:
        model, metadata = load_best_model()

        X_train, X_test, y_train, y_test = split_data(df)
        results = evaluate_model(model, X_test, y_test, threshold=metadata.get("selected_threshold", 0.5))

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
        st.dataframe(cm_df, use_container_width=True)

        st.subheader("Curva Precision-Recall")
        precision, recall, _ = get_precision_recall_data(y_test, results["y_probs"])
        fig_pr, ax_pr = plt.subplots(figsize=(6, 4))
        ax_pr.plot(recall, precision)
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title("Curva Precision-Recall")
        st.pyplot(fig_pr)

        st.subheader("Curva ROC")
        fpr, tpr, _ = get_roc_data(y_test, results["y_probs"])
        fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
        ax_roc.plot(fpr, tpr)
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("Curva ROC")
        st.pyplot(fig_roc)

        st.subheader("Reporte de clasificación")
        st.text(results["classification_report"])

elif menu == "Umbral":
    st.subheader("Análisis de umbral de decisión")
    show_section_note(
        "Esta sección ayuda a decidir qué umbral conviene usar según el equilibrio "
        "entre capturar más fraudes y generar más falsas alarmas."
    )

    if not os.path.exists("models/best_model.pkl"):
        st.warning("Primero debes entrenar el modelo en la sección Modelado.")
    else:
        model, metadata = load_best_model()

        X_train, X_test, y_train, y_test = split_data(df)
        y_probs = model.predict_proba(X_test)[:, 1]

        recommendations = get_threshold_recommendations(y_test, y_probs)
        best_f1 = recommendations["best_f1"]
        best_recall = recommendations["best_recall"]
        best_precision = recommendations["best_precision"]
        threshold_table = recommendations["table"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Mejor F1", f"{best_f1['threshold']:.2f}")
        c2.metric("Mayor Recall", f"{best_recall['threshold']:.2f}")
        c3.metric("Mayor Precision", f"{best_precision['threshold']:.2f}")

        st.write("Recomendación inicial: usar el mejor umbral por F1 como punto de equilibrio.")

        threshold = st.slider("Selecciona el umbral", 0.01, 0.99, float(best_f1["threshold"]), 0.01)
        threshold_metrics = evaluate_threshold(y_test, y_probs, threshold)

        c4, c5, c6 = st.columns(3)
        c4.metric("Precision", f"{threshold_metrics['precision']:.4f}")
        c5.metric("Recall", f"{threshold_metrics['recall']:.4f}")
        c6.metric("F1", f"{threshold_metrics['f1']:.4f}")

        c7, c8, c9 = st.columns(3)
        c7.metric("Fraudes capturados", threshold_metrics["frauds_captured"])
        c8.metric("Falsas alarmas", threshold_metrics["fp"])
        c9.metric("Fraudes perdidos", threshold_metrics["missed_frauds"])

        st.subheader("Tabla de umbrales")
        st.dataframe(threshold_table, use_container_width=True)

        fig_threshold, ax_threshold = plt.subplots(figsize=(8, 4))
        ax_threshold.plot(threshold_table["threshold"], threshold_table["precision"], label="Precision")
        ax_threshold.plot(threshold_table["threshold"], threshold_table["recall"], label="Recall")
        ax_threshold.plot(threshold_table["threshold"], threshold_table["f1"], label="F1")
        ax_threshold.set_xlabel("Umbral")
        ax_threshold.set_ylabel("Valor")
        ax_threshold.set_title("Métricas según el umbral")
        ax_threshold.legend()
        st.pyplot(fig_threshold)

elif menu == "Simulación":
    st.subheader("Simulación de flujo de transacciones")
    show_section_note(
        "Aquí se muestran algunas transacciones del dataset y cómo el modelo las clasifica."
    )

    if not os.path.exists("models/best_model.pkl"):
        st.warning("Primero debes entrenar el modelo en la sección Modelado.")
    else:
        model, metadata = load_best_model()

        sample_df = df.sample(15, random_state=42).copy()
        X_sample = sample_df.drop(columns=["isFraud", "TransactionID"], errors="ignore").copy()

        # Para asegurar el mismo conjunto usado en entrenamiento:
        X_train, _, _, _ = split_data(df)
        X_sample = X_sample[X_train.columns]

        probs = model.predict_proba(X_sample)[:, 1]
        threshold = metadata.get("selected_threshold", 0.5)
        preds = (probs >= threshold).astype(int)

        sample_df["Prob_Fraude"] = probs
        sample_df["Predicción"] = preds
        sample_df["Estado"] = sample_df["Predicción"].map({0: "🟢 Legítima", 1: "🔴 Sospechosa"})

        show_cols = [
            col for col in [
                "TransactionID", "TransactionDT", "TransactionAmt", "ProductCD",
                "card4", "card6", "P_emaildomain", "DeviceType",
                "isFraud", "Prob_Fraude", "Predicción", "Estado"
            ] if col in sample_df.columns
        ]

        st.dataframe(
            sample_df[show_cols].sort_values(by="Prob_Fraude", ascending=False),
            use_container_width=True
        )

elif menu == "Predicción manual":
    st.subheader("Predicción de una transacción")
    show_section_note(
        "Puedes probar una fila real del dataset o ingresar algunos campos principales manualmente."
    )

    if not os.path.exists("models/best_model.pkl"):
        st.warning("Primero debes entrenar el modelo en la sección Modelado.")
    else:
        model, metadata = load_best_model()
        threshold = metadata.get("selected_threshold", 0.5)

        tab1, tab2 = st.tabs(["Probar con ejemplo", "Modo manual simple"])

        with tab1:
            sample_index = st.slider(
                "Selecciona el índice de la transacción de ejemplo",
                min_value=0,
                max_value=len(df) - 1,
                value=0,
                step=1
            )

            sample_row = df.iloc[sample_index].copy()

            preview_cols = [col for col in ["TransactionID", "TransactionAmt", "ProductCD", "card4", "card6", "isFraud"] if col in df.columns]
            st.dataframe(pd.DataFrame([sample_row[preview_cols]]), use_container_width=True)

            if st.button("Predecir ejemplo seleccionado"):
                X_train, _, _, _ = split_data(df)
                input_data = sample_row[X_train.columns].to_frame().T

                probability = float(model.predict_proba(input_data)[0][1])
                prediction = int(probability >= threshold)

                c1, c2, c3 = st.columns(3)
                c1.metric("Probabilidad de fraude", f"{probability:.6f}")
                c2.metric("Umbral usado", f"{threshold:.2f}")
                c3.metric("Predicción", "Fraude" if prediction == 1 else "No fraude")

                st.write("Clase real del dataset:", "Fraude" if int(sample_row["isFraud"]) == 1 else "No fraude")

                if prediction == 1:
                    st.markdown('<div class="result-alert">🔴 La transacción fue clasificada como sospechosa de fraude.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-ok">🟢 La transacción fue clasificada como legítima.</div>', unsafe_allow_html=True)

        with tab2:
            with st.form("manual_prediction_form"):
                c1, c2 = st.columns(2)
                with c1:
                    transaction_dt = st.number_input("TransactionDT", value=0.0, format="%.2f")
                    transaction_amt = st.number_input("TransactionAmt", value=0.0, format="%.2f")
                    addr1 = st.number_input("addr1", value=0.0, format="%.2f")
                    addr2 = st.number_input("addr2", value=0.0, format="%.2f")
                with c2:
                    product_cd = st.selectbox("ProductCD", ["W", "C", "R", "H", "S"])
                    card4 = st.selectbox("card4", ["visa", "mastercard", "discover", "american express"])
                    card6 = st.selectbox("card6", ["debit", "credit", "charge debit"])
                    p_email = st.text_input("P_emaildomain", value="gmail.com")

                submitted = st.form_submit_button("Predecir transacción manual")

            if submitted:
                overrides = {
                    "TransactionDT": transaction_dt,
                    "TransactionAmt": transaction_amt,
                    "addr1": addr1,
                    "addr2": addr2,
                    "ProductCD": product_cd,
                    "card4": card4,
                    "card6": card6,
                    "P_emaildomain": p_email,
                }

                input_data = build_manual_input_dataframe(df, overrides)
                probability = float(model.predict_proba(input_data)[0][1])
                prediction = int(probability >= threshold)

                c1, c2, c3 = st.columns(3)
                c1.metric("Probabilidad de fraude", f"{probability:.6f}")
                c2.metric("Umbral usado", f"{threshold:.2f}")
                c3.metric("Predicción", "Fraude" if prediction == 1 else "No fraude")

                if prediction == 1:
                    st.markdown('<div class="result-alert">🔴 La transacción fue clasificada como sospechosa de fraude.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-ok">🟢 La transacción fue clasificada como legítima.</div>', unsafe_allow_html=True)