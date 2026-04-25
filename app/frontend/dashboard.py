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
from app.backend.threshold import (
    evaluate_threshold,
    get_threshold_recommendations,
    generate_threshold_table
)

def show_section_note(text):
    st.markdown(
        f"""
        <div style="
            padding: 14px;
            border-radius: 12px;
            background-color: rgba(59, 130, 246, 0.10);
            border: 1px solid rgba(59, 130, 246, 0.35);
            margin-bottom: 18px;
        ">
            {text}
        </div>
        """,
        unsafe_allow_html=True
    )

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("Detección de fraude con tarjetas de crédito")
st.write("Sistema analítico con Python, FastAPI y Streamlit")

st.markdown("""
<style>
.metric-card {
    padding: 14px;
    border-radius: 14px;
    background-color: #111827;
    border: 1px solid #374151;
    margin-bottom: 10px;
}
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
.section-note {
    padding: 14px;
    border-radius: 12px;
    background-color: rgba(59, 130, 246, 0.12);
    border: 1px solid rgba(59, 130, 246, 0.35);
}
</style>
""", unsafe_allow_html=True)

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
    show_section_note("Esta sección presenta el contexto del problema, la magnitud del fraude financiero y el objetivo principal del sistema.")
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
    show_section_note("Aquí se analiza el dataset para entender el desbalance de clases y el comportamiento de variables importantes como Amount y Time.")
    st.write("Primeras filas del dataset")
    st.dataframe(df.head(10), use_container_width=True)

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
    show_section_note("En esta sección se comparan distintos modelos y estrategias de balanceo para identificar cuál ofrece mejor desempeño.")
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_scaled, X_test_scaled, scaler = scale_time_amount(X_train, X_test)

    st.write("Distribución de clases antes y después del reesampling")
    sampling_preview = get_sampling_preview(X_train_scaled, y_train)
    st.dataframe(sampling_preview, use_container_width=True)

    fig_sampling, ax_sampling = plt.subplots()
    sampling_preview.plot(kind="bar", ax=ax_sampling)
    ax_sampling.set_title("Comparación de clases: original vs reesampling")
    ax_sampling.set_ylabel("Cantidad")
    ax_sampling.set_xlabel("Clase")
    st.pyplot(fig_sampling)

    if st.button("Entrenar y comparar modelos"):
        results_df, trained_models = compare_models(X_train_scaled, y_train, X_test_scaled, y_test)

        st.write("Resultados comparativos")
        st.dataframe(results_df, use_container_width=True)

        st.subheader("Promedios de validación cruzada")
        st.write("""
        Estas métricas provienen de StratifiedKFold, por lo que muestran un rendimiento
        más estable y confiable que una sola partición.
        """)

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

        save_best_model(best_model, scaler, metadata)

        st.success(
            f"Mejor modelo guardado: {best_row['model_name']} + {best_row['strategy']}"
        )

elif menu == "Evaluación":
    st.subheader("Evaluación del mejor modelo")
    show_section_note("Aquí se visualiza el rendimiento del mejor modelo mediante métricas, matriz de confusión y curvas de evaluación.")
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
        st.dataframe(cm_df, use_container_width=True)

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
    show_section_note("Esta sección permite analizar cómo cambia el equilibrio entre detectar más fraudes y generar más falsas alarmas al mover el umbral.")
    if not os.path.exists("models/best_model.pkl"):
        st.warning("Primero debes entrenar el modelo en la sección Modelado.")
    else:
        model, scaler, metadata = load_best_model()

        X_train, X_test, y_train, y_test = split_data(df)
        X_train_scaled, X_test_scaled, scaler = scale_time_amount(X_train, X_test)

        y_probs = model.predict_proba(X_test_scaled)[:, 1]

        recommendations = get_threshold_recommendations(y_test, y_probs)
        best_f1 = recommendations["best_f1"]
        best_recall = recommendations["best_recall"]
        best_precision = recommendations["best_precision"]
        threshold_table = recommendations["table"]

        st.markdown("### Recomendación automática de umbral")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Mejor equilibrio (F1)", f"{best_f1['threshold']:.2f}")
            st.write(f"Precision: {best_f1['precision']:.4f}")
            st.write(f"Recall: {best_f1['recall']:.4f}")
            st.write(f"F1: {best_f1['f1']:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Mayor Recall", f"{best_recall['threshold']:.2f}")
            st.write(f"Precision: {best_recall['precision']:.4f}")
            st.write(f"Recall: {best_recall['recall']:.4f}")
            st.write(f"F1: {best_recall['f1']:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)

        with c3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Mayor Precision", f"{best_precision['threshold']:.2f}")
            st.write(f"Precision: {best_precision['precision']:.4f}")
            st.write(f"Recall: {best_precision['recall']:.4f}")
            st.write(f"F1: {best_precision['f1']:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="section-note">
        Recomendación sugerida para el proyecto: usar el umbral con mejor F1 como punto de equilibrio inicial,
        porque balancea la detección de fraudes y las falsas alarmas. Luego la gerencia puede moverlo según
        el nivel de riesgo que quiera aceptar.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Simulación interactiva")
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

        st.markdown("### Tabla de umbrales")
        st.dataframe(threshold_table, use_container_width=True)

        fig_threshold, ax_threshold = plt.subplots()
        ax_threshold.plot(threshold_table["threshold"], threshold_table["precision"], label="Precision")
        ax_threshold.plot(threshold_table["threshold"], threshold_table["recall"], label="Recall")
        ax_threshold.plot(threshold_table["threshold"], threshold_table["f1"], label="F1")
        ax_threshold.set_xlabel("Umbral")
        ax_threshold.set_ylabel("Valor")
        ax_threshold.set_title("Comportamiento de métricas según el umbral")
        ax_threshold.legend()
        st.pyplot(fig_threshold)

elif menu == "Simulación":
    st.subheader("Simulación de flujo de transacciones")
    show_section_note("Aquí se muestran transacciones de ejemplo procesadas por el modelo para observar cómo se comporta la clasificación en casos simulados.")
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
            .sort_values(by="Prob_Fraude", ascending=False),
            use_container_width=True
        )

elif menu == "Predicción manual":
    st.subheader("Predicción de una transacción")
    show_section_note("Esta sección permite probar una transacción individual, usando un ejemplo real del dataset o ingresando valores manualmente.")
    if not os.path.exists("models/best_model.pkl"):
        st.warning("Primero debes entrenar el modelo en la sección Modelado.")
    else:
        model, scaler, metadata = load_best_model()
        threshold = metadata.get("selected_threshold", 0.5)

        tab1, tab2 = st.tabs(["Probar con ejemplo", "Modo avanzado"])

        with tab1:
            st.write("Selecciona una fila del dataset para simular una predicción.")

            sample_index = st.slider(
                "Selecciona el índice de la transacción de ejemplo",
                min_value=0,
                max_value=len(df) - 1,
                value=0,
                step=1
            )

            sample_row = df.iloc[sample_index].copy()

            preview_cols = ["Time", "Amount", "Class"]
            st.write("Vista rápida del ejemplo seleccionado:")
            st.dataframe(pd.DataFrame([sample_row[preview_cols]]), use_container_width=True)

            if st.button("Predecir ejemplo seleccionado"):
                input_data = sample_row.drop(labels=["Class"]).to_frame().T
                input_scaled = input_data.copy()
                input_scaled[["Time", "Amount"]] = scaler.transform(input_scaled[["Time", "Amount"]])

                probability = float(model.predict_proba(input_scaled)[0][1])
                prediction = int(probability >= threshold)

                c1, c2, c3 = st.columns(3)
                c1.metric("Probabilidad de fraude", f"{probability:.6f}")
                c2.metric("Umbral usado", f"{threshold:.2f}")
                c3.metric("Predicción", "Fraude" if prediction == 1 else "No fraude")

                st.write("Clase real del dataset:", "Fraude" if int(sample_row["Class"]) == 1 else "No fraude")

                if prediction == 1:
                    st.markdown(
                        '<div class="result-alert">🔴 La transacción fue clasificada como sospechosa de fraude.</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="result-ok">🟢 La transacción fue clasificada como legítima.</div>',
                        unsafe_allow_html=True
                    )

        with tab2:
            st.write("""
            Este modo permite ingresar manualmente todos los atributos.
            Está pensado más para pruebas técnicas, porque V1 a V28 son variables transformadas.
            """)

            with st.form("manual_prediction_form"):
                st.markdown("### Datos principales")
                c1, c2 = st.columns(2)
                with c1:
                    time_value = st.number_input("Time", value=0.0, format="%.6f")
                with c2:
                    amount_value = st.number_input("Amount", value=0.0, format="%.6f")

                st.markdown("### Variables técnicas")

                with st.expander("Bloque 1 · V1 a V10"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        v1 = st.number_input("V1", value=0.0, format="%.6f")
                        v2 = st.number_input("V2", value=0.0, format="%.6f")
                        v3 = st.number_input("V3", value=0.0, format="%.6f")
                        v4 = st.number_input("V4", value=0.0, format="%.6f")
                        v5 = st.number_input("V5", value=0.0, format="%.6f")
                    with col_b:
                        v6 = st.number_input("V6", value=0.0, format="%.6f")
                        v7 = st.number_input("V7", value=0.0, format="%.6f")
                        v8 = st.number_input("V8", value=0.0, format="%.6f")
                        v9 = st.number_input("V9", value=0.0, format="%.6f")
                        v10 = st.number_input("V10", value=0.0, format="%.6f")

                with st.expander("Bloque 2 · V11 a V20"):
                    col_c, col_d = st.columns(2)
                    with col_c:
                        v11 = st.number_input("V11", value=0.0, format="%.6f")
                        v12 = st.number_input("V12", value=0.0, format="%.6f")
                        v13 = st.number_input("V13", value=0.0, format="%.6f")
                        v14 = st.number_input("V14", value=0.0, format="%.6f")
                        v15 = st.number_input("V15", value=0.0, format="%.6f")
                    with col_d:
                        v16 = st.number_input("V16", value=0.0, format="%.6f")
                        v17 = st.number_input("V17", value=0.0, format="%.6f")
                        v18 = st.number_input("V18", value=0.0, format="%.6f")
                        v19 = st.number_input("V19", value=0.0, format="%.6f")
                        v20 = st.number_input("V20", value=0.0, format="%.6f")

                with st.expander("Bloque 3 · V21 a V28"):
                    col_e, col_f = st.columns(2)
                    with col_e:
                        v21 = st.number_input("V21", value=0.0, format="%.6f")
                        v22 = st.number_input("V22", value=0.0, format="%.6f")
                        v23 = st.number_input("V23", value=0.0, format="%.6f")
                        v24 = st.number_input("V24", value=0.0, format="%.6f")
                    with col_f:
                        v25 = st.number_input("V25", value=0.0, format="%.6f")
                        v26 = st.number_input("V26", value=0.0, format="%.6f")
                        v27 = st.number_input("V27", value=0.0, format="%.6f")
                        v28 = st.number_input("V28", value=0.0, format="%.6f")

                submitted = st.form_submit_button("Predecir transacción manual")

            if submitted:
                input_data = pd.DataFrame([{
                    "Time": time_value,
                    "V1": v1, "V2": v2, "V3": v3, "V4": v4, "V5": v5,
                    "V6": v6, "V7": v7, "V8": v8, "V9": v9, "V10": v10,
                    "V11": v11, "V12": v12, "V13": v13, "V14": v14, "V15": v15,
                    "V16": v16, "V17": v17, "V18": v18, "V19": v19, "V20": v20,
                    "V21": v21, "V22": v22, "V23": v23, "V24": v24,
                    "V25": v25, "V26": v26, "V27": v27, "V28": v28,
                    "Amount": amount_value
                }])

                input_scaled = input_data.copy()
                input_scaled[["Time", "Amount"]] = scaler.transform(input_scaled[["Time", "Amount"]])

                probability = float(model.predict_proba(input_scaled)[0][1])
                prediction = int(probability >= threshold)

                c1, c2, c3 = st.columns(3)
                c1.metric("Probabilidad de fraude", f"{probability:.6f}")
                c2.metric("Umbral usado", f"{threshold:.2f}")
                c3.metric("Predicción", "Fraude" if prediction == 1 else "No fraude")

                if prediction == 1:
                    st.markdown(
                        '<div class="result-alert">🔴 La transacción fue clasificada como sospechosa de fraude.</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="result-ok">🟢 La transacción fue clasificada como legítima.</div>',
                        unsafe_allow_html=True
                    )