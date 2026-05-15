import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import os
import io
import pandas as pd
import streamlit as st
import plotly.express as px

from app.backend.data_loader import load_data, get_basic_info
from app.backend.evaluate import evaluate_model, get_precision_recall_data, get_roc_data
from app.backend.preprocess import (
    split_data,
    build_manual_input_dataframe,
)
from app.backend.threshold import (
    evaluate_threshold,
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

    # -----------------------------
    # RESUMEN ESTADÍSTICO BONITO
    # -----------------------------
    st.subheader("Resumen estadístico del dataset")

    total_filas = len(df)
    total_columnas = df.shape[1]
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    porcentaje_nulos_total = round(df.isna().mean().mean() * 100, 2)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Filas", f"{total_filas:,}")
    c2.metric("Columnas", total_columnas)
    c3.metric("Numéricas", len(num_cols))
    c4.metric("Categóricas", len(cat_cols))
    c5.metric("% nulos global", f"{porcentaje_nulos_total}%")

    st.markdown("---")

    tab_num, tab_cat = st.tabs(["📊 Variables numéricas", "🧩 Variables categóricas"])

    with tab_num:
        numeric_summary = df.describe(include=["number"]).T.reset_index()
        numeric_summary = numeric_summary.rename(columns={"index": "Variable"})

        numeric_summary = numeric_summary[
            ~numeric_summary["Variable"].isin(["TransactionID", "isFraud"])
        ]

        for col in ["mean", "std", "min", "25%", "50%", "75%", "max"]:
            if col in numeric_summary.columns:
                numeric_summary[col] = numeric_summary[col].round(2)

        if "count" in numeric_summary.columns:
            numeric_summary["count"] = numeric_summary["count"].astype(int)

        st.dataframe(
            numeric_summary,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Variable": st.column_config.TextColumn("Variable"),
                "count": st.column_config.NumberColumn("No nulos", format="%d"),
                "mean": st.column_config.NumberColumn("Media", format="%.2f"),
                "std": st.column_config.NumberColumn("Desv. estándar", format="%.2f"),
                "min": st.column_config.NumberColumn("Mínimo", format="%.2f"),
                "25%": st.column_config.NumberColumn("Q1 (25%)", format="%.2f"),
                "50%": st.column_config.NumberColumn("Mediana", format="%.2f"),
                "75%": st.column_config.NumberColumn("Q3 (75%)", format="%.2f"),
                "max": st.column_config.NumberColumn("Máximo", format="%.2f"),
            }
        )

    with tab_cat:
        categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        if categorical_cols:
            cat_summary_rows = []
            for col in categorical_cols:
                moda = df[col].mode(dropna=True)
                cat_summary_rows.append({
                    "Variable": col,
                    "No nulos": int(df[col].notna().sum()),
                    "Nulos": int(df[col].isna().sum()),
                    "% Nulos": round(df[col].isna().mean() * 100, 2),
                    "Categorías únicas": int(df[col].nunique(dropna=True)),
                    "Moda": str(moda.iloc[0]) if not moda.empty else "Sin moda"
                })

            categorical_summary = pd.DataFrame(cat_summary_rows)
            categorical_summary = categorical_summary.sort_values(
                by=["% Nulos", "Categorías únicas"],
                ascending=[False, False]
            ).reset_index(drop=True)

            st.dataframe(
                categorical_summary,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Variable": st.column_config.TextColumn("Variable"),
                    "No nulos": st.column_config.NumberColumn("No nulos", format="%d"),
                    "Nulos": st.column_config.NumberColumn("Nulos", format="%d"),
                    "% Nulos": st.column_config.NumberColumn("% Nulos", format="%.2f %%"),
                    "Categorías únicas": st.column_config.NumberColumn("Categorías únicas", format="%d"),
                    "Moda": st.column_config.TextColumn("Moda"),
                }
            )
        else:
            st.info("No se encontraron variables categóricas en el dataset.")

    # -----------------------------
    # DESCARGA EXCEL DE FRAUDES
    # -----------------------------
    st.subheader("Descargar transacciones con fraude")

    fraud_df = df[df["isFraud"] == 1].copy()
    st.write(f"Cantidad de transacciones fraudulentas encontradas: {len(fraud_df)}")

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        fraud_df.to_excel(writer, index=False, sheet_name="Fraudes")

    st.download_button(
        label="Descargar Excel de transacciones fraudulentas",
        data=excel_buffer.getvalue(),
        file_name="transacciones_fraude.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # -----------------------------
    # GRÁFICOS
    # -----------------------------
    st.subheader("Distribución de clases (isFraud)")
    fraud_counts = df["isFraud"].value_counts().sort_index().reset_index()
    fraud_counts.columns = ["isFraud", "Cantidad"]
    fraud_counts["isFraud"] = fraud_counts["isFraud"].map({0: "Legítima", 1: "Fraude"})

    fig_class = px.bar(
        fraud_counts,
        x="isFraud",
        y="Cantidad",
        title="Desbalance de clases",
        text="Cantidad"
    )
    fig_class.update_layout(xaxis_title="", yaxis_title="Cantidad")
    st.plotly_chart(fig_class, use_container_width=True)

    st.subheader("Boxplot de TransactionAmt por isFraud")
    fig_amt = px.box(
        df,
        x="isFraud",
        y="TransactionAmt",
        title="TransactionAmt por clase",
        points=False
    )
    fig_amt.update_xaxes(tickvals=[0, 1], ticktext=["Legítima", "Fraude"])
    st.plotly_chart(fig_amt, use_container_width=True)

    st.subheader("Boxplot de TransactionDT por isFraud")
    fig_dt = px.box(
        df,
        x="isFraud",
        y="TransactionDT",
        title="TransactionDT por clase",
        points=False
    )
    fig_dt.update_xaxes(tickvals=[0, 1], ticktext=["Legítima", "Fraude"])
    st.plotly_chart(fig_dt, use_container_width=True)

    if "ProductCD" in df.columns:
        st.subheader("Distribución de ProductCD")
        product_counts = df["ProductCD"].fillna("Missing").value_counts().head(10).reset_index()
        product_counts.columns = ["ProductCD", "Cantidad"]

        fig_product = px.bar(
            product_counts,
            x="ProductCD",
            y="Cantidad",
            title="Frecuencia de ProductCD",
            text="Cantidad"
        )
        st.plotly_chart(fig_product, use_container_width=True)

    if "card4" in df.columns:
        st.subheader("Distribución de card4")
        card4_counts = df["card4"].fillna("Missing").value_counts().head(10).reset_index()
        card4_counts.columns = ["card4", "Cantidad"]

        fig_card4 = px.bar(
            card4_counts,
            x="card4",
            y="Cantidad",
            title="Frecuencia de card4",
            text="Cantidad"
        )
        st.plotly_chart(fig_card4, use_container_width=True)

    st.subheader("Top columnas con más valores nulos")
    missing_df = (
        df.isna().mean()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )
    missing_df.columns = ["Columna", "Porcentaje_nulos"]
    missing_df["Porcentaje_nulos"] = missing_df["Porcentaje_nulos"] * 100

    fig_missing = px.bar(
        missing_df,
        x="Porcentaje_nulos",
        y="Columna",
        orientation="h",
        title="Porcentaje de nulos por columna",
        text="Porcentaje_nulos"
    )
    fig_missing.update_layout(yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig_missing, use_container_width=True)

elif menu == "Modelado":
    st.subheader("Comparación de modelos")
    show_section_note(
        "En esta sección se comparan Logistic Regression, Decision Tree y Random Forest "
        "con class_weight y undersampling. En este dataset grande se quitó SMOTE "
        "para evitar errores de memoria."
    )

    X_train, X_test, y_train, y_test = split_data(df)

    st.write("Vista del efecto del balanceo")
    sampling_preview = get_sampling_preview(y_train)
    st.dataframe(sampling_preview, use_container_width=True)

    sampling_plot_df = sampling_preview.reset_index().rename(columns={"index": "Clase"})
    sampling_melt = sampling_plot_df.melt(id_vars="Clase", var_name="Escenario", value_name="Cantidad")

    fig_sampling = px.bar(
        sampling_melt,
        x="Clase",
        y="Cantidad",
        color="Escenario",
        barmode="group",
        title="Original vs balanceo",
        text="Cantidad"
    )
    st.plotly_chart(fig_sampling, use_container_width=True)

    if st.button("Entrenar y comparar modelos"):
        results_df, trained_models = compare_models(X_train, y_train, X_test, y_test)

        st.write("Resultados comparativos")
        st.dataframe(results_df, use_container_width=True)

        metric_cols = [
            "model_name", "strategy", "cv_pr_auc_mean", "cv_f1_mean", "cv_recall_mean", "cv_precision_mean"
        ]
        chart_df = results_df[metric_cols].copy()
        chart_df["Modelo"] = chart_df["model_name"] + " + " + chart_df["strategy"]

        fig_models = px.bar(
            chart_df,
            x="Modelo",
            y="cv_pr_auc_mean",
            title="Comparación de modelos por CV PR-AUC",
            text="cv_pr_auc_mean"
        )
        st.plotly_chart(fig_models, use_container_width=True)

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
        ).reset_index().rename(columns={"index": "Real"})
        cm_melt = cm_df.melt(id_vars="Real", var_name="Predicción", value_name="Cantidad")

        fig_cm = px.density_heatmap(
            cm_melt,
            x="Predicción",
            y="Real",
            z="Cantidad",
            text_auto=True,
            title="Matriz de confusión"
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        st.subheader("Curva Precision-Recall")
        precision, recall, _ = get_precision_recall_data(y_test, results["y_probs"])
        pr_df = pd.DataFrame({"Recall": recall, "Precision": precision})

        fig_pr = px.line(
            pr_df,
            x="Recall",
            y="Precision",
            title="Curva Precision-Recall"
        )
        st.plotly_chart(fig_pr, use_container_width=True)

        st.subheader("Curva ROC")
        fpr, tpr, _ = get_roc_data(y_test, results["y_probs"])
        roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr})

        fig_roc = px.line(
            roc_df,
            x="FPR",
            y="TPR",
            title="Curva ROC"
        )
        st.plotly_chart(fig_roc, use_container_width=True)

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

        metric_threshold_df = threshold_table[["threshold", "precision", "recall", "f1"]].copy()
        metric_threshold_melt = metric_threshold_df.melt(
            id_vars="threshold",
            var_name="Métrica",
            value_name="Valor"
        )

        fig_threshold = px.line(
            metric_threshold_melt,
            x="threshold",
            y="Valor",
            color="Métrica",
            title="Métricas según el umbral"
        )
        st.plotly_chart(fig_threshold, use_container_width=True)

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

        sim_chart_df = sample_df[show_cols].copy()
        if "TransactionID" in sim_chart_df.columns:
            sim_chart_df["TransactionID"] = sim_chart_df["TransactionID"].astype(str)

        if "TransactionID" in sim_chart_df.columns:
            fig_sim = px.bar(
                sim_chart_df.sort_values(by="Prob_Fraude", ascending=False),
                x="TransactionID",
                y="Prob_Fraude",
                color="Estado",
                title="Probabilidad de fraude por transacción"
            )
            st.plotly_chart(fig_sim, use_container_width=True)

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

                result_df = pd.DataFrame({
                    "Resultado": ["Probabilidad", "Umbral", "Predicción"],
                    "Valor": [probability, threshold, "Fraude" if prediction == 1 else "No fraude"]
                })
                st.dataframe(result_df, use_container_width=True)

                if prediction == 1:
                    st.markdown('<div class="result-alert">🔴 La transacción fue clasificada como sospechosa de fraude.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-ok">🟢 La transacción fue clasificada como legítima.</div>', unsafe_allow_html=True)