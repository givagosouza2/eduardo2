import streamlit as st
import pandas as pd
import numpy as np
import io

# -------------------------
# Funções baseadas em IQR
# -------------------------

def iqr(data):
    return np.percentile(data, 75) - np.percentile(data, 25)

def cv_iqr(data):
    median = np.median(data)
    return (iqr(data) / median) * 100 if median != 0 else np.nan

def bootstrap_se_median(data, n_resamples=1000):
    n = len(data)
    medians = []
    for _ in range(n_resamples):
        sample = np.random.choice(data, size=n, replace=True)
        medians.append(np.median(sample))
    return np.std(medians, ddof=1)

def bootstrap_mdc(differences, n_resamples=1000, confidence=0.95):
    n = len(differences)
    boot_se = []
    for _ in range(n_resamples):
        resample = np.random.choice(differences, size=n, replace=True)
        boot_se.append(np.std(resample, ddof=1))
    mean_se = np.mean(boot_se)
    z_score = 1.96 if confidence == 0.95 else 1.64
    return mean_se * z_score * np.sqrt(2)

def medae(day1, day2):
    return np.median(np.abs(day1 - day2))

def mdape(day1, day2):
    perc_error = 100 * np.abs(day1 - day2) / day1
    return np.median(perc_error)

# -------------------------
# Interface Streamlit
# -------------------------

st.title("Análise Não Paramétrica de Confiabilidade Inter-Dias (Baseada em IQR)")

uploaded_file = st.file_uploader("Carregue um arquivo CSV com duas colunas (Dia 1 e Dia 2)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    try:
        day1 = df.iloc[:, 0].dropna().values
        day2 = df.iloc[:, 1].dropna().values

        st.subheader("Resultados Não Paramétricos (usando IQR)")

        # Calcula os indicadores
        median_day1 = np.median(day1)
        median_day2 = np.median(day2)
        iqr_day1 = iqr(day1)
        iqr_day2 = iqr(day2)
        cv_day1 = cv_iqr(day1)
        cv_day2 = cv_iqr(day2)
        diffs = day1 - day2
        iqr_diffs = iqr(diffs)
        icc_est = 1 - (iqr_diffs / (iqr_day1 + iqr_day2)) if (iqr_day1 + iqr_day2) != 0 else np.nan
        se_median_day1 = bootstrap_se_median(day1)
        se_median_day2 = bootstrap_se_median(day2)
        mdc = bootstrap_mdc(diffs)
        medae_val = medae(day1, day2)
        mdape_val = mdape(day1, day2)

        # Exibição
        st.write(f"Mediana Dia 1: {median_day1:.3f}")
        st.write(f"Mediana Dia 2: {median_day2:.3f}")
        st.write(f"IQR Dia 1: {iqr_day1:.3f}")
        st.write(f"IQR Dia 2: {iqr_day2:.3f}")
        st.write(f"CV (IQR/Mediana) Dia 1: {cv_day1:.2f}%")
        st.write(f"CV (IQR/Mediana) Dia 2: {cv_day2:.2f}%")
        st.write(f"ICC Não Paramétrico (via IQR): {icc_est:.3f}")
        st.write(f"Erro padrão da Mediana (Dia 1): {se_median_day1:.3f}")
        st.write(f"Erro padrão da Mediana (Dia 2): {se_median_day2:.3f}")
        st.write(f"MDC Não Paramétrico (via Bootstrap): {mdc:.3f}")
        st.write(f"MedAE: {medae_val:.3f}")
        st.write(f"MdAPE: {mdape_val:.2f}%")

        # Monta DataFrame de resultados
        results = {
            "Métrica": [
                "Mediana Dia 1", "Mediana Dia 2",
                "IQR Dia 1", "IQR Dia 2",
                "CV Dia 1 (%)", "CV Dia 2 (%)",
                "ICC Não Paramétrico",
                "Erro padrão da Mediana Dia 1",
                "Erro padrão da Mediana Dia 2",
                "MDC Não Paramétrico",
                "MedAE",
                "MdAPE (%)"
            ],
            "Valor": [
                median_day1, median_day2,
                iqr_day1, iqr_day2,
                cv_day1, cv_day2,
                icc_est,
                se_median_day1,
                se_median_day2,
                mdc,
                medae_val,
                mdape_val
            ]
        }

        results_df = pd.DataFrame(results)

        # Exportação CSV
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Baixar Resultados como CSV",
            data=csv,
            file_name='resultados_confiabilidade_nao_parametrica.csv',
            mime='text/csv',
        )

    except Exception as e:
        st.error(f"Erro ao processar os dados: {e}")
