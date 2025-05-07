import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load cleaned data
@st.cache_data
def load_data():
    return pd.read_csv("kpi_cleaned.csv")

df = load_data()

# Title and Description
st.title("📊 Pelindo KPI Dashboard & AI Insight")
st.markdown("""
Dashboard ini menampilkan capaian KPI, prediksi performa, dan rekomendasi peningkatan berbasis AI sederhana.
""")

# Sidebar filter
selected_kpi = st.sidebar.selectbox("Pilih Nama KPI", df["NAMA KPI"].unique())
df_kpi = df[df["NAMA KPI"] == selected_kpi]

# Display data
st.subheader(f"📌 Data untuk KPI: {selected_kpi}")
st.dataframe(df_kpi)

# Hitung Skor dan Prediksi jika cukup data
if df_kpi.shape[0] > 1:
    X = df_kpi[["TARGET TW TERKAIT"]]
    y = df_kpi["REALISASI TW TERKAIT"]

    model = LinearRegression()
    model.fit(X, y)
    df_kpi["PREDIKSI"] = model.predict(X)
    df_kpi["CAPAIAN %"] = (df_kpi["REALISASI TW TERKAIT"] / df_kpi["TARGET TW TERKAIT"]) * 100
    df_kpi["SKOR"] = df_kpi["CAPAIAN %"] * df_kpi["BOBOT"] / 100

    # Plot Prediksi
    st.subheader("📈 Prediksi Realisasi vs Target")
    fig, ax = plt.subplots()
    ax.scatter(df_kpi["TARGET TW TERKAIT"], df_kpi["REALISASI TW TERKAIT"], label="Aktual")
    ax.plot(df_kpi["TARGET TW TERKAIT"], df_kpi["PREDIKSI"], color='red', label="Prediksi")
    ax.set_xlabel("Target")
    ax.set_ylabel("Realisasi")
    ax.legend()
    st.pyplot(fig)

    # Rekomendasi Sederhana
    st.subheader("🤖 Rekomendasi Performa")
    for _, row in df_kpi.iterrows():
        rekom = "✅ Sudah baik" if row["CAPAIAN %"] >= 100 else "⚠️ Perlu coaching & review strategi"
        st.markdown(f"**{row['NAMA KPI']}**: {rekom} (Capaian: {row['CAPAIAN %']:.1f}%)")
else:
    st.warning("Data tidak cukup untuk prediksi. Silakan pilih KPI lain.")

# Footer
st.markdown("""
---
Prototype AI Insight Pelindo 2025 | Dibangun dengan Streamlit & Linear Regression
""")
