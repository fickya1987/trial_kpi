import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load data function
@st.cache_data

def load_data(uploaded_file=None):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_csv("kpi_cleaned.csv")

# Sidebar file uploader
uploaded_file = st.sidebar.file_uploader("Upload Data KPI (CSV)", type=["csv"])

# Load data
df = load_data(uploaded_file)

# Konversi kolom numerik untuk menghindari TypeError
numeric_cols = ["TARGET TW TERKAIT", "REALISASI TW TERKAIT", "BOBOT"]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Title and Description
st.title("📊 Pelindo KPI Dashboard & AI Insight")
st.markdown("""
Dashboard ini menampilkan capaian KPI, prediksi performa, penilaian 360 AKHLAK, dan rekomendasi peningkatan.
""")

# Filter by Divisi/Unit if available
if "POSISI PEKERJA" in df.columns:
    unit_filter = st.sidebar.selectbox("Filter berdasarkan Unit/Posisi", ["All"] + sorted(df["POSISI PEKERJA"].unique()))
    if unit_filter != "All":
        df = df[df["POSISI PEKERJA"] == unit_filter]

# Filter KPI
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

    # Penilaian 360 AKHLAK (simulasi sederhana)
    st.subheader("💬 Penilaian 360 AKHLAK (Simulasi)")
    for _, row in df_kpi.iterrows():
        nilai_atasan = st.slider(f"Nilai AKHLAK oleh Atasan untuk {row['POSISI PEKERJA']}", 1, 6, 4)
        nilai_rekan = st.slider(f"Nilai oleh Rekan Kerja", 1, 6, 4, key=f"rekan_{row['ID KPI']}")
        nilai_bawahan = st.slider(f"Nilai oleh Bawahan", 1, 6, 4, key=f"bawah_{row['ID KPI']}")
        skor_akhir = round((nilai_atasan*0.45 + nilai_rekan*0.25 + nilai_bawahan*0.30) * (20/6), 2)
        st.markdown(f"**Skor AKHLAK**: {skor_akhir} dari maksimum 20")
else:
    st.warning("Data tidak cukup untuk prediksi. Silakan pilih KPI lain.")

# Footer
st.markdown("""
---
Prototype AI Insight Pelindo 2025 | Dibangun dengan Streamlit, Linear Regression & Simulasi 360 Behavior
""")


