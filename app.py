import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Prediksi Produksi Padi", layout="wide")
st.title("📊 Prediksi Produksi Padi di Pulau Sumatera")
st.markdown("Menggunakan **Random Forest** dan **Linear Regression** berdasarkan data historis.")

# ------------------ LOAD DATA ------------------
try:
    df = pd.read_csv("Data_Tanaman_Padi_Sumatera_version_1.csv")
except FileNotFoundError:
    st.error("❌ File 'Data_Tanaman_Padi_Sumatera_version_1.csv' tidak ditemukan. Pastikan file ada di direktori yang sama.")
    st.stop()

# ------------------ PREPROCESSING ------------------
df.columns = df.columns.str.strip()  # Hilangkan spasi jika ada
df = df.dropna()

# Tentukan fitur dan target
fitur = ['Luas panen', 'Curah hujan', 'Kelembapan', 'Suhu rata-rata']
target = 'Produksi'

# Cek apakah kolom-kolom penting tersedia
missing = [col for col in fitur + [target, 'Tahun', 'Provinsi'] if col not in df.columns]
if missing:
    st.error(f"❌ Kolom berikut tidak ditemukan dalam dataset: {missing}")
    st.stop()

# ------------------ SPLIT TRAIN-PREDIKSI ------------------
df_train = df[df['Tahun'] <= 2020]
df_pred = df[df['Tahun'] > 2020]

X_train = df_train[fitur]
y_train = df_train[target]

# ------------------ TRAINING MODEL ------------------
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_train)

rf = RandomForestRegressor(
    n_estimators=50,
    max_depth=5,
    random_state=42
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_train)

# ------------------ PREDIKSI MASA DEPAN ------------------
if not df_pred.empty:
    X_pred = df_pred[fitur]
    df_pred = df_pred.copy()  # hindari SettingWithCopyWarning
    df_pred['Pred_LR'] = lr.predict(X_pred)
    df_pred['Pred_RF'] = rf.predict(X_pred)

# ------------------ EVALUASI ------------------
st.subheader("📈 Evaluasi Model pada Data Latih")

def tampilkan_evaluasi(nama, y_true, y_pred):
    col1, col2, col3 = st.columns(3)
    col1.metric(f"{nama} - R²", f"{r2_score(y_true, y_pred):.3f}")
    col2.metric(f"{nama} - MAE", f"{mean_absolute_error(y_true, y_pred):,.0f}")
    col3.metric(f"{nama} - MSE", f"{mean_squared_error(y_true, y_pred):,.0f}")

tampilkan_evaluasi("Linear Regression", y_train, y_pred_lr)
tampilkan_evaluasi("Random Forest", y_train, y_pred_rf)

# ------------------ VISUALISASI ------------------
st.subheader("📊 Visualisasi Fitur terhadap Produksi")
fitur_pilihan = st.selectbox("Pilih fitur:", fitur)
fig, ax = plt.subplots()
sns.scatterplot(data=df_train, x=fitur_pilihan, y=target, ax=ax)
ax.set_title(f'{fitur_pilihan} vs Produksi')
st.pyplot(fig)

# ------------------ HASIL PREDIKSI ------------------
if not df_pred.empty:
    st.subheader("📅 Hasil Prediksi Tahun 2021–2025")
    st.dataframe(df_pred[['Provinsi', 'Tahun', 'Produksi', 'Pred_LR', 'Pred_RF']])

    st.download_button(
        label="⬇️ Unduh Hasil Prediksi",
        data=df_pred.to_csv(index=False),
        file_name="prediksi_2021_2025.csv",
        mime="text/csv"
    )

st.caption("© 2025 — Aplikasi ini dikembangkan menggunakan Streamlit dan Scikit-learn.")
