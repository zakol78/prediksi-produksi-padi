import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Judul aplikasi
st.title("Prediksi Produksi Padi di Sumatera (1993–2025)")
st.write("Menggunakan algoritma: **Linear Regression** dan **Random Forest**")

# Load data
df = pd.read_csv("Data_Tanaman_Padi_Sumatera_version_1.csv")

# Menentukan fitur dan target
fitur = ['Luas panen', 'Curah hujan', 'Kelembapan', 'Suhu rata-rata']
target = 'Produksi'

# Data latih (1993–2020)
df_train = df[df['Tahun'] <= 2020]

# Melatih model
lr = LinearRegression()
rf = RandomForestRegressor(random_state=42)

lr.fit(df_train[fitur], df_train[target])
rf.fit(df_train[fitur], df_train[target])

# === Prediksi Historis (1993–2020) ===
df_hist = df.copy()
df_hist['Prediksi (Linear Regression)'] = lr.predict(df_hist[fitur])
df_hist['Prediksi (Random Forest)'] = rf.predict(df_hist[fitur])

# === Prediksi Masa Depan (2021–2025) ===
df_2020 = df[df['Tahun'] == 2020].copy()
df_2020['Tahun'] = 2021

def generate_future_data(df_base, start_year, end_year):
    future_data = []
    for year in range(start_year, end_year + 1):
        temp = df_base.copy()
        temp['Tahun'] = year
        temp['Luas panen'] *= np.random.uniform(0.98, 1.03)
        temp['Curah hujan'] *= np.random.uniform(0.95, 1.05)
        temp['Kelembapan'] *= np.random.uniform(0.98, 1.02)
        temp['Suhu rata-rata'] *= np.random.uniform(0.99, 1.01)
        future_data.append(temp)
    return pd.concat(future_data, ignore_index=True)

df_future = generate_future_data(df_2020, 2022, 2025)
df_future = pd.concat([df_2020, df_future], ignore_index=True)

df_future['Prediksi (Linear Regression)'] = lr.predict(df_future[fitur])
df_future['Prediksi (Random Forest)'] = rf.predict(df_future[fitur])

# Gabungkan data historis dan prediksi masa depan
df_all_pred = pd.concat([
    df_hist[['Provinsi', 'Tahun', 'Prediksi (Linear Regression)', 'Prediksi (Random Forest)']],
    df_future[['Provinsi', 'Tahun', 'Prediksi (Linear Regression)', 'Prediksi (Random Forest)']]
], ignore_index=True)

# === Tabel Prediksi Lengkap ===
st.subheader("Hasil Prediksi Produksi Padi (1993–2025)")
st.dataframe(df_all_pred)

# === Visualisasi Tahunan ===
st.subheader("Grafik Perbandingan Prediksi per Tahun")

for tahun in sorted(df_all_pred['Tahun'].unique()):
    st.markdown(f"### Tahun {tahun}")
    df_tahun = df_all_pred[df_all_pred['Tahun'] == tahun]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df_tahun))
    width = 0.35

    ax.bar(x - width/2, df_tahun['Prediksi (Linear Regression)'], width, label='Linear Regression')
    ax.bar(x + width/2, df_tahun['Prediksi (Random Forest)'], width, label='Random Forest')

    ax.set_xticks(x)
    ax.set_xticklabels(df_tahun['Provinsi'], rotation=45, ha='right')
    ax.set_ylabel("Produksi (Ton)")
    ax.set_title(f"Prediksi Produksi Padi Tahun {tahun}")
    ax.legend()

    st.pyplot(fig)

# === Evaluasi Model pada Data Latih ===
st.subheader("Evaluasi Model pada Data Latih")

y_true = df_train[target]
y_pred_lr = lr.predict(df_train[fitur])
y_pred_rf = rf.predict(df_train[fitur])

eval_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest'],
    'R² Score': [r2_score(y_true, y_pred_lr), r2_score(y_true, y_pred_rf)],
    'MAE (ton)': [mean_absolute_error(y_true, y_pred_lr), mean_absolute_error(y_true, y_pred_rf)],
    'MSE (ton²)': [mean_squared_error(y_true, y_pred_lr), mean_squared_error(y_true, y_pred_rf)],
})

st.dataframe(eval_df.style.format({
    'R² Score': "{:.4f}",
    'MAE (ton)': "{:,.2f}",
    'MSE (ton²)': "{:,.2f}"
}))
