import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Judul aplikasi
st.title("Prediksi Produksi Padi di Sumatera (2021–2025)")
st.write("Menggunakan algoritma: **Linear Regression** dan **Random Forest**")

# Load data (sesuai permintaan)
df = pd.read_csv("Data_Tanaman_Padi_Sumatera_version_1.csv")

# Menentukan fitur dan target
fitur = ['Luas panen', 'Curah hujan', 'Kelembapan', 'Suhu rata-rata']
target = 'Produksi'

# Membagi data latih (hingga tahun 2020)
df_train = df[df['Tahun'] <= 2020]

# Melatih model
lr = LinearRegression()
rf = RandomForestRegressor(random_state=42)

lr.fit(df_train[fitur], df_train[target])
rf.fit(df_train[fitur], df_train[target])

# Prediksi tahun 2021 (duplikasi dari 2020)
df_2021 = df[df['Tahun'] == 2020].copy()
df_2021['Tahun'] = 2021

# Fungsi membuat data masa depan
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

# Simulasi tahun 2022–2025
df_2022_2025 = generate_future_data(df_2021, 2022, 2025)

# Gabungkan untuk prediksi
df_prediksi = pd.concat([df_2021, df_2022_2025], ignore_index=True)

# Prediksi
pred_lr = lr.predict(df_prediksi[fitur])
pred_rf = rf.predict(df_prediksi[fitur])

# Data hasil prediksi
hasil = df_prediksi[['Provinsi', 'Tahun']].copy()
hasil['Prediksi (Linear Regression)'] = pred_lr
hasil['Prediksi (Random Forest)'] = pred_rf

# Tampilkan tabel prediksi
st.subheader("Hasil Prediksi Produksi Padi")
st.dataframe(hasil)

# Visualisasi per tahun
st.subheader("Grafik Perbandingan Prediksi per Tahun")

for tahun in sorted(hasil['Tahun'].unique()):
    st.markdown(f"### Tahun {tahun}")
    df_tahun = hasil[hasil['Tahun'] == tahun]

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

# Evaluasi model
st.subheader("Evaluasi Model pada Data Latih")

# Linear Regression
y_true = df_train[target]
y_pred_lr = lr.predict(df_train[fitur])
r2_lr = r2_score(y_true, y_pred_lr)
mae_lr = mean_absolute_error(y_true, y_pred_lr)
mse_lr = mean_squared_error(y_true, y_pred_lr)

# Random Forest
y_pred_rf = rf.predict(df_train[fitur])
r2_rf = r2_score(y_true, y_pred_rf)
mae_rf = mean_absolute_error(y_true, y_pred_rf)
mse_rf = mean_squared_error(y_true, y_pred_rf)

# Tabel evaluasi model
eval_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest'],
    'R² Score': [r2_lr, r2_rf],
    'MAE (ton)': [mae_lr, mae_rf],
    'MSE (ton²)': [mse_lr, mse_rf]
})

st.dataframe(eval_df.style.format({
    'R² Score': "{:.4f}",
    'MAE (ton)': "{:,.2f}",
    'MSE (ton²)': "{:,.2f}"
}))
