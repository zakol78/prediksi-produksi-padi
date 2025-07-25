import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Load data
df = pd.read_csv("Data_Tanaman_Padi_Sumatera_version_1.csv")

# Fitur dan target
fitur = ['Luas panen', 'Curah hujan', 'Kelembapan', 'Suhu rata-rata']
target = 'Produksi'

# Pisahkan data historis (1993–2020)
df_historis = df[df['Tahun'] <= 2020].copy()

# Model training pada data historis
lr = LinearRegression()
rf = RandomForestRegressor(random_state=42)

lr.fit(df_historis[fitur], df_historis[target])
rf.fit(df_historis[fitur], df_historis[target])

# Prediksi tahun 2021 berdasarkan data 2020
df_2021 = df[df['Tahun'] == 2020].copy()
df_2021['Tahun'] = 2021

# Fungsi membuat data masa depan (2022–2025)
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

# Buat data masa depan
df_future = generate_future_data(df_2021, 2022, 2025)

# Gabungkan semua data prediksi (2021–2025)
df_prediksi = pd.concat([df_2021, df_future], ignore_index=True)

# Prediksi dengan dua model
df_prediksi['Produksi_LR'] = lr.predict(df_prediksi[fitur])
df_prediksi['Produksi_RF'] = rf.predict(df_prediksi[fitur])

# Tampilkan aplikasi
st.title("Prediksi Produksi Padi di Sumatera (1993–2025)")
st.write("Model: Linear Regression & Random Forest")

# Gabungkan historis dan prediksi
df_prediksi_display = df_prediksi[['Provinsi', 'Tahun', 'Produksi_LR', 'Produksi_RF']].copy()
df_prediksi_display = df_prediksi_display.rename(columns={
    'Produksi_LR': 'Prediksi (Linear Regression)',
    'Produksi_RF': 'Prediksi (Random Forest)'
})
df_historis_display = df_historis[['Provinsi', 'Tahun', 'Produksi']].copy()
df_historis_display = df_historis_display.rename(columns={'Produksi': 'Produksi Aktual'})

# Gabungkan historis dan prediksi
df_all = pd.concat([df_historis_display, df_prediksi_display], ignore_index=True)

# Tampilkan tabel semua data
st.subheader("Data Produksi Padi: Aktual & Prediksi")
st.dataframe(df_all)

# Grafik garis per provinsi
st.subheader("Grafik Produksi Padi per Provinsi")

provinsi_list = df_all['Provinsi'].unique()
selected_prov = st.selectbox("Pilih Provinsi", provinsi_list)

df_prov = df_all[df_all['Provinsi'] == selected_prov]

fig, ax = plt.subplots(figsize=(10, 5))

# Plot produksi aktual
if 'Produksi Aktual' in df_prov:
    ax.plot(df_prov['Tahun'], df_prov['Produksi Aktual'], label='Aktual', marker='o')

# Plot prediksi
if 'Prediksi (Linear Regression)' in df_prov:
    ax.plot(df_prov['Tahun'], df_prov['Prediksi (Linear Regression)'], label='Prediksi LR', marker='x')

if 'Prediksi (Random Forest)' in df_prov:
    ax.plot(df_prov['Tahun'], df_prov['Prediksi (Random Forest)'], label='Prediksi RF', marker='s')

ax.set_title(f"Produksi Padi: {selected_prov}")
ax.set_xlabel("Tahun")
ax.set_ylabel("Produksi (Ton)")
ax.legend()
st.pyplot(fig)
