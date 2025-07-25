import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Load data
df = pd.read_csv("Data_Tanaman_Padi_Sumatera_version_1.csv")

# Fitur dan target
fitur = ['Luas panen', 'Curah hujan', 'Kelembapan', 'Suhu rata-rata']
target = 'Produksi'

# Filter data train hingga 2020
df_train = df[df['Tahun'] <= 2020]

# Buat model
lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)

# Latih model
lr.fit(df_train[fitur], df_train[target])
rf.fit(df_train[fitur], df_train[target])

# Simulasi data tahun 2021–2025 berbasis 2020 dengan variasi per provinsi
def simulate_future_data(df, tahun_awal, tahun_akhir):
    data_future = []
    for tahun in range(tahun_awal, tahun_akhir + 1):
        for _, row in df[df['Tahun'] == 2020].iterrows():
            row_baru = row.copy()
            row_baru['Tahun'] = tahun
            row_baru['Luas panen'] *= np.random.uniform(0.95, 1.05)
            row_baru['Curah hujan'] *= np.random.uniform(0.93, 1.08)
            row_baru['Kelembapan'] *= np.random.uniform(0.97, 1.03)
            row_baru['Suhu rata-rata'] *= np.random.uniform(0.98, 1.02)
            data_future.append(row_baru)
    return pd.DataFrame(data_future)

# Buat data prediksi tahun 2021–2025
df_prediksi = simulate_future_data(df, 2021, 2025)

# Prediksi dengan dua model
df_prediksi['Prediksi (Linear Regression)'] = lr.predict(df_prediksi[fitur])
df_prediksi['Prediksi (Random Forest)'] = rf.predict(df_prediksi[fitur])

# Tampilkan di Streamlit
st.title("Prediksi Produksi Padi di Sumatera (2021–2025)")
st.write("Menggunakan model Linear Regression dan Random Forest")
st.dataframe(df_prediksi[['Provinsi', 'Tahun', 'Prediksi (Linear Regression)', 'Prediksi (Random Forest)']])

# Grafik perbandingan
st.subheader("Grafik Perbandingan Prediksi per Tahun")
for tahun in sorted(df_prediksi['Tahun'].unique()):
    df_tahun = df_prediksi[df_prediksi['Tahun'] == tahun]

    st.markdown(f"### Tahun {tahun}")
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
