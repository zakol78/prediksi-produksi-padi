import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv("Data_Tanaman_Padi_Sumatera_version_1.csv")

# Fitur dan target
fitur = ['Luas panen', 'Curah hujan', 'Kelembapan', 'Suhu rata-rata']
target = 'Produksi'

# Data hingga 2020 sebagai data latih
df_train = df[df['Tahun'] <= 2020]

# Model
lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)

# Pelatihan
lr.fit(df_train[fitur], df_train[target])
rf.fit(df_train[fitur], df_train[target])

# Evaluasi RMSE
rmse_lr = mean_squared_error(df_train[target], lr.predict(df_train[fitur]), squared=False)
rmse_rf = mean_squared_error(df_train[target], rf.predict(df_train[fitur]), squared=False)

# Fungsi prediksi tahun 2021–2025
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

# Prediksi 2021–2025
df_prediksi = simulate_future_data(df, 2021, 2025)
df_prediksi['Prediksi (Linear Regression)'] = lr.predict(df_prediksi[fitur])
df_prediksi['Prediksi (Random Forest)'] = rf.predict(df_prediksi[fitur])

# Antarmuka
st.title("📈 Prediksi Produksi Padi 2021–2025 Menggunakan Seaborn")
st.write("Menggunakan model: **Linear Regression** dan **Random Forest**")

st.subheader("🔍 Evaluasi Model (RMSE pada data latih)")
st.write(f"RMSE Linear Regression: {rmse_lr:,.2f}")
st.write(f"RMSE Random Forest: {rmse_rf:,.2f}")

st.subheader("📋 Tabel Hasil Prediksi")
st.dataframe(df_prediksi[['Provinsi', 'Tahun', 'Prediksi (Linear Regression)', 'Prediksi (Random Forest)']])

# Export tombol
csv = df_prediksi.to_csv(index=False)
st.download_button("⬇️ Download Hasil (.csv)", csv, file_name="prediksi_padi.csv", mime="text/csv")

# Visualisasi Seaborn – Grafik Tren Produksi
st.subheader("📈 Visualisasi Tren Produksi dengan Seaborn")

# Opsi provinsi
provinsi_terpilih = st.selectbox("Pilih Provinsi:", sorted(df_prediksi['Provinsi'].unique()))
df_prov = df_prediksi[df_prediksi['Provinsi'] == provinsi_terpilih]

# Plot Seaborn Lineplot
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(data=df_prov, x='Tahun', y='Prediksi (Linear Regression)', label='Linear Regression', marker='o', ax=ax)
sns.lineplot(data=df_prov, x='Tahun', y='Prediksi (Random Forest)', label='Random Forest', marker='s', ax=ax)
ax.set_title(f"Tren Prediksi Produksi Padi Provinsi {provinsi_terpilih} (2021–2025)")
ax.set_ylabel("Produksi (Ton)")
ax.set_xlabel("Tahun")
ax.legend()
st.pyplot(fig)
