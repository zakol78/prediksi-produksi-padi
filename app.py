import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.title("Prediksi Produksi Padi di Pulau Sumatera (2021–2025)")

@st.cache_data
def load_data():
    df = pd.read_csv("Data_Tanaman_Padi_Sumatera_version_1.csv")
    return df

df = load_data()

# Tampilkan data awal
st.subheader("Data Asli")
st.dataframe(df)

# Fitur dan target
fitur = ['Luas panen', 'Curah hujan', 'Kelembapan', 'Suhu rata-rata']
target = 'Produksi'

# Data latih: sebelum 2021
df_latih = df[df['Tahun'] <= 2020]
X = df_latih[fitur]
y = df_latih[target]

# Training model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
lr = LinearRegression()
rf.fit(X, y)
lr.fit(X, y)

# Buat data tahun 2021–2025 berdasarkan rata-rata fitur per provinsi
provinsi_unik = df['Provinsi'].unique()
tahun_prediksi = [2021, 2022, 2023, 2024, 2025]
rows = []

for prov in provinsi_unik:
    df_prov = df[df['Provinsi'] == prov]
    mean_fitur = df_prov[fitur].mean()
    for th in tahun_prediksi:
        row = {'Provinsi': prov, 'Tahun': th}
        row.update(mean_fitur.to_dict())
        rows.append(row)

df_prediksi = pd.DataFrame(rows)

# Prediksi dengan kedua model
X_pred = df_prediksi[fitur]
df_prediksi['Prediksi (Random Forest)'] = rf.predict(X_pred)
df_prediksi['Prediksi (Linear Regression)'] = lr.predict(X_pred)

# Gabung hasil prediksi ke data lama
df_all = pd.concat([df, df_prediksi], ignore_index=True)

# Tampilkan hasil prediksi
st.subheader("Hasil Prediksi Tahun 2021–2025")
st.dataframe(df_prediksi[['Provinsi', 'Tahun', 'Prediksi (Linear Regression)', 'Prediksi (Random Forest)']])

# Visualisasi
st.subheader("Visualisasi Prediksi per Provinsi")
provinsi_pilih = st.selectbox("Pilih Provinsi", provinsi_unik)

df_plot = df_all[df_all['Provinsi'] == provinsi_pilih]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df_plot['Tahun'], df_plot['Produksi'], marker='o', label="Produksi Aktual")
ax.plot(df_plot['Tahun'], df_plot['Prediksi (Random Forest)'], linestyle='--', label="Random Forest")
ax.plot(df_plot['Tahun'], df_plot['Prediksi (Linear Regression)'], linestyle='--', label="Linear Regression")
ax.set_title(f"Produksi Padi - {provinsi_pilih}")
ax.set_xlabel("Tahun")
ax.set_ylabel("Produksi")
ax.legend()
st.pyplot(fig)

# Evaluasi Model
st.subheader("Evaluasi Model pada Data Latih")

y_pred_rf = rf.predict(X)
y_pred_lr = lr.predict(X)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Random Forest**")
    st.write(f"R² Score: {r2_score(y, y_pred_rf):.4f}")
    st.write(f"RMSE: {np.sqrt(mean_squared_error(y, y_pred_rf)):.2f}")

with col2:
    st.markdown("**Linear Regression**")
    st.write(f"R² Score: {r2_score(y, y_pred_lr):.4f}")
    st.write(f"RMSE: {np.sqrt(mean_squared_error(y, y_pred_lr)):.2f}")
