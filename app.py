import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Judul aplikasi
st.title("Prediksi Produksi Padi di Pulau Sumatera (2021–2025)")
st.markdown("Menggunakan algoritma **Random Forest** dan **Linear Regression**")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Data_Tanaman_Padi_Sumatera_version_1.csv")
    return df

df = load_data()

# Tampilkan data
st.subheader("Data Produksi Padi")
st.dataframe(df)

# Preprocessing
fitur = ['Luas panen', 'Curah hujan', 'Kelembapan', 'Suhu rata-rata']
target = 'Produksi'

# Pisahkan data latih dan data prediksi
df_latih = df[df['Tahun'] <= 2020]
df_prediksi = df[df['Tahun'] > 2020]

X = df_latih[fitur]
y = df_latih[target]

# Training model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

lr_model = LinearRegression()
lr_model.fit(X, y)

# Prediksi untuk tahun 2021–2025
if not df_prediksi.empty:
    X_prediksi = df_prediksi[fitur]
    rf_pred = rf_model.predict(X_prediksi)
    lr_pred = lr_model.predict(X_prediksi)

    # Tambahkan kolom prediksi ke DataFrame
    df.loc[df['Tahun'] > 2020, 'Prediksi (Random Forest)'] = rf_pred
    df.loc[df['Tahun'] > 2020, 'Prediksi (Linear Regression)'] = lr_pred

    # Tampilkan hasil prediksi
    st.subheader("Hasil Prediksi 2021–2025")
    st.dataframe(df[df['Tahun'] > 2020][['Provinsi', 'Tahun', 'Produksi', 'Prediksi (Linear Regression)', 'Prediksi (Random Forest)']])
else:
    st.warning("Data untuk tahun 2021–2025 tidak ditemukan. Silakan pastikan file CSV mencakup data tahun tersebut.")

# Visualisasi hasil prediksi
if not df_prediksi.empty:
    st.subheader("Grafik Prediksi vs Aktual")

    provinsi_pilihan = st.selectbox("Pilih Provinsi", df_prediksi['Provinsi'].unique())

    df_plot = df[df['Provinsi'] == provinsi_pilihan]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_plot['Tahun'], df_plot['Produksi'], label='Produksi Aktual', marker='o')
    
    if 'Prediksi (Linear Regression)' in df_plot.columns:
        ax.plot(df_plot['Tahun'], df_plot['Prediksi (Linear Regression)'], label='Prediksi LR', linestyle='--')
    
    if 'Prediksi (Random Forest)' in df_plot.columns:
        ax.plot(df_plot['Tahun'], df_plot['Prediksi (Random Forest)'], label='Prediksi RF', linestyle='--')

    ax.set_xlabel('Tahun')
    ax.set_ylabel('Produksi Padi')
    ax.set_title(f'Prediksi Produksi Padi - {provinsi_pilihan}')
    ax.legend()
    st.pyplot(fig)

# Evaluasi model
st.subheader("Evaluasi Model (Data Latih)")

y_pred_rf = rf_model.predict(X)
y_pred_lr = lr_model.predict(X)

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Random Forest**")
    st.write(f"R² Score: {r2_score(y, y_pred_rf):.4f}")
    st.write(f"RMSE: {mean_squared_error(y, y_pred_rf, squared=False):,.2f}")

with col2:
    st.markdown("**Linear Regression**")
    st.write(f"R² Score: {r2_score(y, y_pred_lr):.4f}")
    st.write(f"RMSE: {mean_squared_error(y, y_pred_lr, squared=False):,.2f}")
