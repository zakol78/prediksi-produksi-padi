import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Judul
st.title("Prediksi Produksi Padi di Sumatera (1993–2030)")
st.markdown("Menggunakan algoritma: **Linear Regression** dan **Random Forest**")

# Load Data
df = pd.read_csv("Data_Tanaman_Padi_Sumatera_version_1.csv")

# Fitur & Target
fitur = ['Luas panen', 'Curah hujan', 'Kelembapan', 'Suhu rata-rata']
target = 'Produksi'

# Bagi data latih (1993–2020)
df_train = df[df['Tahun'] <= 2020]

# Split train-validation
X_train, X_val, y_train, y_val = train_test_split(
    df_train[fitur], df_train[target], test_size=0.2, random_state=42
)

# Latih model
lr = LinearRegression()
rf = RandomForestRegressor(random_state=42)
lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Prediksi data aktual (1993–2020)
df_actual = df.copy()
df_actual['Prediksi (Linear Regression)'] = lr.predict(df[fitur])
df_actual['Prediksi (Random Forest)'] = rf.predict(df[fitur])

# Tampilkan data 1993–2020
st.subheader("Prediksi pada Data Tahun 1993–2020")
st.dataframe(df_actual[df_actual['Tahun'] <= 2020][['Provinsi', 'Tahun', 'Produksi', 'Prediksi (Linear Regression)', 'Prediksi (Random Forest)']])

# Prediksi masa depan 2025–2030
df_2025 = df[df['Tahun'] == 2020].copy()
df_2025['Tahun'] = 2025

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

df_2026_2030 = generate_future_data(df_2025, 2026, 2030)
df_future = pd.concat([df_2025, df_2026_2030], ignore_index=True)

df_future['Prediksi (Linear Regression)'] = lr.predict(df_future[fitur])
df_future['Prediksi (Random Forest)'] = rf.predict(df_future[fitur])

# Tampilkan data 2025–2030
st.subheader("Prediksi Produksi Padi Tahun 2025–2030")
st.dataframe(df_future[['Provinsi', 'Tahun', 'Prediksi (Linear Regression)', 'Prediksi (Random Forest)']])

# Visualisasi per tahun (Bar Chart)
st.subheader("Visualisasi Perbandingan Prediksi")
tahun_terpilih = st.selectbox("Pilih Tahun untuk Ditampilkan", sorted(df_future['Tahun'].unique()))
df_tampil = df_future[df_future['Tahun'] == tahun_terpilih]

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(df_tampil))
width = 0.35

ax.bar(x - width/2, df_tampil['Prediksi (Linear Regression)'], width, label='Linear Regression')
ax.bar(x + width/2, df_tampil['Prediksi (Random Forest)'], width, label='Random Forest')

ax.set_xticks(x)
ax.set_xticklabels(df_tampil['Provinsi'], rotation=45, ha='right')
ax.set_ylabel("Produksi (Ton)")
ax.set_title(f"Prediksi Produksi Padi Tahun {tahun_terpilih}")
ax.legend()

st.pyplot(fig)

# Visualisasi Tren Nasional 1993–2030 (Line Chart)
st.subheader("Grafik Perbandingan Produksi dengan Hasil Prediksi Model Random Forest dan Linear Regression per Tahun (1993–2030)")

df_gabungan = pd.concat([
    df_actual[['Tahun', 'Prediksi (Linear Regression)', 'Prediksi (Random Forest)']],
    df_future[['Tahun', 'Prediksi (Linear Regression)', 'Prediksi (Random Forest)']]
], ignore_index=True)

df_tren = df_gabungan.groupby('Tahun').sum().reset_index()

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(df_tren['Tahun'], df_tren['Prediksi (Linear Regression)'], marker='o', label='Linear Regression')
ax2.plot(df_tren['Tahun'], df_tren['Prediksi (Random Forest)'], marker='s', label='Random Forest')

ax2.set_title("Grafik Perbandingan Produksi dengan Hasil Prediksi Model Random Forest dan Linear Regression per Tahun (1993–2030)")
ax2.set_xlabel("Tahun")
ax2.set_ylabel("Total Produksi (Ton)")
ax2.legend()
ax2.grid(True)

st.pyplot(fig2)

# Evaluasi model
st.subheader("Evaluasi Model pada Data Validasi")
st.caption("Evaluasi dilakukan pada 20% data validasi dari tahun 1993–2020. Data tahun 2025–2030 hanya digunakan untuk prediksi.")

def rrmse(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mean_actual = np.mean(y_true)
    return (rmse / mean_actual) * 100

y_pred_lr = lr.predict(X_val)
y_pred_rf = rf.predict(X_val)

eval_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest'],
    'R² Score': [r2_score(y_val, y_pred_lr), r2_score(y_val, y_pred_rf)],
    'MAE (ton)': [mean_absolute_error(y_val, y_pred_lr), mean_absolute_error(y_val, y_pred_rf)],
    'MSE (ton²)': [mean_squared_error(y_val, y_pred_lr), mean_squared_error(y_val, y_pred_rf)],
    'RRMSE (%)': [rrmse(y_val, y_pred_lr), rrmse(y_val, y_pred_rf)]
})

st.dataframe(eval_df.style.format({
    'R² Score': "{:.4f}",
    'MAE (ton)': "{:,.2f}",
    'MSE (ton²)': "{:,.2f}",
    'RRMSE (%)': "{:.2f}%"
}))
