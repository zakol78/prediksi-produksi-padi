import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Judul
st.title("Prediksi Produksi Padi di Sumatera (1993–2025)")
st.markdown("Menggunakan algoritma: **Linear Regression** dan **Random Forest**")

# Load Data
df = pd.read_csv("Data_Tanaman_Padi_Sumatera_version_1.csv")

# Bersihkan nama kolom dari spasi
df.columns = df.columns.str.strip()

# Fitur & Target
fitur = ['Luas panen', 'Curah hujan', 'Kelembapan', 'Suhu rata-rata']
target = 'Produksi'

# Bagi data latih
df_train = df[df['Tahun'] <= 2020]

# Latih model
lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)  # dibatasi agar tidak overfit
lr.fit(df_train[fitur], df_train[target])
rf.fit(df_train[fitur], df_train[target])

# Prediksi data aktual (1993–2020)
df_actual = df.copy()
df_actual['Prediksi (Linear Regression)'] = lr.predict(df[fitur])
df_actual['Prediksi (Random Forest)'] = rf.predict(df[fitur])

st.subheader("Prediksi pada Data Tahun 1993–2020")
st.dataframe(df_actual[df_actual['Tahun'] <= 2020][['Provinsi', 'Tahun', 'Produksi', 'Prediksi (Linear Regression)', 'Prediksi (Random Forest)']])

# Prediksi masa depan
df_2021 = df[df['Tahun'] == 2020].copy()
df_2021['Tahun'] = 2021

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

df_2022_2025 = generate_future_data(df_2021, 2022, 2025)
df_future = pd.concat([df_2021, df_2022_2025], ignore_index=True)

df_future['Prediksi (Linear Regression)'] = lr.predict(df_future[fitur])
df_future['Prediksi (Random Forest)'] = rf.predict(df_future[fitur])

st.subheader("Prediksi Produksi Padi Tahun 2021–2025")
st.dataframe(df_future[['Provinsi', 'Tahun', 'Prediksi (Linear Regression)', 'Prediksi (Random Forest)']])

# Visualisasi per tahun
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

# Evaluasi model
st.subheader("Evaluasi Model pada Data Latih")
y_true = df_train[target]
y_pred_lr = lr.predict(df_train[fitur])
y_pred_rf = rf.predict(df_train[fitur])

eval_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest'],
    'R² Score': [r2_score(y_true, y_pred_lr), r2_score(y_true, y_pred_rf)],
    'MAE (ton)': [mean_absolute_error(y_true, y_pred_lr), mean_absolute_error(y_true, y_pred_rf)],
    'MSE (ton²)': [mean_squared_error(y_true, y_pred_lr), mean_squared_error(y_true, y_pred_rf)]
})

st.dataframe(eval_df.style.format({
    'R² Score': "{:.4f}",
    'MAE (ton)': "{:,.2f}",
    'MSE (ton²)': "{:,.2f}"
}))
