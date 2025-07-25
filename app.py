import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Prediksi Produksi Padi", layout="wide")
st.title("Prediksi Produksi Padi di Sumatera 2021–2025")
st.markdown("Menggunakan Algoritma **Linear Regression** dan **Random Forest**")

# Membaca data lokal
df = pd.read_csv("Data_Tanaman_Padi_Sumatera_version_1.csv")

st.subheader("Data Asli")
st.dataframe(df)

# Split data training
df_train = df[df['Tahun'] < 2021]

fitur = ['Luas panen', 'Curah hujan', 'Kelembapan', 'Suhu rata-rata']
target = 'Produksi'

X = df_train[fitur]
y = df_train[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

# Model Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

# Prediksi tahun 2021–2025 berdasarkan data 2020
df_base = df[df['Tahun'] == 2020].copy()
hasil_prediksi = []

for tahun in range(2021, 2026):
    df_temp = df_base.copy()
    df_temp['Tahun'] = tahun
    pred_lr = lr.predict(df_temp[fitur])
    pred_rf = rf.predict(df_temp[fitur])
    
    df_result = df_temp[['Provinsi']].copy()
    df_result['Tahun'] = tahun
    df_result['Prediksi (Linear Regression)'] = pred_lr
    df_result['Prediksi (Random Forest)'] = pred_rf
    hasil_prediksi.append(df_result)

# Gabungkan semua prediksi
df_prediksi = pd.concat(hasil_prediksi, ignore_index=True)

# Tampilkan hasil
st.subheader("Hasil Prediksi Produksi Padi 2021–2025")
st.dataframe(df_prediksi.style.format({'Prediksi (Linear Regression)': '{:.2f}', 'Prediksi (Random Forest)': '{:.2f}'}))

# Grafik visualisasi per tahun
st.subheader("Grafik Prediksi Produksi per Tahun dan Provinsi")
tahun_terpilih = st.selectbox("Pilih Tahun", sorted(df_prediksi['Tahun'].unique()))
df_tahun = df_prediksi[df_prediksi['Tahun'] == tahun_terpilih]

fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.35
index = np.arange(len(df_tahun))

ax.bar(index, df_tahun['Prediksi (Linear Regression)'], bar_width, label='Linear Regression')
ax.bar(index + bar_width, df_tahun['Prediksi (Random Forest)'], bar_width, label='Random Forest')

ax.set_xlabel('Provinsi')
ax.set_ylabel('Prediksi Produksi')
ax.set_title(f'Prediksi Produksi Padi Tahun {tahun_terpilih}')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(df_tahun['Provinsi'], rotation=45, ha='right')
ax.legend()
plt.tight_layout()
st.pyplot(fig)

# Evaluasi model
st.subheader("Evaluasi Model (Test Data Split)")
st.markdown("**Linear Regression:**")
st.write(f"- MAE: {mae_lr:.2f}")
st.write(f"- RMSE: {rmse_lr:.2f}")
st.write(f"- R² Score: {r2_lr:.4f}")

st.markdown("**Random Forest:**")
st.write(f"- MAE: {mae_rf:.2f}")
st.write(f"- RMSE: {rmse_rf:.2f}")
st.write(f"- R² Score: {r2_rf:.4f}")
