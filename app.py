import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Prediksi Produksi Padi", layout="wide")
st.title("Prediksi Produksi Padi di Sumatera Tahun 2021")
st.markdown("Menggunakan Algoritma **Linear Regression** dan **Random Forest**")

# Membaca data lokal
df = pd.read_csv("data_padi.csv")

st.subheader("Data Asli")
st.dataframe(df)

# Split data
df_train = df[df['Tahun'] < 2021]
df_2021 = df[df['Tahun'] == 2020].copy()
df_2021['Tahun'] = 2021

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

# Prediksi untuk tahun 2021
pred_lr_2021 = lr.predict(df_2021[fitur])
pred_rf_2021 = rf.predict(df_2021[fitur])

hasil = df_2021[['Provinsi']].copy()
hasil['Tahun'] = 2021
hasil['Prediksi (Linear Regression)'] = pred_lr_2021
hasil['Prediksi (Random Forest)'] = pred_rf_2021

st.subheader("Hasil Prediksi Produksi Padi Tahun 2021")
st.dataframe(hasil.style.format({'Prediksi (Linear Regression)': '{:.2f}', 'Prediksi (Random Forest)': '{:.2f}'}))

# Grafik Visualisasi
st.subheader("Grafik Perbandingan Prediksi Produksi Padi")
fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.35
index = np.arange(len(hasil))

ax.bar(index, hasil['Prediksi (Linear Regression)'], bar_width, label='Linear Regression')
ax.bar(index + bar_width, hasil['Prediksi (Random Forest)'], bar_width, label='Random Forest')

ax.set_xlabel('Provinsi')
ax.set_ylabel('Prediksi Produksi')
ax.set_title('Perbandingan Prediksi Produksi Padi Tahun 2021')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(hasil['Provinsi'], rotation=45, ha='right')
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
