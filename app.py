import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Prediksi Produksi Padi di Sumatera (1993–2025)")

# Load Data
df = pd.read_csv("Data_Tanaman_Padi_Sumatera_version_1.csv")

# Bersihkan nama kolom
df.columns = df.columns.str.strip()
st.write("Kolom di DataFrame:", df.columns.tolist())  # Tampilkan kolom untuk diagnosis

# Ubah nama kolom jika perlu
df = df.rename(columns={
    'Luas panen (ha)': 'Luas panen',
    'Curah hujan (mm)': 'Curah hujan',
    'Kelembapan (%)': 'Kelembapan',
    'Suhu rata-rata (°C)': 'Suhu rata-rata'
})

fitur = ['Luas panen', 'Curah hujan', 'Kelembapan', 'Suhu rata-rata']
target = 'Produksi'

df_train = df[df['Tahun'] <= 2020]

# Cek apakah fitur benar-benar ada
missing_columns = [f for f in fitur if f not in df_train.columns]
if missing_columns:
    st.error(f"Kolom tidak ditemukan dalam data: {missing_columns}")
    st.stop()

# Train model
lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
lr.fit(df_train[fitur], df_train[target])
rf.fit(df_train[fitur], df_train[target])

# Prediksi dan hasil
df['Prediksi LR'] = lr.predict(df[fitur])
df['Prediksi RF'] = rf.predict(df[fitur])

st.subheader("Prediksi 1993–2020")
st.dataframe(df[df['Tahun'] <= 2020][['Provinsi', 'Tahun', 'Produksi', 'Prediksi LR', 'Prediksi RF']])

# Prediksi masa depan
df_2021 = df[df['Tahun'] == 2020].copy()
df_2021['Tahun'] = 2021

def buat_data_tahun_mendatang(df_base, tahun_akhir):
    semua_tahun = []
    for tahun in range(2022, tahun_akhir + 1):
        d = df_base.copy()
        d['Tahun'] = tahun
        for kol in fitur:
            d[kol] *= np.random.uniform(0.97, 1.03)
        semua_tahun.append(d)
    return pd.concat([df_base] + semua_tahun, ignore_index=True)

df_future = buat_data_tahun_mendatang(df_2021, 2025)
df_future['Prediksi LR'] = lr.predict(df_future[fitur])
df_future['Prediksi RF'] = rf.predict(df_future[fitur])

st.subheader("Prediksi 2021–2025")
st.dataframe(df_future[['Provinsi', 'Tahun', 'Prediksi LR', 'Prediksi RF']])

# Visualisasi
st.subheader("Visualisasi Prediksi")
tahun = st.selectbox("Pilih Tahun", sorted(df_future['Tahun'].unique()))
df_plot = df_future[df_future['Tahun'] == tahun]

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(df_plot))
ax.bar(x - 0.2, df_plot['Prediksi LR'], width=0.4, label="Linear Regression")
ax.bar(x + 0.2, df_plot['Prediksi RF'], width=0.4, label="Random Forest")
ax.set_xticks(x)
ax.set_xticklabels(df_plot['Provinsi'], rotation=45, ha='right')
ax.set_ylabel("Produksi")
ax.set_title(f"Prediksi Produksi Padi Tahun {tahun}")
ax.legend()
st.pyplot(fig)

# Evaluasi
st.subheader("Evaluasi Model")
y_true = df_train[target]
y_pred_lr = lr.predict(df_train[fitur])
y_pred_rf = rf.predict(df_train[fitur])
st.dataframe(pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "R²": [r2_score(y_true, y_pred_lr), r2_score(y_true, y_pred_rf)],
    "MAE": [mean_absolute_error(y_true, y_pred_lr), mean_absolute_error(y_true, y_pred_rf)],
    "MSE": [mean_squared_error(y_true, y_pred_lr), mean_squared_error(y_true, y_pred_rf)]
}).round(4))
