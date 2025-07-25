import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(page_title="Prediksi Produksi Padi", layout="wide")
st.title("📈 Prediksi Produksi Padi di Pulau Sumatera")

# --------------------------------------------
# 1. Load Data
# --------------------------------------------
try:
    df = pd.read_csv("Data_Tanaman_Padi_Sumatera_version_1.csv")
    df.columns = df.columns.str.strip()  # hilangkan spasi ekstra
    st.success("✅ Data berhasil dimuat.")
except Exception as e:
    st.error(f"❌ Gagal memuat data: {e}")
    st.stop()

# --------------------------------------------
# 2. Cek Kolom yang Terdeteksi
# --------------------------------------------
st.sidebar.header("📋 Fitur yang Digunakan")
fitur = ['Luas panen', 'Curah hujan', 'Kelembapan', 'Suhu rata-rata']
target = 'Produksi'

missing = [col for col in fitur + [target, 'Tahun', 'Provinsi'] if col not in df.columns]
if missing:
    st.error(f"❌ Kolom hilang di dataset: {missing}")
    st.stop()

# --------------------------------------------
# 3. Filter Tahun
# --------------------------------------------
df_train = df[df['Tahun'] <= 2020]
df_test = df[df['Tahun'] > 2020]

X_train = df_train[fitur]
y_train = df_train[target]
X_test = df_test[fitur]
y_test = df_test[target]

# --------------------------------------------
# 4. Model Training
# --------------------------------------------
st.subheader("🔧 Konfigurasi Model Random Forest")
n_estimators = st.slider("Jumlah Pohon (n_estimators)", 10, 300, 100, 10)
max_depth = st.slider("Kedalaman Maksimal (max_depth)", 1, 30, 10)

rf_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Linear Regression (untuk perbandingan)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# --------------------------------------------
# 5. Evaluasi Model
# --------------------------------------------
def evaluasi_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    st.markdown(f"#### {model_name}")
    st.write(f"- R2 Score: `{r2:.4f}`")
    st.write(f"- MAE: `{mae:.2f}`")
    st.write(f"- MSE: `{mse:.2f}`")

st.subheader("📊 Evaluasi Model (Data Uji 2021–2025)")
evaluasi_model(y_test, y_pred_rf, "🌲 Random Forest")
evaluasi_model(y_test, y_pred_lr, "📉 Linear Regression")

# --------------------------------------------
# 6. Visualisasi Prediksi
# --------------------------------------------
st.subheader("📈 Visualisasi Hasil Prediksi")

fitur_grafik = st.selectbox("Pilih Provinsi", sorted(df_test['Provinsi'].unique()))

df_plot = df_test[df_test['Provinsi'] == fitur_grafik].copy()
df_plot['Prediksi_RF'] = rf_model.predict(df_plot[fitur])
df_plot['Prediksi_LR'] = lr_model.predict(df_plot[fitur])

plt.figure(figsize=(10, 5))
plt.plot(df_plot['Tahun'], df_plot[target], marker='o', label='Data Aktual', linewidth=2)
plt.plot(df_plot['Tahun'], df_plot['Prediksi_RF'], marker='s', label='Random Forest', linestyle='--')
plt.plot(df_plot['Tahun'], df_plot['Prediksi_LR'], marker='^', label='Linear Regression', linestyle='--')
plt.title(f"Prediksi Produksi Padi Provinsi {fitur_grafik}")
plt.xlabel("Tahun")
plt.ylabel("Produksi (ton)")
plt.legend()
plt.grid(True)
st.pyplot(plt)

# --------------------------------------------
# 7. Tampilkan Data
# --------------------------------------------
st.subheader("📄 Dataset yang Digunakan")
with st.expander("Klik untuk melihat data"):
    st.dataframe(df)

