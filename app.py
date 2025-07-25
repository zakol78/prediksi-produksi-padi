import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Judul Aplikasi
st.title("Prediksi Produksi Padi di Pulau Sumatera")
st.write("Menggunakan Algoritma Random Forest dan Linear Regression")

# Load dataset
try:
    df = pd.read_csv("Data_Tanaman_Padi_Sumatera_version_1.csv")
    df.columns = df.columns.str.strip().str.replace('\u200b', '')  # bersihkan kolom
except Exception as e:
    st.error(f"Gagal memuat data: {e}")
    st.stop()

# Cek nama kolom
expected_columns = ['Provinsi', 'Tahun', 'Produksi', 'Luas panen', 'Curah hujan', 'Kelembapan', 'Suhu rata-rata']
if not all(col in df.columns for col in expected_columns):
    st.error(f"Kolom tidak lengkap! Kolom tersedia: {df.columns.tolist()}")
    st.stop()

# Pisahkan fitur dan target
fitur = ['Luas panen', 'Curah hujan', 'Kelembapan', 'Suhu rata-rata']
target = 'Produksi'

# Bagi data latih dan uji
X = df[fitur]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi model
rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
lr = LinearRegression()

# Latih model
rf.fit(X_train, y_train)
lr.fit(X_train, y_train)

# Prediksi
rf_pred_train = rf.predict(X_train)
rf_pred_test = rf.predict(X_test)
lr_pred_train = lr.predict(X_train)
lr_pred_test = lr.predict(X_test)

# Evaluasi Model
def evaluate(model_name, y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return f"**{model_name}** | RMSE: {rmse:.2f} | R²: {r2:.2f}"

st.subheader("Evaluasi Model")
st.markdown(evaluate("Random Forest (Train)", y_train, rf_pred_train))
st.markdown(evaluate("Random Forest (Test)", y_test, rf_pred_test))
st.markdown(evaluate("Linear Regression (Train)", y_train, lr_pred_train))
st.markdown(evaluate("Linear Regression (Test)", y_test, lr_pred_test))

# Visualisasi hasil prediksi
st.subheader("Perbandingan Hasil Prediksi vs Aktual")
fig, ax = plt.subplots()
ax.scatter(y_test, rf_pred_test, label='Random Forest', alpha=0.7)
ax.scatter(y_test, lr_pred_test, label='Linear Regression', alpha=0.7)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
ax.set_xlabel("Produksi Aktual")
ax.set_ylabel("Prediksi")
ax.legend()
st.pyplot(fig)

# Form Prediksi Manual
st.subheader("Prediksi Produksi Padi (Input Manual)")
luas = st.number_input("Luas Panen (ha)", min_value=0.0, step=1.0)
hujan = st.number_input("Curah Hujan (mm)", min_value=0.0, step=1.0)
lembap = st.number_input("Kelembapan (%)", min_value=0.0, max_value=100.0, step=1.0)
suhu = st.number_input("Suhu Rata-rata (°C)", min_value=0.0, step=0.1)

if st.button("Prediksi Sekarang"):
    data_input = np.array([[luas, hujan, lembap, suhu]])
    rf_pred = rf.predict(data_input)[0]
    lr_pred = lr.predict(data_input)[0]
    st.success(f"Hasil Prediksi Produksi (ton):")
    st.markdown(f"- **Random Forest**: {rf_pred:.2f} ton")
    st.markdown(f"- **Linear Regression**: {lr_pred:.2f} ton")

# Prediksi Otomatis Tahun Depan
st.subheader("Prediksi Tahun Berikutnya Otomatis")
tahun_terakhir = df['Tahun'].max()
rata_input = df[fitur].mean().values.reshape(1, -1)
rf_pred_next = rf.predict(rata_input)[0]
lr_pred_next = lr.predict(rata_input)[0]

st.markdown(f"📅 Prediksi produksi untuk tahun {tahun_terakhir + 1}:")
st.markdown(f"- Random Forest: **{rf_pred_next:.2f} ton**")
st.markdown(f"- Linear Regression: **{lr_pred_next:.2f} ton**")
