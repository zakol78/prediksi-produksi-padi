import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# Load data
df = pd.read_csv("Data_Tanaman_Padi_Sumatera_version_1.csv")

# Preprocessing
df['Tahun'] = df['Tahun'].astype(int)
df = df.dropna()

# Fitur dan target
fitur = ['Luas Panen (ha)', 'Produktivitas (kw/ha)']
target = 'Produksi (ton)'

# Pisahkan data latih dan prediksi
df_train = df[df['Tahun'] <= 2020]
df_pred = df[df['Tahun'] > 2020]

# ==================== MODELING ==================== #
# Linear Regression
lr = LinearRegression()
lr.fit(df_train[fitur], df_train[target])
lr_pred_train = lr.predict(df_train[fitur])
lr_pred_future = lr.predict(df_pred[fitur])

# Random Forest (tuned to avoid overfitting)
rf = RandomForestRegressor(
    n_estimators=50,
    max_depth=10,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42
)
rf.fit(df_train[fitur], df_train[target])
rf_pred_train = rf.predict(df_train[fitur])
rf_pred_future = rf.predict(df_pred[fitur])

# ==================== EVALUASI ==================== #
def evaluasi_model(y_true, y_pred):
    return {
        "R² Score": r2_score(y_true, y_pred),
        "MAE (ton)": mean_absolute_error(y_true, y_pred),
        "MSE (ton²)": mean_squared_error(y_true, y_pred)
    }

eval_lr = evaluasi_model(df_train[target], lr_pred_train)
eval_rf = evaluasi_model(df_train[target], rf_pred_train)

# ==================== STREAMLIT ==================== #
st.title("Prediksi Produksi Padi di Pulau Sumatera")
st.markdown("Model: **Linear Regression** dan **Random Forest**")

# === Grafik Interaktif === #
st.subheader("Visualisasi Fitur")
fitur_dipilih = st.selectbox("Pilih fitur yang ingin ditampilkan:", fitur)

fig, ax = plt.subplots()
sns.lineplot(x='Tahun', y=fitur_dipilih, data=df, marker='o', ax=ax)
ax.set_title(f"Perubahan {fitur_dipilih} per Tahun")
st.pyplot(fig)

# === Evaluasi Model === #
st.subheader("Evaluasi Model pada Data Latih")

df_eval = pd.DataFrame([
    {"Model": "Linear Regression", **eval_lr},
    {"Model": "Random Forest", **eval_rf}
])
df_eval["R² Score"] = df_eval["R² Score"].round(4)
df_eval["MAE (ton)"] = df_eval["MAE (ton)"].apply(lambda x: f"{x:,.2f}")
df_eval["MSE (ton²)"] = df_eval["MSE (ton²)"].apply(lambda x: f"{x:,.2f}")
st.table(df_eval)

# === Prediksi Masa Depan === #
st.subheader("Prediksi Produksi Padi (2021–2025)")

df_pred['Prediksi Linear'] = lr_pred_future
df_pred['Prediksi Random Forest'] = rf_pred_future
df_pred_tampil = df_pred[['Tahun', 'Provinsi', 'Prediksi Linear', 'Prediksi Random Forest']]

st.dataframe(df_pred_tampil)

# === Grafik Prediksi === #
st.subheader("Grafik Prediksi Produksi (2021–2025)")

provinsi_terpilih = st.selectbox("Pilih provinsi:", df_pred['Provinsi'].unique())

df_grafik = df_pred[df_pred['Provinsi'] == provinsi_terpilih]

fig2, ax2 = plt.subplots()
ax2.plot(df_grafik['Tahun'], df_grafik['Prediksi Linear'], label='Linear Regression', marker='o')
ax2.plot(df_grafik['Tahun'], df_grafik['Prediksi Random Forest'], label='Random Forest', marker='s')
ax2.set_title(f"Prediksi Produksi di {provinsi_terpilih} (2021–2025)")
ax2.set_ylabel("Produksi (ton)")
ax2.legend()
st.pyplot(fig2)
