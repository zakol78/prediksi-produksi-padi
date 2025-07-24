import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np

st.set_page_config(page_title="Prediksi Produksi Padi", layout="wide")
st.title("Prediksi Produksi Padi di Sumatera Tahun 2021")
st.markdown("Menggunakan Algoritma **Linear Regression** dan **Random Forest**")

# 1. Baca data langsung dari file lokal (tidak perlu upload)
df = pd.read_csv("data_padi.csv")

st.subheader("Data Asli")
st.dataframe(df)

# 2. Pisahkan data latih dan data prediksi
df_train = df[df['Tahun'] < 2021]
df_2021 = df[df['Tahun'] == 2020].copy()
df_2021['Tahun'] = 2021

fitur = ['Luas panen', 'Curah hujan', 'Kelembapan', 'Suhu rata-rata']
target = 'Produksi'

X = df_train[fitur]
y = df_train[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

# 4. Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

# 5. Prediksi 2021
pred_lr_2021 = lr.predict(df_2021[fitur])
pred_rf_2021 = rf.predict(df_2021[fitur])

hasil = df_2021[['Provinsi']].copy()
hasil['Tahun'] = 2021
hasil['Prediksi (Linear Regression)'] = pred_lr_2021
hasil['Prediksi (Random Forest)'] = pred_rf_2021

st.subheader("Hasil Prediksi Produksi Padi Tahun 2021")
st.dataframe(hasil.style.format({
    'Prediksi (Linear Regression)': '{:.2f}',
    'Prediksi (Random Forest)': '{:.2f}'
}))

# 6. Evaluasi Model
st.subheader("Evaluasi Model (Data Uji)")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Linear Regression**")
    st.write(f"- MAE: {mae_lr:.2f}")
    st.write(f"- RMSE: {rmse_lr:.2f}")
    st.write(f"- R² Score: {r2_lr:.4f}")

with col2:
    st.markdown("**Random Forest**")
    st.write(f"- MAE: {mae_rf:.2f}")
    st.write(f"- RMSE: {rmse_rf:.2f}")
    st.write(f"- R² Score: {r2_rf:.4f}")
