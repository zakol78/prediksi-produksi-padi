import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Judul
st.title("Prediksi Produksi Padi di Pulau Sumatera")

# Load data
df = pd.read_csv("Data_Tanaman_Padi_Sumatera_version_1.csv")
df.columns = df.columns.str.strip()  # Hilangkan spasi pada nama kolom

# Preprocessing
df['Tahun'] = df['Tahun'].astype(int)
df = df.dropna()

# Bagi data
df_train = df[df['Tahun'] <= 2020]
df_test = df[df['Tahun'] > 2020]

# Fitur dan Target
fitur = ['Luas Panen (ha)', 'Produktivitas (kw/ha)']
target = 'Produksi (ton)'

X_train = df_train[fitur]
y_train = df_train[target]
X_test = df_test[fitur]
y_test = df_test[target]

# =====================#
# ===== MODELING ===== #
# =====================#

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_train)

# Random Forest dengan parameter konservatif (anti-overfit)
rf = RandomForestRegressor(
    n_estimators=50,
    max_depth=5,
    random_state=42
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_train)

# ========================#
# ===== EVALUASI =========#
# ========================#
def evaluate_model(name, y_true, y_pred):
    return {
        "Model": name,
        "R² Score": round(r2_score(y_true, y_pred), 4),
        "MAE (ton)": f"{mean_absolute_error(y_true, y_pred):,.2f}",
        "MSE (ton²)": f"{mean_squared_error(y_true, y_pred):,.2f}",
    }

results = [
    evaluate_model("Linear Regression", y_train, y_pred_lr),
    evaluate_model("Random Forest", y_train, y_pred_rf),
]

eval_df = pd.DataFrame(results)

st.subheader("Evaluasi Model pada Data Latih")
st.dataframe(eval_df, use_container_width=True)

# ============================#
# ===== VISUALISASI =========#
# ============================#

st.subheader("Visualisasi Fitur")

fitur_pilihan = st.multiselect("Pilih fitur untuk ditampilkan", df.columns.tolist(), default=["Produksi (ton)"])
tahun_range = st.slider("Rentang Tahun", int(df['Tahun'].min()), int(df['Tahun'].max()), (1993, 2025))

if fitur_pilihan:
    df_plot = df[(df['Tahun'] >= tahun_range[0]) & (df['Tahun'] <= tahun_range[1])]
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in fitur_pilihan:
        ax.plot(df_plot['Tahun'], df_plot[col], label=col)
    ax.set_xlabel("Tahun")
    ax.set_ylabel("Nilai")
    ax.set_title("Tren Data per Tahun")
    ax.legend()
    st.pyplot(fig)

# =============================#
# ===== PREDIKSI MASA DEPAN ==#
# =============================#

st.subheader("Prediksi Produksi Padi (2021–2025)")

df_pred = df_test.copy()
df_pred['Pred_LR'] = lr.predict(X_test)
df_pred['Pred_RF'] = rf.predict(X_test)

st.dataframe(df_pred[['Provinsi', 'Tahun', 'Produksi (ton)', 'Pred_LR', 'Pred_RF']], use_container_width=True)

# Visualisasi hasil prediksi
st.subheader("Perbandingan Aktual vs Prediksi")

provinsi_pilih = st.selectbox("Pilih Provinsi", df_pred['Provinsi'].unique())

df_prov = df_pred[df_pred['Provinsi'] == provinsi_pilih]
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(df_prov['Tahun'], df_prov['Produksi (ton)'], label='Aktual', marker='o')
ax2.plot(df_prov['Tahun'], df_prov['Pred_LR'], label='Linear Regression', marker='s')
ax2.plot(df_prov['Tahun'], df_prov['Pred_RF'], label='Random Forest', marker='^')
ax2.set_title(f"Prediksi Produksi Padi di {provinsi_pilih}")
ax2.set_xlabel("Tahun")
ax2.set_ylabel("Produksi (ton)")
ax2.legend()
st.pyplot(fig2)
