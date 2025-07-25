import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load data
df = pd.read_csv("Data_Tanaman_Padi_Sumatera_version_1.csv")
df.columns = df.columns.str.strip()  # hapus spasi ekstra jika ada

# Definisikan fitur dan target
fitur = ['Luas panen', 'Curah hujan', 'Kelembapan', 'Suhu rata-rata']
target = 'Produksi'

# Split data train (1993–2020) dan test (2021–2025)
df_train = df[df['Tahun'] <= 2020]
df_test = df[df['Tahun'] > 2020]

# Model
lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)  # atur agar tidak overfit

# Latih model
lr.fit(df_train[fitur], df_train[target])
rf.fit(df_train[fitur], df_train[target])

# Prediksi
df_train['Pred_LR'] = lr.predict(df_train[fitur])
df_train['Pred_RF'] = rf.predict(df_train[fitur])
df_test['Pred_LR'] = lr.predict(df_test[fitur])
df_test['Pred_RF'] = rf.predict(df_test[fitur])

# Evaluasi
def evaluasi(y_true, y_pred):
    return {
        "R² Score": r2_score(y_true, y_pred),
        "MAE (ton)": mean_absolute_error(y_true, y_pred),
        "MSE (ton²)": mean_squared_error(y_true, y_pred)
    }

eval_lr_train = evaluasi(df_train[target], df_train['Pred_LR'])
eval_rf_train = evaluasi(df_train[target], df_train['Pred_RF'])
eval_lr_test = evaluasi(df_test[target], df_test['Pred_LR'])
eval_rf_test = evaluasi(df_test[target], df_test['Pred_RF'])

# Streamlit UI
st.title("Prediksi Produksi Padi di Sumatera")
st.subheader("Evaluasi Model")

# Tabel evaluasi
def tampilkan_evaluasi(judul, eval_lr, eval_rf):
    st.markdown(f"### {judul}")
    st.dataframe(pd.DataFrame({
        "Linear Regression": eval_lr,
        "Random Forest": eval_rf
    }))

tampilkan_evaluasi("Data Latih (1993–2020)", eval_lr_train, eval_rf_train)
tampilkan_evaluasi("Data Uji (2021–2025)", eval_lr_test, eval_rf_test)

# Visualisasi prediksi vs aktual
st.subheader("Visualisasi Prediksi vs Aktual")
tahun_range = st.selectbox("Pilih rentang data", ["1993–2020", "2021–2025"])
df_plot = df_train if tahun_range == "1993–2020" else df_test

plt.figure(figsize=(10, 5))
plt.plot(df_plot['Tahun'], df_plot[target], label='Aktual', marker='o')
plt.plot(df_plot['Tahun'], df_plot['Pred_LR'], label='Linear Regression', marker='s')
plt.plot(df_plot['Tahun'], df_plot['Pred_RF'], label='Random Forest', marker='^')
plt.xlabel("Tahun")
plt.ylabel("Produksi (ton)")
plt.title("Prediksi vs Aktual Produksi Padi")
plt.legend()
st.pyplot(plt)

# Korelasi fitur (opsional)
st.subheader("Korelasi dan Distribusi Fitur")
fitur_dipilih = st.multiselect("Pilih fitur yang ingin dianalisis", fitur, default=fitur[:2])

if fitur_dipilih:
    st.write("📌 Korelasi antar fitur:")
    corr = df[fitur_dipilih + [target]].corr()
    fig_corr, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='YlGnBu', ax=ax)
    st.pyplot(fig_corr)

    for f in fitur_dipilih:
        fig_feat, ax_feat = plt.subplots()
        sns.scatterplot(data=df, x=f, y=target, ax=ax_feat)
        ax_feat.set_title(f'Hasil Produksi terhadap {f}')
        st.pyplot(fig_feat)

st.caption("© 2025 Prediksi Produksi Padi - Streamlit App")
