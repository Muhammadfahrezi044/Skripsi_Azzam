import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import warnings

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["axes.grid"] = True

# ─────────────────────────────────────────────
# PAGE CONFIG & SESSION STATE
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Saham IHSG",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Prediksi Harga Saham IHSG (^JKSE)")
st.caption("Linear Regression vs Decision Tree Regressor · Versi Integrasi Lengkap Colab & Multi-Source Ingestion")

# Inisialisasi session_state agar aplikasi tidak reset saat interaksi widget (seperti saat klik download/upload)
if "analisis_berjalan" not in st.session_state:
    st.session_state["analisis_berjalan"] = False

# ─────────────────────────────────────────────
# SIDEBAR CONFIGURATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Pengaturan")
    
    # Opsi Sumber Data Utama
    sumber_data = st.radio("Pilih Sumber Data:", ["Yahoo Finance (Otomatis)", "Unggah Dataset CSV (Investing.id / Lainnya)"])
    
    if sumber_data == "Yahoo Finance (Otomatis)":
        start_date = st.date_input("Tanggal Mulai", value=pd.Timestamp("2020-01-01"))
        end_date   = st.date_input("Tanggal Akhir",  value=pd.Timestamp("2026-03-31"))
        file_dataset = None
    else:
        file_dataset = st.file_uploader("Unggah File CSV Dataset Utama", type=["csv"])
        
    test_size  = st.slider("Ukuran Data Uji (%)", min_value=10, max_value=40, value=20, step=5)
    dt_depth   = st.slider("Max Depth Decision Tree", min_value=3, max_value=20, value=10)
    rsi_period = st.slider("Periode RSI", min_value=7, max_value=28, value=14)
    
    # Memicu perubahan state saat tombol diklik
    if st.button("🚀 Jalankan Analisis", use_container_width=True):
        st.session_state["analisis_berjalan"] = True

# ─────────────────────────────────────────────
# CORE HELPERS
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Mengunduh data dari Yahoo Finance…")
def load_data(start, end):
    df_raw = yf.download("^JKSE", start=str(start), end=str(end), auto_adjust=False)
    # Flatten MultiIndex jika versi yfinance baru menggunakannya
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
    # Bersihkan nama kolom dari spasi tersembunyi
    df_raw.columns = [str(col).strip() for col in df_raw.columns]
    df = df_raw[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].copy()
    df.index.name = "Date"
    df = df.sort_index()
    return df


def compute_rsi(series, period=14):
    delta    = series.diff()
    gain     = delta.where(delta > 0, 0.0)
    loss     = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def evaluate(y_true, y_pred, name):
    return {
        "Model":    name,
        "MAPE (%)": mean_absolute_percentage_error(y_true, y_pred) * 100,
        "MAE":      mean_absolute_error(y_true, y_pred),
        "RMSE":     np.sqrt(mean_squared_error(y_true, y_pred)),
        "R²":       r2_score(y_true, y_pred),
    }


def bersihkan_nilai(val):
    if pd.isna(val) or str(val).strip() in ['-', '']:
        return 0.0
    val_str = str(val).strip()
    
    # Konversi format Volume Investing.com (M=Juta, B=Miliar, K=Ribu)
    multiplier = 1
    if val_str.upper().endswith('M'):
        multiplier = 1_000_000
        val_str = val_str[:-1]
    elif val_str.upper().endswith('B'):
        multiplier = 1_000_000_000
        val_str = val_str[:-1]
    elif val_str.upper().endswith('K'):
        multiplier = 1_000
        val_str = val_str[:-1]
        
    # Konversi titik ribuan dan koma desimal ala Indonesia
    if '.' in val_str and ',' in val_str:
        val_str = val_str.replace('.', '').replace(',', '.')
    elif ',' in val_str:
        if len(val_str.split(',')[1]) <= 2:
            val_str = val_str.replace(',', '.')
        else:
            val_str = val_str.replace(',', '')
            
    try:
        return float(val_str) * multiplier
    except:
        return 0.0


def tebak_kolom(pilihan, kata_kunci):
    for p in pilihan:
        if any(kw in str(p).lower() for kw in kata_kunci):
            return pilihan.index(p)
    return 0


# ─────────────────────────────────────────────
# MAIN EXECUTION INTERFACE
# ─────────────────────────────────────────────
if not st.session_state["analisis_berjalan"]:
    st.info("👈 Atur parameter atau unggah file di sidebar lalu klik **Jalankan Analisis**.")
    st.stop()

# ── 1. Load & Preprocess Data ────────────────
if sumber_data == "Yahoo Finance (Otomatis)":
    df = load_data(start_date, end_date)
else:
    if file_dataset is None:
        st.warning("⚠️ Silakan unggah file dataset CSV terlebih dahulu di sidebar lalu klik Jalankan Analisis!")
        st.stop()
    
    # Baca file mentah
    df_raw = pd.read_csv(file_dataset)
    daftar_kolom = list(df_raw.columns)
    
    st.success("📊 Dataset Berhasil Diunggah!")
    st.markdown("### 🔍 Konfigurasi Pemetaan Kolom Dataset Utama")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        col_date = st.selectbox("📅 Kolom Tanggal:", daftar_kolom, index=tebak_kolom(daftar_kolom, ['tang', 'date', 'time', 'tgl']))
        col_open = st.selectbox("📈 Kolom Open / Pembukaan:", daftar_kolom, index=tebak_kolom(daftar_kolom, ['buka', 'open', 'pembukaan']))
    with c2:
        col_high = st.selectbox("🔼 Kolom High / Tertinggi:", daftar_kolom, index=tebak_kolom(daftar_kolom, ['tinggi', 'high', 'max', 'tertinggi']))
        col_low = st.selectbox("🔽 Kolom Low / Terendah:", daftar_kolom, index=tebak_kolom(daftar_kolom, ['rendah', 'low', 'min', 'terendah']))
    with c3:
        col_close = st.selectbox("🎯 Kolom Close / Terakhir:", daftar_kolom, index=tebak_kolom(daftar_kolom, ['akhir', 'close', 'tutup', 'terakhir', 'terupdate']))
        col_volume = st.selectbox("📊 Kolom Volume / Vol.:", daftar_kolom, index=tebak_kolom(daftar_kolom, ['vol', 'volume']))

    try:
        df = pd.DataFrame()
        df['Open'] = df_raw[col_open].apply(bersihkan_nilai)
        df['High'] = df_raw[col_high].apply(bersihkan_nilai)
        df['Low'] = df_raw[col_low].apply(bersihkan_nilai)
        df['Close'] = df_raw[col_close].apply(bersihkan_nilai)
        df['Volume'] = df_raw[col_volume].apply(bersihkan_nilai)
        
        df.index = pd.to_datetime(df_raw[col_date], errors='coerce')
        df.index.name = "Date"
        df = df.dropna().sort_index()
        
        if df.empty:
            st.error("❌ Komponen konversi menghasilkan dataframe kosong. Periksa format kolom.")
            st.stop()
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan pemrosesan struktur file: {e}")
        st.stop()

# ── Tampilan Data ────────────────────────────
st.header("1. Pemahaman Data")
c1, c2, c3 = st.columns(3)
c1.metric("Total Baris Data", f"{df.shape[0]:,}")
c2.metric("Periode Mulai",    str(df.index[0].date()))
c3.metric("Periode Akhir",    str(df.index[-1].date()))

tab1, tab2, tab3 = st.tabs(["Tabel Data", "Statistik Deskriptif", "Missing Values"])
with tab1:
    st.dataframe(df, use_container_width=True, height=250)
with tab2:
    st.dataframe(df.describe().T, use_container_width=True)
with tab3:
    st.dataframe(df.isnull().sum().rename("Jumlah NaN").to_frame(), use_container_width=True)

# ── 2. Eksplorasi Visual ─────────────────────
st.header("2. Eksplorasi Visual")

fig, axes = plt.subplots(2, 1, figsize=(14, 7))
axes[0].plot(df.index, df["Close"], color="navy")
axes[0].set_title("Harga Penutupan IHSG"); axes[0].set_ylabel("IDR")
axes[1].plot(df.index, df["Volume"], color="darkorange")
axes[1].set_title("Volume Perdagangan Saham"); axes[1].set_ylabel("Volume")
plt.tight_layout()
st.pyplot(fig); plt.close()

# Perbaikan pemotongan grafik candlestick setahun terakhir secara aman (mengganti .last())
terakhir = df.index.max()
mulai_tanggal = terakhir - pd.Timedelta(days=365)
last_year = df.loc[mulai_tanggal:].copy()

fig2, ax2 = plt.subplots(figsize=(14, 5))
for date, row in last_year.iterrows():
    color = "green" if row["Close"] >= row["Open"] else "red"
    ax2.vlines(date, row["Low"], row["High"], color=color, linewidth=0.8)
    body_low  = min(row["Open"], row["Close"])
    body_high = max(row["Open"], row["Close"])
    ax2.add_patch(plt.Rectangle(
        (date - pd.Timedelta(days=0.3), body_low),
        pd.Timedelta(days=0.6), body_high - body_low,
        facecolor=color, edgecolor=color))
ax2.set_title("Visualisasi Candlestick (Setahun Terakhir)")
plt.xticks(rotation=45); plt.tight_layout()
st.pyplot(fig2); plt.close()

df_box = df.copy()
df_box["Year"] = df_box.index.year
cv_per_year = df_box.groupby("Year")["Close"].agg(["mean", "std"])
cv_per_year["CV (%)"] = (cv_per_year["std"] / cv_per_year["mean"]) * 100

col_box, col_cv = st.columns(2)
with col_box:
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    sns.boxplot(x="Year", y="Close", data=df_box, palette="Set2", ax=ax3)
    ax3.set_title("Boxplot Nilai Close per Tahun")
    plt.tight_layout(); st.pyplot(fig3); plt.close()
with col_cv:
    fig4, ax4 = plt.subplots(figsize=(7, 4))
    ax4.bar(cv_per_year.index.astype(str), cv_per_year["CV (%)"], color="steelblue", edgecolor="black")
    ax4.set_title("Koefisien Variasi (CV) Tren per Tahun")
    for i, v in enumerate(cv_per_year["CV (%)"]):
        ax4.text(i, v + 0.1, f"{v:.2f}%", ha="center", fontsize=8, fontweight='bold')
    plt.tight_layout(); st.pyplot(fig4); plt.close()

# ── 3. Feature Engineering ───────────────────
st.header("3. Persiapan Data & Fitur")

df["Daily Return"] = df["Close"].pct_change()
df["SMA_50"]       = df["Close"].rolling(50).mean()
df["SMA_200"]      = df["Close"].rolling(200).mean()
df["RSI"]          = compute_rsi(df["Close"], period=rsi_period)
df_model           = df.dropna().copy()

fig5, ax5 = plt.subplots(figsize=(14, 4))
ax5.plot(df_model.index, df_model["Close"],   label="Close Price", color="black", alpha=0.5)
ax5.plot(df_model.index, df_model["SMA_50"],  label="SMA 50",      color="blue")
ax5.plot(df_model.index, df_model["SMA_200"], label="SMA 200",     color="red")
ax5.set_title("Indikator Moving Average Tren"); ax5.legend()
plt.tight_layout(); st.pyplot(fig5); plt.close()

features = ["Open", "High", "Low", "Volume", "Daily Return", "SMA_50", "SMA_200", "RSI"]
target   = "Close"
X        = df_model[features]
y        = df_model[target]

fig6, ax6 = plt.subplots(figsize=(8, 6))
sns.heatmap(df_model[features + [target]].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax6)
ax6.set_title("Matriks Korelasi Sumbu Fitur & Target")
plt.tight_layout(); st.pyplot(fig6); plt.close()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42, shuffle=True)
st.info(f"💾 **Distribusi Data:** Train Set = {X_train.shape[0]} baris | Test Set = {X_test.shape[0]} baris")

# ── 4. Modelling ─────────────────────────────
st.header("4. Pemodelan Machine Learning")

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

dt_model = DecisionTreeRegressor(random_state=42, max_depth=dt_depth)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

col_ml1, col_ml2 = st.columns(2)
with col_ml1:
    coef_df = pd.DataFrame({"Fitur": features, "Koefisien": lr_model.coef_}).sort_values("Koefisien", ascending=False)
    fig7, ax7 = plt.subplots(figsize=(6, 4))
    sns.barplot(x="Koefisien", y="Fitur", data=coef_df, palette="coolwarm", ax=ax7)
    ax7.set_title("Koefisien Pengaruh Linear Regression"); st.pyplot(fig7); plt.close()
with col_ml2:
    imp_df = pd.DataFrame({"Fitur": features, "Importance": dt_model.feature_importances_}).sort_values("Importance", ascending=False)
    fig8, ax8 = plt.subplots(figsize=(6, 4))
    sns.barplot(x="Importance", y="Fitur", data=imp_df, palette="viridis", ax=ax8)
    ax8.set_title("Tingkat Kepentingan Fitur Decision Tree"); st.pyplot(fig8); plt.close()

# ── 5. Evaluasi Model (Sesuai Google Colab) ──
st.header("5. Evaluasi Performa Model")

res_lr  = evaluate(y_test, y_pred_lr, "Regresi Linier")
res_dt  = evaluate(y_test, y_pred_dt, "Regresi Pohon Keputusan")
hasil   = pd.DataFrame([res_lr, res_dt])

st.dataframe(
    hasil.set_index("Model").style.format({
        "MAPE (%)": "{:.4f}%", "MAE": "{:,.2f}", "RMSE": "{:,.2f}", "R²": "{:.4f}"
    }), use_container_width=True
)

# Grafik Grouped Bar Chart Menggunakan metode .melt() dari File Colab
st.subheader("📊 Grafik Perbandingan Metrik Error")
hasil_melt = hasil.melt(id_vars="Model", value_vars=["MAPE (%)", "MAE", "RMSE"], var_name="Metrik", value_name="Nilai")

fig9, ax9 = plt.subplots(figsize=(11, 5))
sns.barplot(x="Metrik", y="Nilai", hue="Model", data=hasil_melt, palette=["steelblue", "tomato"], ax=ax9)
ax9.set_title("Perbandingan Metrik Error Model (Sesuai Hasil Google Colab)")
ax9.set_ylabel("Nilai Parameter")

# Penambahan text di atas batang chart
for p in ax9.patches:
    height = p.get_height()
    if height > 0:
        ax9.annotate(f"{height:,.2f}", (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='center', xytext=(0, 6), textcoords='offset points', fontsize=9, fontweight='bold')
plt.tight_layout(); st.pyplot(fig9); plt.close()

# Line Plot Pergerakan Aktual vs Prediksi
plot_df = pd.DataFrame({"Aktual": y_test, "Linear Regression": y_pred_lr, "Decision Tree": y_pred_dt}, index=X_test.index).sort_index()
fig10, ax10 = plt.subplots(figsize=(14, 4))
ax10.plot(plot_df.index, plot_df["Aktual"], label="Harga Aktual", color="black", linewidth=1.5)
ax10.plot(plot_df.index, plot_df["Linear Regression"], label="Prediksi LR", color="blue", alpha=0.7)
ax10.plot(plot_df.index, plot_df["Decision Tree"], label="Prediksi DT", color="red", alpha=0.7)
ax10.set_title("Plot Garis Komparasi Pergerakan Harga Aktual vs Hasil Prediksi"); ax10.legend()
plt.tight_layout(); st.pyplot(fig10); plt.close()

# ── 6. Prediksi via File CSV Massal ───────────
st.header("6. Prediksi via File CSV (Massal)")
st.markdown("Unggah file `.csv` baru (dari Investing.id atau platform lain) untuk langsung mendapatkan prediksi penutupan secara massal.")

file_prediksi = st.file_uploader("Unggah File CSV Baru Untuk Prediksi", type=["csv"], key="uploader_prediksi")

if file_prediksi is not None:
    df_pred_raw = pd.read_csv(file_prediksi)
    daftar_kolom_pred = list(df_pred_raw.columns)
    
    st.markdown("### 🔍 Konfigurasi Pemetaan Kolom Data Prediksi")
    
    cp1, cp2, cp3 = st.columns(3)
    with cp1:
        col_date_p = st.selectbox("📅 Kolom Tanggal:", daftar_kolom_pred, index=tebak_kolom(daftar_kolom_pred, ['tang', 'date', 'time', 'tgl']), key="p_date")
        col_open_p = st.selectbox("📈 Kolom Open / Pembukaan:", daftar_kolom_pred, index=tebak_kolom(daftar_kolom_pred, ['buka', 'open', 'pembukaan']), key="p_open")
    with cp2:
        col_high_p = st.selectbox("🔼 Kolom High / Tertinggi:", daftar_kolom_pred, index=tebak_kolom(daftar_kolom_pred, ['tinggi', 'high', 'max', 'tertinggi']), key="p_high")
        col_low_p = st.selectbox("🔽 Kolom Low / Terendah:", daftar_kolom_pred, index=tebak_kolom(daftar_kolom_pred, ['rendah', 'low', 'min', 'terendah']), key="p_low")
    with cp3:
        col_close_p = st.selectbox("🎯 Kolom Close / Terakhir (Target):", daftar_kolom_pred, index=tebak_kolom(daftar_kolom_pred, ['akhir', 'close', 'tutup', 'terakhir', 'terupdate']), key="p_close")
        col_volume_p = st.selectbox("📊 Kolom Volume / Vol.:", daftar_kolom_pred, index=tebak_kolom(daftar_kolom_pred, ['vol', 'volume']), key="p_volume")

    try:
        df_p = pd.DataFrame()
        df_p['Open'] = df_pred_raw[col_open_p].apply(bersihkan_nilai)
        df_p['High'] = df_pred_raw[col_high_p].apply(bersihkan_nilai)
        df_p['Low'] = df_pred_raw[col_low_p].apply(bersihkan_nilai)
        df_p['Close_Aktual'] = df_pred_raw[col_close_p].apply(bersihkan_nilai)
        df_p['Volume'] = df_pred_raw[col_volume_p].apply(bersihkan_nilai)
        
        df_p.index = pd.to_datetime(df_pred_raw[col_date_p], errors='coerce')
        df_p.index.name = "Date"
        df_p = df_p.dropna(subset=['Open', 'High', 'Low', 'Volume']).sort_index()

        # Pembuatan Fitur Otomatis pada Skenario Sekuens Data Baru
        df_p["Daily Return"] = df_p["Close_Aktual"].pct_change().fillna(0.0)
        df_p["SMA_50"]       = df_p["Close_Aktual"].rolling(50, min_periods=1).mean()
        df_p["SMA_200"]      = df_p["Close_Aktual"].rolling(200, min_periods=1).mean()
        df_p["RSI"]          = compute_rsi(df_p["Close_Aktual"], period=rsi_period).fillna(50.0)

        # Proses Klasifikasi Prediksi Massal Model
        X_p = df_p[features]
        df_p["Prediksi_Linear_Regression"] = lr_model.predict(X_p)
        df_p["Prediksi_Decision_Tree"]     = dt_model.predict(X_p)

        st.success("🎉 Prediksi Massal Berhasil Diselesaikan!")
        
        kolom_tampil = ["Open", "High", "Low", "Volume", "Close_Aktual", "Prediksi_Linear_Regression", "Prediksi_Decision_Tree"]
        st.dataframe(
            df_p[kolom_tampil].style.format({
                "Open": "Rp {:,.2f}", "High": "Rp {:,.2f}", "Low": "Rp {:,.2f}", "Volume": "{:,.0f}",
                "Close_Aktual": "Rp {:,.2f}", "Prediksi_Linear_Regression": "Rp {:,.2f}", "Prediksi_Decision_Tree": "Rp {:,.2f}"
            }), use_container_width=True
        )

        # Tombol Unduh Output CSV
        csv_download = df_p.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Unduh File Hasil Prediksi (CSV)",
            data=csv_download,
            file_name="hasil_prediksi_saham_massal.csv",
            mime="text/csv",
            use_container_width=True
        )
    except Exception as e:
        st.error(f"❌ Gagal mengeksekusi kalkulasi prediksi pada data baru: {e}")
else:
    st.info("💡 Unggah berkas berkode `.csv` pada area di atas untuk melakukan pengujian data baru.")

st.divider()
st.caption("Dibuat dengan Streamlit · Data Integration Dashboard")
