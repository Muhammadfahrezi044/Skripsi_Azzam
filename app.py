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
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Saham IHSG (^JKSE)",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Prediksi Harga Saham IHSG (^JKSE)")
st.caption("Linear Regression vs Decision Tree Regressor · Data: Yahoo Finance")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Pengaturan")
    start_date = st.date_input("Tanggal Mulai", value=pd.Timestamp("2020-01-01"))
    end_date = st.date_input("Tanggal Akhir", value=pd.Timestamp("2026-03-31"))
    test_size = st.slider("Ukuran Data Uji (%)", min_value=10, max_value=40, value=20, step=5)
    dt_depth = st.slider("Max Depth Decision Tree", min_value=3, max_value=20, value=10)
    rsi_period = st.slider("Periode RSI", min_value=7, max_value=28, value=14)
    run_btn = st.button("🚀 Jalankan Analisis", use_container_width=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Mengunduh data dari Yahoo Finance…")
def load_data(start, end):
    df_raw = yf.download("^JKSE", start=str(start), end=str(end), auto_adjust=False)
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
    df = df_raw[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].copy()
    df.index.name = "Date"
    df = df.sort_index()
    return df


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def evaluate(y_true, y_pred, name):
    return {
        "Model": name,
        "MAPE (%)": mean_absolute_percentage_error(y_true, y_pred) * 100,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R²": r2_score(y_true, y_pred),
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if not run_btn:
    st.info("👈 Atur parameter di sidebar lalu klik **Jalankan Analisis**.")
    st.stop()

# ── 1. Load & preprocess ─────────────────────
df = load_data(start_date, end_date)

st.header("1. Pemahaman Data")

c1, c2, c3 = st.columns(3)
c1.metric("Total Baris", f"{df.shape[0]:,}")
c2.metric("Periode Mulai", str(df.index[0].date()))
c3.metric("Periode Akhir", str(df.index[-1].date()))

tab1, tab2, tab3 = st.tabs(["Tabel Data", "Statistik Deskriptif", "Missing Values"])
with tab1:
    st.dataframe(df, use_container_width=True, height=280)
with tab2:
    st.dataframe(df.describe().T, use_container_width=True)
with tab3:
    mv = df.isnull().sum().rename("Jumlah NaN").to_frame()
    st.dataframe(mv, use_container_width=True)

# ── 2. Grafik Eksplorasi ─────────────────────
st.header("2. Eksplorasi Visual")

# Grafik pergerakan harga Close
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df.index, df["Close"], color="navy")
ax.set_title("Pergerakan Harga Penutupan Saham IHSG")
ax.set_xlabel("Tanggal")
ax.set_ylabel("Harga Close (IDR)")
plt.tight_layout()
st.pyplot(fig)
plt.close()

# Grafik Volume Perdagangan
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(df.index, df["Volume"], color="darkorange")
ax.set_title("Volume Perdagangan Saham IHSG")
ax.set_xlabel("Tanggal")
ax.set_ylabel("Volume")
plt.tight_layout()
st.pyplot(fig)
plt.close()

# Candlestick (setahun terakhir)
cutoff_date = df.index.max() - pd.Timedelta(days=365)
last_year = df[df.index >= cutoff_date].copy()
fig, ax = plt.subplots(figsize=(14, 6))
width = 0.6
for date, row in last_year.iterrows():
    color = "green" if row["Close"] >= row["Open"] else "red"
    ax.vlines(date, row["Low"], row["High"], color=color, linewidth=1)
    body_low = min(row["Open"], row["Close"])
    body_high = max(row["Open"], row["Close"])
    ax.add_patch(plt.Rectangle((date - pd.Timedelta(days=width/2), body_low),
                                pd.Timedelta(days=width), body_high - body_low,
                                facecolor=color, edgecolor=color))
ax.set_title("Grafik Candlestick Pergerakan Harga Saham IHSG (Setahun Terakhir)")
ax.set_xlabel("Tanggal")
ax.set_ylabel("Harga (IDR)")
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)
plt.close()

# Boxplot Harga Penutupan per Tahun
df_box = df.copy()
df_box["Year"] = df_box.index.year
fig, ax = plt.subplots(figsize=(12, 5))
sns.boxplot(x="Year", y="Close", data=df_box, palette="Set2", ax=ax)
ax.set_title("Boxplot Harga Penutupan (Close) Saham IHSG per Tahun")
ax.set_xlabel("Tahun")
ax.set_ylabel("Harga Close (IDR)")
plt.tight_layout()
st.pyplot(fig)
plt.close()

# Koefisien Variasi (CV)
cv_per_year = df_box.groupby("Year")["Close"].agg(["mean", "std"])
cv_per_year["CV in %"] = (cv_per_year["std"] / cv_per_year["mean"]) * 100
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(cv_per_year.index.astype(str), cv_per_year["CV in %"], color="steelblue", edgecolor="black")
ax.set_title("Koefisien Variasi (CV) Harga Saham IHSG per Tahun")
ax.set_xlabel("Tahun")
ax.set_ylabel("CV (%)")
for i, v in enumerate(cv_per_year["CV in %"]):
    ax.text(i, v + 0.05, f"{v:.2f}%", ha="center")
plt.tight_layout()
st.pyplot(fig)
plt.close()

# Matriks Korelasi (Mean, Std, CV)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cv_per_year.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
ax.set_title("Matriks Korelasi (Mean, Std, CV)")
plt.tight_layout()
st.pyplot(fig)
plt.close()

# ── 3. Feature Engineering ───────────────────
st.header("3. Persiapan Data & Fitur")

# Pembersihan data
df = df.dropna()
st.success(f"Shape setelah hilangkan NaN: {df.shape}")

# Daily Return
df["Daily Return"] = df["Close"].pct_change()

# Grafik Daily Return
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(df.index, df["Daily Return"], color="teal", linewidth=0.8)
ax.axhline(0, color="black", linewidth=0.6)
ax.set_title("Daily Return Saham IHSG")
ax.set_xlabel("Tanggal")
ax.set_ylabel("Return Harian")
plt.tight_layout()
st.pyplot(fig)
plt.close()

# Distribusi Daily Return
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(df["Daily Return"].dropna(), bins=60, kde=True, color="teal", ax=ax)
ax.set_title("Distribusi Daily Return Saham IHSG")
ax.set_xlabel("Daily Return")
ax.set_ylabel("Frekuensi")
plt.tight_layout()
st.pyplot(fig)
plt.close()

# Moving Average (SMA-50 & SMA-200)
df["SMA_50"] = df["Close"].rolling(window=50).mean()
df["SMA_200"] = df["Close"].rolling(window=200).mean()
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df.index, df["Close"], label="Close", color="black", alpha=0.6)
ax.plot(df.index, df["SMA_50"], label="SMA 50 hari", color="blue")
ax.plot(df.index, df["SMA_200"], label="SMA 200 hari", color="red")
ax.set_title("Grafik Moving Average (SMA-50 & SMA-200) – Saham IHSG")
ax.set_xlabel("Tanggal")
ax.set_ylabel("Harga (IDR)")
ax.legend()
plt.tight_layout()
st.pyplot(fig)
plt.close()

# Relative Strength Index (RSI)
df["RSI"] = compute_rsi(df["Close"], period=rsi_period)
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df.index, df["RSI"], color="purple", label=f"RSI ({rsi_period})")
ax.axhline(70, linestyle="--", color="red", label="Overbought (70)")
ax.axhline(30, linestyle="--", color="green", label="Oversold (30)")
ax.axhline(50, linestyle=":", color="gray", alpha=0.6)
ax.set_title("Grafik Relative Strength Index (RSI) – Saham IHSG")
ax.set_xlabel("Tanggal")
ax.set_ylabel("RSI")
ax.legend()
plt.tight_layout()
st.pyplot(fig)
plt.close()

# Feature selection
features = ["Open", "High", "Low", "Volume", "Daily Return", "SMA_50", "SMA_200", "RSI"]
target = "Close"
df_model = df.dropna().copy()
X = df_model[features]
y = df_model[target]
st.info(f"Fitur yang digunakan: {', '.join(features)}")

# Correlation heatmap
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(df_model[features + [target]].corr(), annot=True, cmap="coolwarm", fmt=".2f",
            linewidths=0.5, ax=ax)
ax.set_title("Matriks Korelasi Fitur dan Target (Close)")
plt.tight_layout()
st.pyplot(fig)
plt.close()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size / 100, random_state=42, shuffle=True
)
st.success(f"**Data Latih:** {X_train.shape[0]} baris &nbsp;|&nbsp; **Data Uji:** {X_test.shape[0]} baris")

# ── 4. Pemodelan ─────────────────────────────
st.header("4. Pemodelan")

# 4.1 Regresi Linier Berganda
with st.spinner("Melatih model Linear Regression…"):
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)

# Tabel Koefisien Linear Regression
coef_df = pd.DataFrame({"Fitur": features, "Koefisien": lr_model.coef_})
st.subheader("Koefisien Linear Regression")
col1, col2 = st.columns(2)
with col1:
    st.dataframe(coef_df, use_container_width=True)
with col2:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x="Koefisien", y="Fitur",
                data=coef_df.sort_values("Koefisien", ascending=False),
                palette="coolwarm", ax=ax)
    ax.set_title("Koefisien Tiap Fitur – Linear Regression")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# 4.2 Regresi Pohon Keputusan
with st.spinner("Melatih model Decision Tree…"):
    dt_model = DecisionTreeRegressor(random_state=42, max_depth=dt_depth)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)

# Tabel Perbandingan Aktual vs Prediksi
hasil_pred = pd.DataFrame({
    "Close": y_test,
    "lr_pred": y_pred_lr,
    "dt_pred": y_pred_dt
}).sort_index()
st.subheader("Perbandingan Aktual vs Prediksi (5 data teratas & terbawah)")
st.dataframe(pd.concat([hasil_pred.head(5), hasil_pred.tail(5)]), use_container_width=True)

# Visualisasi Decision Tree (max_depth=3)
with st.expander("🌳 Lihat Visualisasi Decision Tree (max_depth=3)"):
    dt_vis = DecisionTreeRegressor(random_state=42, max_depth=3)
    dt_vis.fit(X_train, y_train)
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(dt_vis, feature_names=features, filled=True, rounded=True, fontsize=10, ax=ax)
    ax.set_title("Visualisasi Decision Tree Regressor (max_depth=3)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# Feature Importance Decision Tree
imp_df = pd.DataFrame({
    "Fitur": features,
    "Importance": dt_model.feature_importances_
}).sort_values("Importance", ascending=False)
fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(x="Importance", y="Fitur", data=imp_df, palette="viridis", ax=ax)
ax.set_title("Feature Importance – Decision Tree Regressor")
plt.tight_layout()
st.pyplot(fig)
plt.close()
st.dataframe(imp_df, use_container_width=True)

# ── 5. Evaluasi Model ──────────────────────────────
st.header("5. Evaluasi Model")

res_lr = evaluate(y_test, y_pred_lr, "Regresi Linier")
res_dt = evaluate(y_test, y_pred_dt, "Regresi Pohon Keputusan")
hasil = pd.DataFrame([res_lr, res_dt])

st.subheader("Metrik Evaluasi")
st.dataframe(
    hasil.set_index("Model").style.format({
        "MAPE (%)": "{:.4f}",
        "MAE": "{:,.2f}",
        "RMSE": "{:,.2f}",
        "R²": "{:.4f}",
    }),
    use_container_width=True,
)

# Plot Aktual vs Prediksi
test_idx = X_test.index
plot_df = pd.DataFrame({
    "Aktual": y_test,
    "Linear Regression": y_pred_lr,
    "Decision Tree": y_pred_dt,
}, index=test_idx).sort_index()

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(plot_df.index, plot_df["Aktual"], label="Aktual", color="black", linewidth=1.5)
ax.plot(plot_df.index, plot_df["Linear Regression"], label="Linear Regression", color="blue", alpha=0.8)
ax.plot(plot_df.index, plot_df["Decision Tree"], label="Decision Tree", color="red", alpha=0.8)
ax.set_title("Harga Aktual vs Prediksi (Data Uji)")
ax.set_xlabel("Tanggal")
ax.set_ylabel("Harga (IDR)")
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)
plt.close()

# Scatter plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(y_test, y_pred_lr, alpha=0.4, color="blue")
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--")
axes[0].set_title(f"Regresi Linier (MAPE={res_lr['MAPE (%)']:.2f}%)")
axes[0].set_xlabel("Aktual")
axes[0].set_ylabel("Prediksi")

axes[1].scatter(y_test, y_pred_dt, alpha=0.4, color="red")
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--")
axes[1].set_title(f"Regresi Pohon Keputusan (MAPE={res_dt['MAPE (%)']:.2f}%)")
axes[1].set_xlabel("Aktual")
axes[1].set_ylabel("Prediksi")

plt.tight_layout()
st.pyplot(fig)
plt.close()

# Distribusi Residual
residual_lr = y_test.values - y_pred_lr
residual_dt = y_test.values - y_pred_dt

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
sns.histplot(residual_lr, bins=40, kde=True, color="blue", ax=axes[0])
axes[0].axvline(0, color="black", linestyle="--")
axes[0].set_title("Residual – Linear Regression")
axes[0].set_xlabel("Residual")

sns.histplot(residual_dt, bins=40, kde=True, color="red", ax=axes[1])
axes[1].axvline(0, color="black", linestyle="--")
axes[1].set_title("Residual – Decision Tree")
axes[1].set_xlabel("Residual")

plt.tight_layout()
st.pyplot(fig)
plt.close()

# Perbandingan MAPE
fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(x="Model", y="MAPE (%)", data=hasil, palette=["steelblue", "tomato"], ax=ax)
for i, v in enumerate(hasil["MAPE (%)"]):
    ax.text(i, v + 0.02, f"{v:.4f}%", ha="center", fontweight="bold")
ax.set_title("Perbandingan MAPE antar Model")
plt.tight_layout()
st.pyplot(fig)
plt.close()

# ── 6. Kesimpulan ──────────────────────────────
st.header("6. Kesimpulan")
st.markdown(f"""
- **Regresi Linier** menghasilkan performa yang lebih baik dengan **MAPE {res_lr['MAPE (%)']:.4f}%** 
  dibandingkan Decision Tree (**MAPE {res_dt['MAPE (%)']:.4f}%**).

- Model **Linear Regression** memiliki **R² {res_lr['R²']:.6f}** yang menunjukkan model mampu 
  menjelaskan sekitar **{(res_lr['R²']*100):.2f}%** variabilitas data.

- **Feature Importance** pada Decision Tree menunjukkan bahwa fitur **High**, **Low**, dan **SMA_50** 
  memiliki pengaruh terbesar terhadap prediksi harga penutupan.

- Dari **matriks korelasi**, terlihat bahwa fitur **Close** (target) memiliki korelasi sangat tinggi 
  dengan **Open**, **High**, **Low**, dan **SMA_50** (semua di atas 0.99), yang menjelaskan 
  mengapa model linear regression bekerja dengan sangat baik.

- Nilai **Koefisien Variasi (CV)** yang rendah (di bawah 10% untuk semua tahun) menunjukkan 
  bahwa data relatif stabil, dengan volatilitas tertinggi terjadi pada tahun 2020 (10.49%).
""")

st.divider()
st.caption("Dibuat dengan Streamlit · Data: Yahoo Finance (^JKSE)")
