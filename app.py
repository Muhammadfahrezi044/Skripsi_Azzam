import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.subplots as plt
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

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Saham IHSG",
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
    end_date   = st.date_input("Tanggal Akhir",  value=pd.Timestamp("2026-03-31"))
    test_size  = st.slider("Ukuran Data Uji (%)", min_value=10, max_value=40, value=20, step=5)
    dt_depth   = st.slider("Max Depth Decision Tree", min_value=3, max_value=20, value=10)
    rsi_period = st.slider("Periode RSI", min_value=7, max_value=28, value=14)
    run_btn    = st.button("🚀 Jalankan Analisis", use_container_width=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Mengunduh data dari Yahoo Finance…")
def load_data(start, end):
    df_raw = yf.download("^JKSE", start=str(start), end=str(end), auto_adjust=False)
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
    
    # Membersihkan nama kolom dari spasi gaib agar sinkron dengan fitur model
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
c1.metric("Total Baris",  f"{df.shape[0]:,}")
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

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
axes[0].plot(df.index, df["Close"], color="navy")
axes[0].set_title("Harga Penutupan IHSG"); axes[0].set_ylabel("IDR")
axes[1].plot(df.index, df["Volume"], color="darkorange")
axes[1].set_title("Volume Perdagangan"); axes[1].set_ylabel("Volume")
plt.tight_layout()
st.pyplot(fig); plt.close()

# Candlestick (setahun terakhir)
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
ax2.set_title("Candlestick Setahun Terakhir")
ax2.set_xlabel("Tanggal"); ax2.set_ylabel("Harga (IDR)")
plt.xticks(rotation=45); plt.tight_layout()
st.pyplot(fig2); plt.close()

# Boxplot per tahun + CV
df_box        = df.copy()
df_box["Year"] = df_box.index.year
cv_per_year   = df_box.groupby("Year")["Close"].agg(["mean", "std"])
cv_per_year["CV in %"] = (cv_per_year["std"] / cv_per_year["mean"]) * 100

col_box, col_cv = st.columns(2)
with col_box:
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    sns.boxplot(x="Year", y="Close", data=df_box, palette="Set2", ax=ax3)
    ax3.set_title("Boxplot Close per Tahun")
    plt.tight_layout(); st.pyplot(fig3); plt.close()
with col_cv:
    fig4, ax4 = plt.subplots(figsize=(7, 4))
    ax4.bar(cv_per_year.index.astype(str), cv_per_year["CV in %"], color="steelblue", edgecolor="black")
    ax4.set_title("Koefisien Variasi (CV) per Tahun"); ax4.set_ylabel("CV (%)")
    for i, v in enumerate(cv_per_year["CV in %"]):
        ax4.text(i, v + 0.05, f"{v:.2f}%", ha="center", fontsize=8)
    plt.tight_layout(); st.pyplot(fig4); plt.close()

# ── 3. Feature Engineering ───────────────────
st.header("3. Persiapan Data & Fitur")

df = df.dropna()
df["Daily Return"] = df["Close"].pct_change()
df["SMA_50"]       = df["Close"].rolling(50).mean()
df["SMA_200"]      = df["Close"].rolling(200).mean()
df["RSI"]          = compute_rsi(df["Close"], period=rsi_period)

# Moving Average chart
fig5, ax5 = plt.subplots(figsize=(14, 5))
ax5.plot(df.index, df["Close"],   label="Close",   color="black",  alpha=0.6)
ax5.plot(df.index, df["SMA_50"],  label="SMA 50",  color="blue")
ax5.plot(df.index, df["SMA_200"], label="SMA 200", color="red")
ax5.set_title("Moving Average SMA-50 & SMA-200"); ax5.set_ylabel("IDR")
ax5.legend(); plt.tight_layout(); st.pyplot(fig5); plt.close()

# RSI chart
fig6, ax6 = plt.subplots(figsize=(14, 4))
ax6.plot(df.index, df["RSI"], color="purple", label=f"RSI ({rsi_period})")
ax6.axhline(70, linestyle="--", color="red",   label="Overbought (70)")
ax6.axhline(30, linestyle="--", color="green", label="Oversold (30)")
ax6.set_title("RSI"); ax6.set_ylabel("RSI"); ax6.legend()
plt.tight_layout(); st.pyplot(fig6); plt.close()

# Feature selection
features  = ["Open", "High", "Low", "Volume", "Daily Return", "SMA_50", "SMA_200", "RSI"]
target    = "Close"
df_model  = df.dropna().copy()
X         = df_model[features]
y         = df_model[target]

# Correlation heatmap
fig7, ax7 = plt.subplots(figsize=(9, 7))
sns.heatmap(df_model[features + [target]].corr(), annot=True, cmap="coolwarm", fmt=".2f",
            linewidths=0.5, ax=ax7)
ax7.set_title("Matriks Korelasi Fitur & Target")
plt.tight_layout(); st.pyplot(fig7); plt.close()

# Train/test split (shuffle=False diterapkan di sini untuk data saham)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size / 100, random_state=42, shuffle=False
)
st.info(f"**Data Latih:** {X_train.shape[0]} baris &nbsp;|&nbsp; **Data Uji:** {X_test.shape[0]} baris")

# ── 4. Modelling ─────────────────────────────
st.header("4. Pemodelan")

with st.spinner("Melatih model…"):
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)

    dt_model = DecisionTreeRegressor(random_state=42, max_depth=dt_depth)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)

# Koefisien LR
coef_df = pd.DataFrame({"Fitur": features, "Koefisien": lr_model.coef_})
col_lr, col_dt = st.columns(2)
with col_lr:
    st.subheader("Koefisien Linear Regression")
    fig8, ax8 = plt.subplots(figsize=(6, 4))
    sns.barplot(x="Koefisien", y="Fitur",
                data=coef_df.sort_values("Koefisien", ascending=False),
                palette="coolwarm", ax=ax8)
    ax8.set_title("Koefisien LR"); plt.tight_layout(); st.pyplot(fig8); plt.close()

with col_dt:
    st.subheader("Feature Importance Decision Tree")
    imp_df = pd.DataFrame({
        "Fitur": features,
        "Importance": dt_model.feature_importances_
    }).sort_values("Importance", ascending=False)
    fig9, ax9 = plt.subplots(figsize=(6, 4))
    sns.barplot(x="Importance", y="Fitur", data=imp_df, palette="viridis", ax=ax9)
    ax9.set_title("Feature Importance DT"); plt.tight_layout(); st.pyplot(fig9); plt.close()

# Decision Tree visualisasi
with st.expander("🌳 Lihat Visualisasi Decision Tree (depth=3)"):
    dt_vis = DecisionTreeRegressor(random_state=42, max_depth=3)
    dt_vis.fit(X_train, y_train)
    fig10, ax10 = plt.subplots(figsize=(20, 8))
    plot_tree(dt_vis, feature_names=features, filled=True, rounded=True, fontsize=8, ax=ax10)
    ax10.set_title("Decision Tree (max_depth=3)")
    plt.tight_layout(); st.pyplot(fig10); plt.close()

# ── 5. Evaluasi ──────────────────────────────
st.header("5. Evaluasi Model")

res_lr  = evaluate(y_test, y_pred_lr, "Regresi Linier")
res_dt  = evaluate(y_test, y_pred_dt, "Regresi Pohon Keputusan")
hasil   = pd.DataFrame([res_lr, res_dt])

st.dataframe(
    hasil.set_index("Model").style.format({
        "MAPE (%)": "{:.4f}",
        "MAE":  "{:,.2f}",
        "RMSE": "{:,.2f}",
        "R²":   "{:.4f}",
    }),
    use_container_width=True,
)

# Plot aktual vs prediksi – gabungan
test_idx = X_test.index
plot_df  = pd.DataFrame({
    "Aktual":           y_test,
    "Linear Regression": y_pred_lr,
    "Decision Tree":    y_pred_dt,
}, index=test_idx).sort_index()

fig11, ax11 = plt.subplots(figsize=(14, 5))
ax11.plot(plot_df.index, plot_df["Aktual"],            label="Aktual",            color="black", linewidth=1.5)
ax11.plot(plot_df.index, plot_df["Linear Regression"], label="Linear Regression", color="blue",  alpha=0.8)
ax11.plot(plot_df.index, plot_df["Decision Tree"],     label="Decision Tree",     color="red",   alpha=0.8)
ax11.set_title("Harga Aktual vs Prediksi (Data Uji)"); ax11.set_ylabel("IDR")
ax11.legend(); plt.tight_layout(); st.pyplot(fig11); plt.close()

# Scatter plots
fig12, axes12 = plt.subplots(1, 2, figsize=(14, 5))
for ax, preds, color, name_key in [
    (axes12[0], y_pred_lr, "blue",  "Regresi Linier"),
    (axes12[1], y_pred_dt, "red",   "Regresi Pohon Keputusan"),
]:
    r = hasil[hasil["Model"] == name_key]["MAPE (%)"].values[0]
    ax.scatter(y_test, preds, alpha=0.4, color=color)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--")
    ax.set_title(f"{name_key} (MAPE={r:.2f}%)")
    ax.set_xlabel("Aktual"); ax.set_ylabel("Prediksi")
plt.tight_layout(); st.pyplot(fig12); plt.close()

# Distribusi residual
residual_lr = y_test.values - y_pred_lr
residual_dt = y_test.values - y_pred_dt
fig13, axes13 = plt.subplots(1, 2, figsize=(14, 4))
sns.histplot(residual_lr, bins=40, kde=True, color="blue", ax=axes13[0])
axes13[0].axvline(0, color="black", linestyle="--"); axes13[0].set_title("Residual – Linear Regression")
sns.histplot(residual_dt, bins=40, kde=True, color="red",  ax=axes13[1])
axes13[1].axvline(0, color="black", linestyle="--"); axes13[1].set_title("Residual – Decision Tree")
plt.tight_layout(); st.pyplot(fig13); plt.close()

# Grafik Metrik Menggunakan Melt (Disamakan dengan file Colab)
st.subheader("📊 Grafik Perbandingan Metrik Error")
hasil_melt = hasil.melt(id_vars="Model", value_vars=["MAPE (%)", "MAE", "RMSE"], var_name="Metrik", value_name="Nilai")

fig14, ax14 = plt.subplots(figsize=(10, 5))
sns.barplot(x="Metrik", y="Nilai", hue="Model", data=hasil_melt, palette=["steelblue", "tomato"], ax=ax14)
ax14.set_title("Perbandingan Metrik Error (Regresi Linier vs Pohon Keputusan)")
ax14.set_ylabel("Nilai")
ax14.set_xlabel("Metrik")

for p in ax14.patches:
    height = p.get_height()
    if height > 0:
        ax14.annotate(f"{height:,.2f}", (p.get_x() + p.get_width() / 2., height),
                      ha='center', va='center', xytext=(0, 6), textcoords='offset points', fontsize=9, fontweight='bold')

plt.tight_layout()
st.pyplot(fig14)
plt.close()

# ── 6. Prediksi Manual ───────────────────────
st.header("6. Prediksi Manual")
st.markdown("Masukkan nilai fitur secara manual untuk mendapatkan prediksi harga penutupan.")

latest = df_model[features].iloc[-1]
with st.form("pred_form"):
    cols = st.columns(4)
    vals = {}
    for i, feat in enumerate(features):
        vals[feat] = cols[i % 4].number_input(feat, value=float(latest[feat]), format="%.4f")
    submitted = st.form_submit_button("🔮 Prediksi")

if submitted:
    inp = np.array([[vals[f] for f in features]])
    pred_lr_val = lr_model.predict(inp)[0]
    pred_dt_val = dt_model.predict(inp)[0]
    p1, p2 = st.columns(2)
    p1.metric("Linear Regression", f"Rp {pred_lr_val:,.2f}")
    p2.metric("Decision Tree",     f"Rp {pred_dt_val:,.2f}")

st.divider()
st.caption("Dibuat dengan Streamlit · Data: Yahoo Finance (^JKSE)")
