import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(layout="wide")
st.title("Online Retail Dashboard")

@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
    return pd.read_excel(url)

df = load_data()
df = df.dropna(subset=["CustomerID"])
df["Revenue"] = df["Quantity"] * df["UnitPrice"]

st.subheader("İlk 10 Satır")
st.dataframe(df.head(10))

st.subheader("Veri Seti Bilgisi")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Gözlem Sayısı", df.shape[0])
with col2:
    st.metric("Değişken Sayısı", df.shape[1])
with col3:
    st.write(df.dtypes)

cat_cols = df.select_dtypes(include="object").columns.tolist()
secim = st.sidebar.selectbox("Kategorik Değişken Seç", cat_cols)

degerler = df[secim].value_counts().head(10)
fig, ax = plt.subplots()
ax.pie(degerler, labels=degerler.index, autopct="%1.1f%%")
st.pyplot(fig)

st.subheader("En Çok İşlem Yapılan 10 Ülke")
ulkeler = df["Country"].value_counts().head(10)
fig, ax = plt.subplots()
ax.bar(ulkeler.index, ulkeler.values)
ax.set_xticklabels(ulkeler.index, rotation=45)
st.pyplot(fig)

st.subheader("Quantity - UnitPrice")
fig, ax = plt.subplots()
ax.scatter(df["Quantity"], df["UnitPrice"], alpha=0.3)
st.pyplot(fig)

st.subheader("Revenue Dağılımı")
fig, ax = plt.subplots()
ax.hist(df["Revenue"], bins=50)
st.pyplot(fig)

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
zaman = df.groupby(df["InvoiceDate"].dt.to_period("M")).size()
fig, ax = plt.subplots()
zaman.plot(ax=ax)
st.pyplot(fig)

pca_df = df[["Quantity", "UnitPrice", "Revenue"]]
scaled = StandardScaler().fit_transform(pca_df)
pca = PCA(n_components=2)
pca_sonuc = pca.fit_transform(scaled)
fig, ax = plt.subplots()
ax.scatter(pca_sonuc[:, 0], pca_sonuc[:, 1], alpha=0.4)
st.pyplot(fig)

top5 = df["Country"].value_counts().head(5).index
df2 = df[df["Country"].isin(top5)]
scaled2 = StandardScaler().fit_transform(df2[["Quantity", "UnitPrice", "Revenue"]])
pca2 = pca.fit_transform(scaled2)
fig, ax = plt.subplots()
for ulke in top5:
    idx = df2["Country"] == ulke
    ax.scatter(pca2[idx, 0], pca2[idx, 1], label=ulke, alpha=0.5)
ax.legend()
st.pyplot(fig)

X = df[["Quantity", "UnitPrice"]]
y = df["Revenue"]
secici = SelectKBest(score_func=f_regression, k="all")
secici.fit(X, y)
st.dataframe(pd.DataFrame({"Feature": X.columns, "Score": secici.scores_}))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
tahmin = model.predict(X_test)
st.write("R2:", r2_score(y_test, tahmin))
st.write("RMSE:", np.sqrt(mean_squared_error(y_test, tahmin)))
