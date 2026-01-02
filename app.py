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
st.title("Online Retail Streamlit Dashboard")

@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
    return pd.read_excel(url)

df = load_data()
df = df.dropna(subset=["CustomerID"])
df["Revenue"] = df["Quantity"] * df["UnitPrice"]

st.subheader("First 10 Observations")
st.dataframe(df.head(10))

st.subheader("Dataset Structure")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Number of Observations", df.shape[0])
with col2:
    st.metric("Number of Variables", df.shape[1])
with col3:
    st.write(df.dtypes)

cat_cols = df.select_dtypes(include="object").columns.tolist()
selected_cat = st.sidebar.selectbox("Select Categorical Variable", cat_cols)

values = df[selected_cat].value_counts().head(10)
fig, ax = plt.subplots()
ax.pie(values, labels=values.index, autopct="%1.1f%%")
ax.set_title(f"Distribution of {selected_cat}")
st.pyplot(fig)

st.subheader("Top 10 Countries by Number of Transactions")
countries = df["Country"].value_counts().head(10)
fig, ax = plt.subplots()
ax.bar(countries.index, countries.values)
ax.set_xlabel("Country")
ax.set_ylabel("Number of Transactions")
ax.set_xticklabels(countries.index, rotation=45)
st.pyplot(fig)

st.subheader("Quantity vs Unit Price")
fig, ax = plt.subplots()
ax.scatter(df["Quantity"], df["UnitPrice"], alpha=0.3)
ax.set_xlabel("Quantity")
ax.set_ylabel("Unit Price")
st.pyplot(fig)

st.subheader("Revenue Distribution")
fig, ax = plt.subplots()
ax.hist(df["Revenue"], bins=50)
ax.set_xlabel("Revenue")
ax.set_ylabel("Frequency")
st.pyplot(fig)

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
time_series = df.groupby(df["InvoiceDate"].dt.to_period("M")).size()
fig, ax = plt.subplots()
time_series.plot(ax=ax)
ax.set_xlabel("Time")
ax.set_ylabel("Number of Transactions")
st.pyplot(fig)

st.subheader("PCA Analysis (Quantity, UnitPrice, Revenue)")
pca_data = df[["Quantity", "UnitPrice", "Revenue"]]
scaled = StandardScaler().fit_transform(pca_data)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled)

fig, ax = plt.subplots()
ax.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.4)
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
st.pyplot(fig)

st.subheader("PCA Colored by Top 5 Countries")
top5 = df["Country"].value_counts().head(5).index
df_top5 = df[df["Country"].isin(top5)]
scaled2 = StandardScaler().fit_transform(df_top5[["Quantity", "UnitPrice", "Revenue"]])
pca2 = pca.fit_transform(scaled2)

fig, ax = plt.subplots()
for country in top5:
    idx = df_top5["Country"] == country
    ax.scatter(pca2[idx, 0], pca2[idx, 1], label=country, alpha=0.5)

ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.legend()
st.pyplot(fig)

st.subheader("Feature Selection for Revenue")
X = df[["Quantity", "UnitPrice"]]
y = df["Revenue"]
selector = SelectKBest(score_func=f_regression, k="all")
selector.fit(X, y)

feature_scores = pd.DataFrame({
    "Feature": X.columns,
    "Score": selector.scores_
})
st.dataframe(feature_scores)

st.subheader("Random Forest Regression Results")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

st.write("R² Score:", r2_score(y_test, predictions))
st.write("RMSE:", np.sqrt(mean_squared_error(y_test, predictions)))
