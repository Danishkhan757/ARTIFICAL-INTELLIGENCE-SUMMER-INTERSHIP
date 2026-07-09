import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Google Play Store Dashboard", layout="wide")

st.title("📱 Google Play Store Data Visualization Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("PROJECT 1 DATA VISUALISATION/googleplaystore_v2.csv")

    # Data Cleaning
    df = df.dropna(subset=["Rating"])

    if "Price" in df.columns:
        df["Price"] = (
            df["Price"]
            .astype(str)
            .str.replace("$", "", regex=False)
        )
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    if "Reviews" in df.columns:
        df["Reviews"] = pd.to_numeric(df["Reviews"], errors="coerce")

    if "Installs" in df.columns:
        df["Installs"] = (
            df["Installs"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("+", "", regex=False)
        )
        df["Installs"] = pd.to_numeric(df["Installs"], errors="coerce")

    if "Size" in df.columns:
        df["Size"] = (
            df["Size"]
            .astype(str)
            .str.replace("M", "", regex=False)
            .str.replace("k", "", regex=False)
        )
        df["Size"] = pd.to_numeric(df["Size"], errors="coerce")

    return df

df = load_data()

# Sidebar
st.sidebar.header("Filters")

category = st.sidebar.multiselect(
    "Select Category",
    options=sorted(df["Category"].dropna().unique()),
    default=sorted(df["Category"].dropna().unique())[:5]
)

content_rating = st.sidebar.multiselect(
    "Content Rating",
    options=df["Content Rating"].dropna().unique(),
    default=df["Content Rating"].dropna().unique()
)

filtered_df = df[
    (df["Category"].isin(category))
    & (df["Content Rating"].isin(content_rating))
]

# Metrics
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Apps", len(filtered_df))
col2.metric("Average Rating", round(filtered_df["Rating"].mean(), 2))
col3.metric("Average Reviews", int(filtered_df["Reviews"].mean()))
col4.metric("Average Installs", int(filtered_df["Installs"].mean()))

st.markdown("---")

# Rating Distribution
st.subheader("⭐ Rating Distribution")
fig1 = px.histogram(
    filtered_df,
    x="Rating",
    nbins=20,
    title="Distribution of Ratings"
)
st.plotly_chart(fig1, use_container_width=True)

# Category-wise Average Rating
st.subheader("📊 Category-wise Average Rating")

cat_rating = (
    filtered_df.groupby("Category")["Rating"]
    .mean()
    .reset_index()
    .sort_values("Rating", ascending=False)
)

fig2 = px.bar(
    cat_rating,
    x="Category",
    y="Rating",
    title="Average Rating by Category"
)

st.plotly_chart(fig2, use_container_width=True)

# Reviews vs Rating
st.subheader("📝 Reviews vs Rating")

fig3 = px.scatter(
    filtered_df,
    x="Reviews",
    y="Rating",
    color="Category",
    hover_name="App",
    title="Reviews vs Rating"
)

st.plotly_chart(fig3, use_container_width=True)

# Installs Distribution
st.subheader("📥 Installs Distribution")

fig4 = px.box(
    filtered_df,
    y="Installs",
    title="Installs Box Plot"
)

st.plotly_chart(fig4, use_container_width=True)

# Price vs Rating
st.subheader("💰 Price vs Rating")

paid_apps = filtered_df[filtered_df["Price"] > 0]

fig5 = px.scatter(
    paid_apps,
    x="Price",
    y="Rating",
    color="Category",
    title="Price vs Rating (Paid Apps)"
)

st.plotly_chart(fig5, use_container_width=True)

# Data Preview
st.subheader("📄 Dataset Preview")
st.dataframe(filtered_df.head(20))
