import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Outlier Detection Using Percentiles",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Airbnb NYC Outlier Detection Dashboard")
st.write("Remove outliers using percentile method.")

@st.cache_data
def load_data():
    return pd.read_csv("AB_NYC_2019.csv")

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.sidebar.header("Settings")

column = st.sidebar.selectbox(
    "Select Numerical Column",
    df.select_dtypes(include=["int64", "float64"]).columns
)

lower_percentile = st.sidebar.slider(
    "Lower Percentile",
    0.0,
    10.0,
    1.0,
    0.1
)

upper_percentile = st.sidebar.slider(
    "Upper Percentile",
    90.0,
    100.0,
    99.0,
    0.1
)

lower_limit = df[column].quantile(lower_percentile / 100)
upper_limit = df[column].quantile(upper_percentile / 100)

filtered_df = df[
    (df[column] >= lower_limit) &
    (df[column] <= upper_limit)
]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Original Rows", len(df))

with col2:
    st.metric("Rows After Cleaning", len(filtered_df))

with col3:
    st.metric("Outliers Removed", len(df) - len(filtered_df))

st.subheader("Before Outlier Removal")

fig1 = px.box(
    df,
    y=column,
    title=f"{column} Distribution (Original)"
)
st.plotly_chart(fig1, use_container_width=True)

st.subheader("After Outlier Removal")

fig2 = px.box(
    filtered_df,
    y=column,
    title=f"{column} Distribution (Cleaned)"
)
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Statistics")

c1, c2 = st.columns(2)

with c1:
    st.write("### Original Data")
    st.dataframe(df[column].describe())

with c2:
    st.write("### Cleaned Data")
    st.dataframe(filtered_df[column].describe())

st.subheader("Cleaned Dataset")
st.dataframe(filtered_df.head(20))

csv = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Cleaned Dataset",
    data=csv,
    file_name="cleaned_airbnb_data.csv",
    mime="text/csv"
)
