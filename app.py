
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import joblib
from pathlib import Path

# -------------------- PAGE CONFIGURATION --------------------
st.set_page_config(
    page_title="Sara Enterprises - Sales Analysis",
    layout="wide"
)

# -------------------- PROJECT PATHS --------------------
BASE_DIR = Path(__file__).resolve().parent

CUSTOMER_FILE = BASE_DIR / "Dataset" / "customer.csv"
ORDERS_FILE = BASE_DIR / "Dataset" / "orders.csv"
MODEL_FILE = BASE_DIR / "model" / "model.pkl"

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    customers = pd.read_csv(CUSTOMER_FILE)
    orders = pd.read_csv(ORDERS_FILE)

    data = pd.merge(
        customers,
        orders,
        on="CustomerID",
        how="inner"
    )
    return data

try:
    df = load_data()
except Exception as e:
    st.error(f"Unable to load dataset: {e}")
    st.stop()

# -------------------- HEADER --------------------
st.title("Sara Enterprises - Sales Data Analysis")
st.write(f"**Shape of Data:** {df.shape}")

required_columns = [
    "CustomerID",
    "Product Category",
    "Product Sub-Category",
    "State or Province",
    "City",
    "Gender",
    "Profit",
    "Amount",
    "Qty"
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error(f"Missing columns in dataset: {missing_columns}")
    st.stop()

# -------------------- FILTERS --------------------
filtered_df = df.copy()

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    categories = ["All"] + sorted(
        df["Product Category"].dropna().unique().tolist()
    )
    cat_filter = st.selectbox("Select Product Category:", categories)

with col2:
    subcategories = ["All"] + sorted(
        df["Product Sub-Category"].dropna().unique().tolist()
    )
    scat_filter = st.selectbox(
        "Select Product Sub-Category:", subcategories
    )

with col3:
    states = ["All"] + sorted(
        df["State or Province"].dropna().unique().tolist()
    )
    state_filter = st.selectbox("Select State:", states)

with col4:
    cities = ["All"] + sorted(
        df["City"].dropna().unique().tolist()
    )
    city_filter = st.selectbox("Select City:", cities)

with col5:
    genders = ["All"] + sorted(
        df["Gender"].dropna().unique().tolist()
    )
    gender_filter = st.selectbox("Select Gender:", genders)

if cat_filter != "All":
    filtered_df = filtered_df[
        filtered_df["Product Category"] == cat_filter
    ]

if scat_filter != "All":
    filtered_df = filtered_df[
        filtered_df["Product Sub-Category"] == scat_filter
    ]

if state_filter != "All":
    filtered_df = filtered_df[
        filtered_df["State or Province"] == state_filter
    ]

if city_filter != "All":
    filtered_df = filtered_df[
        filtered_df["City"] == city_filter
    ]

if gender_filter != "All":
    filtered_df = filtered_df[
        filtered_df["Gender"] == gender_filter
    ]

# -------------------- KPI METRICS --------------------
st.subheader("Sales Overview")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Orders", f"{len(filtered_df):,}")

with col2:
    st.metric("Total Sales", f"{filtered_df['Amount'].sum():,.2f}")

with col3:
    st.metric("Total Profit", f"{filtered_df['Profit'].sum():,.2f}")

with col4:
    st.metric("Total Sold Quantity", f"{filtered_df['Qty'].sum():,.0f}")

with col5:
    st.metric(
        "Total Customers",
        f"{filtered_df['CustomerID'].nunique():,}"
    )

# -------------------- GRAPHS --------------------
st.subheader("Sales and Profit Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    profit_category = (
        filtered_df.groupby("Product Category")["Profit"]
        .sum()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    profit_category.plot(kind="bar", ax=ax)
    ax.set_title("Profit by Product Category")
    ax.set_xlabel("Product Category")
    ax.set_ylabel("Profit")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with col2:
    profit_subcategory = (
        filtered_df.groupby("Product Sub-Category")["Profit"]
        .sum()
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    profit_subcategory.plot(kind="barh", ax=ax)
    ax.set_title("Profit by Product Sub-Category")
    ax.set_xlabel("Profit")
    ax.set_ylabel("Product Sub-Category")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with col3:
    profit_state = (
        filtered_df.groupby("State or Province")["Profit"]
        .sum()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    profit_state.plot(kind="bar", ax=ax)
    ax.set_title("Profit by State")
    ax.set_xlabel("State")
    ax.set_ylabel("Profit")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# -------------------- DATA TABLE --------------------
st.subheader("Detailed Sales Data")
st.dataframe(filtered_df, height=300, use_container_width=True)

# -------------------- MACHINE LEARNING MODEL --------------------
st.divider()
st.subheader("Profit Prediction")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load ML model: {e}")
    st.stop()

st.write(
    "Select a product category and sub-category, "
    "then enter the sales amount to predict profit."
)

col1, col2, col3 = st.columns(3)

with col1:
    prediction_categories = sorted(
        df["Product Category"].dropna().unique().tolist()
    )
    prediction_category = st.selectbox(
        "Prediction - Product Category:",
        prediction_categories
    )

with col2:
    prediction_subcategories = sorted(
        df["Product Sub-Category"].dropna().unique().tolist()
    )
    prediction_subcategory = st.selectbox(
        "Prediction - Product Sub-Category:",
        prediction_subcategories
    )

with col3:
    sales_amount = st.number_input(
        "Enter Sales Amount:",
        min_value=0.0,
        value=0.0,
        step=100.0
    )

if st.button("Predict Profit", type="primary"):
    input_data = pd.DataFrame({
        "Product Category": [prediction_category],
        "Product Sub-Category": [prediction_subcategory],
        "Amount": [sales_amount]
    })

    try:
        predicted_profit = model.predict(input_data)
        st.success(f"Predicted Profit: {predicted_profit[0]:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info(
            "Check that the model was trained using the same input "
            "columns and preprocessing."
        )
