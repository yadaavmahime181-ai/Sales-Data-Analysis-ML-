
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Sara Enterprises | Sales Analysis",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
CUSTOMER_FILE = BASE_DIR / "Dataset" / "customer.csv"
ORDERS_FILE = BASE_DIR / "Dataset" / "orders.csv"
MODEL_FILE = BASE_DIR / "model" / "model.pkl"


# =====================================================
# 2. LOAD AND MERGE DATA
# =====================================================
@st.cache_data
def load_data():
    if not CUSTOMER_FILE.exists():
        raise FileNotFoundError(
            "Dataset/customer.csv file was not found."
        )

    if not ORDERS_FILE.exists():
        raise FileNotFoundError(
            "Dataset/orders.csv file was not found."
        )

    customers = pd.read_csv(CUSTOMER_FILE)
    orders = pd.read_csv(ORDERS_FILE)

    # Clean column names
    customers.columns = customers.columns.str.strip()
    orders.columns = orders.columns.str.strip()

    if "CustomerID" not in customers.columns:
        raise ValueError(
            "CustomerID column is missing from customer.csv."
        )

    if "CustomerID" not in orders.columns:
        raise ValueError(
            "CustomerID column is missing from orders.csv."
        )

    # Standardize CustomerID for merging
    customers["CustomerID"] = (
        customers["CustomerID"].astype("string").str.strip()
    )
    orders["CustomerID"] = (
        orders["CustomerID"].astype("string").str.strip()
    )

    # Remove blank IDs before merging
    customers = customers.dropna(subset=["CustomerID"])
    orders = orders.dropna(subset=["CustomerID"])

    customers = customers[
        customers["CustomerID"] != ""
    ]
    orders = orders[
        orders["CustomerID"] != ""
    ]

    data = pd.merge(
        customers,
        orders,
        on="CustomerID",
        how="inner"
    )

    if data.empty:
        raise ValueError(
            "No matching CustomerID values were found in the two CSV files. "
            "Please check the IDs in both datasets."
        )

    return data


try:
    df = load_data()
except Exception as e:
    st.error("Unable to load the datasets.")
    st.code(str(e))
    st.stop()


# =====================================================
# 3. VALIDATE REQUIRED COLUMNS
# =====================================================
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

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(
        f"Required columns missing from merged data: {missing_columns}"
    )
    st.write("Available columns:", df.columns.tolist())
    st.stop()

# Convert numeric fields safely
for column in ["Profit", "Amount", "Qty"]:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# =====================================================
# 4. HEADER
# =====================================================
st.title("Sara Enterprises - Sales Data Analysis")
st.caption(
    "Sales analytics dashboard with machine-learning profit prediction"
)

st.write(
    f"**Dataset Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns"
)


# =====================================================
# 5. FILTERS
# =====================================================
st.subheader("Filter Sales Data")


def get_options(series):
    values = series.dropna().unique().tolist()
    values = sorted(values, key=lambda value: str(value).lower())
    return ["All"] + values


col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    category_filter = st.selectbox(
        "Product Category",
        get_options(df["Product Category"]),
        key="category_filter"
    )

with col2:
    subcategory_filter = st.selectbox(
        "Product Sub-Category",
        get_options(df["Product Sub-Category"]),
        key="subcategory_filter"
    )

with col3:
    state_filter = st.selectbox(
        "State",
        get_options(df["State or Province"]),
        key="state_filter"
    )

with col4:
    city_filter = st.selectbox(
        "City",
        get_options(df["City"]),
        key="city_filter"
    )

with col5:
    gender_filter = st.selectbox(
        "Gender",
        get_options(df["Gender"]),
        key="gender_filter"
    )


filtered_df = df.copy()

selected_filters = [
    ("Product Category", category_filter),
    ("Product Sub-Category", subcategory_filter),
    ("State or Province", state_filter),
    ("City", city_filter),
    ("Gender", gender_filter)
]

for column, selected_value in selected_filters:
    if selected_value != "All":
        filtered_df = filtered_df[
            filtered_df[column] == selected_value
        ]


# =====================================================
# 6. KPI METRICS
# =====================================================
st.subheader("Sales Overview")

if filtered_df.empty:
    st.warning(
        "No data matches these filters. "
        "Please select different filters."
    )

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Total Orders / Rows",
        f"{len(filtered_df):,}"
    )

with col2:
    st.metric(
        "Total Sales",
        f"{filtered_df['Amount'].sum():,.2f}"
    )

with col3:
    st.metric(
        "Total Profit",
        f"{filtered_df['Profit'].sum():,.2f}"
    )

with col4:
    st.metric(
        "Total Sold Quantity",
        f"{filtered_df['Qty'].sum():,.0f}"
    )

with col5:
    st.metric(
        "Total Customers",
        f"{filtered_df['CustomerID'].nunique():,}"
    )


# =====================================================
# 7. SAFE CHART FUNCTION
# =====================================================
def show_bar_chart(
    data,
    title,
    xlabel,
    ylabel,
    horizontal=False
):
    # Never plot an empty Series
    if data.empty:
        st.info(
            f"No data available for '{title}'. "
            "Try changing the filters."
        )
        return

    # Ignore non-finite values
    data = data[
        pd.to_numeric(data, errors="coerce").notna()
    ]

    if data.empty:
        st.info(f"No valid numeric values available for '{title}'.")
        return

    fig, ax = plt.subplots(figsize=(7, 4))

    try:
        if horizontal:
            ax.barh(
                data.index.astype(str),
                data.values
            )
        else:
            ax.bar(
                data.index.astype(str),
                data.values
            )

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if not horizontal:
            ax.tick_params(
                axis="x",
                rotation=45
            )

        fig.tight_layout()
        st.pyplot(fig)

    finally:
        plt.close(fig)


# =====================================================
# 8. PROFIT CHARTS
# =====================================================
st.subheader("Sales and Profit Analysis")

if not filtered_df.empty:

    col1, col2, col3 = st.columns(3)

    with col1:
        profit_category = (
            filtered_df
            .dropna(subset=["Product Category", "Profit"])
            .groupby("Product Category")["Profit"]
            .sum()
            .sort_values(ascending=False)
        )

        show_bar_chart(
            profit_category,
            "Profit by Product Category",
            "Product Category",
            "Profit"
        )

    with col2:
        profit_subcategory = (
            filtered_df
            .dropna(subset=["Product Sub-Category", "Profit"])
            .groupby("Product Sub-Category")["Profit"]
            .sum()
            .sort_values()
        )

        show_bar_chart(
            profit_subcategory,
            "Profit by Product Sub-Category",
            "Product Sub-Category",
            "Profit",
            horizontal=True
        )

    with col3:
        profit_state = (
            filtered_df
            .dropna(subset=["State or Province", "Profit"])
            .groupby("State or Province")["Profit"]
            .sum()
            .sort_values(ascending=False)
        )

        show_bar_chart(
            profit_state,
            "Profit by State",
            "State",
            "Profit"
        )


# =====================================================
# 9. DATA TABLE
# =====================================================
st.subheader("Detailed Sales Data")

st.dataframe(
    filtered_df,
    height=300,
    use_container_width=True
)


# =====================================================
# 10. LOAD ML MODEL
# =====================================================
st.divider()
st.subheader("Machine Learning - Profit Prediction")


@st.cache_resource
def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "model/model.pkl file was not found."
        )

    return joblib.load(MODEL_FILE)


model = None

try:
    model = load_model()
except Exception as e:
    st.warning(
        "The dashboard is available, but the prediction model "
        "could not be loaded."
    )
    st.code(str(e))


# =====================================================
# 11. PROFIT PREDICTION
# =====================================================
if model is not None:

    st.write(
        "Select the product details and enter the sales amount."
    )

    prediction_categories = sorted(
        df["Product Category"].dropna().unique().tolist(),
        key=lambda value: str(value).lower()
    )

    prediction_subcategories = sorted(
        df["Product Sub-Category"].dropna().unique().tolist(),
        key=lambda value: str(value).lower()
    )

    if (
        not prediction_categories
        or not prediction_subcategories
    ):
        st.warning(
            "Prediction options are unavailable because the dataset "
            "has no valid category or sub-category values."
        )

    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            prediction_category = st.selectbox(
                "Prediction - Product Category",
                prediction_categories,
                key="prediction_category"
            )

        with col2:
            prediction_subcategory = st.selectbox(
                "Prediction - Product Sub-Category",
                prediction_subcategories,
                key="prediction_subcategory"
            )

        with col3:
            sales_amount = st.number_input(
                "Enter Sales Amount",
                min_value=0.0,
                value=0.0,
                step=100.0,
                key="prediction_amount"
            )

        if st.button("Predict Profit", type="primary"):

            input_data = pd.DataFrame({
                "Product Category": [prediction_category],
                "Product Sub-Category": [prediction_subcategory],
                "Amount": [sales_amount]
            })

            try:
                predictions = model.predict(input_data)

                if predictions is None or len(predictions) == 0:
                    st.error(
                        "The model returned no prediction. "
                        "Check the model training code."
                    )
                else:
                    predicted_profit = float(predictions[0])

                    st.success(
                        f"Predicted Profit: {predicted_profit:,.2f}"
                    )

            except Exception as e:
                st.error(
                    "Profit prediction failed. The model's expected "
                    "input format may differ from the supplied columns."
                )
                st.code(str(e))
                st.info(
                    "To fix the model correctly, check "
                    "preprocess/train.py and the preprocessing used "
                    "when model.pkl was created."
                )

st.caption("Sara Enterprises | Sales Analytics Project")
