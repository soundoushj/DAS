import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Cars Price Dashboard",
    layout="wide"
)

# ==================================================
# LOAD DATA
# ==================================================

@st.cache_data
def load_data():
    return pd.read_csv("Cars_dataset.csv")

df = load_data()

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("Cars Dashboard")

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "EDA", "ML Model", "Predict"]
)

# ==================================================
# OVERVIEW
# ==================================================

if page == "Overview":

    st.title("Cars Price Analysis Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Cars", len(df))
    col2.metric("Average Price", f"${df['Price'].mean():,.0f}")
    col3.metric("Maximum Price", f"${df['Price'].max():,.0f}")

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    st.subheader("Statistics")

    st.dataframe(df.describe())

# ==================================================
# EDA
# ==================================================

elif page == "EDA":

    st.title("Exploratory Data Analysis")

    # Price Distribution
    st.subheader("Price Distribution")

    fig, ax = plt.subplots()

    ax.hist(df["Price"], bins=30)

    ax.set_xlabel("Price")
    ax.set_ylabel("Count")

    st.pyplot(fig)

    # Brand Count
    st.subheader("Cars by Brand")

    fig, ax = plt.subplots()

    df["Brand"].value_counts().plot(
        kind="bar",
        ax=ax
    )

    st.pyplot(fig)

    # Mileage vs Price
    st.subheader("Mileage vs Price")

    fig, ax = plt.subplots()

    ax.scatter(
        df["Mileage"],
        df["Price"]
    )

    ax.set_xlabel("Mileage")
    ax.set_ylabel("Price")

    st.pyplot(fig)

# ==================================================
# MACHINE LEARNING
# ==================================================

elif page == "ML Model":

    st.title("Machine Learning Model")

    # Remove unnecessary column
    df_model = df.drop(
        columns=["Car ID"],
        errors="ignore"
    )

    # Convert text columns to numbers
    df_encoded = pd.get_dummies(df_model)

    # Features and target
    X = df_encoded.drop("Price", axis=1)
    y = df_encoded["Price"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Train model
    model = LinearRegression()

    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Score
    score = r2_score(y_test, y_pred)

    st.metric("R² Score", f"{score:.4f}")

    # Graph
    st.subheader("Actual vs Predicted")

    fig, ax = plt.subplots()

    ax.scatter(y_test, y_pred)

    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")

    st.pyplot(fig)

# ==================================================
# PREDICT PAGE
# ==================================================

elif page == "Predict":

    st.title("Predict Car Price")

    # Prepare data
    df_model = df.drop(
        columns=["Car ID"],
        errors="ignore"
    )

    df_encoded = pd.get_dummies(df_model)

    X = df_encoded.drop("Price", axis=1)
    y = df_encoded["Price"]

    model = LinearRegression()

    model.fit(X, y)

    # Inputs
    year = st.slider(
        "Year",
        int(df["Year"].min()),
        int(df["Year"].max()),
        2020
    )

    mileage = st.number_input(
        "Mileage",
        0,
        500000,
        50000
    )

    engine = st.slider(
        "Engine Size",
        float(df["Engine Size"].min()),
        float(df["Engine Size"].max()),
        2.0
    )

    # Input dataframe
    input_data = {
        "Year": year,
        "Mileage": mileage,
        "Engine Size": engine
    }

    input_df = pd.DataFrame([input_data])

    # Match training columns
    input_df = input_df.reindex(
        columns=X.columns,
        fill_value=0
    )

    # Predict
    if st.button("Predict Price"):

        prediction = model.predict(input_df)[0]

        st.success(
            f"Estimated Price: ${prediction:,.0f}"
        )

# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.write("Built with Streamlit and Scikit-learn")