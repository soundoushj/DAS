import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Cars Price Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #0f1117;
    color: white;
}

.metric-card {
    background: #1a2035;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid #2d3a52;
}

.metric-value {
    font-size: 2rem;
    font-weight: bold;
    color: #60a5fa;
}

.metric-label {
    color: #94a3b8;
    font-size: 0.9rem;
}

.section-header {
    font-size: 1.5rem;
    font-weight: bold;
    color: #f1f5f9;
    border-left: 5px solid #3b82f6;
    padding-left: 12px;
    margin-top: 30px;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# COLORS
# =========================================================

CARD_BG = "#1a2035"
BORDER = "#2d3a52"
BLUE = "#3b82f6"
TEAL = "#10b981"
ROSE = "#f43f5e"
TEXT = "#e8eaf0"
MUTED = "#94a3b8"

# =========================================================
# DARK STYLE
# =========================================================

def apply_dark_style(ax, fig):

    fig.patch.set_facecolor(CARD_BG)
    ax.set_facecolor(CARD_BG)

    ax.tick_params(colors=MUTED)

    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)

    ax.title.set_color(TEXT)

    for spine in ax.spines.values():
        spine.set_color(BORDER)

# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    return pd.read_csv("Cars_dataset.csv")

df = load_data()

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🚗 Cars Dashboard")

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "EDA",
            "ML Model",
            "Predict"
        ]
    )

    st.markdown("---")

    st.subheader("Filters")

    # Brand filter
    brands = sorted(df["Brand"].dropna().unique())

    selected_brands = st.multiselect(
        "Brand",
        brands,
        default=brands
    )

    # Fuel filter
    fuels = sorted(df["Fuel Type"].dropna().unique())

    selected_fuels = st.multiselect(
        "Fuel Type",
        fuels,
        default=fuels
    )

    # Year filter
    year_range = st.slider(
        "Year Range",
        int(df["Year"].min()),
        int(df["Year"].max()),
        (
            int(df["Year"].min()),
            int(df["Year"].max())
        )
    )

# =========================================================
# FILTER DATA
# =========================================================

dff = df.copy()

dff = dff[dff["Brand"].isin(selected_brands)]

dff = dff[dff["Fuel Type"].isin(selected_fuels)]

dff = dff[
    (dff["Year"] >= year_range[0]) &
    (dff["Year"] <= year_range[1])
]

# =========================================================
# OVERVIEW PAGE
# =========================================================

if page == "Overview":

    st.title("🚗 Cars Price Analysis Dashboard")

    st.markdown(
        "End-to-end data analysis and machine learning dashboard."
    )

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)

    metrics = [
        ("Total Cars", f"{len(dff):,}"),
        ("Average Price", f"${dff['Price'].mean():,.0f}"),
        ("Maximum Price", f"${dff['Price'].max():,.0f}"),
        ("Brands", f"{dff['Brand'].nunique()}")
    ]

    for col, (label, value) in zip(
        [c1, c2, c3, c4],
        metrics
    ):

        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    # Dataset preview
    st.markdown(
        '<div class="section-header">Dataset Preview</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        dff.head(10),
        use_container_width=True
    )

    # Statistics
    st.markdown(
        '<div class="section-header">Statistical Summary</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        dff.describe(),
        use_container_width=True
    )

# =========================================================
# EDA PAGE
# =========================================================

elif page == "EDA":

    st.markdown(
        '<div class="section-header">Exploratory Data Analysis</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    # Price Distribution
    with col1:

        fig, ax = plt.subplots(figsize=(6, 4))

        apply_dark_style(ax, fig)

        sns.histplot(
            dff["Price"],
            kde=True,
            color=BLUE,
            bins=30,
            ax=ax
        )

        ax.set_title("Price Distribution")

        st.pyplot(fig)

    # Cars by Brand
    with col2:

        fig, ax = plt.subplots(figsize=(6, 4))

        apply_dark_style(ax, fig)

        dff["Brand"].value_counts().plot(
            kind="bar",
            color=TEAL,
            ax=ax
        )

        ax.set_title("Cars by Brand")

        st.pyplot(fig)

    # Heatmap
    st.markdown(
        '<div class="section-header">Correlation Heatmap</div>',
        unsafe_allow_html=True
    )

    numeric_df = dff.select_dtypes(include=np.number)

    numeric_df = numeric_df.drop(
        columns=["Car ID"],
        errors="ignore"
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    apply_dark_style(ax, fig)

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        linewidths=0.5,
        ax=ax
    )

    st.pyplot(fig)

    # Mileage vs Price
    st.markdown(
        '<div class="section-header">Mileage vs Price</div>',
        unsafe_allow_html=True
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    apply_dark_style(ax, fig)

    ax.scatter(
        dff["Mileage"],
        dff["Price"],
        color=ROSE,
        alpha=0.5
    )

    ax.set_xlabel("Mileage")
    ax.set_ylabel("Price")

    st.pyplot(fig)

# =========================================================
# MACHINE LEARNING PAGE
# =========================================================

elif page == "ML Model":

    st.markdown(
        '<div class="section-header">Machine Learning Model</div>',
        unsafe_allow_html=True
    )

    # Prepare data
    df_model = dff.drop(
        columns=["Car ID"],
        errors="ignore"
    )

    # Encode categorical variables
    df_encoded = pd.get_dummies(
        df_model,
        drop_first=True
    )

    # Fill missing values
    df_encoded = df_encoded.fillna(
        df_encoded.median(numeric_only=True)
    )

    # Features and target
    y = df_encoded["Price"]

    X = df_encoded.drop(columns=["Price"])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Scaling
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model selection
    model_name = st.selectbox(
        "Select Model",
        [
            "Linear Regression",
            "Ridge Regression",
            "Lasso Regression"
        ]
    )

    if model_name == "Linear Regression":

        model = LinearRegression()

    elif model_name == "Ridge Regression":

        alpha = st.slider(
            "Ridge Alpha",
            0.01,
            10.0,
            1.0
        )

        model = Ridge(alpha=alpha)

    else:

        alpha = st.slider(
            "Lasso Alpha",
            0.01,
            10.0,
            0.1
        )

        model = Lasso(alpha=alpha)

    # Train
    model.fit(
        X_train_scaled,
        y_train
    )

    # Predict
    y_pred = model.predict(X_test_scaled)

    # Metrics
    r2 = r2_score(y_test, y_pred)

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )

    cv = cross_val_score(
        model,
        scaler.transform(X),
        y,
        cv=5,
        scoring="r2"
    ).mean()

    # Display metrics
    m1, m2, m3, m4 = st.columns(4)

    model_metrics = [
        ("R² Score", f"{r2:.4f}"),
        ("MAE", f"${mae:,.0f}"),
        ("RMSE", f"${rmse:,.0f}"),
        ("CV Score", f"{cv:.4f}")
    ]

    for col, (label, value) in zip(
        [m1, m2, m3, m4],
        model_metrics
    ):

        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    # Actual vs Predicted
    st.markdown(
        '<div class="section-header">Actual vs Predicted</div>',
        unsafe_allow_html=True
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    apply_dark_style(ax, fig)

    ax.scatter(
        y_test,
        y_pred,
        color=BLUE,
        alpha=0.5
    )

    lim = [
        min(y_test.min(), y_pred.min()),
        max(y_test.max(), y_pred.max())
    ]

    ax.plot(
        lim,
        lim,
        linestyle="--",
        color=ROSE
    )

    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")

    st.pyplot(fig)

    # Feature importance
    st.markdown(
        '<div class="section-header">Feature Importance</div>',
        unsafe_allow_html=True
    )

    coef_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_
    })

    coef_df = coef_df.sort_values(
        by="Coefficient",
        ascending=False
    )

    st.dataframe(
        coef_df.head(15),
        use_container_width=True
    )

# =========================================================
# PREDICTION PAGE
# =========================================================

elif page == "Predict":

    st.markdown(
        '<div class="section-header">Price Predictor</div>',
        unsafe_allow_html=True
    )

    # Prepare data
    df_model = df.drop(
        columns=["Car ID"],
        errors="ignore"
    )

    df_encoded = pd.get_dummies(
        df_model,
        drop_first=True
    )

    y_full = df_encoded["Price"]

    X_full = df_encoded.drop(columns=["Price"])

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X_full)

    model = LinearRegression()

    model.fit(X_scaled, y_full)

    col1, col2 = st.columns(2)

    inputs = {}

    with col1:

        inputs["Year"] = st.slider(
            "Year",
            int(df["Year"].min()),
            int(df["Year"].max()),
            int(df["Year"].median())
        )

        inputs["Mileage"] = st.number_input(
            "Mileage",
            min_value=0,
            max_value=500000,
            value=50000
        )

        inputs["Engine Size"] = st.slider(
            "Engine Size",
            float(df["Engine Size"].min()),
            float(df["Engine Size"].max()),
            float(df["Engine Size"].median()),
            step=0.1
        )

    with col2:

        brand_choice = st.selectbox(
            "Brand",
            sorted(df["Brand"].dropna().unique())
        )

        fuel_choice = st.selectbox(
            "Fuel Type",
            sorted(df["Fuel Type"].dropna().unique())
        )

        transmission_choice = st.selectbox(
            "Transmission",
            sorted(df["Transmission"].dropna().unique())
        )

        condition_choice = st.selectbox(
            "Condition",
            sorted(df["Condition"].dropna().unique())
        )

        model_choice = st.selectbox(
            "Model",
            sorted(df["Model"].dropna().unique())
        )

    # Predict button
    if st.button(
        "Estimate Price",
        use_container_width=True
    ):

        input_row = pd.DataFrame([inputs])

        input_row["Brand"] = brand_choice
        input_row["Fuel Type"] = fuel_choice
        input_row["Transmission"] = transmission_choice
        input_row["Condition"] = condition_choice
        input_row["Model"] = model_choice

        # Encode input
        input_encoded = pd.get_dummies(input_row)

        input_encoded = input_encoded.reindex(
            columns=X_full.columns,
            fill_value=0
        )

        # Scale input
        input_scaled = scaler.transform(
            input_encoded
        )

        # Predict
        prediction = model.predict(
            input_scaled
        )[0]

        # Display prediction
        st.markdown(f"""
        <div style="
            background:#1a2035;
            border:1px solid #3b82f6;
            padding:30px;
            border-radius:15px;
            text-align:center;
            margin-top:20px;
        ">
            <div style="
                color:#94a3b8;
                font-size:1rem;
            ">
                Estimated Price
            </div>

            <div style="
                color:#60a5fa;
                font-size:3rem;
                font-weight:bold;
                margin-top:10px;
            ">
                ${prediction:,.0f}
            </div>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <p style='text-align:center;color:#64748b'>
    Built with Streamlit & Scikit-learn
    </p>
    """,
    unsafe_allow_html=True
)