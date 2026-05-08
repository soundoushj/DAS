import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


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
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Syne', sans-serif;
    }

    .stApp {
        background-color: #0f1117;
        color: #e8eaf0;
    }

    .metric-card {
        background: linear-gradient(135deg, #1a2035 0%, #1e2740 100%);
        border: 1px solid #2d3a52;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #60a5fa;
        font-family: 'Syne', sans-serif;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .section-header {
        font-family: 'Syne', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #f0f4ff;
        border-left: 4px solid #3b82f6;
        padding-left: 14px;
        margin: 32px 0 16px 0;
    }

    .insight-box {
        background: #1a2035;
        border: 1px solid #2d3a52;
        border-left: 4px solid #10b981;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 10px 0;
        font-size: 0.92rem;
        color: #cbd5e1;
    }

    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
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
AMBER = "#f59e0b"
ROSE = "#f43f5e"
MUTED = "#94a3b8"
TEXT = "#e8eaf0"


# =========================================================
# PLOT STYLE
# =========================================================

def apply_dark_style(ax, fig):
    fig.patch.set_facecolor(CARD_BG)
    ax.set_facecolor(CARD_BG)

    ax.tick_params(colors=MUTED, labelsize=9)

    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)

    ax.title.set_color(TEXT)

    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Data_Set/Cars_dataset.csv")

    except FileNotFoundError:

        np.random.seed(42)

        n = 300

        brands = ["Toyota", "BMW", "Honda", "Ford", "Mercedes"]
        fuel = ["Petrol", "Diesel", "Electric"]

        df = pd.DataFrame({
            "Car ID": range(1, n + 1),
            "Brand": np.random.choice(brands, n),
            "Fuel Type": np.random.choice(fuel, n),
            "Year": np.random.randint(2010, 2024, n),
            "Mileage": np.random.randint(5000, 120000, n),
            "Engine Size": np.random.uniform(1.0, 4.5, n).round(1),
            "HP": np.random.randint(80, 400, n),
            "Price": np.random.randint(8000, 80000, n),
        })

    return df


df = load_data()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("Cars Dashboard")

    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["Overview", "EDA", "ML Model", "Predict"]
    )

    st.markdown("---")

    st.subheader("Filters")

    # Brand filter
    if "Brand" in df.columns:

        brands_available = sorted(df["Brand"].dropna().unique())

        selected_brands = st.multiselect(
            "Brand",
            brands_available,
            default=brands_available
        )

    else:
        selected_brands = []

    # Fuel filter
    if "Fuel Type" in df.columns:

        fuel_available = sorted(df["Fuel Type"].dropna().unique())

        selected_fuel = st.multiselect(
            "Fuel Type",
            fuel_available,
            default=fuel_available
        )

    else:
        selected_fuel = []

    # Year filter
    if "Year" in df.columns:

        year_min = int(df["Year"].min())
        year_max = int(df["Year"].max())

        year_range = st.slider(
            "Year Range",
            year_min,
            year_max,
            (year_min, year_max)
        )

    else:
        year_range = (0, 9999)


# =========================================================
# FILTER DATA
# =========================================================

dff = df.copy()

if selected_brands and "Brand" in dff.columns:
    dff = dff[dff["Brand"].isin(selected_brands)]

if selected_fuel and "Fuel Type" in dff.columns:
    dff = dff[dff["Fuel Type"].isin(selected_fuel)]

if "Year" in dff.columns:
    dff = dff[
        (dff["Year"] >= year_range[0]) &
        (dff["Year"] <= year_range[1])
    ]


# =========================================================
# PAGE 1 — OVERVIEW
# =========================================================

if page == "Overview":

    st.title("Cars Price Analysis")

    st.markdown(
        "End-to-end data science pipeline from raw data to predictive modeling."
    )

    st.markdown("---")

    c1, c2, c3, c4, c5 = st.columns(5)

    kpis = [
        ("Total Cars", f"{len(dff):,}"),
        ("Average Price", f"${dff['Price'].mean():,.0f}"),
        ("Maximum Price", f"${dff['Price'].max():,.0f}"),
        ("Minimum Price", f"${dff['Price'].min():,.0f}"),
        ("Brands", str(dff['Brand'].nunique()))
    ]

    for col, (label, value) in zip([c1, c2, c3, c4, c5], kpis):

        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-header">Dataset Preview</div>',
        unsafe_allow_html=True
    )

    st.dataframe(dff.head(10), use_container_width=True)

    st.markdown(
        '<div class="section-header">Statistical Summary</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        dff.describe().T.style.format("{:.2f}"),
        use_container_width=True
    )

    st.markdown("""
    <div class="insight-box">
    The dataset includes multiple brands, fuel types,
    and production years. Price variation suggests
    strong feature influence, making the dataset
    suitable for regression modeling.
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# PAGE 2 — EDA
# =========================================================

elif page == "EDA":

    st.markdown(
        '<div class="section-header">Exploratory Data Analysis</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    # Price distribution
    with col1:

        st.subheader("Price Distribution")

        fig, ax = plt.subplots(figsize=(6, 3.5))

        apply_dark_style(ax, fig)

        sns.histplot(
            dff["Price"],
            kde=True,
            color=BLUE,
            bins=30,
            alpha=0.7,
            ax=ax
        )

        ax.set_title("Price Distribution")
        ax.set_xlabel("Price ($)")

        st.pyplot(fig)

        plt.close()

    # Brand counts
    with col2:

        st.subheader("Cars by Brand")

        fig, ax = plt.subplots(figsize=(6, 3.5))

        apply_dark_style(ax, fig)

        brand_counts = dff["Brand"].value_counts()

        brand_counts.plot(
            kind="bar",
            color=[BLUE, TEAL, AMBER, ROSE],
            ax=ax
        )

        ax.set_title("Cars by Brand")

        plt.xticks(rotation=30)

        st.pyplot(fig)

        plt.close()

    # Correlation heatmap
    st.markdown(
        '<div class="section-header">Correlation Heatmap</div>',
        unsafe_allow_html=True
    )

    numeric_df = dff.select_dtypes(include=np.number)

    numeric_df = numeric_df.drop(
        columns=["Car ID"],
        errors="ignore"
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    apply_dark_style(ax, fig)

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        ax=ax
    )

    ax.set_title("Feature Correlation Matrix")

    st.pyplot(fig)

    plt.close()

    # Mileage vs Price
    if "Mileage" in dff.columns:

        st.markdown(
            '<div class="section-header">Mileage vs Price</div>',
            unsafe_allow_html=True
        )

        fig, ax = plt.subplots(figsize=(9, 4))

        apply_dark_style(ax, fig)

        ax.scatter(
            dff["Mileage"],
            dff["Price"],
            alpha=0.5,
            color=BLUE
        )

        z = np.polyfit(
            dff["Mileage"],
            dff["Price"],
            1
        )

        p = np.poly1d(z)

        xs = np.linspace(
            dff["Mileage"].min(),
            dff["Mileage"].max(),
            200
        )

        ax.plot(xs, p(xs), color=ROSE)

        ax.set_xlabel("Mileage")
        ax.set_ylabel("Price")

        st.pyplot(fig)

        plt.close()


# =========================================================
# PAGE 3 — MACHINE LEARNING
# =========================================================

elif page == "ML Model":

    st.markdown(
        '<div class="section-header">Linear Regression Modeling</div>',
        unsafe_allow_html=True
    )

    df_model = dff.drop(columns=["Car ID"], errors="ignore").copy()

    df_encoded = pd.get_dummies(df_model, drop_first=True)

    df_encoded = df_encoded.fillna(
        df_encoded.median(numeric_only=True)
    )

    y = df_encoded["Price"]

    X = df_encoded.drop(columns=["Price"])

    test_size = st.slider(
        "Test Set Size",
        0.1,
        0.4,
        0.2,
        0.05
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_choice = st.selectbox(
        "Model",
        [
            "Linear Regression",
            "Ridge Regression",
            "Lasso Regression"
        ]
    )

    if model_choice == "Linear Regression":

        model = LinearRegression()

    elif model_choice == "Ridge Regression":

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

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    r2 = r2_score(y_test, y_pred)

    mae = mean_absolute_error(y_test, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_test, y_pred)
    )

    cv = cross_val_score(
        model,
        scaler.transform(X),
        y,
        cv=5,
        scoring="r2"
    ).mean()

    st.markdown(
        '<div class="section-header">Model Performance</div>',
        unsafe_allow_html=True
    )

    m1, m2, m3, m4 = st.columns(4)

    metrics = [
        ("R² Score", f"{r2:.4f}"),
        ("MAE", f"${mae:,.0f}"),
        ("RMSE", f"${rmse:,.0f}"),
        ("CV R²", f"{cv:.4f}")
    ]

    for col, (label, value) in zip(
        [m1, m2, m3, m4],
        metrics
    ):

        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    # Actual vs Predicted
    st.subheader("Actual vs Predicted")

    fig, ax = plt.subplots(figsize=(6, 4))

    apply_dark_style(ax, fig)

    ax.scatter(
        y_test,
        y_pred,
        alpha=0.5,
        color=BLUE
    )

    lim = (
        min(y_test.min(), y_pred.min()),
        max(y_test.max(), y_pred.max())
    )

    ax.plot(lim, lim, linestyle="--", color=ROSE)

    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")

    st.pyplot(fig)

    plt.close()

    # Feature importance
    st.markdown(
        '<div class="section-header">Feature Importance</div>',
        unsafe_allow_html=True
    )

    coef_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_
    })

    coef_df = coef_df.reindex(
        coef_df["Coefficient"]
        .abs()
        .sort_values(ascending=False)
        .index
    )

    top_coef = coef_df.head(12)

    fig, ax = plt.subplots(figsize=(9, 4))

    apply_dark_style(ax, fig)

    colors = [
        TEAL if x >= 0 else ROSE
        for x in top_coef["Coefficient"]
    ]

    ax.barh(
        top_coef["Feature"],
        top_coef["Coefficient"],
        color=colors
    )

    ax.invert_yaxis()

    st.pyplot(fig)

    plt.close()


# =========================================================
# PAGE 4 — PREDICTION
# =========================================================

elif page == "Predict":

    st.markdown(
        '<div class="section-header">Price Predictor</div>',
        unsafe_allow_html=True
    )

    df_model = df.drop(columns=["Car ID"], errors="ignore").copy()

    df_encoded = pd.get_dummies(
        df_model,
        drop_first=True
    )

    df_encoded = df_encoded.fillna(
        df_encoded.median(numeric_only=True)
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
            0,
            300000,
            50000
        )

        inputs["HP"] = st.slider(
            "Horsepower",
            int(df["HP"].min()),
            int(df["HP"].max()),
            int(df["HP"].median())
        )

    with col2:

        inputs["Engine Size"] = st.slider(
            "Engine Size",
            float(df["Engine Size"].min()),
            float(df["Engine Size"].max()),
            float(df["Engine Size"].median()),
            0.1
        )

        brand_choice = st.selectbox(
            "Brand",
            sorted(df["Brand"].unique())
        )

        fuel_choice = st.selectbox(
            "Fuel Type",
            sorted(df["Fuel Type"].unique())
        )

    if st.button("Estimate Price", use_container_width=True):

        input_row = pd.DataFrame([inputs])

        input_row["Brand"] = brand_choice
        input_row["Fuel Type"] = fuel_choice

        input_encoded = pd.get_dummies(input_row)

        input_encoded = input_encoded.reindex(
            columns=X_full.columns,
            fill_value=0
        )

        input_scaled = scaler.transform(input_encoded)

        prediction = model.predict(input_scaled)[0]

        st.markdown(f"""
        <div style="
            background:linear-gradient(135deg,#1a2035,#1e2740);
            border:1px solid #3b82f6;
            border-radius:14px;
            padding:30px;
            text-align:center;
            margin-top:20px;
        ">
            <div style="
                font-size:0.9rem;
                color:#94a3b8;
                text-transform:uppercase;
                letter-spacing:0.1em;
            ">
                Estimated Price
            </div>

            <div style="
                font-size:3rem;
                font-weight:800;
                color:#60a5fa;
                font-family:'Syne',sans-serif;
                margin:8px 0;
            ">
                ${prediction:,.0f}
            </div>

            <div style="
                font-size:0.8rem;
                color:#475569;
            ">
                Prediction generated using linear regression
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown(
        """
        <p style="
            color:#475569;
            font-size:0.8rem;
            text-align:center;
        ">
        Built with Streamlit and Scikit-learn
        </p>
        """,
        unsafe_allow_html=True
    )