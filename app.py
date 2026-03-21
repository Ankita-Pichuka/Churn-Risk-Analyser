"""
Customer Churn Risk Analyzer
Personal Project
Author: Ankita Pichuka · MS Data Analytics @ Northeastern
Dataset: Kaggle Bank Customer Churn Dataset
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score, f1_score
)
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import io

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Risk Analyzer",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

  .main { background: #0a0e1a; }
  .stApp { background: #0a0e1a; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #0f1525 !important;
    border-right: 1px solid #1e2d4a;
  }

  /* Cards */
  .metric-card {
    background: linear-gradient(135deg, #111827 0%, #1a2540 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,100,255,0.08);
  }
  .metric-card h2 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    margin: 0;
    color: #4da6ff;
  }
  .metric-card p { color: #8ba3c7; font-size: 0.85rem; margin: 4px 0 0; }

  /* Risk badge */
  .risk-high   { background:#ff3b3b22; color:#ff6b6b; border:1px solid #ff3b3b55; border-radius:8px; padding:6px 14px; font-weight:600; }
  .risk-medium { background:#ff960022; color:#ffa94d; border:1px solid #ff960055; border-radius:8px; padding:6px 14px; font-weight:600; }
  .risk-low    { background:#00d68f22; color:#00d68f; border:1px solid #00d68f55; border-radius:8px; padding:6px 14px; font-weight:600; }

  h1, h2, h3 { color: #e8eef8 !important; }
  p, li      { color: #8ba3c7; }
  .stMarkdown { color: #8ba3c7; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { background: #111827; border-radius: 10px; }
  .stTabs [data-baseweb="tab"]      { color: #8ba3c7; }
  .stTabs [aria-selected="true"]    { color: #4da6ff !important; }

  /* Slider label */
  .stSlider label { color: #8ba3c7 !important; }

  /* Section header */
  .section-header {
    font-family: 'IBM Plex Mono', monospace;
    color: #4da6ff !important;
    font-size: 0.78rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 6px;
    margin-bottom: 18px;
  }
  .hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    color: #e8eef8 !important;
    line-height: 1.15;
  }
  .hero-sub {
    color: #4da6ff;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.12em;
  }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Load the Kaggle Bank Churn dataset.
    Primary: direct URL  →  fallback: synthetic replica with same schema.
    """
    url = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/bank_churn.csv"
    try:
        df = pd.read_csv(url)
        # normalise common column name variants
        rename = {
            "Exited": "Churn", "exited": "Churn",
            "CustomerId": "CustomerID", "customerid": "CustomerID",
            "CreditScore": "CreditScore", "creditscore": "CreditScore",
            "Geography": "Geography", "geography": "Geography",
            "Gender": "Gender", "gender": "Gender",
            "Age": "Age", "age": "Age",
            "Tenure": "Tenure", "tenure": "Tenure",
            "Balance": "Balance", "balance": "Balance",
            "NumOfProducts": "NumOfProducts", "numofproducts": "NumOfProducts",
            "HasCrCard": "HasCrCard", "hascrcard": "HasCrCard",
            "IsActiveMember": "IsActiveMember", "isactivemember": "IsActiveMember",
            "EstimatedSalary": "EstimatedSalary", "estimatedsalary": "EstimatedSalary",
            "Surname": "Surname", "surname": "Surname",
            "RowNumber": "RowNumber", "rownumber": "RowNumber",
        }
        df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)
        if "Churn" not in df.columns:
            raise ValueError("Column 'Churn' not found")
        return df
    except Exception:
        return _synthetic_dataset()


def _synthetic_dataset(n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    geos    = rng.choice(["France", "Germany", "Spain"], n, p=[0.5, 0.25, 0.25])
    genders = rng.choice(["Male", "Female"], n)
    age     = rng.normal(38, 10, n).clip(18, 75).astype(int)
    tenure  = rng.integers(0, 11, n)
    balance = np.where(rng.random(n) < 0.3, 0, rng.normal(76_500, 62_000, n).clip(0))
    salary  = rng.normal(100_000, 32_000, n).clip(10_000)
    products= rng.choice([1, 2, 3, 4], n, p=[0.5, 0.46, 0.03, 0.01])
    credit  = rng.normal(650, 97, n).clip(350, 850).astype(int)
    crcard  = rng.integers(0, 2, n)
    active  = rng.integers(0, 2, n)
    # logistic churn probability
    log_odds = (
        -3.5
        + 0.04 * (age - 38)
        + 0.8  * (geos == "Germany").astype(float)
        - 0.6  * active
        + 0.4  * (balance == 0).astype(float)
        - 0.3  * (tenure > 5).astype(float)
    )
    p_churn = 1 / (1 + np.exp(-log_odds))
    churn   = (rng.random(n) < p_churn).astype(int)
    return pd.DataFrame({
        "RowNumber": np.arange(1, n + 1),
        "CustomerID": rng.integers(10_000_000, 20_000_000, n),
        "Surname": ["Surname"] * n,
        "CreditScore": credit,
        "Geography": geos,
        "Gender": genders,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance.round(2),
        "NumOfProducts": products,
        "HasCrCard": crcard,
        "IsActiveMember": active,
        "EstimatedSalary": salary.round(2),
        "Churn": churn,
    })


@st.cache_data
def preprocess(df: pd.DataFrame):
    drop_cols = [c for c in ["RowNumber", "CustomerID", "Surname"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Feature engineering
    df["BalanceToSalary"]   = df["Balance"] / (df["EstimatedSalary"] + 1)
    df["AgeGroup"]          = pd.cut(df["Age"], bins=[0, 30, 45, 60, 100],
                                     labels=["<30", "30-45", "45-60", "60+"])
    df["TenurePerProduct"]  = df["Tenure"] / (df["NumOfProducts"] + 1)
    df["IsZeroBalance"]     = (df["Balance"] == 0).astype(int)
    df["ProductsPerTenure"] = df["NumOfProducts"] / (df["Tenure"] + 1)

    # Encode categoricals
    le = LabelEncoder()
    for col in ["Geography", "Gender"]:
        if col in df.columns:
            df[col + "_enc"] = le.fit_transform(df[col])

    feature_cols = [
        "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
        "HasCrCard", "IsActiveMember", "EstimatedSalary",
        "Geography_enc", "Gender_enc",
        "BalanceToSalary", "TenurePerProduct", "IsZeroBalance", "ProductsPerTenure",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    y = df["Churn"]
    return df, X, y, feature_cols


@st.cache_resource
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10,
                                                 random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=4,
                                                         learning_rate=0.1, random_state=42),
        "Logistic Regression": LogisticRegression(C=0.5, max_iter=1000, random_state=42),
    }
    results = {}
    for name, mdl in models.items():
        if name == "Logistic Regression":
            mdl.fit(X_train_sc, y_train)
            preds    = mdl.predict(X_test_sc)
            proba    = mdl.predict_proba(X_test_sc)[:, 1]
            cv_X, cv_sc = X_train_sc, True
        else:
            mdl.fit(X_train, y_train)
            preds = mdl.predict(X_test)
            proba = mdl.predict_proba(X_test)[:, 1]
            cv_X, cv_sc = X_train, False

        cv = cross_val_score(mdl,
                             X_train_sc if cv_sc else X_train,
                             y_train,
                             cv=StratifiedKFold(5), scoring="roc_auc")
        results[name] = {
            "model": mdl, "preds": preds, "proba": proba,
            "auc": roc_auc_score(y_test, proba),
            "f1":  f1_score(y_test, preds),
            "cv_auc_mean": cv.mean(), "cv_auc_std": cv.std(),
            "report": classification_report(y_test, preds, output_dict=True),
            "cm": confusion_matrix(y_test, preds),
        }
    return results, scaler, X_train, X_test, y_train, y_test


# ══════════════════════════════════════════════════════════════════════════════
# PLOT HELPERS  (dark-themed Plotly)
# ══════════════════════════════════════════════════════════════════════════════

DARK = dict(
    paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
    font_color="#8ba3c7", font_family="IBM Plex Sans",
    xaxis=dict(gridcolor="#1e2d4a", zerolinecolor="#1e2d4a"),
    yaxis=dict(gridcolor="#1e2d4a", zerolinecolor="#1e2d4a"),
)

def apply_dark(fig):
    fig.update_layout(**DARK)
    fig.update_xaxes(gridcolor="#1e2d4a", zerolinecolor="#1e2d4a")
    fig.update_yaxes(gridcolor="#1e2d4a", zerolinecolor="#1e2d4a")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<p class="hero-sub">▸ ANKITA PICHUKA  ▸  NORTHEASTERN UNIVERSITY</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-title">Churn Risk<br>Analyzer</p>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<p class="section-header">Navigation</p>', unsafe_allow_html=True)
    page = st.radio("", [
        "📊  Overview & EDA",
        "🤖  Model Training",
        "🔍  Feature Insights",
        "🎯  Predict Single Customer",
        "📋  Model Report",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown('<p class="section-header">About</p>', unsafe_allow_html=True)
    st.markdown("""
    **Dataset**: Kaggle Bank Churn (10k rows)  
    **Models**: Random Forest · GBM · Logistic  
    **Stack**: Streamlit · Scikit-learn · Plotly  
    **Author**: Ankita Pichuka · MS Data Analytics @ Northeastern  
    """)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD & TRAIN
# ══════════════════════════════════════════════════════════════════════════════

with st.spinner("Loading data & training models…"):
    raw_df = load_data()
    df, X, y, feature_cols = preprocess(raw_df)
    results, scaler, X_train, X_test, y_train, y_test = train_models(X, y)

best_model_name = max(results, key=lambda k: results[k]["auc"])
best = results[best_model_name]

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW & EDA
# ══════════════════════════════════════════════════════════════════════════════

if page == "📊  Overview & EDA":
    st.markdown('<p class="hero-sub">▸ EXPLORATORY DATA ANALYSIS</p>', unsafe_allow_html=True)
    st.markdown("# Customer Churn — Dataset Overview")

    churn_rate = y.mean()
    col1, col2, col3, col4 = st.columns(4)
    for col, val, label in zip(
        [col1, col2, col3, col4],
        [f"{len(df):,}", f"{churn_rate:.1%}", f"{df['Age'].mean():.1f}", f"{df['Balance'].mean():,.0f}"],
        ["Total Customers", "Overall Churn Rate", "Avg Age", "Avg Balance ($)"]
    ):
        col.markdown(f"""
        <div class="metric-card"><h2>{val}</h2><p>{label}</p></div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["  Distribution  ", "  Correlations  ", "  Churn Drivers  "])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.pie(
                values=y.value_counts().values,
                names=["Retained", "Churned"],
                color_discrete_sequence=["#4da6ff", "#ff6b6b"],
                title="Churn Distribution",
                hole=0.55,
            )
            fig.update_layout(**DARK, title_font_color="#e8eef8")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.histogram(
                df, x="Age", color="Churn",
                barmode="overlay", nbins=40,
                color_discrete_map={0: "#4da6ff", 1: "#ff6b6b"},
                title="Age Distribution by Churn",
                labels={"Churn": "Churned"},
            )
            apply_dark(fig)
            fig.update_layout(title_font_color="#e8eef8")
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            if "Geography" in df.columns:
                geo_churn = df.groupby("Geography")["Churn"].mean().reset_index()
                fig = px.bar(geo_churn, x="Geography", y="Churn",
                             color="Churn", color_continuous_scale="Blues",
                             title="Churn Rate by Geography",
                             labels={"Churn": "Churn Rate"})
                apply_dark(fig)
                fig.update_layout(title_font_color="#e8eef8", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        with c4:
            fig = px.box(df, x="Churn", y="Balance",
                         color="Churn",
                         color_discrete_map={0: "#4da6ff", 1: "#ff6b6b"},
                         title="Balance vs Churn",
                         labels={"Churn": "Churned"})
            apply_dark(fig)
            fig.update_layout(title_font_color="#e8eef8")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        num_df = df[feature_cols + ["Churn"]].copy()
        corr   = num_df.corr()
        fig = px.imshow(
            corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Feature Correlation Matrix", aspect="auto",
        )
        fig.update_layout(**DARK, title_font_color="#e8eef8", height=600)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        churn_corr = num_df.corr()["Churn"].drop("Churn").sort_values()
        colors = ["#ff6b6b" if v > 0 else "#4da6ff" for v in churn_corr.values]
        fig = go.Figure(go.Bar(
            x=churn_corr.values, y=churn_corr.index,
            orientation="h",
            marker_color=colors,
        ))
        fig.update_layout(**DARK, title="Correlation with Churn",
                          title_font_color="#e8eef8", height=500,
                          xaxis_title="Pearson r", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

        if "AgeGroup" in df.columns:
            age_churn = df.groupby("AgeGroup", observed=True)["Churn"].mean().reset_index()
            fig2 = px.bar(age_churn, x="AgeGroup", y="Churn",
                          color="Churn", color_continuous_scale="Reds",
                          title="Churn Rate by Age Group",
                          labels={"Churn": "Churn Rate"})
            apply_dark(fig2)
            fig2.update_layout(title_font_color="#e8eef8", showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🤖  Model Training":
    st.markdown('<p class="hero-sub">▸ MODEL PERFORMANCE</p>', unsafe_allow_html=True)
    st.markdown("# Model Training & Evaluation")

    # Summary metrics
    cols = st.columns(3)
    for i, (name, res) in enumerate(results.items()):
        tag = " ⭐" if name == best_model_name else ""
        cols[i].markdown(f"""
        <div class="metric-card">
          <h2>{res['auc']:.3f}</h2>
          <p>ROC-AUC — {name}{tag}</p>
          <p style="color:#4da6ff">F1: {res['f1']:.3f} &nbsp;|&nbsp; CV: {res['cv_auc_mean']:.3f}±{res['cv_auc_std']:.3f}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ROC curves
    fig_roc = go.Figure()
    colors_roc = ["#4da6ff", "#ffa94d", "#00d68f"]
    for (name, res), c in zip(results.items(), colors_roc):
        fpr, tpr, _ = roc_curve(y_test, res["proba"])
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, name=f"{name} (AUC={res['auc']:.3f})",
            line=dict(color=c, width=2)
        ))
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Random",
                                  line=dict(dash="dash", color="#555")))
    fig_roc.update_layout(**DARK, title="ROC Curves — All Models",
                           title_font_color="#e8eef8",
                           xaxis_title="False Positive Rate",
                           yaxis_title="True Positive Rate", height=420)
    st.plotly_chart(fig_roc, use_container_width=True)

    # Confusion matrix for best model
    st.markdown(f"### Confusion Matrix — {best_model_name}")
    cm = best["cm"]
    fig_cm = px.imshow(
        cm, text_auto=True,
        x=["Predicted: Retained", "Predicted: Churned"],
        y=["Actual: Retained", "Actual: Churned"],
        color_continuous_scale=[[0, "#0a0e1a"], [1, "#4da6ff"]],
        title=f"Confusion Matrix — {best_model_name}",
    )
    fig_cm.update_layout(**DARK, title_font_color="#e8eef8", height=360)
    st.plotly_chart(fig_cm, use_container_width=True)

    # Precision-Recall
    prec, rec, _ = precision_recall_curve(y_test, best["proba"])
    ap = average_precision_score(y_test, best["proba"])
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=rec, y=prec,
                                 fill="tozeroy", fillcolor="rgba(77,166,255,0.1)",
                                 line=dict(color="#4da6ff"),
                                 name=f"AP={ap:.3f}"))
    fig_pr.update_layout(**DARK, title=f"Precision-Recall Curve — {best_model_name}",
                          title_font_color="#e8eef8",
                          xaxis_title="Recall", yaxis_title="Precision", height=380)
    st.plotly_chart(fig_pr, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — FEATURE INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔍  Feature Insights":
    st.markdown('<p class="hero-sub">▸ FEATURE IMPORTANCE</p>', unsafe_allow_html=True)
    st.markdown("# What Drives Churn?")

    rf_model = results["Random Forest"]["model"]
    importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=True)

    fig = go.Figure(go.Bar(
        x=importances.values, y=importances.index,
        orientation="h",
        marker=dict(
            color=importances.values,
            colorscale=[[0, "#1e3a5f"], [1, "#4da6ff"]],
        ),
    ))
    fig.update_layout(**DARK, title="Random Forest — Feature Importances",
                       title_font_color="#e8eef8",
                       xaxis_title="Importance", yaxis_title="", height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Churn probability by top features
    top2 = importances.sort_values(ascending=False).index[:2].tolist()
    c1, c2 = st.columns(2)
    for ax, feat in zip([c1, c2], top2):
        fig2 = px.violin(df, x="Churn", y=feat, color="Churn",
                         color_discrete_map={0: "#4da6ff", 1: "#ff6b6b"},
                         box=True, title=f"{feat} by Churn Status",
                         labels={"Churn": "Churned"})
        apply_dark(fig2)
        fig2.update_layout(title_font_color="#e8eef8")
        ax.plotly_chart(fig2, use_container_width=True)

    # Products breakdown
    if "NumOfProducts" in df.columns:
        prod_churn = df.groupby("NumOfProducts")["Churn"].mean().reset_index()
        fig3 = px.bar(prod_churn, x="NumOfProducts", y="Churn",
                      color="Churn", color_continuous_scale="Blues",
                      title="Churn Rate by Number of Products",
                      labels={"Churn": "Churn Rate", "NumOfProducts": "# Products"})
        apply_dark(fig3)
        fig3.update_layout(title_font_color="#e8eef8", showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PREDICT SINGLE CUSTOMER
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🎯  Predict Single Customer":
    st.markdown('<p class="hero-sub">▸ INDIVIDUAL RISK SCORING</p>', unsafe_allow_html=True)
    st.markdown("# Predict Churn Risk for a Single Customer")

    col1, col2, col3 = st.columns(3)
    with col1:
        credit_score  = st.slider("Credit Score",  300, 850, 650)
        age           = st.slider("Age",             18,  92,  38)
        tenure        = st.slider("Tenure (years)",   0,  10,   5)
        balance       = st.number_input("Account Balance ($)", 0.0, 300_000.0, 76_500.0, step=500.0)
    with col2:
        num_products  = st.selectbox("Number of Products", [1, 2, 3, 4])
        has_cr_card   = st.selectbox("Has Credit Card",   [1, 0], format_func=lambda x: "Yes" if x else "No")
        is_active     = st.selectbox("Active Member",     [1, 0], format_func=lambda x: "Yes" if x else "No")
        salary        = st.number_input("Estimated Salary ($)", 0.0, 300_000.0, 100_000.0, step=1_000.0)
    with col3:
        geography     = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender        = st.selectbox("Gender",    ["Male", "Female"])

    geo_enc    = {"France": 0, "Germany": 1, "Spain": 2}[geography]
    gender_enc = {"Male": 1, "Female": 0}[gender]
    bal_sal    = balance / (salary + 1)
    ten_prod   = tenure / (num_products + 1)
    zero_bal   = int(balance == 0)
    prod_ten   = num_products / (tenure + 1)

    input_dict = {
        "CreditScore": credit_score, "Age": age, "Tenure": tenure,
        "Balance": balance, "NumOfProducts": num_products,
        "HasCrCard": has_cr_card, "IsActiveMember": is_active,
        "EstimatedSalary": salary, "Geography_enc": geo_enc, "Gender_enc": gender_enc,
        "BalanceToSalary": bal_sal, "TenurePerProduct": ten_prod,
        "IsZeroBalance": zero_bal, "ProductsPerTenure": prod_ten,
    }
    input_df = pd.DataFrame([{c: input_dict.get(c, 0) for c in feature_cols}])

    if st.button("🔮  Run Churn Prediction", use_container_width=True):
        rf_mdl = results["Random Forest"]["model"]
        prob   = rf_mdl.predict_proba(input_df)[0][1]

        level = "HIGH" if prob >= 0.6 else ("MEDIUM" if prob >= 0.35 else "LOW")
        css   = "risk-high" if level == "HIGH" else ("risk-medium" if level == "MEDIUM" else "risk-low")

        st.markdown("<br>", unsafe_allow_html=True)
        r1, r2, r3 = st.columns([1, 1, 1])
        r1.markdown(f"""
        <div class="metric-card">
          <h2>{prob:.1%}</h2>
          <p>Churn Probability</p>
        </div>""", unsafe_allow_html=True)
        r2.markdown(f"""
        <div class="metric-card">
          <p style="margin-top:18px"><span class="{css}">{level} RISK</span></p>
          <p>Risk Classification</p>
        </div>""", unsafe_allow_html=True)
        r3.markdown(f"""
        <div class="metric-card">
          <h2>{1-prob:.1%}</h2>
          <p>Retention Probability</p>
        </div>""", unsafe_allow_html=True)

        # Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Churn Risk Score", "font": {"color": "#e8eef8", "size": 18}},
            number={"suffix": "%", "font": {"color": "#4da6ff", "size": 36}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8ba3c7"},
                "bar":  {"color": "#ff6b6b" if level == "HIGH" else ("#ffa94d" if level == "MEDIUM" else "#00d68f")},
                "bgcolor": "#111827",
                "steps": [
                    {"range": [0, 35],  "color": "#00d68f22"},
                    {"range": [35, 60], "color": "#ff960022"},
                    {"range": [60, 100],"color": "#ff3b3b22"},
                ],
                "threshold": {"line": {"color": "#e8eef8", "width": 2}, "value": prob * 100},
            }
        ))
        fig_gauge.update_layout(**DARK, height=350)
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Actionable recommendations
        st.markdown("### 💡 Recommended Actions")
        actions = []
        if prob >= 0.6:
            actions += ["🚨 **High Priority** — Flag for immediate retention outreach",
                        "📞 Assign dedicated relationship manager",
                        "🎁 Offer personalised loyalty incentive or rate upgrade"]
        if is_active == 0:
            actions.append("📲 Re-engagement campaign — encourage app/online banking usage")
        if zero_bal:
            actions.append("💰 Offer high-yield savings product to build balance engagement")
        if num_products == 1:
            actions.append("🔗 Cross-sell second product (e.g. credit card, investment account)")
        if age > 55:
            actions.append("🏦 Offer dedicated senior banking advisor consultation")
        if not actions:
            actions.append("✅ Customer appears low-risk — maintain standard relationship cadence")
        for a in actions:
            st.markdown(f"- {a}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL REPORT
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📋  Model Report":
    st.markdown('<p class="hero-sub">▸ FULL EVALUATION REPORT</p>', unsafe_allow_html=True)
    st.markdown("# Model Performance Report")

    for name, res in results.items():
        tag = " ⭐ Best Model" if name == best_model_name else ""
        with st.expander(f"**{name}**{tag} — AUC {res['auc']:.4f}", expanded=(name == best_model_name)):
            rep = res["report"]
            rows = []
            for cls in ["0", "1"]:
                rows.append({
                    "Class": "Retained" if cls == "0" else "Churned",
                    "Precision": f"{rep[cls]['precision']:.4f}",
                    "Recall":    f"{rep[cls]['recall']:.4f}",
                    "F1-Score":  f"{rep[cls]['f1-score']:.4f}",
                    "Support":   int(rep[cls]["support"]),
                })
            rows.append({
                "Class": "Macro Avg",
                "Precision": f"{rep['macro avg']['precision']:.4f}",
                "Recall":    f"{rep['macro avg']['recall']:.4f}",
                "F1-Score":  f"{rep['macro avg']['f1-score']:.4f}",
                "Support":   int(rep['macro avg']['support']),
            })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            c1, c2 = st.columns(2)
            c1.metric("ROC-AUC",  f"{res['auc']:.4f}")
            c2.metric("CV AUC",   f"{res['cv_auc_mean']:.4f} ± {res['cv_auc_std']:.4f}")

    # Download report
    st.markdown("---")
    st.markdown("### 📥 Export Report")
    report_lines = ["# Churn Risk Analyzer — Model Report\n"]
    report_lines.append(f"Dataset: {len(df):,} customers | Churn rate: {y.mean():.1%}\n")
    report_lines.append(f"Best model: {best_model_name} (AUC={best['auc']:.4f})\n\n")
    for name, res in results.items():
        report_lines.append(f"## {name}\n")
        report_lines.append(f"- ROC-AUC : {res['auc']:.4f}\n")
        report_lines.append(f"- F1      : {res['f1']:.4f}\n")
        report_lines.append(f"- CV AUC  : {res['cv_auc_mean']:.4f} ± {res['cv_auc_std']:.4f}\n\n")
    st.download_button(
        label="⬇️  Download Markdown Report",
        data="".join(report_lines),
        file_name="churn_model_report.md",
        mime="text/markdown",
        use_container_width=True,
    )
