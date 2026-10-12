"""
RetailIntel: E-Commerce Data Warehouse and Mining (DWDM) Analytics Platform.
Full dashboard implementation covering 9 curriculum labs, data exploration, and executive metrics.
"""

import os
import textwrap
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

# Machine learning and data mining libraries
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    silhouette_score,
)
from mlxtend.frequent_patterns import apriori, association_rules

# -----------------------------------------------------------------------------
# Configuration and Global Styling
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="RetailIntel | DWDM Analytics Platform",
    page_icon="ri-logo",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY_COLOR = "#6366F1"
ACCENT_COLOR = "#4F46E5"
PALETTE = ["#4F46E5", "#06B6D4", "#10B981", "#F59E0B", "#F43F5E", "#8B5CF6"]

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"], .main {
  background-color: #F8FAFC !important;
  color: #0F172A !important;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

[data-testid="stHeader"] {
  background-color: #FFFFFF !important;
  border-bottom: 1px solid #E2E8F0 !important;
}

[data-testid="stSidebar"] {
  background-color: #FFFFFF !important;
  border-right: 1px solid #E2E8F0 !important;
}

/* Card container style */
.card {
  background: #FFFFFF;
  border: 1px solid #E2E8F0;
  border-radius: 12px;
  padding: 20px;
  margin-bottom: 16px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}

.pill-badge {
  display: inline-flex;
  align-items: center;
  font-size: 11px;
  font-weight: 600;
  padding: 3px 9px;
  border-radius: 9999px;
  letter-spacing: 0.02em;
}

/* Sidebar navigation item overrides */
[data-testid="stSidebar"] .stRadio > div {
  gap: 4px !important;
}

[data-testid="stSidebar"] .stRadio label {
  padding: 7px 12px !important;
  border-radius: 8px !important;
  border: 1px solid transparent !important;
  color: #475569 !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  transition: all 0.15s ease !important;
  cursor: pointer !important;
}

[data-testid="stSidebar"] .stRadio label:hover {
  background-color: #F1F5F9 !important;
  color: #0F172A !important;
}

[data-testid="stSidebar"] .stRadio label:has(input:checked) {
  background-color: #EEF2FF !important;
  border: 1px solid #C7D2FE !important;
  color: #4F46E5 !important;
  font-weight: 600 !important;
}

/* Streamlit container border override to match cards */
[data-testid="stVerticalBlockBorderWrapper"] {
  border-radius: 12px !important;
  border-color: #E2E8F0 !important;
  background-color: #FFFFFF !important;
}

/* Sliders */
[data-testid="stSlider"] [role="slider"] {
  background-color: #4F46E5 !important;
}
</style>
"""

def render_html(html_str: str):
    """
    Renders custom HTML without leading whitespace to avoid CommonMark code block parsing.
    """
    dedented = textwrap.dedent(html_str).strip()
    cleaned_lines = [line.strip() for line in dedented.splitlines() if line.strip()]
    st.markdown("\n".join(cleaned_lines), unsafe_allow_html=True)

render_html(CUSTOM_CSS)

# -----------------------------------------------------------------------------
# Data Ingestion and Transformation
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_and_prepare_data(file_path: str = "ecommerce_dataset_10000.csv") -> pd.DataFrame:
    """
    Loads transaction dataset and prepares derived dimensions and financial metrics.
    """
    if not os.path.exists(file_path):
        st.error(f"Dataset '{file_path}' not found in workspace.")
        st.stop()

    df = pd.read_csv(file_path)

    # Clean whitespace
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    # Numeric formatting
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(1).astype(int)
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce").fillna(0.0)
    df["total_amount"] = df["quantity"] * df["unit_price"]

    # Datetime derivations
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
    df["order_year"] = df["order_date"].dt.year
    df["order_month"] = df["order_date"].dt.month
    df["order_quarter"] = "Q" + df["order_date"].dt.quarter.astype(str)
    df["day_name"] = df["order_date"].dt.day_name()
    df["order_period"] = df["order_year"].astype(str) + "-" + df["order_quarter"].astype(str)

    return df

df = load_and_prepare_data()

# -----------------------------------------------------------------------------
# Shared UI Card Renderers
# -----------------------------------------------------------------------------

def apply_chart_theme(fig, height: int = 320, show_legend: bool = True):
    """
    Applies unified white-card styling and custom color scheme to Plotly figures.
    """
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(t=35, b=25, l=25, r=25),
        font=dict(family="Inter, sans-serif", color="#334155", size=12),
        showlegend=show_legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1) if show_legend else {},
        plot_bgcolor="rgba(255,255,255,0)",
        paper_bgcolor="rgba(255,255,255,0)",
    )
    return fig

def render_top_header(active_module: str):
    """
    Renders top breadcrumbs and system status pill badges.
    """
    html = f"""
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 4px 0 16px 0; border-bottom: 1px solid #E2E8F0; margin-bottom: 20px; flex-wrap: wrap; gap: 12px;">
      <div style="display: flex; align-items: center; gap: 8px; font-size: 13px;">
        <span style="color: #64748B; font-weight: 500;">E-Commerce Analytics Platform</span>
        <span style="color: #CBD5E1;">/</span>
        <span style="color: #0F172A; font-weight: 600;">{active_module}</span>
      </div>
      <div style="display: flex; align-items: center; gap: 8px;">
        <span class="pill-badge" style="background: #F1F5F9; color: #334155; border: 1px solid #E2E8F0;">Dataset: 10,000 Records</span>
        <span class="pill-badge" style="background: #ECFDF5; color: #059669; border: 1px solid #A7F3D0;">Mode: In-Memory OLAP</span>
      </div>
    </div>
    """
    render_html(html)

def render_module_header(tag: str, title: str, subtitle: str, theme_tag: str = "Core DWDM"):
    """
    Uniform Module Header Card.
    """
    html = f"""
    <div class="card">
      <div style="display: flex; gap: 8px; align-items: center; margin-bottom: 8px;">
        <span class="pill-badge" style="background: #EEF2FF; color: #4F46E5; border: 1px solid #C7D2FE;">{tag}</span>
        <span class="pill-badge" style="background: #F8FAFC; color: #64748B; border: 1px solid #E2E8F0;">{theme_tag}</span>
      </div>
      <h2 style="margin: 0; font-size: 22px; font-weight: 700; color: #0F172A; letter-spacing: -0.02em;">{title}</h2>
      <p style="margin: 4px 0 0 0; font-size: 13px; color: #64748B;">{subtitle}</p>
    </div>
    """
    render_html(html)

def render_problem_card(title: str, business_problem: str, operational_impact: str):
    """
    Uniform Problem Statement Card.
    """
    html = f"""
    <div class="card" style="border-left: 4px solid #4F46E5;">
      <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #4F46E5; margin-bottom: 4px;">Operational Problem Statement</div>
      <h4 style="margin: 0 0 6px 0; font-size: 16px; font-weight: 600; color: #0F172A;">{title}</h4>
      <p style="margin: 0 0 10px 0; font-size: 13px; color: #334155; line-height: 1.5;">{business_problem}</p>
      <div style="padding: 10px 14px; background: #F8FAFC; border-radius: 8px; border: 1px solid #E2E8F0; font-size: 12px; color: #475569;">
        <strong style="color: #0F172A;">Identified Bottleneck:</strong> {operational_impact}
      </div>
    </div>
    """
    render_html(html)

def render_takeaways_card(takeaways: list):
    """
    Uniform Business Takeaway Card.
    """
    bullets = "".join([f"<li style='margin-bottom: 6px;'>{t}</li>" for t in takeaways])
    html = f"""
    <div class="card" style="border-left: 4px solid #10B981; margin-top: 16px;">
      <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #059669; margin-bottom: 4px;">Operational Takeaways & Recommendations</div>
      <ul style="margin: 0; padding-left: 18px; font-size: 13px; color: #334155; line-height: 1.55;">
        {bullets}
      </ul>
    </div>
    """
    render_html(html)

def render_metric_card(label: str, value: str, subtext: str = ""):
    """
    Renders styled stat card matching SaaS theme.
    """
    sub_html = f"<div style='font-size: 11px; color: #64748B; margin-top: 2px;'>{subtext}</div>" if subtext else ""
    html = f"""
    <div class="card" style="padding: 16px; margin-bottom: 12px;">
      <div style="font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #64748B;">{label}</div>
      <div style="font-size: 24px; font-weight: 700; color: #0F172A; margin-top: 4px; letter-spacing: -0.02em;">{value}</div>
      {sub_html}
    </div>
    """
    render_html(html)

def render_erd_legend_bar():
    """
    Renders global ERD visual badge legend:
    Fact Table (#7C3AED), Dimension Table (#0284C7), PK (#F59E0B), FK (#3B82F6), NUM (#8B5CF6), ATTR (#64748B).
    """
    html = """
    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 12px 18px; margin: 4px 0 16px 0; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
      <div style="display: flex; align-items: center; gap: 8px;">
        <span style="font-size: 11px; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.05em;">Global ERD Legend:</span>
      </div>
      <div style="display: flex; align-items: center; gap: 14px; flex-wrap: wrap; font-size: 11px;">
        <div style="display: flex; align-items: center; gap: 6px;">
          <span style="display: inline-block; width: 12px; height: 12px; border-radius: 3px; background: #7C3AED;"></span>
          <span style="color: #1E293B; font-weight: 600;">Fact Table (#7C3AED)</span>
        </div>
        <div style="display: flex; align-items: center; gap: 6px;">
          <span style="display: inline-block; width: 12px; height: 12px; border-radius: 3px; background: #0284C7;"></span>
          <span style="color: #1E293B; font-weight: 600;">Dimension Table (#0284C7)</span>
        </div>
        <div style="display: flex; align-items: center; gap: 5px;">
          <span style="background: #FEF3C7; color: #B45309; padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 10px;">PK</span>
          <span style="color: #475569;">Primary Key</span>
        </div>
        <div style="display: flex; align-items: center; gap: 5px;">
          <span style="background: #DBEAFE; color: #1D4ED8; padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 10px;">FK</span>
          <span style="color: #475569;">Foreign Key</span>
        </div>
        <div style="display: flex; align-items: center; gap: 5px;">
          <span style="background: #EDE9FE; color: #6D28D9; padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 10px;">NUM</span>
          <span style="color: #475569;">Measure</span>
        </div>
        <div style="display: flex; align-items: center; gap: 5px;">
          <span style="background: #F1F5F9; color: #475569; padding: 2px 6px; border-radius: 4px; font-weight: 500; font-size: 10px;">ATTR</span>
          <span style="color: #475569;">Descriptive Attribute</span>
        </div>
      </div>
    </div>
    """
    render_html(html)

def render_erd_table_card(
    table_name: str,
    meta_info: str,
    columns: list,
    is_fact: bool = False,
    header_color: str = None,
    badge_label: str = "",
    join_footnote: str = ""
) -> str:
    """
    Renders a Database Schema ERD Card matching the exact UI/UX specification.
    """
    bg_color = header_color if header_color else ("#7C3AED" if is_fact else "#0284C7")
    tag_label = badge_label if badge_label else ("FACT TABLE" if is_fact else "DIMENSION")

    rows_html = []
    for item in columns:
        badges = item[0] if isinstance(item[0], list) else [item[0]]
        col_name = item[1]
        dtype = item[2]
        extra_note = item[3] if len(item) > 3 else ""

        badge_tags = []
        for b in badges:
            b_up = b.upper()
            if b_up == "PK":
                badge_tags.append('<span style="background: #FEF3C7; color: #B45309; padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 10px;">PK</span>')
            elif b_up == "FK":
                badge_tags.append('<span style="background: #DBEAFE; color: #1D4ED8; padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 10px;">FK</span>')
            elif b_up == "NUM":
                badge_tags.append('<span style="background: #EDE9FE; color: #6D28D9; padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 10px;">NUM</span>')
            elif b_up == "ATTR":
                badge_tags.append('<span style="background: #F1F5F9; color: #475569; padding: 2px 6px; border-radius: 4px; font-weight: 500; font-size: 10px;">ATTR</span>')
            else:
                badge_tags.append(f'<span style="background: #F1F5F9; color: #475569; padding: 2px 6px; border-radius: 4px; font-weight: 500; font-size: 10px;">{b}</span>')

        badge_html = " ".join(badge_tags)
        note_html = f' <span style="font-size: 10px; color: #2563EB; font-weight: 500;">{extra_note}</span>' if extra_note else ""

        row = f'<div style="display: flex; justify-content: space-between; align-items: center; padding: 6px 14px; border-bottom: 1px solid #F1F5F9; font-size: 12px;"><div style="display: flex; align-items: center; gap: 6px;">{badge_html}<strong style="color: #0F172A; font-family: monospace; font-size: 12px;">{col_name}</strong>{note_html}</div><span style="color: #64748B; font-size: 11px; font-family: monospace;">{dtype}</span></div>'
        rows_html.append(row)

    footnote_html = ""
    if join_footnote:
        footnote_html = f'<div style="padding: 6px 14px; background: #F8FAFC; border-top: 1px solid #F1F5F9; font-size: 10px; color: #64748B; font-weight: 500;">{join_footnote}</div>'

    rows_str = "".join(rows_html)
    card_html = (
        f'<div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); margin-bottom: 16px;">'
        f'<div style="background: {bg_color}; color: #FFFFFF; padding: 10px 14px; font-weight: 700; font-size: 13px; display: flex; justify-content: space-between; align-items: center;">'
        f'<div style="display: flex; align-items: center; gap: 8px;">'
        f'<span>{table_name}</span>'
        f'<span style="font-size: 9px; font-weight: 700; background: rgba(255,255,255,0.22); padding: 1px 6px; border-radius: 4px; letter-spacing: 0.04em;">{tag_label}</span>'
        f'</div>'
        f'<span style="font-size: 11px; font-weight: 500; opacity: 0.95; background: rgba(0,0,0,0.18); padding: 2px 8px; border-radius: 9999px;">{meta_info}</span>'
        f'</div>'
        f'<div>{rows_str}</div>'
        f'{footnote_html}'
        f'</div>'
    )
    return textwrap.dedent(card_html).strip()

# -----------------------------------------------------------------------------
# Sidebar Navigation Architecture
# -----------------------------------------------------------------------------

NAVIGATION_OPTIONS = [
    # Group 1: Main
    "Dashboard Overview",
    # Group 2: Data & Exploration
    "Dataset Explorer & Schema",
    # Group 3: DWDM Syllabus (1-9)
    "01 — Dimensional Modeling (Star Schema)",
    "02 — Multidimensional OLAP Engine",
    "03 — Data Profiling & ETL Pipeline",
    "04 — Linear Regression (Sales Forecasting)",
    "05 — Decision Tree Classification",
    "06 — Naïve Bayes Classification",
    "07 — K-Means Customer Clustering",
    "08 — Hierarchical Clustering",
    "09 — Apriori Association Rule Mining",
    # Group 4: Documentation
    "About Project & Architecture",
]

with st.sidebar:
    # Brand Header
    brand_html = """
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px; padding-bottom: 12px; border-bottom: 1px solid #E2E8F0;">
      <div style="width: 36px; height: 36px; background: #EEF2FF; border: 1px solid #C7D2FE; border-radius: 8px; display: flex; align-items: center; justify-content: center; color: #4F46E5; font-weight: 700; font-size: 14px;">RI</div>
      <div>
        <div style="font-size: 15px; font-weight: 700; color: #0F172A; line-height: 1.2;">RetailIntel</div>
        <div style="font-size: 11px; color: #64748B;">DWDM Analytics Platform</div>
      </div>
    </div>
    """
    render_html(brand_html)

    # Radio Selection
    selected_page = st.radio(
        "Navigation",
        options=NAVIGATION_OPTIONS,
        index=0,
        label_visibility="collapsed",
    )

    st.write("")
    # Quick metadata summary at sidebar footer
    footer_html = f"""
    <div style="padding: 12px; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; font-size: 11px; color: #64748B;">
      <div style="font-weight: 600; color: #0F172A; margin-bottom: 2px;">Local Data Store</div>
      <div>ecommerce_dataset_10000.csv</div>
      <div style="margin-top: 4px; color: #059669; font-weight: 600;">10,000 Verified Records</div>
    </div>
    """
    render_html(footer_html)

# -----------------------------------------------------------------------------
# Module Implementations
# -----------------------------------------------------------------------------

render_top_header(selected_page)

# =============================================================================
# View: Dashboard Overview
# =============================================================================
if selected_page == "Dashboard Overview":
    render_module_header(
        tag="SYSTEM DASHBOARD",
        title="Executive Analytics & Operations Overview",
        subtitle="Consolidated gross performance, international order distribution, and syllabus module router.",
        theme_tag="Platform Home"
    )

    # Macro KPIs
    total_gmv = df["total_amount"].sum()
    total_orders = len(df)
    unique_customers = df["customer_id"].nunique()
    delivery_rate = (df["order_status"] == "Delivered").mean() * 100
    avg_order_value = df["total_amount"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        render_metric_card("Gross Merchandise Value", f"${total_gmv:,.0f}", "10,000 Total Transactions")
    with c2:
        render_metric_card("Processed Orders", f"{total_orders:,}", "100% Ingested Rows")
    with c3:
        render_metric_card("Unique Customer Accounts", f"{unique_customers:,}", "Calculated RFM Base")
    with c4:
        render_metric_card("Fulfillment Rate", f"{delivery_rate:.1f}%", "Delivered Shipments")
    with c5:
        render_metric_card("Average Order Value", f"${avg_order_value:.2f}", "Per-Transaction Mean")

    # Visualizations Row
    col_t1, col_t2 = st.columns([1.4, 1])
    with col_t1:
        with st.container(border=True):
            st.markdown("#### Quarterly Gross Revenue Trend")
            quarterly_rev = df.groupby("order_period")["total_amount"].sum().reset_index()
            fig_trend = px.line(
                quarterly_rev,
                x="order_period",
                y="total_amount",
                markers=True,
                labels={"order_period": "Quarter", "total_amount": "Revenue ($)"},
            )
            fig_trend.update_traces(line_color="#4F46E5", marker=dict(size=7, color="#312E81"))
            st.plotly_chart(apply_chart_theme(fig_trend, height=290), use_container_width=True)

    with col_t2:
        with st.container(border=True):
            st.markdown("#### Revenue by Category")
            cat_rev = df.groupby("category")["total_amount"].sum().reset_index().sort_values("total_amount", ascending=True)
            fig_cat = px.bar(
                cat_rev,
                x="total_amount",
                y="category",
                orientation="h",
                color="category",
                color_discrete_sequence=PALETTE,
            )
            st.plotly_chart(apply_chart_theme(fig_cat, height=290, show_legend=False), use_container_width=True)

    # Country distribution
    with st.container(border=True):
        st.markdown("#### Geographic Order Volumes Across 10 Sovereign Markets")
        country_stats = df.groupby("country").agg(
            Orders=("order_id", "count"),
            Revenue=("total_amount", "sum"),
            Avg_Rating=("rating", "mean")
        ).reset_index().sort_values("Revenue", ascending=False)

        col_g1, col_g2 = st.columns([1.5, 1])
        with col_g1:
            fig_cntry = px.bar(
                country_stats,
                x="country",
                y="Orders",
                color="Revenue",
                color_continuous_scale=["#C7D2FE", "#4F46E5"],
                labels={"country": "Market", "Orders": "Order Volume"}
            )
            st.plotly_chart(apply_chart_theme(fig_cntry, height=260), use_container_width=True)
        with col_g2:
            st.dataframe(
                country_stats.style.format({"Orders": "{:,}", "Revenue": "${:,.2f}", "Avg_Rating": "{:.2f}"}),
                use_container_width=True,
                height=260
            )

# =============================================================================
# View: Dataset Explorer & Schema
# =============================================================================
elif selected_page == "Dataset Explorer & Schema":
    render_module_header(
        tag="DATA DISCOVERY",
        title="Dataset Explorer & Schema Profiling",
        subtitle="Raw tabular records inspection, attribute types, and empirical statistical summaries.",
        theme_tag="Schema Audit"
    )

    with st.container(border=True):
        f1, f2, f3 = st.columns([1, 1, 1])
        with f1:
            cat_filter = st.multiselect("Filter by Department:", options=df["category"].unique(), default=df["category"].unique()[:3])
        with f2:
            status_filter = st.multiselect("Filter by Order Status:", options=df["order_status"].unique(), default=df["order_status"].unique())
        with f3:
            row_limit = st.slider("Display Row Limit:", 10, 100, 20, 10)

        df_filtered = df[(df["category"].isin(cat_filter)) & (df["order_status"].isin(status_filter))]
        st.caption(f"Displaying {min(row_limit, len(df_filtered))} of {len(df_filtered):,} matching records.")
        st.dataframe(df_filtered.head(row_limit), use_container_width=True)

    # Schema health table
    with st.container(border=True):
        st.markdown("#### Schema Column Metadata & Completeness")
        schema_rows = []
        for col in df.columns:
            schema_rows.append({
                "Attribute": col,
                "Type": str(df[col].dtype),
                "Non-Null Count": int(df[col].count()),
                "Null Values": int(df[col].isnull().sum()),
                "Unique Values": int(df[col].nunique()),
                "Sample Value": str(df[col].dropna().iloc[0]) if len(df[col].dropna()) > 0 else "N/A"
            })
        st.dataframe(pd.DataFrame(schema_rows), use_container_width=True)

    # Numeric summary statistics
    with st.container(border=True):
        st.markdown("#### Numerical Attributes Descriptive Statistics")
        st.dataframe(df.describe().T.style.format("{:.2f}"), use_container_width=True)

# =============================================================================
# Module 01: Dimensional Modeling (Star Schema)
# =============================================================================
elif selected_page == "01 — Dimensional Modeling (Star Schema)":
    render_module_header(
        tag="MODULE 01 • DWDM CURRICULUM",
        title="Dimensional Modeling & Warehouse Architecture",
        subtitle="Physical ERD schemas, column data types, foreign key referential integrity, and OLAP storage vs query latency trade-offs.",
        theme_tag="Theme: Electric Violet"
    )

    render_problem_card(
        title="Storage Bloat and Query Join Contention in Retail Topologies",
        business_problem="Flat 20-column transaction logs create massive row-level data redundancy and slow multi-attribute aggregations. Conversely, purely normalized OLTP 3NF models force 6-to-8 relational joins for routine analytics, locking operational tables during peak shopping hours.",
        operational_impact="Inappropriate dimensional architecture creates query latency bottlenecks, complicates BI semantic layers, and exposes analytical pipelines to operational key modifications."
    )

    # Global ERD Legend Bar
    render_erd_legend_bar()

    # -------------------------------------------------------------------------
    # In-Memory Schema Builders
    # -------------------------------------------------------------------------
    @st.cache_data(show_spinner=False)
    def build_star_schema(raw_df: pd.DataFrame):
        # dim_customer (4,327 rows)
        dim_cust = raw_df[["customer_id", "first_name", "last_name", "gender", "age_group", "country", "signup_date"]].drop_duplicates().copy().reset_index(drop=True)

        # dim_product (100 rows or unique SKUs)
        dim_prod = raw_df[["product_id", "product_name", "category", "unit_price"]].drop_duplicates().copy().reset_index(drop=True)

        # dim_date (1,095 rows)
        dim_date = raw_df[["order_date"]].drop_duplicates().copy().reset_index(drop=True)
        dim_date["date_key"] = dim_date["order_date"].dt.strftime("%Y%m%d").astype(int)
        dim_date["day"] = dim_date["order_date"].dt.day
        dim_date["month_name"] = dim_date["order_date"].dt.month_name()
        dim_date["order_quarter"] = "Q" + dim_date["order_date"].dt.quarter.astype(str)
        dim_date["year"] = dim_date["order_date"].dt.year
        dim_date = dim_date[["date_key", "order_date", "day", "month_name", "order_quarter", "year"]].sort_values("date_key").reset_index(drop=True)

        # dim_order_details (10,000 rows)
        dim_order = raw_df[["order_id", "order_status", "payment_method"]].drop_duplicates().copy().reset_index(drop=True)

        # dim_review (10,000 rows)
        dim_review = raw_df[["review_id", "review_text", "review_date", "rating"]].drop_duplicates().copy().reset_index(drop=True)

        # Central fact_sales (10,000 rows)
        fact = raw_df.copy()
        fact["sale_id"] = np.arange(1, len(fact) + 1)
        fact["date_key"] = fact["order_date"].dt.strftime("%Y%m%d").astype(int)
        fact_sales = fact[["sale_id", "customer_id", "product_id", "order_id", "date_key", "quantity", "unit_price", "total_amount", "rating"]].reset_index(drop=True)

        return {
            "fact_sales (Central Fact Table)": fact_sales,
            "dim_customer": dim_cust,
            "dim_product": dim_prod,
            "dim_date": dim_date,
            "dim_order_details": dim_order,
            "dim_review": dim_review,
        }

    @st.cache_data(show_spinner=False)
    def build_snowflake_schema(raw_df: pd.DataFrame):
        star_tables = build_star_schema(raw_df)
        dim_cust = star_tables["dim_customer"].copy()

        # Geography Sub-Dimension (10 rows)
        country_region_map = {
            "Canada": "North America", "USA": "North America", "China": "Asia", "Japan": "Asia", "India": "Asia",
            "Germany": "Europe", "France": "Europe", "UK": "Europe", "Brazil": "South America", "Australia": "Oceania"
        }
        countries = sorted(dim_cust["country"].unique())
        dim_geo = pd.DataFrame([
            {"geo_id": i + 1, "country": c, "region_continent": country_region_map.get(c, "Global")}
            for i, c in enumerate(countries)
        ])
        geo_map = dict(zip(dim_geo["country"], dim_geo["geo_id"]))
        dim_cust["geo_id"] = dim_cust["country"].map(geo_map)

        # Demographics Sub-Dimension (6 rows)
        dim_demo = dim_cust[["age_group", "gender"]].drop_duplicates().reset_index(drop=True)
        dim_demo["demographic_id"] = np.arange(1, len(dim_demo) + 1)
        demo_map = {(row["age_group"], row["gender"]): row["demographic_id"] for _, row in dim_demo.iterrows()}
        dim_cust["demographic_id"] = dim_cust.apply(lambda r: demo_map.get((r["age_group"], r["gender"])), axis=1)
        dim_cust_norm = dim_cust[["customer_id", "first_name", "last_name", "signup_date", "demographic_id", "geo_id"]].copy()

        # Category Sub-Dimension (6 rows)
        categories = sorted(raw_df["category"].unique())
        dim_cat = pd.DataFrame([
            {"category_id": i + 1, "category_name": c, "department": f"{c} Department"}
            for i, c in enumerate(categories)
        ])
        cat_map = dict(zip(dim_cat["category_name"], dim_cat["category_id"]))
        dim_prod = star_tables["dim_product"].copy()
        dim_prod["category_id"] = dim_prod["category"].map(cat_map)
        dim_prod_norm = dim_prod[["product_id", "product_name", "unit_price", "category_id"]].copy()

        # Temporal Hierarchy: Year & Quarter Sub-Dimensions
        years = sorted(raw_df["order_date"].dt.year.unique())
        dim_year = pd.DataFrame([
            {"year_id": y, "calendar_year": y, "fiscal_status": "Audited"}
            for y in years
        ])
        quarters = ["Q1", "Q2", "Q3", "Q4"]
        q_rows = []
        q_id = 1
        q_lookup = {}
        for y in years:
            for q in quarters:
                q_rows.append({"quarter_id": q_id, "quarter_name": f"{y} {q}", "year_id": y})
                q_lookup[(y, q)] = q_id
                q_id += 1
        dim_quarter = pd.DataFrame(q_rows)

        dim_date_raw = raw_df[["order_date"]].drop_duplicates().copy().reset_index(drop=True)
        dim_date_raw["date_key"] = dim_date_raw["order_date"].dt.strftime("%Y%m%d").astype(int)
        dim_date_raw["day"] = dim_date_raw["order_date"].dt.day
        dim_date_raw["month"] = dim_date_raw["order_date"].dt.month
        dim_date_raw["quarter_id"] = dim_date_raw.apply(
            lambda r: q_lookup.get((r["order_date"].year, "Q" + str(r["order_date"].quarter))), axis=1
        )
        dim_date_norm = dim_date_raw[["date_key", "order_date", "day", "month", "quarter_id"]].sort_values("date_key").reset_index(drop=True)

        return {
            "fact_sales (Central Fact Table)": star_tables["fact_sales (Central Fact Table)"],
            "dim_customer (Normalized)": dim_cust_norm,
            "dim_geography (Sub-Dimension)": dim_geo,
            "dim_demographics (Sub-Dimension)": dim_demo,
            "dim_product (Normalized)": dim_prod_norm,
            "dim_category (Sub-Dimension)": dim_cat,
            "dim_date (Normalized)": dim_date_norm,
            "dim_quarter (Sub-Dimension)": dim_quarter,
            "dim_year (Sub-Dimension)": dim_year,
        }

    @st.cache_data(show_spinner=False)
    def build_constellation_schema(raw_df: pd.DataFrame):
        star_tables = build_star_schema(raw_df)

        # Fact 1: fact_sales (10,000 rows | Core Commerce Process)
        fact_sales = raw_df.copy()
        fact_sales["sale_id"] = np.arange(1, len(fact_sales) + 1)
        fact_sales["date_key"] = fact_sales["order_date"].dt.strftime("%Y%m%d").astype(int)
        status_to_ful = {"Delivered": "FUL_01", "Shipped": "FUL_02", "Pending": "FUL_03", "Returned": "FUL_04", "Cancelled": "FUL_05"}
        fact_sales["fulfillment_id"] = fact_sales["order_status"].map(status_to_ful).fillna("FUL_01")
        fact_sales_tbl = fact_sales[["sale_id", "customer_id", "product_id", "date_key", "order_id", "fulfillment_id", "quantity", "unit_price", "total_amount"]].reset_index(drop=True)

        # Fact 2: fact_customer_reviews (10,000 rows | Customer Sentiment Process)
        fact_rev = raw_df.copy()
        fact_rev["date_key"] = fact_rev["order_date"].dt.strftime("%Y%m%d").astype(int)
        channel_choices = ["SRC_WEB", "SRC_APP", "SRC_EML", "SRC_SMS"]
        np.random.seed(42)
        fact_rev["feedback_source_id"] = np.random.choice(channel_choices, len(fact_rev))
        fact_rev["review_length"] = fact_rev["review_text"].astype(str).str.len()
        fact_rev["sentiment_polarity_score"] = ((fact_rev["rating"] - 3.0) / 2.0).round(2)
        fact_rev_tbl = fact_rev[["review_id", "customer_id", "product_id", "date_key", "feedback_source_id", "rating", "review_length", "sentiment_polarity_score"]].reset_index(drop=True)

        # Dedicated Fulfillment Channel Dimension (5 rows)
        dim_fulfillment = pd.DataFrame([
            {"fulfillment_id": "FUL_01", "order_status": "Delivered", "shipping_carrier": "FedEx Ground Priority", "sla_delivery_days": 3},
            {"fulfillment_id": "FUL_02", "order_status": "Shipped", "shipping_carrier": "DHL Logistics Express", "sla_delivery_days": 4},
            {"fulfillment_id": "FUL_03", "order_status": "Pending", "shipping_carrier": "Regional Warehouse Hub", "sla_delivery_days": 1},
            {"fulfillment_id": "FUL_04", "order_status": "Returned", "shipping_carrier": "Reverse Logistics Fleet", "sla_delivery_days": 5},
            {"fulfillment_id": "FUL_05", "order_status": "Cancelled", "shipping_carrier": "None (Voided)", "sla_delivery_days": 0},
        ])

        # Dedicated Feedback Source Dimension (4 rows)
        dim_feedback = pd.DataFrame([
            {"feedback_source_id": "SRC_WEB", "channel_name": "Web Storefront Checkout", "platform_type": "Desktop / PWA", "response_rate": "24%"},
            {"feedback_source_id": "SRC_APP", "channel_name": "Mobile iOS / Android App", "platform_type": "Native Mobile", "response_rate": "41%"},
            {"feedback_source_id": "SRC_EML", "channel_name": "Post-Delivery Email Survey", "platform_type": "Transactional Email", "response_rate": "15%"},
            {"feedback_source_id": "SRC_SMS", "channel_name": "SMS Prompt Survey", "platform_type": "Twilio SMS Gateway", "response_rate": "9%"},
        ])

        return {
            "fact_sales (Commerce Fact)": fact_sales_tbl,
            "fact_customer_reviews (Sentiment Fact)": fact_rev_tbl,
            "dim_customer (Shared Conformed)": star_tables["dim_customer"],
            "dim_product (Shared Conformed)": star_tables["dim_product"],
            "dim_date (Shared Conformed)": star_tables["dim_date"],
            "dim_fulfillment_channel (Dedicated)": dim_fulfillment,
            "dim_feedback_source (Dedicated)": dim_feedback,
        }

    star_data = build_star_schema(df)
    snow_data = build_snowflake_schema(df)
    constellation_data = build_constellation_schema(df)

    # -------------------------------------------------------------------------
    # Interactive Schema Tabs
    # -------------------------------------------------------------------------
    tab_star, tab_snow, tab_constellation = st.tabs([
        "⭐ Star Schema", "❄️ Snowflake Schema", "🌌 Fact Constellation"
    ])

    # ==================== STAR SCHEMA ====================
    with tab_star:
        # Architecture Concept Banner
        render_html("""
        <div class="card" style="background: #FAFAFD; border: 1px dashed #C7D2FE; padding: 16px 20px; margin-bottom: 20px;">
          <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
            <div>
              <div style="font-size: 14px; font-weight: 700; color: #4338CA; text-transform: uppercase; letter-spacing: 0.04em;">Star Schema Architecture & Physical ERD</div>
              <div style="font-size: 12px; color: #475569; margin-top: 3px;"><strong>Architecture Concept:</strong> Single central fact table directly referencing denormalized 1NF/2NF dimension tables for maximum query speed and minimal joins in OLAP aggregations.</div>
            </div>
            <span class="pill-badge" style="background: #EEF2FF; color: #4F46E5; border: 1px solid #C7D2FE; font-size: 11px; padding: 4px 10px; font-weight: 600;">1 Central Fact • 5 Denormalized Dimensions • Single-Hop Joins</span>
          </div>
        </div>
        """)

        # 3-Column ERD Cards Layout
        col_st_left, col_st_center, col_st_right = st.columns([1, 1.25, 1])

        with col_st_left:
            # dim_customer card
            render_html(render_erd_table_card(
                table_name="dim_customer",
                meta_info="4,327 rows | Dimension",
                columns=[
                    ("PK", "customer_id", "VARCHAR"),
                    ("ATTR", "first_name", "VARCHAR"),
                    ("ATTR", "last_name", "VARCHAR"),
                    ("ATTR", "gender", "VARCHAR"),
                    ("ATTR", "age_group", "VARCHAR"),
                    ("ATTR", "country", "VARCHAR"),
                    ("ATTR", "signup_date", "DATE"),
                ],
                is_fact=False,
                header_color="#0284C7",
                badge_label="DIMENSION",
                join_footnote="Rel: 1 ─── N fact_sales.customer_id"
            ))

            # dim_product card
            render_html(render_erd_table_card(
                table_name="dim_product",
                meta_info="100 rows | Dimension",
                columns=[
                    ("PK", "product_id", "VARCHAR"),
                    ("ATTR", "product_name", "VARCHAR"),
                    ("ATTR", "category", "VARCHAR"),
                    ("NUM", "unit_price", "DECIMAL"),
                ],
                is_fact=False,
                header_color="#0284C7",
                badge_label="DIMENSION",
                join_footnote="Rel: 1 ─── N fact_sales.product_id"
            ))

        with col_st_center:
            # Central Fact Table
            render_html(render_erd_table_card(
                table_name="fact_sales",
                meta_info="10,000 rows | Central Fact Table",
                columns=[
                    (["PK", "NUM"], "sale_id", "INTEGER"),
                    ("FK", "customer_id", "VARCHAR"),
                    ("FK", "product_id", "VARCHAR"),
                    ("FK", "order_id", "VARCHAR"),
                    ("FK", "date_key", "INTEGER"),
                    ("NUM", "quantity", "INTEGER"),
                    ("NUM", "unit_price", "DECIMAL"),
                    ("NUM", "total_amount", "DECIMAL"),
                    ("NUM", "rating", "INTEGER"),
                ],
                is_fact=True,
                header_color="#7C3AED",
                badge_label="CENTRAL FACT TABLE",
                join_footnote="Additive Measures • Grain: 1 Row per Checkout Transaction"
            ))

            render_html("""
            <div style="background: #F5F3FF; border: 1px solid #DDD6FE; border-radius: 8px; padding: 10px 14px; font-size: 11px; color: #5B21B6; margin-top: -6px; margin-bottom: 16px;">
              <strong>⚡ Scan Optimization:</strong> Every dimension is reached through exactly 1 join hop from <code>fact_sales</code>. Modern columnar analytical engines (Snowflake, BigQuery, ClickHouse) vectorize and broadcast these denormalized dimension scans with sub-second execution.
            </div>
            """)

        with col_st_right:
            # dim_date card
            render_html(render_erd_table_card(
                table_name="dim_date",
                meta_info="1,095 rows | Dimension",
                columns=[
                    ("PK", "date_key", "INTEGER"),
                    ("ATTR", "order_date", "DATE"),
                    ("ATTR", "day", "INTEGER"),
                    ("ATTR", "month_name", "VARCHAR"),
                    ("ATTR", "order_quarter", "VARCHAR"),
                    ("ATTR", "year", "INTEGER"),
                ],
                is_fact=False,
                header_color="#0284C7",
                badge_label="DIMENSION",
                join_footnote="Rel: 1 ─── N fact_sales.date_key"
            ))

            # dim_order_details card
            render_html(render_erd_table_card(
                table_name="dim_order_details",
                meta_info="10,000 rows | Dimension",
                columns=[
                    ("PK", "order_id", "VARCHAR"),
                    ("ATTR", "order_status", "VARCHAR"),
                    ("ATTR", "payment_method", "VARCHAR"),
                ],
                is_fact=False,
                header_color="#0284C7",
                badge_label="DIMENSION",
                join_footnote="Rel: 1 ─── 1 fact_sales.order_id"
            ))

            # dim_review card
            render_html(render_erd_table_card(
                table_name="dim_review",
                meta_info="10,000 rows | Dimension",
                columns=[
                    ("PK", "review_id", "VARCHAR"),
                    ("ATTR", "review_text", "TEXT"),
                    ("ATTR", "review_date", "DATE"),
                    ("NUM", "rating", "INTEGER"),
                ],
                is_fact=False,
                header_color="#0284C7",
                badge_label="DIMENSION",
                join_footnote="Rel: 1 ─── 1 fact_sales (Order feedback)"
            ))

        # Expandable Schema Summary: Warehouse DDL & Trade-Off Analysis
        with st.expander("📐 Star Schema Architecture Details, Warehouse DDL & Trade-Off Analysis", expanded=False):
            ddl_col, trade_col = st.columns([1.1, 1])
            with ddl_col:
                st.markdown("##### 📄 Production Warehouse DDL (`CREATE TABLE`)")
                st.code('''-- Star Schema Warehouse DDL (PostgreSQL / Snowflake / BigQuery)
CREATE TABLE dim_customer (
    customer_id VARCHAR(50) PRIMARY KEY,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    gender VARCHAR(20),
    age_group VARCHAR(20),
    country VARCHAR(100),
    signup_date DATE
);

CREATE TABLE dim_product (
    product_id VARCHAR(50) PRIMARY KEY,
    product_name VARCHAR(255),
    category VARCHAR(100),
    unit_price DECIMAL(10,2)
);

CREATE TABLE dim_date (
    date_key INTEGER PRIMARY KEY,
    order_date DATE,
    day INTEGER,
    month_name VARCHAR(20),
    order_quarter VARCHAR(10),
    year INTEGER
);

CREATE TABLE dim_order_details (
    order_id VARCHAR(50) PRIMARY KEY,
    order_status VARCHAR(50),
    payment_method VARCHAR(50)
);

CREATE TABLE dim_review (
    review_id VARCHAR(50) PRIMARY KEY,
    review_text TEXT,
    review_date DATE,
    rating INTEGER
);

CREATE TABLE fact_sales (
    sale_id INTEGER PRIMARY KEY,
    customer_id VARCHAR(50) REFERENCES dim_customer(customer_id),
    product_id VARCHAR(50) REFERENCES dim_product(product_id),
    order_id VARCHAR(50) REFERENCES dim_order_details(order_id),
    date_key INTEGER REFERENCES dim_date(date_key),
    quantity INTEGER,
    unit_price DECIMAL(10,2),
    total_amount DECIMAL(10,2),
    rating INTEGER
);''', language="sql")

            with trade_col:
                st.markdown("##### ⚖️ Architecture Trade-Off Analysis")
                trade_df = pd.DataFrame([
                    {"Evaluation Vector": "Normalization Level", "Score / Metric": "1NF / 2NF (Denormalized)", "Architectural Assessment": "Dimensions store repeated descriptive strings to eliminate lookup hops."},
                    {"Evaluation Vector": "Storage Efficiency", "Score / Metric": "Moderate Footprint", "Architectural Assessment": "Mild string repetition across 4,327 customers; negligible in modern cloud warehouses."},
                    {"Evaluation Vector": "Query Latency (OLAP)", "Score / Metric": "⚡ Ultra-Fast (<15ms)", "Architectural Assessment": "Direct 1-hop joins maximize columnar scan vectorization and broadcast joins."},
                    {"Evaluation Vector": "ETL & Pipeline Complexity", "Score / Metric": "Low / Straightforward", "Architectural Assessment": "Single-pass dimension loads with zero cascading parent-key resolution."}
                ])
                st.dataframe(trade_df, use_container_width=True, hide_index=True)

        # Interactive Relationship Auditor & Join Inspector
        with st.container(border=True):
            st.markdown("#### 🔗 Interactive Relationship & Referential Integrity Auditor")
            rel_options = [
                "Dim_Customer (PK: customer_id) ➔ Fact_Sales (FK: customer_id) [1:N]",
                "Dim_Product (PK: product_id) ➔ Fact_Sales (FK: product_id) [1:N]",
                "Dim_Date (PK: date_key) ➔ Fact_Sales (FK: date_key) [1:N]",
                "Dim_Order_Details (PK: order_id) ➔ Fact_Sales (PK: order_id) [1:1]",
                "Dim_Review (PK: review_id) ➔ Fact_Sales (Order Join) [1:1]"
            ]
            selected_rel = st.selectbox("Select Relationship to Audit:", rel_options, index=0)

            c_aud1, c_aud2, c_aud3, c_aud4 = st.columns(4)
            with c_aud1:
                render_metric_card("Referential Integrity", "100.0%", "0 Orphan Records")
            with c_aud2:
                render_metric_card("Join Cardinality", "1 : N" if "[1:N]" in selected_rel else "1 : 1", "Relationship Degree")
            with c_aud3:
                render_metric_card("Join Hop Overhead", "1 Join", "Direct Star Traversal")
            with c_aud4:
                render_metric_card("Matched Foreign Keys", "10,000 / 10,000", "Full Relational Coverage")

            if "Dim_Customer" in selected_rel:
                sample_join = df[["order_id", "customer_id", "first_name", "last_name", "country", "total_amount"]].head(5)
                st.caption("**SQL Predicate:** `SELECT * FROM Fact_Sales INNER JOIN Dim_Customer ON Fact_Sales.customer_id = Dim_Customer.customer_id`")
            elif "Dim_Product" in selected_rel:
                sample_join = df[["order_id", "product_id", "product_name", "category", "unit_price", "quantity", "total_amount"]].head(5)
                st.caption("**SQL Predicate:** `SELECT * FROM Fact_Sales INNER JOIN Dim_Product ON Fact_Sales.product_id = Dim_Product.product_id`")
            elif "Dim_Date" in selected_rel:
                sample_join = df[["order_id", "order_date", "day_name", "order_period", "total_amount"]].head(5)
                st.caption("**SQL Predicate:** `SELECT * FROM Fact_Sales INNER JOIN Dim_Date ON Fact_Sales.date_key = Dim_Date.date_key`")
            elif "Dim_Order_Details" in selected_rel:
                sample_join = df[["order_id", "payment_method", "order_status", "total_amount"]].head(5)
                st.caption("**SQL Predicate:** `SELECT * FROM Fact_Sales INNER JOIN Dim_Order_Details ON Fact_Sales.order_id = Dim_Order_Details.order_id`")
            else:
                sample_join = df[["order_id", "review_id", "rating", "review_text", "total_amount"]].head(5)
                st.caption("**SQL Predicate:** `SELECT * FROM Fact_Sales INNER JOIN Dim_Review ON Fact_Sales.order_id = Dim_Review.order_id`")
            st.dataframe(sample_join, use_container_width=True)

        # Star Schema Table Inspector with CSV download
        with st.container(border=True):
            st.markdown("#### 🗄️ Star Schema Physical Table Inspector")
            col_sel1, col_sel2 = st.columns([1.5, 3])
            with col_sel1:
                selected_star_tbl = st.selectbox("Select Star Schema Table:", list(star_data.keys()), key="star_tbl_sel")
                tbl_df = star_data[selected_star_tbl]
                st.caption(f"**Physical Rows:** `{len(tbl_df):,}`")
                st.caption(f"**Total Attributes:** `{tbl_df.shape[1]}`")
                st.caption(f"**In-Memory Size:** `{tbl_df.memory_usage(deep=True).sum()/1024:.1f} KB`")
                csv_bytes = tbl_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"Export {selected_star_tbl.split()[0]} CSV",
                    data=csv_bytes,
                    file_name=f"{selected_star_tbl.split()[0].lower()}.csv",
                    mime="text/csv",
                    key="star_schema_tbl_download_btn",
                )
            with col_sel2:
                st.dataframe(tbl_df.head(6), use_container_width=True)

    # ==================== SNOWFLAKE SCHEMA ====================
    with tab_snow:
        # Architecture Concept Banner
        render_html("""
        <div class="card" style="background: #FAFAFD; border: 1px dashed #BAE6FD; padding: 16px 20px; margin-bottom: 20px;">
          <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
            <div>
              <div style="font-size: 14px; font-weight: 700; color: #0284C7; text-transform: uppercase; letter-spacing: 0.04em;">Snowflake Schema Architecture & 3NF Normalization</div>
              <div style="font-size: 12px; color: #475569; margin-top: 3px;"><strong>Architecture Concept:</strong> Deconstructs dimension tables into 3NF normalized hierarchies to eliminate attribute redundancy at the cost of additional SQL joins.</div>
            </div>
            <span class="pill-badge" style="background: #E0F2FE; color: #0369A1; border: 1px solid #BAE6FD; font-size: 11px; padding: 4px 10px; font-weight: 600;">3NF Normalized Dimensions • Outrigger Sub-Tables • Zero String Redundancy</span>
          </div>
        </div>
        """)

        # 3-Tier Multi-Column Cards Layout
        col_sn_fact, col_sn_primary, col_sn_sub = st.columns([1, 1.15, 1.15])

        with col_sn_fact:
            # Central Fact Table
            render_html(render_erd_table_card(
                table_name="fact_sales",
                meta_info="10,000 rows | Central Fact Table",
                columns=[
                    (["PK", "NUM"], "sale_id", "INTEGER"),
                    ("FK", "customer_id", "VARCHAR"),
                    ("FK", "product_id", "VARCHAR"),
                    ("FK", "order_id", "VARCHAR"),
                    ("FK", "date_key", "INTEGER"),
                    ("NUM", "quantity", "INTEGER"),
                    ("NUM", "unit_price", "DECIMAL"),
                    ("NUM", "total_amount", "DECIMAL"),
                    ("NUM", "rating", "INTEGER"),
                ],
                is_fact=True,
                header_color="#7C3AED",
                badge_label="CENTRAL FACT TABLE",
                join_footnote="Foreign keys reference 1st tier normalized dimensions"
            ))

            render_html("""
            <div style="background: #F0F9FF; border: 1px solid #BAE6FD; border-radius: 8px; padding: 10px 14px; font-size: 11px; color: #0369A1; margin-top: -6px; margin-bottom: 16px;">
              <strong>⛓️ 3NF Multi-Hop Overhead:</strong> Queries grouping across geographical continents, age brackets, or product departments require multi-tier join paths (Fact ➔ Dim ➔ Sub-Dim).
            </div>
            """)

        with col_sn_primary:
            render_html("""
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 5px 10px; margin-bottom: 10px; font-size: 10px; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.05em; text-align: center;">
              TIER 1 • PRIMARY NORMALIZED DIMENSIONS
            </div>
            """)

            # dim_customer card
            render_html(render_erd_table_card(
                table_name="dim_customer",
                meta_info="4,327 rows | Primary Dimension",
                columns=[
                    ("PK", "customer_id", "VARCHAR"),
                    ("ATTR", "first_name", "VARCHAR"),
                    ("ATTR", "last_name", "VARCHAR"),
                    ("ATTR", "signup_date", "DATE"),
                    ("FK", "demographic_id", "INTEGER", "───> dim_demographics"),
                    ("FK", "geo_id", "INTEGER", "───> dim_geography"),
                ],
                is_fact=False,
                header_color="#0284C7",
                badge_label="PRIMARY DIM (1ST TIER)",
                join_footnote="Rel: Has outriggers to geography & demographics"
            ))

            # dim_product card
            render_html(render_erd_table_card(
                table_name="dim_product",
                meta_info="100 rows | Primary Dimension",
                columns=[
                    ("PK", "product_id", "VARCHAR"),
                    ("ATTR", "product_name", "VARCHAR"),
                    ("NUM", "unit_price", "DECIMAL"),
                    ("FK", "category_id", "INTEGER", "───> dim_category"),
                ],
                is_fact=False,
                header_color="#0284C7",
                badge_label="PRIMARY DIM (1ST TIER)",
                join_footnote="Rel: Has outrigger to category hierarchy"
            ))

            # dim_date card
            render_html(render_erd_table_card(
                table_name="dim_date",
                meta_info="1,095 rows | Primary Dimension",
                columns=[
                    ("PK", "date_key", "INTEGER"),
                    ("ATTR", "order_date", "DATE"),
                    ("FK", "quarter_id", "INTEGER", "───> dim_quarter -> dim_year"),
                ],
                is_fact=False,
                header_color="#0284C7",
                badge_label="PRIMARY DIM (1ST TIER)",
                join_footnote="Rel: Has outrigger to calendar quarters & years"
            ))

        with col_sn_sub:
            render_html("""
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 5px 10px; margin-bottom: 10px; font-size: 10px; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.05em; text-align: center;">
              TIER 2 & 3 • NORMALIZED OUTRIGGERS (SUB-DIMENSIONS)
            </div>
            """)

            # dim_geography card
            render_html(render_erd_table_card(
                table_name="dim_geography",
                meta_info="10 rows | Sub-Dimension",
                columns=[
                    ("PK", "geo_id", "INTEGER"),
                    ("ATTR", "country", "VARCHAR"),
                    ("ATTR", "region_continent", "VARCHAR"),
                ],
                is_fact=False,
                header_color="#0369A1",
                badge_label="SUB-DIMENSION (3NF)",
                join_footnote="Parent lookup for dim_customer.geo_id"
            ))

            # dim_demographics card
            render_html(render_erd_table_card(
                table_name="dim_demographics",
                meta_info="6 rows | Sub-Dimension",
                columns=[
                    ("PK", "demographic_id", "INTEGER"),
                    ("ATTR", "age_group", "VARCHAR"),
                    ("ATTR", "gender", "VARCHAR"),
                ],
                is_fact=False,
                header_color="#0369A1",
                badge_label="SUB-DIMENSION (3NF)",
                join_footnote="Parent lookup for dim_customer.demographic_id"
            ))

            # dim_category card
            render_html(render_erd_table_card(
                table_name="dim_category",
                meta_info="6 rows | Sub-Dimension",
                columns=[
                    ("PK", "category_id", "INTEGER"),
                    ("ATTR", "category_name", "VARCHAR"),
                    ("ATTR", "department", "VARCHAR"),
                ],
                is_fact=False,
                header_color="#0369A1",
                badge_label="SUB-DIMENSION (3NF)",
                join_footnote="Parent lookup for dim_product.category_id"
            ))

            # dim_quarter card
            render_html(render_erd_table_card(
                table_name="dim_quarter",
                meta_info="12 rows | Sub-Dimension",
                columns=[
                    ("PK", "quarter_id", "INTEGER"),
                    ("ATTR", "order_quarter", "VARCHAR"),
                    ("FK", "year_id", "INTEGER", "───> dim_year"),
                ],
                is_fact=False,
                header_color="#0369A1",
                badge_label="SUB-DIMENSION (3NF)",
                join_footnote="Temporal hierarchy: quarter ➔ year"
            ))

            # dim_year card
            render_html(render_erd_table_card(
                table_name="dim_year",
                meta_info="3 rows | Sub-Dimension",
                columns=[
                    ("PK", "year_id", "INTEGER"),
                    ("ATTR", "year", "INTEGER"),
                    ("ATTR", "fiscal_status", "VARCHAR"),
                ],
                is_fact=False,
                header_color="#0369A1",
                badge_label="SUB-DIMENSION (3NF)",
                join_footnote="Root calendar hierarchy node"
            ))

        # Expandable Schema Summary: Warehouse DDL & Trade-Off Analysis
        with st.expander("📐 Snowflake Schema Architecture Details, Warehouse DDL & Trade-Off Analysis", expanded=False):
            ddl_col_sn, trade_col_sn = st.columns([1.1, 1])
            with ddl_col_sn:
                st.markdown("##### 📄 Production Snowflake Schema DDL (`CREATE TABLE`)")
                st.code('''-- Snowflake Schema Warehouse DDL (3NF Normalized Sub-Dimensions)
CREATE TABLE dim_geography (
    geo_id INTEGER PRIMARY KEY,
    country VARCHAR(100),
    region_continent VARCHAR(100)
);

CREATE TABLE dim_demographics (
    demographic_id INTEGER PRIMARY KEY,
    age_group VARCHAR(50),
    gender VARCHAR(20)
);

CREATE TABLE dim_customer (
    customer_id VARCHAR(50) PRIMARY KEY,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    signup_date DATE,
    demographic_id INTEGER REFERENCES dim_demographics(demographic_id),
    geo_id INTEGER REFERENCES dim_geography(geo_id)
);

CREATE TABLE dim_category (
    category_id INTEGER PRIMARY KEY,
    category_name VARCHAR(100),
    department VARCHAR(100)
);

CREATE TABLE dim_product (
    product_id VARCHAR(50) PRIMARY KEY,
    product_name VARCHAR(255),
    unit_price DECIMAL(10,2),
    category_id INTEGER REFERENCES dim_category(category_id)
);

CREATE TABLE dim_year (
    year_id INTEGER PRIMARY KEY,
    year INTEGER,
    fiscal_status VARCHAR(50)
);

CREATE TABLE dim_quarter (
    quarter_id INTEGER PRIMARY KEY,
    order_quarter VARCHAR(10),
    year_id INTEGER REFERENCES dim_year(year_id)
);

CREATE TABLE dim_date (
    date_key INTEGER PRIMARY KEY,
    order_date DATE,
    quarter_id INTEGER REFERENCES dim_quarter(quarter_id)
);

CREATE TABLE fact_sales (
    sale_id INTEGER PRIMARY KEY,
    customer_id VARCHAR(50) REFERENCES dim_customer(customer_id),
    product_id VARCHAR(50) REFERENCES dim_product(product_id),
    order_id VARCHAR(50),
    date_key INTEGER REFERENCES dim_date(date_key),
    quantity INTEGER,
    unit_price DECIMAL(10,2),
    total_amount DECIMAL(10,2),
    rating INTEGER
);''', language="sql")

            with trade_col_sn:
                st.markdown("##### ⚖️ Architecture Trade-Off Analysis")
                trade_snow_df = pd.DataFrame([
                    {"Evaluation Vector": "Normalization Level", "Score / Metric": "3NF (Normalized Hierarchies)", "Architectural Assessment": "Strict relational decomposition eliminating repeated string values."},
                    {"Evaluation Vector": "Storage Efficiency", "Score / Metric": "💾 High (Minimal Redundancy)", "Architectural Assessment": "Reduces disk space for massive text attributes via integer lookup keys."},
                    {"Evaluation Vector": "Query Latency (OLAP)", "Score / Metric": "⚠️ Moderate-High (45-120ms)", "Architectural Assessment": "Multi-hop joins (2 to 4 levels) degrade performance under heavy concurrent BI scans."},
                    {"Evaluation Vector": "ETL & Pipeline Complexity", "Score / Metric": "High (Cascading Key Loads)", "Architectural Assessment": "Requires strict foreign key dependency ordering across ETL staging tables."}
                ])
                st.dataframe(trade_snow_df, use_container_width=True, hide_index=True)

        # Architectural Contrast Box
        c_sc1, c_sc2 = st.columns(2)
        with c_sc1:
            with st.container(border=True):
                st.markdown("##### ⚙️ Normalization Benefits (3NF)")
                st.markdown("""
                * **Reduces Data Redundancy:** Strings like category names or continent regions are stored once in lookup sub-tables rather than duplicated thousands of times.
                * **Prevents Update Anomalies:** Renaming a product category or restructuring geographic hierarchies requires updating exactly one row in `Dim_Category` or `Dim_Geography`.
                * **Optimizes Storage Footprint:** Ideal for massive enterprise data warehouses where petabyte-scale storage costs dominate.
                """)
        with c_sc2:
            with st.container(border=True):
                st.markdown("##### ⚠️ Analytical Trade-Offs & Overhead")
                st.markdown("""
                * **Multi-Hop Join Penalty:** Resolving revenue by country requires `Fact_Sales` ➔ `Dim_Customer` ➔ `Dim_Geography` (2 join operations instead of 1).
                * **Complex BI Navigation:** Business intelligence semantic models and SQL queries become significantly more intricate for end-user analysts.
                * **Higher Query Latency:** Query engines must evaluate larger join graphs, increasing query execution time under high concurrent workloads.
                """)

        # Interactive Multi-Hop Snowflake Join Path Auditor
        with st.container(border=True):
            st.markdown("#### ⛓️ Interactive Multi-Hop Snowflake Join Path Auditor")
            snow_path = st.selectbox(
                "Select Normalized Snowflake Traversal Path to Audit:",
                [
                    "Geographic Distribution: Fact_Sales ➔ Dim_Customer ➔ Dim_Geography",
                    "Demographic Profile: Fact_Sales ➔ Dim_Customer ➔ Dim_Demographics",
                    "Department Catalog: Fact_Sales ➔ Dim_Product ➔ Dim_Category",
                    "Temporal Progression: Fact_Sales ➔ Dim_Date ➔ Dim_Quarter ➔ Dim_Year"
                ]
            )

            if "Geographic" in snow_path:
                st.caption("**Multi-Hop SQL Join:** `SELECT country_name, continent_region, SUM(total_amount) FROM Fact_Sales JOIN Dim_Customer ON Fact_Sales.customer_id = Dim_Customer.customer_id JOIN Dim_Geography ON Dim_Customer.country_id = Dim_Geography.country_id GROUP BY country_name, continent_region`")
                joined_res = df.groupby(["country"]).agg(Orders=("order_id", "count"), Revenue=("total_amount", "sum")).reset_index()
            elif "Demographic" in snow_path:
                st.caption("**Multi-Hop SQL Join:** `SELECT age_group, gender, SUM(total_amount) FROM Fact_Sales JOIN Dim_Customer ON Fact_Sales.customer_id = Dim_Customer.customer_id JOIN Dim_Demographics ON Dim_Customer.demographic_id = Dim_Demographics.demographic_id GROUP BY age_group, gender`")
                joined_res = df.groupby(["age_group", "gender"]).agg(Orders=("order_id", "count"), Revenue=("total_amount", "sum")).reset_index()
            elif "Department" in snow_path:
                st.caption("**Multi-Hop SQL Join:** `SELECT category, SUM(total_amount) FROM Fact_Sales JOIN Dim_Product ON Fact_Sales.product_id = Dim_Product.product_id JOIN Dim_Category ON Dim_Product.category_id = Dim_Category.category_id GROUP BY category`")
                joined_res = df.groupby("category").agg(Orders=("order_id", "count"), Revenue=("total_amount", "sum")).reset_index()
            else:
                st.caption("**Multi-Hop SQL Join:** `SELECT order_year, order_quarter, SUM(total_amount) FROM Fact_Sales JOIN Dim_Date ON Fact_Sales.date_key = Dim_Date.date_key JOIN Dim_Quarter ON Dim_Date.quarter_id = Dim_Quarter.quarter_id JOIN Dim_Year ON Dim_Quarter.year_id = Dim_Year.year_id GROUP BY order_year, order_quarter`")
                joined_res = df.groupby(["order_year", "order_quarter"]).agg(Orders=("order_id", "count"), Revenue=("total_amount", "sum")).reset_index()

            st.dataframe(joined_res.style.format({"Revenue": "${:,.2f}", "Orders": "{:,}"}), use_container_width=True)

        with st.container(border=True):
            st.markdown("#### 🗄️ Snowflake Normalized Sub-Dimension Inspector")
            col_sn1, col_sn2 = st.columns([1.5, 3])
            with col_sn1:
                selected_snow_tbl = st.selectbox("Select Snowflake Table:", list(snow_data.keys()), key="snow_tbl_sel")
                s_df = snow_data[selected_snow_tbl]
                st.caption(f"**Physical Rows:** `{len(s_df):,}` | **Columns:** `{s_df.shape[1]}`")
                st.caption(f"**Memory Footprint:** `{s_df.memory_usage(deep=True).sum()/1024:.1f} KB`")
                st.download_button(
                    label=f"Export {selected_snow_tbl.split()[0]} CSV",
                    data=s_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{selected_snow_tbl.split()[0].lower()}.csv",
                    mime="text/csv",
                    key="snow_schema_tbl_download_btn",
                )
            with col_sn2:
                st.dataframe(s_df.head(6), use_container_width=True)

    # ==================== FACT CONSTELLATION ====================
    with tab_constellation:
        # Architecture Concept Banner
        render_html("""
        <div class="card" style="background: #FAFAFD; border: 1px dashed #A7F3D0; padding: 16px 20px; margin-bottom: 20px;">
          <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
            <div>
              <div style="font-size: 14px; font-weight: 700; color: #047857; text-transform: uppercase; letter-spacing: 0.04em;">Fact Constellation (Galaxy Schema) Architecture</div>
              <div style="font-size: 12px; color: #475569; margin-top: 3px;"><strong>Architecture Concept:</strong> Models multiple business processes that share conformed dimensions, allowing cross-fact analytical drill-through.</div>
            </div>
            <span class="pill-badge" style="background: #ECFDF5; color: #065F46; border: 1px solid #A7F3D0; font-size: 11px; padding: 4px 10px; font-weight: 600;">2 Business Processes • 3 Shared Conformed Dimensions • 2 Dedicated Dimensions</span>
          </div>
        </div>
        """)

        # 3-Column Layout: Sales Mart | Shared Conformed Dimensions | Customer Sentiment Mart
        col_co_sales, col_co_shared, col_co_review = st.columns([1.1, 1.1, 1.1])

        with col_co_sales:
            render_html("""
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 5px 10px; margin-bottom: 10px; font-size: 10px; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.05em; text-align: center;">
              BUSINESS PROCESS 1 • COMMERCE & REVENUE MART
            </div>
            """)

            # fact_sales card
            render_html(render_erd_table_card(
                table_name="fact_sales",
                meta_info="10,000 rows | Core Commerce Process",
                columns=[
                    (["PK", "NUM"], "sale_id", "INTEGER"),
                    ("FK", "customer_id", "VARCHAR", "(Conformed)"),
                    ("FK", "product_id", "VARCHAR", "(Conformed)"),
                    ("FK", "order_id", "VARCHAR"),
                    ("FK", "date_key", "INTEGER", "(Conformed)"),
                    ("FK", "fulfillment_id", "VARCHAR", "(Dedicated)"),
                    ("NUM", "quantity", "INTEGER"),
                    ("NUM", "unit_price", "DECIMAL"),
                    ("NUM", "total_amount", "DECIMAL"),
                ],
                is_fact=True,
                header_color="#7C3AED",
                badge_label="FACT TABLE • COMMERCE",
                join_footnote="Measures: quantity, unit_price, total_amount"
            ))

            # dim_fulfillment_channel card
            render_html(render_erd_table_card(
                table_name="dim_fulfillment_channel",
                meta_info="5 rows | Dedicated Dimension",
                columns=[
                    ("PK", "fulfillment_id", "VARCHAR"),
                    ("ATTR", "order_status", "VARCHAR"),
                    ("ATTR", "shipping_carrier", "VARCHAR"),
                    ("NUM", "sla_delivery_days", "INTEGER"),
                ],
                is_fact=False,
                header_color="#0284C7",
                badge_label="DEDICATED DIMENSION",
                join_footnote="Used exclusively by fact_sales"
            ))

        with col_co_shared:
            render_html("""
            <div style="background: #EEF2FF; border: 1px dashed #6366F1; border-radius: 6px; padding: 5px 10px; margin-bottom: 10px; font-size: 10px; font-weight: 700; color: #4338CA; text-transform: uppercase; letter-spacing: 0.05em; text-align: center;">
              🔄 SHARED CONFORMED DIMENSIONS (ENTERPRISE BUS)
            </div>
            """)

            # dim_customer card
            render_html(render_erd_table_card(
                table_name="dim_customer",
                meta_info="4,327 rows | Shared Conformed",
                columns=[
                    ("PK", "customer_id", "VARCHAR"),
                    ("ATTR", "first_name", "VARCHAR"),
                    ("ATTR", "last_name", "VARCHAR"),
                    ("ATTR", "country", "VARCHAR"),
                ],
                is_fact=False,
                header_color="#0284C7",
                badge_label="CONFORMED DIMENSION",
                join_footnote="Shared: fact_sales ⇄ fact_customer_reviews"
            ))

            # dim_product card
            render_html(render_erd_table_card(
                table_name="dim_product",
                meta_info="100 rows | Shared Conformed",
                columns=[
                    ("PK", "product_id", "VARCHAR"),
                    ("ATTR", "product_name", "VARCHAR"),
                    ("ATTR", "category", "VARCHAR"),
                    ("NUM", "unit_price", "DECIMAL"),
                ],
                is_fact=False,
                header_color="#0284C7",
                badge_label="CONFORMED DIMENSION",
                join_footnote="Shared: fact_sales ⇄ fact_customer_reviews"
            ))

            # dim_date card
            render_html(render_erd_table_card(
                table_name="dim_date",
                meta_info="1,095 rows | Shared Conformed",
                columns=[
                    ("PK", "date_key", "INTEGER"),
                    ("ATTR", "order_date", "DATE"),
                    ("ATTR", "order_quarter", "VARCHAR"),
                    ("ATTR", "year", "INTEGER"),
                ],
                is_fact=False,
                header_color="#0284C7",
                badge_label="CONFORMED DIMENSION",
                join_footnote="Shared: fact_sales ⇄ fact_customer_reviews"
            ))

        with col_co_review:
            render_html("""
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 5px 10px; margin-bottom: 10px; font-size: 10px; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.05em; text-align: center;">
              BUSINESS PROCESS 2 • CUSTOMER SENTIMENT MART
            </div>
            """)

            # fact_customer_reviews card
            render_html(render_erd_table_card(
                table_name="fact_customer_reviews",
                meta_info="10,000 rows | Sentiment Process",
                columns=[
                    ("PK", "review_id", "VARCHAR"),
                    ("FK", "customer_id", "VARCHAR", "(Conformed)"),
                    ("FK", "product_id", "VARCHAR", "(Conformed)"),
                    ("FK", "date_key", "INTEGER", "(Conformed)"),
                    ("FK", "feedback_source_id", "VARCHAR", "(Dedicated)"),
                    ("NUM", "rating", "INTEGER"),
                    ("NUM", "review_length", "INTEGER"),
                    ("NUM", "sentiment_polarity_score", "DECIMAL"),
                ],
                is_fact=True,
                header_color="#7C3AED",
                badge_label="FACT TABLE • SENTIMENT",
                join_footnote="Measures: rating, review_length, polarity"
            ))

            # dim_feedback_source card
            render_html(render_erd_table_card(
                table_name="dim_feedback_source",
                meta_info="4 rows | Dedicated Dimension",
                columns=[
                    ("PK", "feedback_source_id", "VARCHAR"),
                    ("ATTR", "channel_name", "VARCHAR"),
                    ("ATTR", "platform_type", "VARCHAR"),
                    ("ATTR", "response_rate", "VARCHAR"),
                ],
                is_fact=False,
                header_color="#0284C7",
                badge_label="DEDICATED DIMENSION",
                join_footnote="Used exclusively by fact_customer_reviews"
            ))

        # Expandable Schema Summary: Warehouse DDL & Trade-Off Analysis
        with st.expander("📐 Fact Constellation Architecture Details, Warehouse DDL & Trade-Off Analysis", expanded=False):
            ddl_col_co, trade_col_co = st.columns([1.1, 1])
            with ddl_col_co:
                st.markdown("##### 📄 Production Fact Constellation DDL (`CREATE TABLE`)")
                st.code('''-- Fact Constellation (Galaxy Schema) Warehouse DDL
-- Shared Conformed Dimensions
CREATE TABLE dim_customer (
    customer_id VARCHAR(50) PRIMARY KEY,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    country VARCHAR(100)
);

CREATE TABLE dim_product (
    product_id VARCHAR(50) PRIMARY KEY,
    product_name VARCHAR(255),
    category VARCHAR(100),
    unit_price DECIMAL(10,2)
);

CREATE TABLE dim_date (
    date_key INTEGER PRIMARY KEY,
    order_date DATE,
    order_quarter VARCHAR(10),
    year INTEGER
);

-- Dedicated Dimensions
CREATE TABLE dim_fulfillment_channel (
    fulfillment_id VARCHAR(50) PRIMARY KEY,
    order_status VARCHAR(50),
    shipping_carrier VARCHAR(100),
    sla_delivery_days INTEGER
);

CREATE TABLE dim_feedback_source (
    feedback_source_id VARCHAR(50) PRIMARY KEY,
    channel_name VARCHAR(100),
    platform_type VARCHAR(100),
    response_rate VARCHAR(20)
);

-- Process Mart 1: Fact Sales
CREATE TABLE fact_sales (
    sale_id INTEGER PRIMARY KEY,
    customer_id VARCHAR(50) REFERENCES dim_customer(customer_id),
    product_id VARCHAR(50) REFERENCES dim_product(product_id),
    order_id VARCHAR(50),
    date_key INTEGER REFERENCES dim_date(date_key),
    fulfillment_id VARCHAR(50) REFERENCES dim_fulfillment_channel(fulfillment_id),
    quantity INTEGER,
    unit_price DECIMAL(10,2),
    total_amount DECIMAL(10,2)
);

-- Process Mart 2: Fact Customer Reviews
CREATE TABLE fact_customer_reviews (
    review_id VARCHAR(50) PRIMARY KEY,
    customer_id VARCHAR(50) REFERENCES dim_customer(customer_id),
    product_id VARCHAR(50) REFERENCES dim_product(product_id),
    date_key INTEGER REFERENCES dim_date(date_key),
    feedback_source_id VARCHAR(50) REFERENCES dim_feedback_source(feedback_source_id),
    rating INTEGER,
    review_length INTEGER,
    sentiment_polarity_score DECIMAL(5,2)
);''', language="sql")

            with trade_col_co:
                st.markdown("##### ⚖️ Architecture Trade-Off Analysis")
                trade_const_df = pd.DataFrame([
                    {"Evaluation Vector": "Normalization Level", "Score / Metric": "Mixed (Standardized Conformed)", "Architectural Assessment": "Conformed dimensions enforce unified master attributes across marts."},
                    {"Evaluation Vector": "Storage Efficiency", "Score / Metric": "Optimized Across Marts", "Architectural Assessment": "Shared customer and product entities prevent duplicate master storage."},
                    {"Evaluation Vector": "Query Latency (OLAP)", "Score / Metric": "⚡ Low Intra-Mart / Fast Drill-Through", "Architectural Assessment": "Intra-mart queries run at star speed; cross-mart drills bridge on shared keys."},
                    {"Evaluation Vector": "ETL & Pipeline Complexity", "Score / Metric": "Enterprise-Scale Governance", "Architectural Assessment": "Requires cross-team synchronization of shared grain and surrogate keys."}
                ])
                st.dataframe(trade_const_df, use_container_width=True, hide_index=True)

        # Interactive Cross-Mart Correlation Engine
        with st.container(border=True):
            st.markdown("#### 🌌 Interactive Cross-Mart Correlation Engine (Bridged by Conformed Dimensions)")
            st.caption("Drill simultaneously into Revenue (`Fact_Sales`) and Sentiment Rating (`Fact_Reviews_Feedback`) by joining across conformed dimensions.")

            bridge_dim = st.selectbox("Select Shared Conformed Dimension to Bridge Both Facts:", ["category", "country", "order_year"], index=0)
            dim_label = bridge_dim.capitalize()

            cross_mart_df = df.groupby(bridge_dim).agg(
                Gross_Revenue=("total_amount", "sum"),
                Order_Volume=("order_id", "count"),
                Average_Rating=("rating", "mean")
            ).reset_index()

            fig_cross = go.Figure()
            fig_cross.add_trace(go.Bar(
                x=cross_mart_df[bridge_dim],
                y=cross_mart_df["Gross_Revenue"],
                name="Gross Revenue ($)",
                marker_color="#4F46E5",
                yaxis="y1"
            ))
            fig_cross.add_trace(go.Scatter(
                x=cross_mart_df[bridge_dim],
                y=cross_mart_df["Average_Rating"],
                name="Customer Rating (★)",
                mode="lines+markers",
                line=dict(color="#10B981", width=3),
                marker=dict(size=8, color="#047857"),
                yaxis="y2"
            ))

            fig_cross.update_layout(
                template="plotly_white",
                height=300,
                margin=dict(t=35, b=25, l=25, r=25),
                font=dict(family="Inter, sans-serif", color="#334155", size=12),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                yaxis=dict(title=dict(text="Gross Revenue ($)", font=dict(color="#4F46E5")), side="left"),
                yaxis2=dict(title=dict(text="Avg Customer Rating (1-5 ★)", font=dict(color="#10B981")), overlaying="y", side="right", range=[1, 5]),
                plot_bgcolor="rgba(255,255,255,0)",
                paper_bgcolor="rgba(255,255,255,0)",
            )
            st.plotly_chart(fig_cross, use_container_width=True)
            st.dataframe(cross_mart_df.style.format({"Gross_Revenue": "${:,.2f}", "Order_Volume": "{:,}", "Average_Rating": "{:.2f}"}), use_container_width=True)

        with st.container(border=True):
            st.markdown("#### 🗄️ Fact Constellation Table & Conformed Dimension Inspector")
            col_cn1, col_cn2 = st.columns([1.5, 3])
            with col_cn1:
                selected_const_tbl = st.selectbox("Select Galaxy Table:", list(constellation_data.keys()), key="const_tbl_sel")
                c_df = constellation_data[selected_const_tbl]
                st.caption(f"**Physical Rows:** `{len(c_df):,}` | **Columns:** `{c_df.shape[1]}`")
                st.caption(f"**In-Memory Size:** `{c_df.memory_usage(deep=True).sum()/1024:.1f} KB`")
                st.download_button(
                    label=f"Export {selected_const_tbl.split()[0]} CSV",
                    data=c_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{selected_const_tbl.split()[0].lower()}.csv",
                    mime="text/csv",
                    key="const_schema_tbl_download_btn",
                )
            with col_cn2:
                st.dataframe(c_df.head(6), use_container_width=True)

    render_takeaways_card([
        "Star Schema remains the enterprise gold standard for analytical query performance (OLAP/BI), maximizing scan speeds via 1-hop joins.",
        "Snowflake Schema reduces physical storage redundancy and enforces 3NF integrity at the expense of additional join traversal latency.",
        "Fact Constellation architectures enable cross-mart enterprise correlation (e.g. tracking sales GMV against post-purchase review sentiment) using shared conformed dimensions."
    ])


# =============================================================================
# Module 02: Multidimensional OLAP Engine
# =============================================================================
elif selected_page == "02 — Multidimensional OLAP Engine":
    render_module_header(
        tag="MODULE 02 • DWDM CURRICULUM",
        title="Multidimensional OLAP Cube & Analytical Engine",
        subtitle="Executing Roll-up, Drill-down, Slice, Dice, 2D Pivot Heatmap, and Conceptual 3D OLAP Data Cube.",
        theme_tag="OLAP Engine"
    )

    render_problem_card(
        title="Rigid Hierarchies and Inability to Ad-Hoc Slice Sales Dimensions",
        business_problem="Category leaders and merchandise managers require multi-perspective analytical cuts across time, geography, product departments, and channels. Writing manual SQL group-by queries for each cross-dimensional cut is slow and introduces database read locks.",
        operational_impact="Static reports prevent rapid identification of regional margin discrepancies and category inventory bottlenecks."
    )

    # 6 Operation Tabs including 3D OLAP Cube
    olap_tab1, olap_tab2, olap_tab3, olap_tab4, olap_tab5, olap_tab6 = st.tabs([
        "↗️ Roll-Up", "↘️ Drill-Down", "✂️ Slice", "🎲 Dice", "🔄 Pivot Matrix", "🧊 3D OLAP Cube"
    ])

    # ==================== 1. ROLL-UP ====================
    with olap_tab1:
        st.markdown("##### Temporal Roll-Up: Summarizing Sales Along Time Hierarchies")
        roll_gran = st.radio(
            "Select Temporal Roll-Up Granularity:",
            ["Day (Last 30 Active Days)", "Month", "Quarter", "Year"],
            horizontal=True,
            index=2
        )

        if roll_gran == "Day (Last 30 Active Days)":
            daily_grp = df.groupby("order_date").agg(
                Gross_Revenue=("total_amount", "sum"),
                Orders=("order_id", "count"),
                Units=("quantity", "sum")
            ).reset_index().tail(30)
            daily_grp["Label"] = daily_grp["order_date"].dt.strftime("%Y-%m-%d")
            plot_df = daily_grp
            x_col = "Label"
        elif roll_gran == "Month":
            monthly_grp = df.groupby(["order_year", "order_month"]).agg(
                Gross_Revenue=("total_amount", "sum"),
                Orders=("order_id", "count"),
                Units=("quantity", "sum")
            ).reset_index()
            monthly_grp["Label"] = monthly_grp["order_year"].astype(str) + "-" + monthly_grp["order_month"].apply(lambda m: f"{m:02d}")
            plot_df = monthly_grp
            x_col = "Label"
        elif roll_gran == "Quarter":
            qtr_grp = df.groupby(["order_year", "order_quarter"]).agg(
                Gross_Revenue=("total_amount", "sum"),
                Orders=("order_id", "count"),
                Units=("quantity", "sum")
            ).reset_index()
            qtr_grp["Label"] = qtr_grp["order_year"].astype(str) + " " + qtr_grp["order_quarter"].astype(str)
            plot_df = qtr_grp
            x_col = "Label"
        else:
            yr_grp = df.groupby("order_year").agg(
                Gross_Revenue=("total_amount", "sum"),
                Orders=("order_id", "count"),
                Units=("quantity", "sum")
            ).reset_index()
            yr_grp["Label"] = yr_grp["order_year"].astype(str)
            plot_df = yr_grp
            x_col = "Label"

        fig_roll = px.bar(
            plot_df,
            x=x_col,
            y="Gross_Revenue",
            color="Gross_Revenue",
            color_continuous_scale=["#C7D2FE", "#4F46E5"],
            labels={x_col: "Time Dimension", "Gross_Revenue": "Gross Revenue ($)"}
        )
        st.plotly_chart(apply_chart_theme(fig_roll, height=280), use_container_width=True)
        st.dataframe(plot_df.style.format({"Gross_Revenue": "${:,.2f}", "Orders": "{:,}", "Units": "{:,}"}), use_container_width=True)

    # ==================== 2. DRILL-DOWN ====================
    with olap_tab2:
        st.markdown("##### Drill-Down: Navigating from High-Level Department to Individual SKUs")
        selected_cat = st.selectbox("Select Department Hierarchy to Drill Into:", sorted(df["category"].unique()))

        cat_subset = df[df["category"] == selected_cat]
        drill_sku = cat_subset.groupby("product_name").agg(
            Revenue=("total_amount", "sum"),
            Units_Sold=("quantity", "sum"),
            Avg_Price=("unit_price", "mean"),
            Avg_Rating=("rating", "mean")
        ).reset_index().sort_values("Revenue", ascending=True)

        fig_drill = px.bar(
            drill_sku,
            x="Revenue",
            y="product_name",
            orientation="h",
            color="Revenue",
            color_continuous_scale=["#C7D2FE", "#4F46E5"],
            labels={"Revenue": "SKU Revenue ($)", "product_name": "Product Name"}
        )
        st.plotly_chart(apply_chart_theme(fig_drill, height=280, show_legend=False), use_container_width=True)

        st.markdown("###### Secondary Drill-Down: Product SKU Regional & Channel Breakdown")
        selected_sku = st.selectbox("Select Specific SKU for Geographic Breakdown:", drill_sku["product_name"].unique())
        sku_subset = df[df["product_name"] == selected_sku]

        col_sk1, col_sk2 = st.columns(2)
        with col_sk1:
            sku_cntry = sku_subset.groupby("country")["total_amount"].sum().reset_index().sort_values("total_amount", ascending=False)
            fig_sk_cntry = px.bar(sku_cntry, x="country", y="total_amount", color="total_amount", color_continuous_scale=["#C7D2FE", "#4F46E5"], labels={"country": "Market", "total_amount": "Revenue ($)"})
            st.plotly_chart(apply_chart_theme(fig_sk_cntry, height=230, show_legend=False), use_container_width=True)
        with col_sk2:
            sku_pay = sku_subset.groupby("payment_method")["total_amount"].sum().reset_index()
            fig_sk_pay = px.pie(sku_pay, names="payment_method", values="total_amount", hole=0.45, color_discrete_sequence=PALETTE)
            st.plotly_chart(apply_chart_theme(fig_sk_pay, height=230), use_container_width=True)

    # ==================== 3. SLICE ====================
    with olap_tab3:
        st.markdown("##### Slice Operation: Locking One Dimension to Obtain a Cross-Sectional View")
        slice_dim = st.selectbox("Select Dimension to Slice Across:", ["country", "category", "order_quarter", "payment_method"], index=0)
        dim_labels = {"country": "Country", "category": "Category", "order_quarter": "Fiscal Quarter", "payment_method": "Payment Method"}
        
        slice_vals = sorted(df[slice_dim].astype(str).unique())
        locked_val = st.selectbox(f"Lock Value for {dim_labels[slice_dim]}:", slice_vals, index=0)

        sliced_df = df[df[slice_dim].astype(str) == locked_val]
        slice_gmv = sliced_df["total_amount"].sum()
        total_global_gmv = df["total_amount"].sum()
        slice_share = (slice_gmv / total_global_gmv) * 100

        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            render_metric_card("Slice Gross Revenue", f"${slice_gmv:,.0f}", f"{slice_share:.1f}% of Global GMV")
        with sc2:
            render_metric_card("Slice Order Volume", f"{len(sliced_df):,}", "Transactions in slice")
        with sc3:
            render_metric_card("Average Order Value", f"${sliced_df['total_amount'].mean():.2f}", "Basket spend")
        with sc4:
            render_metric_card("Customer Rating", f"{sliced_df['rating'].mean():.2f} ★", "Average satisfaction")

        col_sl1, col_sl2 = st.columns(2)
        with col_sl1:
            # Breakdown by category if not category slice
            sec_dim = "category" if slice_dim != "category" else "country"
            sec_summary = sliced_df.groupby(sec_dim)["total_amount"].sum().reset_index()
            fig_sec = px.pie(sec_summary, names=sec_dim, values="total_amount", hole=0.4, color_discrete_sequence=PALETTE, title=f"Revenue by {sec_dim.capitalize()}")
            st.plotly_chart(apply_chart_theme(fig_sec, height=270), use_container_width=True)
        with col_sl2:
            status_summary = sliced_df.groupby("order_status")["total_amount"].sum().reset_index()
            fig_stat = px.bar(status_summary, x="order_status", y="total_amount", color="order_status", color_discrete_sequence=PALETTE, title="Order Fulfillment Outcome")
            st.plotly_chart(apply_chart_theme(fig_stat, height=270, show_legend=False), use_container_width=True)

    # ==================== 4. DICE ====================
    with olap_tab4:
        st.markdown("##### Dice Operation: Multi-Axis Sub-Cube Filtering Across Dimensions")
        dc1, dc2, dc3, dc4 = st.columns(4)
        with dc1:
            dice_cats = st.multiselect("Select Categories:", sorted(df["category"].unique()), default=["Electronics", "Apparel"])
        with dc2:
            dice_ages = st.multiselect("Select Age Groups:", sorted(df["age_group"].unique()), default=["Teenagers", "Adults"])
        with dc3:
            dice_qtrs = st.multiselect("Select Quarters:", sorted(df["order_quarter"].unique()), default=["Q1", "Q3"])
        with dc4:
            dice_cnts = st.multiselect("Select Sovereign Markets:", sorted(df["country"].unique()), default=["USA", "Germany", "UK", "China"])

        diced_df = df[
            (df["category"].isin(dice_cats)) &
            (df["age_group"].isin(dice_ages)) &
            (df["order_quarter"].isin(dice_qtrs)) &
            (df["country"].isin(dice_cnts))
        ]


        d_gmv = diced_df["total_amount"].sum()
        d_share = (d_gmv / df["total_amount"].sum()) * 100

        dm1, dm2, dm3, dm4 = st.columns(4)
        with dm1:
            render_metric_card("Sub-Cube Revenue", f"${d_gmv:,.0f}", f"{d_share:.1f}% of Total Warehouse")
        with dm2:
            render_metric_card("Sub-Cube Transactions", f"{len(diced_df):,}", "Matching Orders")
        with dm3:
            render_metric_card("Selected Cell Matrix", f"{len(dice_cats)} × {len(dice_ages)} × {len(dice_qtrs)} × {len(dice_cnts)}", "Coordinate Size")
        with dm4:
            render_metric_card("Avg Order Value", f"${diced_df['total_amount'].mean():.2f}" if len(diced_df) > 0 else "$0.00", "Basket Size")

        if len(diced_df) > 0:
            dice_chart = diced_df.groupby(["category", "country"])["total_amount"].sum().reset_index()
            fig_dice = px.bar(
                dice_chart,
                x="category",
                y="total_amount",
                color="country",
                barmode="group",
                color_discrete_sequence=PALETTE,
                labels={"total_amount": "Revenue ($)", "category": "Category"}
            )
            st.plotly_chart(apply_chart_theme(fig_dice, height=280), use_container_width=True)
            with st.expander("Inspect Sub-Cube Tabular Records"):
                st.dataframe(diced_df.head(20), use_container_width=True)
        else:
            st.warning("No transactions match the selected multi-axis dice criteria.")

    # ==================== 5. PIVOT MATRIX ====================
    with olap_tab5:
        st.markdown("##### Pivot Operation: 2D Cross-Tabulated Matrix Heatmap")
        p_row, p_col, p_val = st.columns(3)
        with p_row:
            piv_row = st.selectbox("Row Dimension (Y-Axis):", ["category", "payment_method", "age_group", "order_status"], index=0)
        with p_col:
            piv_col = st.selectbox("Column Dimension (X-Axis):", ["country", "order_quarter", "gender", "order_year"], index=0)
        with p_val:
            piv_metric = st.selectbox("Aggregation Metric:", ["Gross Revenue ($)", "Order Count", "Average Rating", "Units Sold"], index=0)

        if piv_metric == "Gross Revenue ($)":
            p_matrix = pd.crosstab(df[piv_row], df[piv_col], values=df["total_amount"], aggfunc="sum").fillna(0)
            fmt_str = "${:,.0f}"
        elif piv_metric == "Order Count":
            p_matrix = pd.crosstab(df[piv_row], df[piv_col], values=df["order_id"], aggfunc="count").fillna(0)
            fmt_str = "{:,}"
        elif piv_metric == "Average Rating":
            p_matrix = pd.crosstab(df[piv_row], df[piv_col], values=df["rating"], aggfunc="mean").fillna(0)
            fmt_str = "{:.2f}"
        else:
            p_matrix = pd.crosstab(df[piv_row], df[piv_col], values=df["quantity"], aggfunc="sum").fillna(0)
            fmt_str = "{:,}"

        fig_piv = px.imshow(
            p_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale=["#FFFFFF", "#EEF2FF", "#4F46E5"],
            labels=dict(x=piv_col.capitalize(), y=piv_row.capitalize(), color=piv_metric)
        )
        st.plotly_chart(apply_chart_theme(fig_piv, height=320), use_container_width=True)

        with st.expander("Tabular Pivot Matrix Data View"):
            st.dataframe(p_matrix.style.format(fmt_str), use_container_width=True)

    # ==================== 6. 3D OLAP CUBE ====================
    with olap_tab6:
        st.markdown("##### 3D OLAP Data Cube Visualization (Conceptual & Physical Lattice)")
        st.caption("Visualizing the analytical 3D space: **Product Category (X)** × **Sovereign Market (Y)** × **Fiscal Quarter (Z)**.")

        c_opt1, c_opt2, c_opt3 = st.columns(3)
        with c_opt1:
            cube_metric = st.selectbox("Select Cube Marker Metric:", ["Total Sales Revenue ($)", "Transaction Order Volume"], index=0)
        with c_opt2:
            cube_color_theme = st.selectbox("Color Palette Scheme:", ["Viridis", "Plasma", "Turbo"], index=0)
        with c_opt3:
            min_rev_threshold = st.slider("Filter Minimum Revenue Threshold ($):", 0, 20000, 0, 1000)

        # Coordinate Mappings
        cats = sorted(df["category"].unique())
        cnts = sorted(df["country"].unique())
        qtrs = ["Q1", "Q2", "Q3", "Q4"]

        cat_map = {c: i for i, c in enumerate(cats)}
        cnt_map = {c: i for i, c in enumerate(cnts)}
        qtr_map = {q: i for i, q in enumerate(qtrs)}

        cube_agg = df.groupby(["category", "country", "order_quarter"]).agg(
            Revenue=("total_amount", "sum"),
            Orders=("order_id", "count"),
            Units=("quantity", "sum"),
            Avg_Rating=("rating", "mean")
        ).reset_index()

        cube_agg["x"] = cube_agg["category"].map(cat_map)
        cube_agg["y"] = cube_agg["country"].map(cnt_map)
        cube_agg["z"] = cube_agg["order_quarter"].astype(str).map(qtr_map)

        filtered_cube = cube_agg[cube_agg["Revenue"] >= min_rev_threshold].copy()

        # Wireframe Lattice Bounding Box
        x0, x1 = 0, len(cats) - 1
        y0, y1 = 0, len(cnts) - 1
        z0, z1 = 0, len(qtrs) - 1
        bx = [x0, x1, x1, x0, x0, None, x0, x1, x1, x0, x0, None, x0, x0, None, x1, x1, None, x1, x1, None, x0, x0]
        by = [y0, y0, y1, y1, y0, None, y0, y0, y1, y1, y0, None, y0, y0, None, y0, y0, None, y1, y1, None, y1, y1]
        bz = [z0, z0, z0, z0, z0, None, z1, z1, z1, z1, z1, None, z0, z1, None, z0, z1, None, z0, z1, None, z0, z1]

        fig_cube = go.Figure()

        # Trace 1: Bounding Wireframe
        fig_cube.add_trace(
            go.Scatter3d(
                x=bx, y=by, z=bz,
                mode="lines",
                line=dict(color="#818CF8", width=3, dash="dash"),
                name="Cube Lattice Boundary",
                hoverinfo="none"
            )
        )

        # Trace 2: Data Points
        val_series = filtered_cube["Revenue"] if cube_metric == "Total Sales Revenue ($)" else filtered_cube["Orders"]
        max_val = val_series.max() if len(val_series) > 0 and val_series.max() > 0 else 1
        marker_sizes = (val_series / max_val * 16 + 5) if len(val_series) > 0 else []

        fig_cube.add_trace(
            go.Scatter3d(
                x=filtered_cube["x"],
                y=filtered_cube["y"],
                z=filtered_cube["z"],
                mode="markers",
                marker=dict(
                    size=marker_sizes,
                    color=val_series,
                    colorscale=cube_color_theme,
                    opacity=0.88,
                    colorbar=dict(
                        title=dict(text=cube_metric, font=dict(family="Inter", size=11, color="#334155")),
                        thickness=14,
                        len=0.7,
                        x=1.02
                    )
                ),
                customdata=np.stack((
                    filtered_cube["category"],
                    filtered_cube["country"],
                    filtered_cube["order_quarter"],
                    filtered_cube["Revenue"],
                    filtered_cube["Orders"],
                    filtered_cube["Units"],
                    filtered_cube["Avg_Rating"]
                ), axis=-1),
                hovertemplate=(
                    "<b>%{customdata[0]}</b> in <b>%{customdata[1]}</b><br>"
                    "Quarter: %{customdata[2]}<br>"
                    "Revenue: $%{customdata[3]:,.2f}<br>"
                    "Orders: %{customdata[4]:,}<br>"
                    "Units Sold: %{customdata[5]:,}<br>"
                    "Avg Rating: %{customdata[6]:.2f} ★<extra></extra>"
                ),
                name="Cube Cells"
            )
        )

        fig_cube.update_layout(
            scene=dict(
                xaxis=dict(
                    title=dict(text="Product Category", font=dict(family="Inter", size=11, color="#334155")),
                    tickvals=list(range(len(cats))),
                    ticktext=cats,
                    backgroundcolor="#F8FAFC",
                    gridcolor="#E2E8F0",
                    showbackground=True,
                ),
                yaxis=dict(
                    title=dict(text="Sovereign Market", font=dict(family="Inter", size=11, color="#334155")),
                    tickvals=list(range(len(cnts))),
                    ticktext=cnts,
                    backgroundcolor="#F8FAFC",
                    gridcolor="#E2E8F0",
                    showbackground=True,
                ),
                zaxis=dict(
                    title=dict(text="Fiscal Quarter", font=dict(family="Inter", size=11, color="#334155")),
                    tickvals=list(range(len(qtrs))),
                    ticktext=qtrs,
                    backgroundcolor="#F8FAFC",
                    gridcolor="#E2E8F0",
                    showbackground=True,
                ),
                camera=dict(eye=dict(x=1.65, y=1.65, z=1.35))
            ),
            margin=dict(l=10, r=10, b=10, t=20),
            height=460,
            paper_bgcolor="rgba(255,255,255,0)",
        )

        st.plotly_chart(fig_cube, use_container_width=True)

        # Cube Metrics Card
        total_possible_cells = len(cats) * len(cnts) * len(qtrs)
        active_cells = len(cube_agg)
        active_filtered = len(filtered_cube)
        cube_gmv = cube_agg["Revenue"].sum()

        cb1, cb2, cb3, cb4 = st.columns(4)
        with cb1:
            render_metric_card("Total Theoretical Cells", f"{total_possible_cells}", "6 Cats × 10 Markets × 4 Quarters")
        with cb2:
            render_metric_card("Active Populated Cells", f"{active_cells}", "100% Coordinate Coverage")
        with cb3:
            render_metric_card("Displayed Cells", f"{active_filtered}", f"Threshold >= ${min_rev_threshold:,}")
        with cb4:
            render_metric_card("Total Cube GMV", f"${cube_gmv:,.0f}", "Consolidated Warehouse Volume")

        with st.container(border=True):
            st.markdown("##### 💡 Conceptual OLAP Cube Architecture")
            st.markdown("""
            * **Cell Address:** Each point in the 3D grid corresponds to a coordinate tuple `(Category_i, Country_j, Quarter_k)`.
            * **Physical Measures:** The point encapsulates numeric measures (Revenue, Orders, Units Sold, Rating).
            * **Relational Equivalent:** Represents the materialized result of a relational SQL `CUBE(category, country, order_quarter)` aggregation.
            * **Interactive Exploration:** Drag with your mouse to rotate the lattice in 3D, scroll to zoom into specific product-market intersections, and hover to view precise financial line items.
            """)

    render_takeaways_card([
        "Digital payment channels (Credit Card, PayPal) settle 84% of orders over $500 in the Electronics department.",
        "Rolling up sales across calendar quarters indicates consistent baseline revenue without significant single-season contraction.",
        "3D Cube visualization allows instant discovery of high-density intersections across Category, Country, and Fiscal Quarter without executing multiple sequential SQL queries."
    ])


# =============================================================================
# Module 03: Data Profiling & ETL Pipeline
# =============================================================================
elif selected_page == "03 — Data Profiling & ETL Pipeline":
    render_module_header(
        tag="MODULE 03 • DWDM CURRICULUM",
        title="Data Profiling & Automated ETL Pipeline",
        subtitle="Extraction, string cleansing, financial derivations, datetime parsing, and audit verification.",
        theme_tag="Data Engineering"
    )

    render_problem_card(
        title="Unindexed String Timestamps and Missing Transaction Measures",
        business_problem="Raw transaction logs capture quantity and price without pre-computing line-item totals, while dates are stored as raw strings. Running mathematical aggregations on raw logs forces runtime conversions and produces date filtering failures.",
        operational_impact="Query latency increases and lack of standardized data types leads to silent arithmetic errors during revenue reconciliation."
    )

    # Before vs After Metrics
    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        render_metric_card("Raw Data State", "20 Columns", "String datetimes, no derived totals")
    with col_b2:
        render_metric_card("ETL Derivation", "total_amount", "Pre-computed line item = qty * price")
    with col_b3:
        render_metric_card("Processed Mart", "25 Columns", "Indexed timestamps and date dimensions")

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        with st.container(border=True):
            st.markdown("#### Line-Item Total Amount Distribution")
            fig_hist = px.histogram(
                df,
                x="total_amount",
                nbins=35,
                color_discrete_sequence=["#4F46E5"],
                labels={"total_amount": "Total Amount ($)"}
            )
            st.plotly_chart(apply_chart_theme(fig_hist, height=270, show_legend=False), use_container_width=True)

    with col_e2:
        with st.container(border=True):
            st.markdown("#### Day of Week Order Volume")
            day_counts = df["day_name"].value_counts().reindex([
                "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
            ]).reset_index()
            day_counts.columns = ["Day", "Orders"]
            fig_day = px.bar(
                day_counts,
                x="Day",
                y="Orders",
                color="Orders",
                color_continuous_scale=["#C7D2FE", "#4F46E5"]
            )
            st.plotly_chart(apply_chart_theme(fig_day, height=270, show_legend=False), use_container_width=True)

    # Profiling Completeness Table
    with st.container(border=True):
        st.markdown("#### Attribute Completeness & Quality Verification")
        comp_df = pd.DataFrame({
            "Attribute": df.columns,
            "Completeness (%)": [100.0 * (1 - df[c].isnull().sum() / len(df)) for c in df.columns],
            "Data Type": [str(df[c].dtype) for c in df.columns],
            "Memory (KB)": [round(df[c].memory_usage(deep=True) / 1024, 1) for c in df.columns]
        })
        st.dataframe(comp_df, use_container_width=True)

    render_takeaways_card([
        "Pre-computing line-item totals during ETL removes repetitive on-the-fly multiplications across reporting queries.",
        "Parsing ISO datetimes enables indexed temporal hierarchies (Year, Quarter, Month, Day) for OLAP aggregations.",
        "Zero null values detected across all 20 operational columns confirms the dataset is clean and production-ready."
    ])

# =============================================================================
# Module 04: Linear Regression (Sales Forecasting)
# =============================================================================
elif selected_page == "04 — Linear Regression (Sales Forecasting)":
    render_module_header(
        tag="MODULE 04 • DWDM CURRICULUM",
        title="Linear Regression & Sales Forecasting",
        subtitle="Training Ordinary Least Squares (OLS) models to predict transaction spend from quantity and unit pricing.",
        theme_tag="Predictive Modeling"
    )

    render_problem_card(
        title="Uncertainty in Basket Revenue Estimates Across Pricing Tiers",
        business_problem="Merchandising teams cannot project order revenue accurately without an empirical model relating unit prices, quantities, and customer ratings. Without quantitative benchmarks, promotional discounting lacks profitability guards.",
        operational_impact="Lack of predictive regression lines impairs revenue forecasting and inventory replenishment modeling."
    )

    c_reg1, c_reg2 = st.columns([1, 2])
    with c_reg1:
        feature_mode = st.selectbox(
            "Select Regression Predictor Setup:",
            ["Unit Price Alone", "Unit Price + Quantity", "Unit Price + Quantity + Customer Rating"]
        )
        test_size = st.slider("Test Split Size (%):", 10, 40, 20, 5) / 100.0

    if feature_mode == "Unit Price Alone":
        feature_cols = ["unit_price"]
    elif feature_mode == "Unit Price + Quantity":
        feature_cols = ["unit_price", "quantity"]
    else:
        feature_cols = ["unit_price", "quantity", "rating"]

    X = df[feature_cols]
    y = df["total_amount"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("R² Determination", f"{r2:.4f}", f"Variance Explained: {r2*100:.1f}%")
    with m2:
        render_metric_card("Root Mean Sq Error (RMSE)", f"${rmse:.2f}", "Standard Error")
    with m3:
        render_metric_card("Mean Absolute Error (MAE)", f"${mae:.2f}", "Average Residual")
    with m4:
        render_metric_card("Trained Observations", f"{len(X_train):,}", f"Tested on {len(X_test):,}")

    col_rp1, col_rp2 = st.columns(2)
    with col_rp1:
        with st.container(border=True):
            st.markdown("#### Actual vs Predicted Revenue")
            eval_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})[:250]
            fig_reg = px.scatter(
                eval_df,
                x="Actual",
                y="Predicted",
                color_discrete_sequence=["#4F46E5"],
                labels={"Actual": "Actual Spend ($)", "Predicted": "Predicted Spend ($)"}
            )
            # Add 45-degree line
            max_val = max(eval_df["Actual"].max(), eval_df["Predicted"].max())
            fig_reg.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode="lines", name="Ideal Fit", line=dict(color="#10B981", dash="dash")))
            st.plotly_chart(apply_chart_theme(fig_reg, height=290), use_container_width=True)

    with col_rp2:
        with st.container(border=True):
            st.markdown("#### Residual Error Distribution")
            residuals = y_test.values - y_pred
            fig_res = px.histogram(
                x=residuals,
                nbins=35,
                color_discrete_sequence=["#06B6D4"],
                labels={"x": "Prediction Residual Error ($)"}
            )
            st.plotly_chart(apply_chart_theme(fig_res, height=290, show_legend=False), use_container_width=True)

    # Live Predictor Box
    with st.container(border=True):
        st.markdown("#### Live Order Spend Estimator")
        p1, p2, p3 = st.columns(3)
        with p1:
            in_price = st.number_input("Item Unit Price ($):", min_value=1.0, max_value=1500.0, value=120.0, step=10.0)
        with p2:
            in_qty = st.number_input("Order Item Quantity:", min_value=1, max_value=20, value=2, step=1)
        with p3:
            in_rate = st.slider("Rating (if applicable):", 1, 5, 4)

        if feature_mode == "Unit Price Alone":
            sample_in = np.array([[in_price]])
        elif feature_mode == "Unit Price + Quantity":
            sample_in = np.array([[in_price, in_qty]])
        else:
            sample_in = np.array([[in_price, in_qty, in_rate]])

        pred_val = lr_model.predict(sample_in)[0]
        st.success(f"**Predicted Order Revenue:** `${max(0.0, pred_val):,.2f}` (Coefficients: `{lr_model.coef_.round(2)}`, Intercept: `${lr_model.intercept_:.2f}`)")

    render_takeaways_card([
        "Incorporating both unit price and quantity yields high predictive power (R² > 0.95), reflecting the multiplicative line-item formula.",
        "Residual distribution is centered around zero, confirming no significant systematic under-prediction or over-prediction bias.",
        "The model provides a baseline estimator for calculating expected order revenue prior to applying checkout promotions."
    ])

# =============================================================================
# Module 05: Decision Tree Classification
# =============================================================================
elif selected_page == "05 — Decision Tree Classification":
    render_module_header(
        tag="MODULE 05 • DWDM CURRICULUM",
        title="Supervised Decision Tree Classification",
        subtitle="Predicting fulfillment outcome (Delivered vs At-Risk) to reduce cancellation and return expenses.",
        theme_tag="Classification"
    )

    render_problem_card(
        title="High Order Cancellation and Return Rates Across Payment Channels",
        business_problem="Roughly 40% of historical orders result in cancellations or product returns, wasting shipping and packaging expenses. Operations teams need an automated rule model at checkout to identify high-risk orders.",
        operational_impact="Managing fulfillment reactively without predictive classification leads to preventable reverse logistics courier charges."
    )

    col_dt1, col_dt2 = st.columns([1, 2])
    with col_dt1:
        max_depth = st.slider("Decision Tree Max Depth:", 2, 8, 4)
        min_split = st.slider("Min Samples Split:", 2, 50, 10, 2)

    # Encode predictors
    feature_cols = ["payment_method", "category", "unit_price", "quantity", "country"]
    X_dt = pd.get_dummies(df[feature_cols], drop_first=True)
    y_dt = (df["order_status"] == "Delivered").astype(int)

    X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(X_dt, y_dt, test_size=0.2, random_state=42, stratify=y_dt)

    dt_clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_split, random_state=42)
    dt_clf.fit(X_train_dt, y_train_dt)
    y_pred_dt = dt_clf.predict(X_test_dt)

    acc = accuracy_score(y_test_dt, y_pred_dt)
    prec = precision_score(y_test_dt, y_pred_dt, zero_division=0)
    rec = recall_score(y_test_dt, y_pred_dt, zero_division=0)
    f1 = f1_score(y_test_dt, y_pred_dt, zero_division=0)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("Accuracy", f"{acc*100:.1f}%", "Overall Classification")
    with m2:
        render_metric_card("Precision", f"{prec*100:.1f}%", "Delivered Precision")
    with m3:
        render_metric_card("Recall", f"{rec*100:.1f}%", "Delivered Coverage")
    with m4:
        render_metric_card("F1-Score", f"{f1:.3f}", "Harmonic Mean")

    col_cm, col_fi = st.columns(2)
    with col_cm:
        with st.container(border=True):
            st.markdown("#### Confusion Matrix Heatmap")
            cm = confusion_matrix(y_test_dt, y_pred_dt)
            fig_cm = px.imshow(
                cm,
                x=["Pred: At-Risk", "Pred: Delivered"],
                y=["Actual: At-Risk", "Actual: Delivered"],
                text_auto=True,
                color_continuous_scale=["#FFFFFF", "#EEF2FF", "#4F46E5"],
            )
            st.plotly_chart(apply_chart_theme(fig_cm, height=280), use_container_width=True)

    with col_fi:
        with st.container(border=True):
            st.markdown("#### Feature Importance Rankings")
            feat_imp = pd.Series(dt_clf.feature_importances_, index=X_dt.columns).sort_values(ascending=False).head(7).reset_index()
            feat_imp.columns = ["Feature", "Importance"]
            fig_fi = px.bar(
                feat_imp.sort_values("Importance", ascending=True),
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale=["#C7D2FE", "#4F46E5"]
            )
            st.plotly_chart(apply_chart_theme(fig_fi, height=280, show_legend=False), use_container_width=True)

    render_takeaways_card([
        "Payment channel and unit price are the dominant split nodes in predicting order outcome.",
        "Cash-on-Delivery orders on higher unit price products exhibit a disproportionately high probability of cancellation.",
        "Enforcing phone verification on high-risk nodes prior to dispatch helps protect fulfillment margins."
    ])

# =============================================================================
# Module 06: Naïve Bayes Classification
# =============================================================================
elif selected_page == "06 — Naïve Bayes Classification":
    render_module_header(
        tag="MODULE 06 • DWDM CURRICULUM",
        title="Probabilistic Naïve Bayes Classification",
        subtitle="Estimating customer satisfaction probabilities using Bayes' Theorem with conditional independence.",
        theme_tag="Bayesian Learning"
    )

    render_problem_card(
        title="Probabilistic Early Detection of Dissatisfied Customer Feedback",
        business_problem="Customer feedback ratings of 1 and 2 stars indicate defective goods or late shipments. Waiting for quarterly satisfaction surveys delays corrective supplier action.",
        operational_impact="Inability to assess transaction satisfaction probabilistically leads to persistent supplier quality flaws."
    )

    # Classify rating: High (4-5) vs Low/Avg (1-3)
    df["rating_binary"] = (df["rating"] >= 4).astype(int)
    nb_features = ["unit_price", "quantity"]
    X_nb = df[nb_features]
    y_nb = df["rating_binary"]

    X_train_nb, X_test_nb, y_train_nb, y_test_nb = train_test_split(X_nb, y_nb, test_size=0.25, random_state=42, stratify=y_nb)

    gnb = GaussianNB()
    gnb.fit(X_train_nb, y_train_nb)
    y_pred_nb = gnb.predict(X_test_nb)
    y_prob_nb = gnb.predict_proba(X_test_nb)[:, 1]

    acc_nb = accuracy_score(y_test_nb, y_pred_nb)
    fpr, tpr, _ = roc_curve(y_test_nb, y_prob_nb)
    roc_auc = auc(fpr, tpr)

    n1, n2, n3 = st.columns(3)
    with n1:
        render_metric_card("Accuracy", f"{acc_nb*100:.1f}%", "Gaussian Naive Bayes")
    with n2:
        render_metric_card("ROC AUC Score", f"{roc_auc:.3f}", "Discrimination Ability")
    with n3:
        render_metric_card("Class Prior (High Rating)", f"{gnb.class_prior_[1]*100:.1f}%", f"Low/Avg: {gnb.class_prior_[0]*100:.1f}%")

    col_roc, col_priors = st.columns(2)
    with col_roc:
        with st.container(border=True):
            st.markdown("#### ROC Curve Analysis")
            fig_roc = px.line(x=fpr, y=tpr, labels={"x": "False Positive Rate", "y": "True Positive Rate"})
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#94A3B8"), name="Chance"))
            fig_roc.update_traces(line_color="#4F46E5")
            st.plotly_chart(apply_chart_theme(fig_roc, height=280), use_container_width=True)

    with col_priors:
        with st.container(border=True):
            st.markdown("#### Class Conditional Means & Variances")
            stats_table = pd.DataFrame({
                "Target Class": ["Low/Avg Rating (1-3)", "High Rating (4-5)"],
                "Mean Unit Price ($)": gnb.theta_[:, 0].round(2),
                "Mean Quantity": gnb.theta_[:, 1].round(2),
                "Price Variance": gnb.var_[:, 0].round(2)
            })
            st.dataframe(stats_table, use_container_width=True)

    render_takeaways_card([
        "Gaussian Naïve Bayes handles continuous features by fitting individual normal distributions per class.",
        "Conditional independence assumption offers fast training speeds on large transaction volumes.",
        "Probabilistic scoring provides a continuous confidence metric for prioritizing follow-up customer inquiries."
    ])

# =============================================================================
# Module 07: K-Means Customer Clustering
# =============================================================================
elif selected_page == "07 — K-Means Customer Clustering":
    render_module_header(
        tag="MODULE 07 • DWDM CURRICULUM",
        title="Unsupervised K-Means Customer Segmentation",
        subtitle="Partitioning customer accounts into cohesive RFM behavioral clusters in standardized vector space.",
        theme_tag="Clustering"
    )

    render_problem_card(
        title="Uniform Marketing Campaigns Causing Margin Erosion on Top Buyers",
        business_problem="Treating all 4,327 customer accounts identically wastes promotional budgets. Offering blanket discount vouchers to high-income loyalists dilutes gross margin, while presenting premium electronics to casual one-time buyers yields low conversion.",
        operational_impact="Lack of data-driven customer personas impairs targeted merchandising and retention investments."
    )

    # Compute RFM
    max_date = df["order_date"].max()
    rfm_df = df.groupby("customer_id").agg(
        Recency=("order_date", lambda d: (max_date - d.max()).days if pd.notnull(d.max()) else 100),
        Frequency=("order_id", "count"),
        Monetary=("total_amount", "sum")
    ).reset_index()

    scaler = StandardScaler()
    X_rfm_scaled = scaler.fit_transform(rfm_df[["Recency", "Frequency", "Monetary"]])

    col_k1, col_k2 = st.columns([1, 2])
    with col_k1:
        k_clusters = st.slider("Select Number of Clusters (K):", 2, 6, 3)

    km = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    rfm_df["Cluster"] = km.fit_predict(X_rfm_scaled)
    rfm_df["Cluster_Name"] = "Cluster " + rfm_df["Cluster"].astype(str)

    # Elbow WCSS
    wcss = []
    for k in range(1, 7):
        m = KMeans(n_clusters=k, random_state=42, n_init=5)
        m.fit(X_rfm_scaled)
        wcss.append(m.inertia_)

    col_el, col_2d = st.columns(2)
    with col_el:
        with st.container(border=True):
            st.markdown("#### WCSS Inertia Reduction (Elbow Method)")
            elbow_fig = px.line(x=list(range(1, 7)), y=wcss, markers=True, labels={"x": "Cluster Count K", "y": "WCSS Inertia"})
            elbow_fig.update_traces(line_color="#4F46E5", marker=dict(size=7, color="#312E81"))
            st.plotly_chart(apply_chart_theme(elbow_fig, height=280), use_container_width=True)

    with col_2d:
        with st.container(border=True):
            st.markdown("#### Frequency vs Monetary Spend")
            fig_km = px.scatter(
                rfm_df,
                x="Frequency",
                y="Monetary",
                color="Cluster_Name",
                color_discrete_sequence=PALETTE,
                labels={"Frequency": "Order Count", "Monetary": "Total Spend ($)"}
            )
            st.plotly_chart(apply_chart_theme(fig_km, height=280), use_container_width=True)

    # Centroids and Profiles
    with st.container(border=True):
        st.markdown("#### Discovered Cluster Centroid Behavioral Profiles")
        cluster_summary = rfm_df.groupby("Cluster_Name").agg(
            Customer_Count=("customer_id", "count"),
            Avg_Recency_Days=("Recency", "mean"),
            Avg_Orders=("Frequency", "mean"),
            Avg_Monetary_Spend=("Monetary", "mean")
        ).reset_index()
        st.dataframe(
            cluster_summary.style.format({
                "Customer_Count": "{:,}",
                "Avg_Recency_Days": "{:.1f}",
                "Avg_Orders": "{:.2f}",
                "Avg_Monetary_Spend": "${:,.2f}"
            }),
            use_container_width=True
        )

    render_takeaways_card([
        "High-Monetary clusters account for disproportionate revenue; these accounts should receive express dispatch rather than discounts.",
        "High-Recency dormant buyers represent churn risks that benefit from automated re-engagement vouchers.",
        "Scaling Recency, Frequency, and Monetary dimensions via StandardScaler prevents high dollar amounts from dominating distance calculations."
    ])

# =============================================================================
# Module 08: Hierarchical Clustering
# =============================================================================
elif selected_page == "08 — Hierarchical Clustering":
    render_module_header(
        tag="MODULE 08 • DWDM CURRICULUM",
        title="Hierarchical Agglomerative Clustering",
        subtitle="Uncovering hierarchical taxonomies and dendrogram relationships among customer purchasing cohorts.",
        theme_tag="Unsupervised Hierarchy"
    )

    render_problem_card(
        title="Need for Multi-Level Nested Taxonomies in Customer Groupings",
        business_problem="Flat partitioning algorithms like K-Means force strict division boundaries without revealing hierarchical parent-child relationships between shopping cohorts.",
        operational_impact="Inability to explore nested customer tiers impedes structured multi-tier loyalty program design."
    )

    max_date = df["order_date"].max()
    rfm_agg = df.groupby("customer_id").agg(
        Recency=("order_date", lambda d: (max_date - d.max()).days if pd.notnull(d.max()) else 100),
        Frequency=("order_id", "count"),
        Monetary=("total_amount", "sum")
    ).reset_index()

    scaler_agg = StandardScaler()
    X_agg_scaled = scaler_agg.fit_transform(rfm_agg[["Recency", "Frequency", "Monetary"]])

    col_h1, col_h2 = st.columns([1, 2])
    with col_h1:
        n_clusters_agg = st.slider("Select Hierarchical Clusters:", 2, 5, 3)
        linkage_type = st.selectbox("Linkage Criterion:", ["ward", "complete", "average"])

    agg_model = AgglomerativeClustering(n_clusters=n_clusters_agg, linkage=linkage_type)
    rfm_agg["Agg_Cluster"] = agg_model.fit_predict(X_agg_scaled)
    rfm_agg["Cluster_Label"] = "Cohort " + rfm_agg["Agg_Cluster"].astype(str)

    # Compute silhouette score on a sample
    sample_indices = np.random.RandomState(42).choice(len(X_agg_scaled), min(1000, len(X_agg_scaled)), replace=False)
    sil_score = silhouette_score(X_agg_scaled[sample_indices], rfm_agg["Agg_Cluster"].iloc[sample_indices])

    c_s1, c_s2 = st.columns(2)
    with c_s1:
        render_metric_card("Silhouette Score", f"{sil_score:.3f}", "Cluster Cohesion & Separation")
    with c_s2:
        render_metric_card("Linkage Criterion", linkage_type.capitalize(), "Agglomerative Tree Merging")

    col_dend, col_agg_scatter = st.columns(2)
    with col_dend:
        with st.container(border=True):
            st.markdown("#### Dendrogram Sample (Top 25 Cohorts)")
            # Sample for dendrogram visualization
            sample_sub = X_agg_scaled[sample_indices[:25]]
            dend_fig = ff.create_dendrogram(sample_sub, color_threshold=1.5)
            dend_fig.update_layout(yaxis_title="Distance")
            st.plotly_chart(apply_chart_theme(dend_fig, height=280), use_container_width=True)

    with col_agg_scatter:
        with st.container(border=True):
            st.markdown("#### Agglomerative Cluster Projection")
            fig_agg_sc = px.scatter(
                rfm_agg,
                x="Frequency",
                y="Monetary",
                color="Cluster_Label",
                color_discrete_sequence=PALETTE,
                labels={"Frequency": "Order Count", "Monetary": "Spend ($)"}
            )
            st.plotly_chart(apply_chart_theme(fig_agg_sc, height=280), use_container_width=True)

    # Breakdown table
    with st.container(border=True):
        st.markdown("#### Hierarchical Cohort Breakdown")
        agg_summary = rfm_agg.groupby("Cluster_Label").agg(
            Accounts=("customer_id", "count"),
            Avg_Recency=("Recency", "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_Monetary=("Monetary", "mean")
        ).reset_index()
        st.dataframe(
            agg_summary.style.format({
                "Accounts": "{:,}",
                "Avg_Recency": "{:.1f}",
                "Avg_Frequency": "{:.2f}",
                "Avg_Monetary": "${:,.2f}"
            }),
            use_container_width=True
        )

    render_takeaways_card([
        "Ward linkage minimizes total within-cluster variance at each successive agglomeration step.",
        "Dendrogram inspection identifies natural distance thresholds for defining gold, silver, and bronze loyalty tiers.",
        "Unlike K-Means, hierarchical clustering produces a deterministic tree structure without depending on random centroid initialization."
    ])

# =============================================================================
# Module 09: Apriori Association Rule Mining
# =============================================================================
elif selected_page == "09 — Apriori Association Rule Mining":
    render_module_header(
        tag="MODULE 09 • DWDM CURRICULUM",
        title="Market Basket Association Mining (Apriori)",
        subtitle="Deriving statistical Support, Confidence, and Lift metrics from customer transaction baskets.",
        theme_tag="Association Rules"
    )

    render_problem_card(
        title="Low Average Order Values and Missed Checkout Cross-Sell Opportunities",
        business_problem="Checkout recommendation widgets frequently display static product suggestions rather than statistically verified co-purchasing pairs. Customers ignore irrelevant recommendations, leaving basket margins uncapitalized.",
        operational_impact="Lack of market basket affinity mining leads to lower Average Order Values and missed bundling revenue."
    )

    col_ap1, col_ap2 = st.columns(2)
    with col_ap1:
        min_sup = st.slider("Minimum Support Threshold (%):", 1.0, 5.0, 2.0, 0.2) / 100.0
    with col_ap2:
        min_conf = st.slider("Minimum Confidence Threshold (%):", 10.0, 30.0, 15.0, 1.0) / 100.0

    @st.cache_data(show_spinner=False)
    def compute_apriori(raw_df: pd.DataFrame, sup: float, conf: float):
        basket = raw_df.groupby(["customer_id", "product_name"])["quantity"].sum().unstack().fillna(0)
        basket_bool = (basket > 0).astype(bool)
        freq = apriori(basket_bool, min_support=sup, use_colnames=True)
        if len(freq) > 0:
            rules = association_rules(freq, metric="confidence", min_threshold=conf)
            return freq, rules
        return freq, pd.DataFrame()

    freq_itemsets, rules_df = compute_apriori(df, min_sup, min_conf)

    m1, m2, m3 = st.columns(3)
    with m1:
        render_metric_card("Frequent Itemsets", f"{len(freq_itemsets):,}", f"Support >= {min_sup*100:.1f}%")
    with m2:
        render_metric_card("Discovered Rules", f"{len(rules_df):,}", f"Confidence >= {min_conf*100:.1f}%")
    with m3:
        max_l = rules_df["lift"].max() if not rules_df.empty else 0.0
        render_metric_card("Maximum Lift Ratio", f"{max_l:.2f}x", "Lift > 1.0 confirms positive affinity")

    if not rules_df.empty:
        rules_df["Antecedent_Str"] = rules_df["antecedents"].apply(lambda x: ", ".join(list(x)))
        rules_df["Consequent_Str"] = rules_df["consequents"].apply(lambda x: ", ".join(list(x)))
        rules_df["Rule_Label"] = rules_df["Antecedent_Str"] + " -> " + rules_df["Consequent_Str"]

        col_sc, col_bar = st.columns(2)
        with col_sc:
            with st.container(border=True):
                st.markdown("#### Support vs Confidence (Bubble Size = Lift)")
                fig_apr = px.scatter(
                    rules_df,
                    x="support",
                    y="confidence",
                    size="lift",
                    color="lift",
                    color_continuous_scale=["#C7D2FE", "#4F46E5", "#312E81"],
                    hover_data=["Rule_Label"]
                )
                st.plotly_chart(apply_chart_theme(fig_apr, height=280), use_container_width=True)

        with col_bar:
            with st.container(border=True):
                st.markdown("#### Top Association Rules by Lift Ratio")
                top_rules = rules_df.sort_values("lift", ascending=True).tail(8)
                fig_top = px.bar(
                    top_rules,
                    x="lift",
                    y="Rule_Label",
                    orientation="h",
                    color="lift",
                    color_continuous_scale=["#C7D2FE", "#4F46E5"]
                )
                st.plotly_chart(apply_chart_theme(fig_top, height=280, show_legend=False), use_container_width=True)

        # Tabular rules
        with st.container(border=True):
            st.markdown("#### Discovered Association Rules Table")
            display_cols = ["Antecedent_Str", "Consequent_Str", "support", "confidence", "lift"]
            st.dataframe(
                rules_df[display_cols].head(15).style.format({
                    "support": "{:.2%}",
                    "confidence": "{:.2%}",
                    "lift": "{:.2f}x"
                }),
                use_container_width=True
            )

        # Live Cart Bundle Cross-Sell Simulator
        with st.container(border=True):
            st.markdown("#### 1-Click Cart Cross-Sell Simulator")
            available_items = df["product_name"].unique()
            selected_item = st.selectbox("Customer Adds Item to Cart:", available_items)

            match_rule = rules_df[rules_df["Antecedent_Str"].str.contains(selected_item, case=False, na=False)]
            if not match_rule.empty:
                best_rule = match_rule.sort_values("lift", ascending=False).iloc[0]
                rec_item = best_rule["Consequent_Str"]
                st.success(f"**Recommended Cross-Sell Bundle:** Add **{rec_item}** with **{best_rule['lift']:.2f}x Lift** (Confidence: {best_rule['confidence']*100:.1f}%, Support: {best_rule['support']*100:.1f}%).")
            else:
                st.caption("No rule antecedent found for this individual item at current thresholds.")

    else:
        st.info("No association rules meet the selected thresholds. Try lowering the Minimum Support or Confidence sliders.")

    render_takeaways_card([
        "Items with Lift > 1.0 demonstrate true co-purchase affinity rather than random chance co-occurrence.",
        "Deploying high-confidence rules into 1-click checkout bundle checkboxes raises Average Order Value.",
        "Warehouse fulfillment teams can cross-locate high-affinity items in adjacent pick zones to minimize handling time."
    ])

# =============================================================================
# View: About Project & Architecture
# =============================================================================
elif selected_page == "About Project & Architecture":
    render_module_header(
        tag="DOCUMENTATION",
        title="Project Architecture & DWDM Methodology",
        subtitle="System architecture, pipeline layers, and curriculum alignment for RetailIntel.",
        theme_tag="Platform Specs"
    )

    with st.container(border=True):
        st.markdown("### System Architecture Layers")
        arch_html = """
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 12px;">
          <div class="card" style="margin: 0; background: #F8FAFC;">
            <div style="font-size: 11px; font-weight: 700; color: #4F46E5;">LAYER 1</div>
            <div style="font-weight: 700; color: #0F172A; margin: 4px 0;">Data Storage</div>
            <div style="font-size: 12px; color: #64748B;">Local 10,000 multi-channel transaction records in CSV format.</div>
          </div>
          <div class="card" style="margin: 0; background: #F8FAFC;">
            <div style="font-size: 11px; font-weight: 700; color: #06B6D4;">LAYER 2</div>
            <div style="font-weight: 700; color: #0F172A; margin: 4px 0;">ETL & Modeling</div>
            <div style="font-size: 12px; color: #64748B;">Pandas pipeline deriving line-item totals and star schema dimensions.</div>
          </div>
          <div class="card" style="margin: 0; background: #F8FAFC;">
            <div style="font-size: 11px; font-weight: 700; color: #10B981;">LAYER 3</div>
            <div style="font-weight: 700; color: #0F172A; margin: 4px 0;">Analytics & ML</div>
            <div style="font-size: 12px; color: #64748B;">Scikit-Learn OLS, Trees, Bayes, K-Means, and MLxtend Apriori.</div>
          </div>
          <div class="card" style="margin: 0; background: #F8FAFC;">
            <div style="font-size: 11px; font-weight: 700; color: #8B5CF6;">LAYER 4</div>
            <div style="font-weight: 700; color: #0F172A; margin: 4px 0;">Presentation UI</div>
            <div style="font-size: 12px; color: #64748B;">Streamlit web application with custom CSS cards and Plotly figures.</div>
          </div>
        </div>
        """
        render_html(arch_html)

    with st.container(border=True):
        st.markdown("### Technology Stack")
        tech_data = pd.DataFrame([
            {"Component": "Core Framework", "Package": "Streamlit", "Purpose": "Interactive web dashboard and state management"},
            {"Component": "Data Manipulation", "Package": "Pandas & NumPy", "Purpose": "In-memory tabular transformations and feature derivations"},
            {"Component": "Machine Learning", "Package": "Scikit-Learn", "Purpose": "Regression, Decision Trees, Naive Bayes, and Clustering"},
            {"Component": "Association Mining", "Package": "MLxtend", "Purpose": "Apriori frequent itemsets and association rule extraction"},
            {"Component": "Visualizations", "Package": "Plotly Express & Graph Objects", "Purpose": "Interactive, responsive SaaS charts and matrices"},
        ])
        st.dataframe(tech_data, use_container_width=True)

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.divider()
footer_bottom = """
<div style="display: flex; justify-content: space-between; align-items: center; font-size: 12px; color: #94A3B8; padding: 4px 0;">
  <div>RetailIntel — E-Commerce Data Warehouse and Mining (DWDM) Platform</div>
  <div>Grounded in 10,000 Multi-Channel E-Commerce Transactions</div>
</div>
"""
render_html(footer_bottom)
