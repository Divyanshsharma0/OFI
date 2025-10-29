"""
NexGen Logistics - Delivery Delay Prediction System
A comprehensive Streamlit application for predicting delivery delays and optimizing logistics operations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="NexGen Logistics - Delivery Delay Predictor",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional, aesthetic, minimal calm UI
st.markdown("""
    <style>
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #f0f3f8 100%);
    }
    
    /* Main header - Minimal and elegant */
    .main-header {
        font-size: 2.2rem;
        font-weight: 600;
        color: #000000;
        text-align: center;
        padding: 1.5rem 1rem;
        background: linear-gradient(90deg, #ffffff 0%, #f8fafb 100%);
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(26, 31, 54, 0.06);
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    /* Subtitle */
    .subtitle {
        color: #6b7280;
        font-size: 0.95rem;
        font-weight: 400;
        letter-spacing: 0.3px;
        margin-top: -0.5rem;
    }
    
    /* Metric cards - Clean and minimal */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #fafbfc 100%);
        padding: 1.25rem 1rem;
        border-radius: 10px;
        border: 1px solid #e5e9f0;
        color: #1a1f36;
        text-align: center;
        box-shadow: 0 1px 3px rgba(26, 31, 54, 0.04);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: #c5d0dd;
        box-shadow: 0 4px 12px rgba(26, 31, 54, 0.08);
        transform: translateY(-2px);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 600;
        margin: 0.5rem 0 0.25rem 0;
        color: #2563eb;
    }
    
    .metric-label {
        font-size: 0.8rem;
        font-weight: 500;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }
    
    .metric-sub {
        font-size: 0.75rem;
        color: #9ca3af;
        margin-top: 0.3rem;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #000000;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #e5e9f0;
    }

    /* Global heading visibility */
    h1, h2, h3, h4, h5, h6,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
    .stMarkdown h4, .stMarkdown h5, .stMarkdown h6,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4,
    [data-testid="stMarkdownContainer"] h5,
    [data-testid="stMarkdownContainer"] h6 {
        color: #1f2937 !important;
    }

    .stMarkdown strong {
        color: #1f2937;
    }
    
    /* Sidebar styling */
    .stSidebar {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafb 100%);
        border-right: 1px solid #e5e9f0;
    }
    
    .stSidebar [data-testid="stMarkdownContainer"] {
        color: #374151;
    }
    
    /* Tab styling - Minimal */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: linear-gradient(90deg, #ffffff 0%, #f8fafb 100%);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e5e9f0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f3f4f6;
        border: 1px solid #e5e9f0;
        border-radius: 8px;
        padding: 0.8rem 1.5rem;
        font-weight: 500;
        color: #6b7280;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2563eb;
        color: white;
        border-color: #2563eb;
    }
    
    /* Button styling - Minimal */
    .stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
    
    /* Input styling */
    .stSelectbox, .stTextInput, .stNumberInput {
        background-color: #f9fafb;
        border: 1px solid #e5e9f0 !important;
        border-radius: 8px;
    }
    
    /* Alert/Warning boxes - Calm colors */
    .stAlert[data-testid="stAlert"] {
        background: linear-gradient(135deg, #fef3c7 0%, #fef9e7 100%);
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        color: #92400e;
    }
    
    .stAlert[data-testid="stAlert"] > div > div > div {
        color: #92400e;
    }
    
    /* Success alerts */
    [data-testid="stAlert"] {
        border-radius: 8px;
        border-left: 4px solid #10b981;
    }
    
    /* Professional text */
    .professional-text {
        font-size: 0.95rem;
        line-height: 1.5;
        color: #6b7280;
        font-weight: 400;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #e5e9f0;
    }
    
    /* Divider */
    .divider {
        border-top: 1px solid #e5e9f0;
        margin: 1.5rem 0;
    }
    
    /* Container styling */
    .stContainer {
        background: #ffffff;
        border-radius: 10px;
        padding: 1.5rem;
        border: 1px solid #e5e9f0;
    }
    
    /* Color coding for values */
    .value-success {
        color: #10b981;
        font-weight: 600;
    }
    
    .value-warning {
        color: #f59e0b;
        font-weight: 600;
    }
    
    .value-danger {
        color: #ef4444;
        font-weight: 600;
    }
    
    /* Minimal spacing */
    .stMarkdown {
        line-height: 1.6;
    }
    </style>
""", unsafe_allow_html=True)

# Load and cache data
@st.cache_data
def load_data():
    """Load all datasets and perform initial preprocessing"""
    try:
        # Load all datasets
        orders = pd.read_csv('dataset/orders.csv')
        delivery = pd.read_csv('dataset/delivery_performance.csv')
        routes = pd.read_csv('dataset/routes_distance.csv')
        vehicles = pd.read_csv('dataset/vehicle_fleet.csv')
        costs = pd.read_csv('dataset/cost_breakdown.csv')
        
        return orders, delivery, routes, vehicles, costs
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None

@st.cache_data
def prepare_data(orders, delivery, routes, vehicles, costs):
    """Merge and engineer features for analysis and modeling"""
    
    # Merge datasets
    df = orders.merge(delivery, on='Order_ID', how='left')
    df = df.merge(routes, on='Order_ID', how='left')
    df = df.merge(costs, on='Order_ID', how='left')
    
    # Feature Engineering
    
    # 1. Delay metrics
    df['delay_days'] = df['Actual_Delivery_Days'] - df['Promised_Delivery_Days']
    df['on_time'] = (df['delay_days'] <= 0).astype(int)
    df['delay_category'] = pd.cut(df['delay_days'], 
                                   bins=[-np.inf, 0, 2, 5, np.inf],
                                   labels=['On-Time', 'Slight', 'Moderate', 'Severe'])
    
    # 2. Cost metrics
    df['total_cost'] = (df['Fuel_Cost'] + df['Labor_Cost'] + 
                       df['Vehicle_Maintenance'] + df['Insurance'] + 
                       df['Packaging_Cost'] + df['Technology_Platform_Fee'] + 
                       df['Other_Overhead'])
    df['cost_per_km'] = df['total_cost'] / (df['Distance_KM'] + 1)  # +1 to avoid division by zero
    df['profit_margin'] = df['Order_Value_INR'] - df['total_cost'] - df['Delivery_Cost_INR']
    
    # 3. Efficiency metrics
    df['fuel_efficiency'] = df['Distance_KM'] / (df['Fuel_Consumption_L'] + 1)
    df['cost_efficiency'] = df['Order_Value_INR'] / (df['total_cost'] + 1)
    
    # 4. Traffic impact
    df['traffic_hours'] = df['Traffic_Delay_Minutes'] / 60
    df['high_traffic'] = (df['Traffic_Delay_Minutes'] > 60).astype(int)
    
    # 5. Date features
    df['Order_Date'] = pd.to_datetime(df['Order_Date'])
    df['day_of_week'] = df['Order_Date'].dt.dayofweek
    df['month'] = df['Order_Date'].dt.month
    df['week_of_year'] = df['Order_Date'].dt.isocalendar().week
    
    # 6. Priority encoding
    priority_map = {'Express': 3, 'Standard': 2, 'Economy': 1}
    df['priority_score'] = df['Priority'].map(priority_map)
    
    # 7. Risk factors
    df['distance_risk'] = pd.cut(df['Distance_KM'], 
                                 bins=[0, 500, 1000, 2000, np.inf],
                                 labels=[1, 2, 3, 4])
    df['distance_risk'] = df['distance_risk'].astype(float)
    
    # 8. Quality issues
    df['has_quality_issue'] = (df['Quality_Issue'] != 'Perfect').astype(int)
    
    # 9. Carrier performance (historical average)
    carrier_performance = df.groupby('Carrier')['on_time'].mean().to_dict()
    df['carrier_reliability'] = df['Carrier'].map(carrier_performance)
    
    # 10. Route complexity score
    df['route_complexity'] = (
        (df['Distance_KM'] / df['Distance_KM'].max()) * 0.4 +
        (df['Traffic_Delay_Minutes'] / df['Traffic_Delay_Minutes'].max()) * 0.3 +
        (df['Fuel_Consumption_L'] / df['Fuel_Consumption_L'].max()) * 0.3
    )
    
    return df

@st.cache_data
def get_kpis(df):
    """Calculate key performance indicators"""
    total_orders = len(df)
    on_time_pct = (df['on_time'].sum() / total_orders) * 100
    avg_delay = df['delay_days'].mean()
    avg_rating = df['Customer_Rating'].mean()
    total_revenue = df['Order_Value_INR'].sum()
    total_profit = df['profit_margin'].sum()
    avg_cost_per_km = df['cost_per_km'].mean()
    
    return {
        'total_orders': total_orders,
        'on_time_pct': on_time_pct,
        'avg_delay': avg_delay,
        'avg_rating': avg_rating,
        'total_revenue': total_revenue,
        'total_profit': total_profit,
        'avg_cost_per_km': avg_cost_per_km
    }

def main():
    """Main application function"""
    
    # Header with improved styling
    st.markdown('<h1 class="main-header">🚚 NexGen Logistics Delivery Intelligence</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-Powered Delivery Delay Prediction & Optimization Platform</p>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading data..."):
        orders, delivery, routes, vehicles, costs = load_data()
        
        if orders is None:
            st.error("Failed to load data. Please check if all CSV files are in the 'dataset' folder.")
            return
        
        df = prepare_data(orders, delivery, routes, vehicles, costs)
        kpis = get_kpis(df)
    
    # Sidebar with professional styling
    st.sidebar.title("⚙️ Dashboard Controls")
    st.sidebar.markdown('<p style="color: #666; font-size: 0.9rem; margin-bottom: 1rem;">Configure your dashboard view and filters</p>', 
                       unsafe_allow_html=True)
    st.sidebar.markdown("---")
    
    # Page selection
    page = st.sidebar.selectbox(
        "Select Page",
        ["📈 Overview Dashboard", "🎯 Delay Prediction", "📊 Deep Analytics", "🔍 Order Lookup"]
    )
    
    # Filters
    st.sidebar.markdown("### 🔎 Filters")
    
    # Date range filter
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(df['Order_Date'].min(), df['Order_Date'].max()),
        min_value=df['Order_Date'].min(),
        max_value=df['Order_Date'].max()
    )
    
    # Priority filter
    priorities = st.sidebar.multiselect(
        "Priority",
        options=df['Priority'].unique(),
        default=df['Priority'].unique()
    )
    
    # Carrier filter
    carriers = st.sidebar.multiselect(
        "Carrier",
        options=df['Carrier'].unique(),
        default=df['Carrier'].unique()
    )
    
    # Apply filters
    if len(date_range) == 2:
        mask = (
            (df['Order_Date'] >= pd.Timestamp(date_range[0])) &
            (df['Order_Date'] <= pd.Timestamp(date_range[1])) &
            (df['Priority'].isin(priorities)) &
            (df['Carrier'].isin(carriers))
        )
        df_filtered = df[mask].copy()
    else:
        df_filtered = df.copy()
    
    # Update KPIs with filtered data
    kpis_filtered = get_kpis(df_filtered)
    
    # Route to selected page
    if page == "📈 Overview Dashboard":
        show_overview(df_filtered, kpis_filtered)
    elif page == "🎯 Delay Prediction":
        show_prediction(df, df_filtered, vehicles)
    elif page == "📊 Deep Analytics":
        show_analytics(df_filtered)
    elif page == "🔍 Order Lookup":
        show_order_lookup(df)

def show_overview(df, kpis):
    """Display the overview dashboard with KPIs and main visualizations"""
    
    st.header("📊 Performance Overview")
    st.markdown('<p class="professional-text">Real-time insights into delivery performance and operational metrics</p>', 
                unsafe_allow_html=True)
    
    # KPI Cards with improved styling
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📦 Total Orders</div>
            <div class="metric-value">{kpis['total_orders']:,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        color = "#10b981" if kpis['on_time_pct'] >= 70 else "#ef4444"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">✅ On-Time Delivery %</div>
            <div class="metric-value" style="color: {color};">{kpis['on_time_pct']:.1f}%</div>
            <div class="metric-sub">Target: 90%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        color = "#10b981" if kpis['avg_delay'] <= 0 else "#ef4444"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">⏱️ Avg Delay</div>
            <div class="metric-value" style="color: {color};">{kpis['avg_delay']:.2f}d</div>
            <div class="metric-sub">Days</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        color = "#10b981" if kpis['avg_rating'] >= 4.0 else "#f59e0b"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">⭐ Avg Rating</div>
            <div class="metric-value" style="color: {color};">{kpis['avg_rating']:.2f}</div>
            <div class="metric-sub">Out of 5</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Second row of KPIs
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">💰 Total Revenue</div>
            <div class="metric-value">₹{kpis['total_revenue']/1000:.1f}K</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col6:
        color = "#10b981" if kpis['total_profit'] > 0 else "#ef4444"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📈 Total Profit</div>
            <div class="metric-value" style="color: {color};">₹{kpis['total_profit']/1000:.1f}K</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col7:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🚗 Cost per KM</div>
            <div class="metric-value">₹{kpis['avg_cost_per_km']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col8:
        delayed_orders = len(df[df['delay_days'] > 0])
        delay_pct = (delayed_orders/len(df)*100)
        color = "#10b981" if delay_pct < 30 else ("#f59e0b" if delay_pct < 50 else "#ef4444")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">⚠️ Delayed Orders</div>
            <div class="metric-value" style="color: {color};">{delayed_orders}</div>
            <div class="metric-sub">{delay_pct:.1f}% of total</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main visualizations
    st.subheader("📊 Key Visualizations")
    col1, col2 = st.columns(2)
    
    with col1:
        # Delivery status distribution
        with st.container():
            st.markdown("**📊 Delivery Status Distribution**")
        status_counts = df['Delivery_Status'].value_counts()
        fig = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Orders by Delivery Status",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Delay by carrier
        with st.container():
            st.markdown("**📦 Average Delay by Carrier**")
        carrier_delay = df.groupby('Carrier')['delay_days'].mean().sort_values(ascending=False)
        fig = px.bar(
            x=carrier_delay.index,
            y=carrier_delay.values,
            title="Carrier Performance Comparison",
            labels={'x': 'Carrier', 'y': 'Average Delay (days)'},
            color=carrier_delay.values,
            color_continuous_scale='RdYlGn_r'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Distance vs Delay scatter
    st.markdown("---")
    with st.container():
        st.markdown("**🗺️ Distance vs Delivery Delay Analysis**")
    fig = px.scatter(
        df,
        x='Distance_KM',
        y='delay_days',
        color='Priority',
        size='Order_Value_INR',
        hover_data=['Order_ID', 'Carrier', 'Customer_Rating'],
        title="Impact of Distance on Delivery Delays",
        labels={'Distance_KM': 'Distance (KM)', 'delay_days': 'Delay (days)'}
    )
    fig.add_hline(y=0, line_dash="dash", line_color="green", annotation_text="On-Time Threshold")
    st.plotly_chart(fig, use_container_width=True)
    
    # Time series
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container():
            st.markdown("**📅 Delivery Performance Over Time**")
        daily_performance = df.groupby(df['Order_Date'].dt.date).agg({
            'on_time': 'mean',
            'Order_ID': 'count'
        }).reset_index()
        daily_performance.columns = ['Date', 'On-Time Rate', 'Order Count']
        
        fig = px.line(
            daily_performance,
            x='Date',
            y='On-Time Rate',
            title="Daily On-Time Delivery Rate",
            labels={'On-Time Rate': 'On-Time Rate (%)'}
        )
        fig.update_traces(line_color='#1f77b4', line_width=3)
        fig.update_yaxes(tickformat='.0%')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🚦 Traffic Impact on Delays")
        fig = px.scatter(
            df,
            x='Traffic_Delay_Minutes',
            y='delay_days',
            color='Delivery_Status',
            title="Traffic Delays vs Delivery Delays",
            labels={'Traffic_Delay_Minutes': 'Traffic Delay (minutes)', 'delay_days': 'Delivery Delay (days)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Quality issues analysis
    st.subheader("🔍 Quality Issues Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        quality_counts = df['Quality_Issue'].value_counts()
        fig = px.bar(
            x=quality_counts.index,
            y=quality_counts.values,
            title="Quality Issues Distribution",
            labels={'x': 'Issue Type', 'y': 'Count'},
            color=quality_counts.values,
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Customer rating distribution
        rating_counts = df['Customer_Rating'].value_counts().sort_index()
        fig = px.bar(
            x=rating_counts.index,
            y=rating_counts.values,
            title="Customer Rating Distribution",
            labels={'x': 'Rating', 'y': 'Count'},
            color=rating_counts.index,
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)

def show_prediction(df_full, df_filtered, vehicles):
    """Machine Learning prediction page"""
    
    st.header("🎯 Delivery Delay Prediction & Recommendations")
    st.markdown('<p class="professional-text">AI-powered machine learning model to identify at-risk orders and suggest corrective actions</p>', 
                unsafe_allow_html=True)
    
    # Import ML libraries
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
    from sklearn.preprocessing import LabelEncoder
    
    # Prepare features for modeling
    @st.cache_resource
    def train_model(df):
        """Train the delay prediction model"""
        
        # Select features for modeling
        feature_cols = [
            'Distance_KM', 'Fuel_Consumption_L', 'Toll_Charges_INR',
            'Traffic_Delay_Minutes', 'priority_score', 'Order_Value_INR',
            'total_cost', 'cost_per_km', 'fuel_efficiency',
            'day_of_week', 'month', 'carrier_reliability', 'route_complexity'
        ]
        
        # Handle categorical variables
        df_model = df.copy()
        
        # Prepare data
        X = df_model[feature_cols].fillna(0)
        y = df_model['on_time']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            class_weight='balanced'
        )
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return model, feature_importance, X_test, y_test, y_pred, y_pred_proba, feature_cols
    
    with st.spinner("Training prediction model..."):
        model, feature_importance, X_test, y_test, y_pred, y_pred_proba, feature_cols = train_model(df_full)
    
    # Model Performance
    st.subheader("📊 Model Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    with col1:
        st.metric("🎯 Accuracy", f"{accuracy:.2%}")
        st.metric("✅ Precision", f"{precision:.2%}")
    
    with col2:
        st.metric("🔍 Recall", f"{recall:.2%}")
        st.metric("⚖️ F1 Score", f"{f1:.2%}")
    
    with col3:
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
            st.metric("📈 ROC AUC", f"{auc:.2%}")
        except:
            st.metric("📈 ROC AUC", "N/A")
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Confusion Matrix
        st.subheader("🔢 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Delayed', 'On-Time'],
            y=['Delayed', 'On-Time'],
            text_auto=True,
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Feature Importance
        st.subheader("⭐ Top Feature Importance")
        fig = px.bar(
            feature_importance.head(10),
            x='importance',
            y='feature',
            orientation='h',
            title="Key Factors Affecting Delays",
            labels={'importance': 'Importance Score', 'feature': 'Feature'},
            color='importance',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ROC Curve
    try:
        st.subheader("📉 ROC Curve")
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC = {auc:.2f})',
                                line=dict(color='#1f77b4', width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier',
                                line=dict(color='red', width=2, dash='dash')))
        fig.update_layout(
            title='ROC Curve - Model Performance',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    except:
        pass
    
    st.markdown("---")
    
    # Prediction Interface
    st.subheader("🔮 Predict Delay Risk for New Orders")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        distance = st.number_input("Distance (KM)", min_value=0.0, max_value=5000.0, value=500.0)
        fuel = st.number_input("Fuel Consumption (L)", min_value=0.0, max_value=500.0, value=50.0)
        traffic = st.number_input("Traffic Delay (minutes)", min_value=0, max_value=300, value=30)
    
    with col2:
        order_value = st.number_input("Order Value (INR)", min_value=0.0, max_value=50000.0, value=1000.0)
        priority = st.selectbox("Priority", ['Express', 'Standard', 'Economy'])
        toll = st.number_input("Toll Charges (INR)", min_value=0.0, max_value=2000.0, value=100.0)
    
    with col3:
        carrier = st.selectbox("Carrier", df_full['Carrier'].unique())
        day_of_week = st.selectbox("Day of Week", 
                                   ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        month = st.slider("Month", 1, 12, 10)
    
    if st.button("🔍 Predict Delay Risk", type="primary"):
        # Prepare input
        priority_map = {'Express': 3, 'Standard': 2, 'Economy': 1}
        day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
                  'Friday': 4, 'Saturday': 5, 'Sunday': 6}
        
        carrier_reliability = df_full.groupby('Carrier')['on_time'].mean().get(carrier, 0.5)
        
        # Calculate derived features
        total_cost_est = (fuel * 50) + (distance * 0.5) + toll + 200  # Simplified estimation
        cost_per_km = total_cost_est / (distance + 1)
        fuel_efficiency = distance / (fuel + 1)
        route_complexity = (distance / 5000) * 0.4 + (traffic / 300) * 0.3 + (fuel / 500) * 0.3
        
        input_data = pd.DataFrame({
            'Distance_KM': [distance],
            'Fuel_Consumption_L': [fuel],
            'Toll_Charges_INR': [toll],
            'Traffic_Delay_Minutes': [traffic],
            'priority_score': [priority_map[priority]],
            'Order_Value_INR': [order_value],
            'total_cost': [total_cost_est],
            'cost_per_km': [cost_per_km],
            'fuel_efficiency': [fuel_efficiency],
            'day_of_week': [day_map[day_of_week]],
            'month': [month],
            'carrier_reliability': [carrier_reliability],
            'route_complexity': [route_complexity]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 1:
                st.success("✅ **Predicted: ON-TIME DELIVERY**")
            else:
                st.error("⚠️ **Predicted: DELAY RISK**")
        
        with col2:
            st.metric("On-Time Probability", f"{probability[1]:.1%}")
        
        with col3:
            st.metric("Delay Risk", f"{probability[0]:.1%}")
        
        # Risk gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability[0] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Delay Risk Score"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred" if probability[0] > 0.5 else "green"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.subheader("💡 Recommended Actions")
        
        if probability[0] > 0.5:  # High delay risk
            st.warning("**⚠️ High Risk of Delay Detected!**")
            
            recommendations = []
            
            if traffic > 60:
                recommendations.append("🚦 **High traffic detected**: Consider rescheduling delivery to off-peak hours or using alternative routes")
            
            if distance > 1000:
                recommendations.append("🗺️ **Long distance route**: Consider breaking into multiple segments or using faster vehicle type")
            
            if priority == 'Express':
                recommendations.append("⚡ **Express priority**: Assign to most reliable carrier and premium vehicle")
            
            if carrier_reliability < 0.7:
                recommendations.append(f"📦 **Low carrier reliability ({carrier_reliability:.1%})**: Consider switching to a more reliable carrier")
            
            if fuel_efficiency < 8:
                recommendations.append("⛽ **Low fuel efficiency**: Use more efficient vehicle to reduce costs and improve speed")
            
            # Vehicle recommendations
            st.markdown("**🚗 Recommended Vehicles:**")
            available_vehicles = vehicles[vehicles['Status'] == 'Available'].sort_values('Fuel_Efficiency_KM_per_L', ascending=False)
            if len(available_vehicles) > 0:
                for idx, row in available_vehicles.head(3).iterrows():
                    st.info(f"• {row['Vehicle_ID']} - {row['Vehicle_Type']} (Efficiency: {row['Fuel_Efficiency_KM_per_L']:.2f} km/L, Location: {row['Current_Location']})")
            
            for rec in recommendations:
                st.markdown(f"• {rec}")
        else:
            st.success("**✅ Low Risk - Proceed with Standard Process**")
            st.markdown("• Current parameters are optimal for on-time delivery")
            st.markdown("• Continue monitoring traffic and weather conditions")
            st.markdown("• Maintain current carrier and vehicle assignment")
    
    # At-risk orders identification
    st.markdown("---")
    st.subheader("⚠️ Current At-Risk Orders")
    
    # Predict for all filtered orders
    df_predict = df_filtered.copy()
    X_predict = df_predict[feature_cols].fillna(0)
    df_predict['delay_risk'] = model.predict_proba(X_predict)[:, 0]
    df_predict['predicted_on_time'] = model.predict(X_predict)
    
    # Show high-risk orders
    at_risk = df_predict[df_predict['delay_risk'] > 0.6].sort_values('delay_risk', ascending=False)
    
    if len(at_risk) > 0:
        st.warning(f"Found {len(at_risk)} orders with high delay risk (>60%)")
        
        display_cols = ['Order_ID', 'Order_Date', 'Priority', 'Carrier', 'Distance_KM', 
                       'Traffic_Delay_Minutes', 'delay_risk', 'Delivery_Status']
        
        at_risk_display = at_risk[display_cols].copy()
        at_risk_display['delay_risk'] = at_risk_display['delay_risk'].apply(lambda x: f"{x:.1%}")
        at_risk_display = at_risk_display.rename(columns={
            'Order_ID': 'Order ID',
            'Order_Date': 'Date',
            'Distance_KM': 'Distance (KM)',
            'Traffic_Delay_Minutes': 'Traffic Delay (min)',
            'delay_risk': 'Delay Risk',
            'Delivery_Status': 'Status'
        })
        
        st.dataframe(at_risk_display.head(20), use_container_width=True)
        
        # Download button
        csv = at_risk.to_csv(index=False)
        st.download_button(
            label="📥 Download At-Risk Orders CSV",
            data=csv,
            file_name="at_risk_orders.csv",
            mime="text/csv"
        )
    else:
        st.success("✅ No high-risk orders found in current filter!")

def show_analytics(df):
    """Deep analytics page with advanced visualizations"""
    
    st.header("📊 Deep Analytics & Insights")
    st.markdown('<p class="professional-text">Advanced analysis of operational metrics, trends, and performance indicators</p>', 
                unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "🗺️ Routes", "💰 Financial", "🎯 Performance"])
    
    with tab1:
        st.subheader("Time-based Trends Analysis")
        
        # Weekly trends
        col1, col2 = st.columns(2)
        
        with col1:
            weekly_data = df.groupby('week_of_year').agg({
                'on_time': 'mean',
                'delay_days': 'mean',
                'Order_ID': 'count'
            }).reset_index()
            weekly_data.columns = ['Week', 'On-Time Rate', 'Avg Delay', 'Order Count']
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Bar(x=weekly_data['Week'], y=weekly_data['Order Count'], name="Orders",
                      marker_color='lightblue'),
                secondary_y=False
            )
            fig.add_trace(
                go.Scatter(x=weekly_data['Week'], y=weekly_data['On-Time Rate'], 
                          name="On-Time Rate", line=dict(color='green', width=3)),
                secondary_y=True
            )
            fig.update_layout(title="Weekly Order Volume and On-Time Performance")
            fig.update_yaxis(title_text="Order Count", secondary_y=False)
            fig.update_yaxis(title_text="On-Time Rate", secondary_y=True, tickformat='.0%')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Day of week analysis
            dow_data = df.groupby('day_of_week').agg({
                'delay_days': 'mean',
                'on_time': 'mean'
            }).reset_index()
            dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            dow_data['day_name'] = dow_data['day_of_week'].apply(lambda x: dow_names[int(x)] if x < 7 else 'Unknown')
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=dow_data['day_name'],
                y=dow_data['delay_days'],
                marker_color=dow_data['delay_days'],
                marker_colorscale='RdYlGn_r',
                text=dow_data['delay_days'].round(2),
                textposition='auto',
                name='Avg Delay'
            ))
            fig.update_layout(
                title="Average Delay by Day of Week",
                xaxis_title="Day",
                yaxis_title="Average Delay (days)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Monthly trends
        monthly_data = df.groupby('month').agg({
            'Order_Value_INR': 'sum',
            'total_cost': 'sum',
            'profit_margin': 'sum',
            'on_time': 'mean'
        }).reset_index()
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Revenue vs Costs by Month", "On-Time Performance by Month"),
            vertical_spacing=0.15
        )
        
        fig.add_trace(
            go.Bar(x=monthly_data['month'], y=monthly_data['Order_Value_INR'], 
                  name='Revenue', marker_color='green'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=monthly_data['month'], y=monthly_data['total_cost'], 
                  name='Costs', marker_color='red'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=monthly_data['month'], y=monthly_data['on_time'], 
                      name='On-Time Rate', line=dict(color='blue', width=3),
                      mode='lines+markers'),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Month", row=2, col=1)
        fig.update_yaxes(title_text="Amount (INR)", row=1, col=1)
        fig.update_yaxes(title_text="On-Time Rate", row=2, col=1, tickformat='.0%')
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Route and Geographic Analysis")
        
        # Top routes by volume
        df['route_name'] = df['Origin'] + ' → ' + df['Destination']
        route_stats = df.groupby('route_name').agg({
            'Order_ID': 'count',
            'delay_days': 'mean',
            'Distance_KM': 'mean',
            'on_time': 'mean'
        }).reset_index()
        route_stats.columns = ['Route', 'Orders', 'Avg Delay', 'Distance', 'On-Time Rate']
        route_stats = route_stats.sort_values('Orders', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                route_stats.head(15),
                x='Orders',
                y='Route',
                orientation='h',
                title="Top 15 Routes by Volume",
                color='Avg Delay',
                color_continuous_scale='RdYlGn_r',
                labels={'Orders': 'Number of Orders', 'Avg Delay': 'Avg Delay (days)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(
                route_stats,
                x='Distance',
                y='Avg Delay',
                size='Orders',
                color='On-Time Rate',
                hover_data=['Route'],
                title="Route Performance: Distance vs Delay",
                labels={'Distance': 'Distance (KM)', 'Avg Delay': 'Avg Delay (days)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Origin and destination analysis
        col1, col2 = st.columns(2)
        
        with col1:
            origin_stats = df.groupby('Origin')['Order_ID'].count().sort_values(ascending=False)
            fig = px.bar(
                x=origin_stats.head(10).values,
                y=origin_stats.head(10).index,
                orientation='h',
                title="Top 10 Origin Cities",
                labels={'x': 'Number of Orders', 'y': 'City'},
                color=origin_stats.head(10).values,
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            dest_stats = df.groupby('Destination')['Order_ID'].count().sort_values(ascending=False)
            fig = px.bar(
                x=dest_stats.head(10).values,
                y=dest_stats.head(10).index,
                orientation='h',
                title="Top 10 Destination Cities",
                labels={'x': 'Number of Orders', 'y': 'City'},
                color=dest_stats.head(10).values,
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Traffic analysis
        st.subheader("Traffic Impact Analysis")
        traffic_bins = pd.cut(df['Traffic_Delay_Minutes'], bins=[0, 30, 60, 120, 300], 
                             labels=['Low (0-30min)', 'Medium (30-60min)', 'High (60-120min)', 'Severe (>120min)'])
        traffic_analysis = df.groupby(traffic_bins).agg({
            'on_time': 'mean',
            'delay_days': 'mean',
            'Order_ID': 'count'
        }).reset_index()
        traffic_analysis.columns = ['Traffic Level', 'On-Time Rate', 'Avg Delay', 'Count']
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("On-Time Rate by Traffic Level", "Average Delay by Traffic Level")
        )
        
        fig.add_trace(
            go.Bar(x=traffic_analysis['Traffic Level'], y=traffic_analysis['On-Time Rate'],
                  marker_color=['green', 'yellow', 'orange', 'red'], name='On-Time Rate'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=traffic_analysis['Traffic Level'], y=traffic_analysis['Avg Delay'],
                  marker_color=['green', 'yellow', 'orange', 'red'], name='Avg Delay'),
            row=1, col=2
        )
        
        fig.update_yaxes(title_text="On-Time Rate", row=1, col=1, tickformat='.0%')
        fig.update_yaxes(title_text="Delay (days)", row=1, col=2)
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Financial Performance Analysis")
        
        # Cost breakdown
        cost_components = {
            'Fuel': df['Fuel_Cost'].sum(),
            'Labor': df['Labor_Cost'].sum(),
            'Maintenance': df['Vehicle_Maintenance'].sum(),
            'Insurance': df['Insurance'].sum(),
            'Packaging': df['Packaging_Cost'].sum(),
            'Technology': df['Technology_Platform_Fee'].sum(),
            'Other': df['Other_Overhead'].sum()
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                values=list(cost_components.values()),
                names=list(cost_components.keys()),
                title="Cost Breakdown by Category",
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Profitability by segment
            segment_profit = df.groupby('Customer_Segment').agg({
                'Order_Value_INR': 'sum',
                'profit_margin': 'sum',
                'Order_ID': 'count'
            }).reset_index()
            segment_profit['profit_per_order'] = segment_profit['profit_margin'] / segment_profit['Order_ID']
            
            fig = px.bar(
                segment_profit,
                x='Customer_Segment',
                y=['Order_Value_INR', 'profit_margin'],
                title="Revenue vs Profit by Customer Segment",
                barmode='group',
                labels={'value': 'Amount (INR)', 'Customer_Segment': 'Segment'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Category analysis
        category_analysis = df.groupby('Product_Category').agg({
            'Order_Value_INR': 'sum',
            'profit_margin': 'sum',
            'Order_ID': 'count',
            'on_time': 'mean'
        }).reset_index()
        category_analysis['profit_margin_pct'] = (category_analysis['profit_margin'] / 
                                                  category_analysis['Order_Value_INR'] * 100)
        
        fig = px.scatter(
            category_analysis,
            x='Order_Value_INR',
            y='profit_margin_pct',
            size='Order_ID',
            color='on_time',
            text='Product_Category',
            title="Profitability and Performance by Product Category",
            labels={
                'Order_Value_INR': 'Total Revenue (INR)',
                'profit_margin_pct': 'Profit Margin (%)',
                'Order_ID': 'Order Count',
                'on_time': 'On-Time Rate'
            }
        )
        fig.update_traces(textposition='top center')
        st.plotly_chart(fig, use_container_width=True)
        
        # Cost efficiency
        st.subheader("Cost Efficiency Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_cost_per_km = df['cost_per_km'].mean()
            st.metric("Avg Cost per KM", f"₹{avg_cost_per_km:.2f}")
            
            # Cost per KM by priority
            priority_cost = df.groupby('Priority')['cost_per_km'].mean().sort_values()
            fig = px.bar(
                x=priority_cost.values,
                y=priority_cost.index,
                orientation='h',
                title="Cost per KM by Priority",
                labels={'x': 'Cost per KM (INR)', 'y': 'Priority'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            avg_profit_margin_pct = (df['profit_margin'].sum() / df['Order_Value_INR'].sum() * 100)
            st.metric("Avg Profit Margin", f"{avg_profit_margin_pct:.1f}%")
            
            # Profit distribution
            fig = px.histogram(
                df,
                x='profit_margin',
                nbins=50,
                title="Profit Margin Distribution",
                labels={'profit_margin': 'Profit Margin (INR)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            total_profit = df['profit_margin'].sum()
            st.metric("Total Profit", f"₹{total_profit/1000:.1f}K")
            
            # Profit trend
            profit_trend = df.groupby(df['Order_Date'].dt.date)['profit_margin'].sum().reset_index()
            profit_trend.columns = ['Date', 'Profit']
            fig = px.line(
                profit_trend,
                x='Date',
                y='Profit',
                title="Daily Profit Trend",
                labels={'Profit': 'Profit (INR)'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Carrier and Quality Performance")
        
        # Carrier comparison
        carrier_stats = df.groupby('Carrier').agg({
            'on_time': 'mean',
            'delay_days': 'mean',
            'Customer_Rating': 'mean',
            'Order_ID': 'count',
            'Delivery_Cost_INR': 'mean'
        }).reset_index()
        carrier_stats.columns = ['Carrier', 'On-Time Rate', 'Avg Delay', 'Avg Rating', 'Orders', 'Avg Cost']
        
        # Radar chart for carrier comparison
        fig = go.Figure()
        
        for idx, row in carrier_stats.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[
                    row['On-Time Rate'] * 100,
                    (5 - row['Avg Delay']) * 20,  # Inverse and scale
                    row['Avg Rating'] * 20,
                    (row['Orders'] / carrier_stats['Orders'].max()) * 100,
                    100 - (row['Avg Cost'] / carrier_stats['Avg Cost'].max() * 100)  # Inverse cost
                ],
                theta=['On-Time %', 'Speed', 'Rating', 'Volume', 'Cost Efficiency'],
                fill='toself',
                name=row['Carrier']
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            title="Carrier Performance Comparison (Radar Chart)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed carrier table
        st.dataframe(
            carrier_stats.style.format({
                'On-Time Rate': '{:.1%}',
                'Avg Delay': '{:.2f}',
                'Avg Rating': '{:.2f}',
                'Orders': '{:.0f}',
                'Avg Cost': '₹{:.2f}'
            }).background_gradient(subset=['On-Time Rate', 'Avg Rating'], cmap='RdYlGn')
            .background_gradient(subset=['Avg Delay'], cmap='RdYlGn_r'),
            use_container_width=True
        )
        
        # Priority performance
        col1, col2 = st.columns(2)
        
        with col1:
            priority_stats = df.groupby('Priority').agg({
                'on_time': 'mean',
                'delay_days': 'mean',
                'Customer_Rating': 'mean'
            }).reset_index()
            
            fig = px.bar(
                priority_stats,
                x='Priority',
                y=['on_time', 'Customer_Rating'],
                barmode='group',
                title="Performance by Priority Level",
                labels={'value': 'Score', 'variable': 'Metric'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Quality issue impact
            quality_impact = df.groupby('Quality_Issue').agg({
                'Customer_Rating': 'mean',
                'Order_ID': 'count'
            }).reset_index()
            
            fig = px.bar(
                quality_impact,
                x='Quality_Issue',
                y='Customer_Rating',
                title="Customer Rating by Quality Issue",
                color='Customer_Rating',
                color_continuous_scale='RdYlGn',
                labels={'Customer_Rating': 'Avg Rating', 'Quality_Issue': 'Quality Issue Type'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Feature Correlation Analysis")
        
        corr_features = ['Distance_KM', 'Traffic_Delay_Minutes', 'delay_days', 'Customer_Rating',
                        'Order_Value_INR', 'total_cost', 'cost_per_km', 'on_time']
        corr_matrix = df[corr_features].corr()
        
        fig = px.imshow(
            corr_matrix,
            labels=dict(color="Correlation"),
            x=corr_features,
            y=corr_features,
            color_continuous_scale='RdBu_r',
            aspect='auto',
            title="Feature Correlation Heatmap"
        )
        st.plotly_chart(fig, use_container_width=True)

def show_order_lookup(df):
    """Order lookup and details page"""
    
    st.header("🔍 Order Lookup & Details")
    st.markdown('<p class="professional-text">Search for specific orders and view comprehensive information</p>', 
                unsafe_allow_html=True)
    
    # Search options
    search_type = st.radio(
        "Search by:",
        ["Order ID", "Date Range", "Customer Segment", "Route"]
    )
    
    if search_type == "Order ID":
        order_id = st.text_input("Enter Order ID:", placeholder="ORD000001")
        
        if order_id and st.button("Search"):
            order = df[df['Order_ID'] == order_id]
            
            if len(order) > 0:
                order = order.iloc[0]
                
                st.success(f"✅ Found order: {order_id}")
                
                # Order details
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### 📦 Order Information")
                    st.write(f"**Order Date:** {order['Order_Date']}")
                    st.write(f"**Customer Segment:** {order['Customer_Segment']}")
                    st.write(f"**Priority:** {order['Priority']}")
                    st.write(f"**Category:** {order['Product_Category']}")
                    st.write(f"**Value:** ₹{order['Order_Value_INR']:.2f}")
                
                with col2:
                    st.markdown("### 🚚 Delivery Details")
                    st.write(f"**Carrier:** {order['Carrier']}")
                    st.write(f"**Promised Days:** {order['Promised_Delivery_Days']}")
                    st.write(f"**Actual Days:** {order['Actual_Delivery_Days']}")
                    st.write(f"**Status:** {order['Delivery_Status']}")
                    st.write(f"**Rating:** {order['Customer_Rating']} ⭐")
                
                with col3:
                    st.markdown("### 🗺️ Route Information")
                    st.write(f"**Origin:** {order['Origin']}")
                    st.write(f"**Destination:** {order['Destination']}")
                    st.write(f"**Distance:** {order['Distance_KM']:.2f} KM")
                    st.write(f"**Traffic Delay:** {order['Traffic_Delay_Minutes']} min")
                    st.write(f"**Route:** {order['Route']}")
                
                # Cost breakdown
                st.markdown("### 💰 Cost Breakdown")
                
                cost_data = {
                    'Component': ['Fuel', 'Labor', 'Maintenance', 'Insurance', 'Packaging', 'Technology', 'Other', 'Delivery'],
                    'Cost (INR)': [
                        order['Fuel_Cost'],
                        order['Labor_Cost'],
                        order['Vehicle_Maintenance'],
                        order['Insurance'],
                        order['Packaging_Cost'],
                        order['Technology_Platform_Fee'],
                        order['Other_Overhead'],
                        order['Delivery_Cost_INR']
                    ]
                }
                
                cost_df = pd.DataFrame(cost_data)
                
                fig = px.bar(
                    cost_df,
                    x='Component',
                    y='Cost (INR)',
                    title=f"Cost Breakdown for {order_id}",
                    color='Cost (INR)',
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Performance metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    delay = order['delay_days']
                    if delay <= 0:
                        st.success(f"✅ On Time\n\n{delay} days")
                    else:
                        st.error(f"⚠️ Delayed\n\n+{delay} days")
                
                with col2:
                    st.info(f"💵 Profit\n\n₹{order['profit_margin']:.2f}")
                
                with col3:
                    st.info(f"📊 Cost/KM\n\n₹{order['cost_per_km']:.2f}")
                
                with col4:
                    st.info(f"⛽ Efficiency\n\n{order['fuel_efficiency']:.2f} km/L")
                
            else:
                st.error(f"❌ Order ID '{order_id}' not found")
    
    elif search_type == "Date Range":
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input("Start Date", df['Order_Date'].min())
        
        with col2:
            end_date = st.date_input("End Date", df['Order_Date'].max())
        
        if st.button("Search"):
            mask = (df['Order_Date'] >= pd.Timestamp(start_date)) & (df['Order_Date'] <= pd.Timestamp(end_date))
            results = df[mask]
            
            st.success(f"Found {len(results)} orders between {start_date} and {end_date}")
            
            display_cols = ['Order_ID', 'Order_Date', 'Priority', 'Carrier', 'Delivery_Status', 
                          'delay_days', 'Customer_Rating', 'Order_Value_INR']
            
            st.dataframe(results[display_cols].sort_values('Order_Date', ascending=False), 
                        use_container_width=True)
    
    elif search_type == "Customer Segment":
        segment = st.selectbox("Select Segment:", df['Customer_Segment'].unique())
        
        if st.button("Search"):
            results = df[df['Customer_Segment'] == segment]
            
            st.success(f"Found {len(results)} orders for {segment} segment")
            
            # Segment statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Orders", len(results))
            
            with col2:
                st.metric("On-Time %", f"{results['on_time'].mean():.1%}")
            
            with col3:
                st.metric("Avg Rating", f"{results['Customer_Rating'].mean():.2f}")
            
            with col4:
                st.metric("Total Revenue", f"₹{results['Order_Value_INR'].sum()/1000:.1f}K")
            
            # Orders table
            display_cols = ['Order_ID', 'Order_Date', 'Priority', 'Product_Category', 
                          'Delivery_Status', 'Customer_Rating', 'Order_Value_INR']
            
            st.dataframe(results[display_cols].sort_values('Order_Date', ascending=False), 
                        use_container_width=True)
    
    elif search_type == "Route":
        col1, col2 = st.columns(2)
        
        with col1:
            origin = st.selectbox("Origin:", sorted(df['Origin'].unique()))
        
        with col2:
            destination = st.selectbox("Destination:", sorted(df['Destination'].unique()))
        
        if st.button("Search"):
            results = df[(df['Origin'] == origin) & (df['Destination'] == destination)]
            
            if len(results) > 0:
                st.success(f"Found {len(results)} orders for route {origin} → {destination}")
                
                # Route statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Avg Distance", f"{results['Distance_KM'].mean():.1f} KM")
                
                with col2:
                    st.metric("Avg Delay", f"{results['delay_days'].mean():.2f} days")
                
                with col3:
                    st.metric("On-Time %", f"{results['on_time'].mean():.1%}")
                
                with col4:
                    st.metric("Avg Traffic", f"{results['Traffic_Delay_Minutes'].mean():.0f} min")
                
                # Orders table
                display_cols = ['Order_ID', 'Order_Date', 'Priority', 'Carrier', 
                              'Distance_KM', 'delay_days', 'Customer_Rating']
                
                st.dataframe(results[display_cols].sort_values('Order_Date', ascending=False), 
                            use_container_width=True)
            else:
                st.warning(f"No orders found for route {origin} → {destination}")

if __name__ == "__main__":
    main()
