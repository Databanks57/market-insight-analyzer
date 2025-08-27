import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from core_analytical import BusinessAnalytics
from datetime import datetime, timedelta
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Market Insight Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 3px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_analytics_data():
    """Load and cache analytics data"""
    try:
        analytics = BusinessAnalytics("retail_business.db")
        analytics.load_data()
        
        sales_data = analytics.sales_performance_analysis()
        customer_data = analytics.customer_analytics()
        product_data = analytics.product_analysis()
        churn_data = analytics.churn_prediction()
        cohort_data = analytics.cohort_analysis()
        elasticity_data = analytics.price_elasticity_analysis()
        insights = analytics.generate_insights()
        
        analytics.close_connection()
        
        return sales_data, customer_data, product_data, churn_data, cohort_data, elasticity_data, insights
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None, None, None, []

def main():
    st.title("📊 Market Insight Analyzer")
    st.markdown("**Transforming business data into actionable insights**")
    
    # Load data
    sales_data, customer_data, product_data, churn_data, cohort_data, elasticity_data, insights = load_analytics_data()
    
    if sales_data is None:
        st.error("Unable to load data. Please ensure the database file exists.")
        st.info("Run the data generator script first to create sample data.")
        return
    
    # Sidebar
    st.sidebar.title("Navigation")
    selected_page = st.sidebar.selectbox(
        "Choose Analysis",
        ["Executive Summary", "Sales Performance", "Customer Analytics", 
         "Product Intelligence", "Churn Analysis", "Cohort Analysis", "Price Optimization"]
    )
    
    # Executive Summary
    if selected_page == "Executive Summary":
        st.header("🎯 Executive Summary")
        
        # Key Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Revenue",
                value=f"${sales_data['summary_metrics']['total_revenue']:,.0f}",
                delta=None
            )
        
        with col2:
            st.metric(
                label="Total Customers",
                value=f"{sales_data['summary_metrics']['unique_customers']:,}",
                delta=None
            )
            
        with col3:
            st.metric(
                label="Average Order Value",
                value=f"${sales_data['summary_metrics']['avg_order_value']:.2f}",
                delta=None
            )
            
        with col4:
            st.metric(
                label="Total Transactions",
                value=f"{sales_data['summary_metrics']['total_transactions']:,}",
                delta=None
            )
        
        # Insights Section
        st.subheader("🔍 Key Insights")
        for insight in insights:
            st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
        
        # Quick Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Monthly Revenue Trend")
            monthly_data = sales_data['monthly_trends']
            fig = px.line(monthly_data, x='month', y='revenue', 
                         title="Revenue Over Time")
            fig.update_layout(xaxis_title="Month", yaxis_title="Revenue ($)")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Customer Segments")
            segment_data = customer_data['segment_summary']
            fig = px.pie(segment_data, values='customer_id', names='segment',
                        title="Customer Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    # Sales Performance
    elif selected_page == "Sales Performance":
        st.header("📈 Sales Performance Analysis")
        
        # Revenue Trends
        st.subheader("Revenue Trends")
        monthly_data = sales_data['monthly_trends']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Monthly Revenue", "Monthly Growth Rate", 
                           "Transaction Count", "Average Order Value"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Monthly Revenue
        fig.add_trace(
            go.Scatter(x=monthly_data['month'], y=monthly_data['revenue'],
                      mode='lines+markers', name='Revenue'),
            row=1, col=1
        )
        
        # Growth Rate
        fig.add_trace(
            go.Bar(x=monthly_data['month'], y=monthly_data['revenue_growth'],
                   name='Growth %'),
            row=1, col=2
        )
        
        # Transaction Count
        fig.add_trace(
            go.Scatter(x=monthly_data['month'], y=monthly_data['transactions'],
                      mode='lines+markers', name='Transactions'),
            row=2, col=1
        )
        
        # AOV
        fig.add_trace(
            go.Scatter(x=monthly_data['month'], y=monthly_data['aov'],
                      mode='lines+markers', name='AOV'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Day of Week Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sales by Day of Week")
            dow_data = sales_data['dow_analysis'].reset_index()
            fig = px.bar(dow_data, x='day_of_week', y='total_amount',
                        title="Revenue by Day of Week")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Top Performing Products")
            top_products = sales_data['top_products'].head(5)
            fig = px.bar(top_products, x='total_amount', y='product_name',
                        orientation='h', title="Top 5 Products by Revenue")
            st.plotly_chart(fig, use_container_width=True)
    
    # Customer Analytics
    elif selected_page == "Customer Analytics":
        st.header("👥 Customer Analytics")
        
        # Customer Segments Overview
        st.subheader("Customer Segmentation (RFM Analysis)")
        segment_summary = customer_data['segment_summary']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(segment_summary, x='segment', y='customer_id',
                        title="Number of Customers by Segment")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(segment_summary, x='segment', y='clv',
                        title="Average Customer Lifetime Value by Segment")
            st.plotly_chart(fig, use_container_width=True)
        
        # RFM Distribution
        st.subheader("RFM Score Distribution")
        rfm_data = customer_data['rfm_data']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig = px.histogram(rfm_data, x='recency', bins=20, title="Recency Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(rfm_data, x='frequency', bins=20, title="Frequency Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            fig = px.histogram(rfm_data, x='monetary', bins=20, title="Monetary Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        # Customer Segment Details
        st.subheader("Segment Analysis")
        st.dataframe(segment_summary)
        
        # CLV Analysis
        st.subheader("Customer Lifetime Value Analysis")
        clv_data = customer_data['clv_distribution']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(clv_data, x='clv', bins=30, title="CLV Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(clv_data, x='order_count', y='avg_order_value', 
                           size='clv', title="Order Count vs AOV (Size = CLV)")
            st.plotly_chart(fig, use_container_width=True)
    
    # Product Intelligence
    elif selected_page == "Product Intelligence":
        st.header("🛍️ Product Intelligence")
        
        product_performance = product_data['product_performance']
        
        # ABC Analysis
        st.subheader("ABC Analysis (Pareto Principle)")
        abc_summary = product_data['abc_analysis']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(abc_summary, x='abc_category', y='total_revenue',
                        title="Revenue by ABC Category")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.pie(abc_summary, values='product_id', names='abc_category',
                        title="Product Count by ABC Category")
            st.plotly_chart(fig, use_container_width=True)
        
        # Product Performance Matrix
        st.subheader("Product Performance Matrix")
        fig = px.scatter(product_performance, x='total_quantity', y='profit_margin',
                        size='total_revenue', color='category',
                        hover_data=['product_name'],
                        title="Quantity vs Profit Margin (Size = Revenue)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Category Performance
        st.subheader("Category Performance")
        category_perf = product_data['category_performance']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(category_perf, x='category', y='total_amount',
                        title="Revenue by Category")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(category_perf, x='category', y='profit',
                        title="Profit by Category")
            st.plotly_chart(fig, use_container_width=True)
        
        # Cross-selling Opportunities
        st.subheader("Cross-selling Opportunities")
        cross_sell = product_data['cross_selling_opportunities']
        if not cross_sell.empty:
            st.dataframe(cross_sell.head(10))
        else:
            st.info("No significant cross-selling patterns found in the data.")
        
        # Product Performance Table
        st.subheader("Detailed Product Performance")
        st.dataframe(product_performance)
    
    # Churn Analysis
    elif selected_page == "Churn Analysis":
        st.header("⚠️ Customer Churn Analysis")
        
        churn_predictions = churn_data['churn_predictions']
        
        # Churn Overview
        st.subheader("Churn Risk Overview")
        
        col1, col2, col3 = st.columns(3)
        
        high_risk = len(churn_predictions[churn_predictions['risk_category'] == 'High Risk'])
        medium_risk = len(churn_predictions[churn_predictions['risk_category'] == 'Medium Risk'])
        low_risk = len(churn_predictions[churn_predictions['risk_category'] == 'Low Risk'])
        
        with col1:
            st.metric("High Risk Customers", high_risk)
        with col2:
            st.metric("Medium Risk Customers", medium_risk)
        with col3:
            st.metric("Low Risk Customers", low_risk)
        
        # Risk Distribution
        col1, col2 = st.columns(2)
        
        with col1:
            risk_counts = churn_predictions['risk_category'].value_counts()
            fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                        title="Customer Risk Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(churn_predictions, x='churn_probability', bins=20,
                              title="Churn Probability Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature Importance
        st.subheader("Churn Prediction Model Insights")
        feature_importance = churn_data['feature_importance']
        
        features_df = pd.DataFrame([
            {"Feature": k, "Importance": v} for k, v in feature_importance.items()
        ])
        
        fig = px.bar(features_df, x='Importance', y='Feature', orientation='h',
                    title="Feature Importance for Churn Prediction")
        st.plotly_chart(fig, use_container_width=True)
        
        # High Risk Customers Details
        st.subheader("High Risk Customers")
        high_risk_customers = churn_predictions[
            churn_predictions['risk_category'] == 'High Risk'
        ].sort_values('churn_probability', ascending=False)
        
        if not high_risk_customers.empty:
            display_cols = ['customer_id', 'churn_probability', 'days_since_last_purchase', 
                           'total_spent', 'purchase_count', 'avg_order_value']
            st.dataframe(high_risk_customers[display_cols].head(20))
        else:
            st.success("Great! No customers are at high risk of churning.")
        
        # Recommendations
        st.subheader("📋 Retention Recommendations")
        st.write("**For High Risk Customers:**")
        st.write("- Send personalized win-back campaigns")
        st.write("- Offer special discounts or incentives")
        st.write("- Reach out with customer service calls")
        
        st.write("**For Medium Risk Customers:**")
        st.write("- Implement loyalty programs")
        st.write("- Send targeted product recommendations")
        st.write("- Increase engagement through email marketing")
    
    # Cohort Analysis
    elif selected_page == "Cohort Analysis":
        st.header("📊 Cohort Analysis")
        
        retention_matrix = cohort_data['retention_matrix']
        
        # Heatmap of retention rates
        fig = px.imshow(retention_matrix * 100, 
                       aspect='auto',
                       title="Customer Retention Rates by Cohort (%)",
                       color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
        
        # Cohort revenue analysis
        st.subheader("Cohort Revenue Analysis")
        cohort_revenue = cohort_data['cohort_data'].groupby('acquisition_month')['total_amount'].sum()
        fig = px.bar(x=cohort_revenue.index.astype(str), y=cohort_revenue.values,
                    title="Total Revenue by Acquisition Cohort")
        st.plotly_chart(fig, use_container_width=True)
        
        # Cohort size analysis
        st.subheader("Cohort Sizes Over Time")
        cohort_sizes = cohort_data['cohort_sizes']
        fig = px.line(x=cohort_sizes.index.astype(str), y=cohort_sizes.values,
                     title="Customer Acquisition by Month")
        st.plotly_chart(fig, use_container_width=True)
    
    # Price Optimization
    elif selected_page == "Price Optimization":
        st.header("💰 Price Optimization Analysis")
        
        st.subheader("Price Elasticity by Product")
        fig = px.scatter(elasticity_data, x='avg_price', y='elasticity',
                        size='total_quantity', color='elasticity',
                        hover_data=['product_name'],
                        title="Price vs Elasticity")
        st.plotly_chart(fig, use_container_width=True)
        
        # Optimal pricing suggestions
        st.subheader("Pricing Recommendations")
        for _, row in elasticity_data.nlargest(5, 'elasticity').iterrows():
            if row['elasticity'] > 1.0:
                st.info(f"**{row['product_name']}**: Highly elastic (ε={row['elasticity']:.2f}). Consider lowering price to increase revenue.")
            elif row['elasticity'] < 0.5:
                st.success(f"**{row['product_name']}**: Inelastic (ε={row['elasticity']:.2f}). You could potentially increase price.")
        
        # Price distribution analysis
        st.subheader("Price Distribution by Category")
        fig = px.box(elasticity_data, x='category', y='avg_price',
                    title="Price Distribution Across Categories")
        st.plotly_chart(fig, use_container_width=True)
        
        # Top elastic products
        st.subheader("Most Price-Sensitive Products")
        elastic_products = elasticity_data.nlargest(10, 'elasticity')
        st.dataframe(elastic_products[['product_name', 'avg_price', 'elasticity', 'total_quantity']])

if __name__ == "__main__":
    main()