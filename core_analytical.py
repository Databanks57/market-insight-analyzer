import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

class BusinessAnalytics:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        
    def load_data(self):
        """Load data from SQLite database"""
        self.customers = pd.read_sql_query("SELECT * FROM customers", self.conn)
        self.products = pd.read_sql_query("SELECT * FROM products", self.conn)
        self.transactions = pd.read_sql_query("SELECT * FROM transactions", self.conn)
        
        # Convert date columns using ISO8601 format
        self.customers['acquisition_date'] = pd.to_datetime(
            self.customers['acquisition_date'], 
            format='ISO8601',
            errors='coerce'
        )
        self.transactions['date'] = pd.to_datetime(
            self.transactions['date'], 
            format='ISO8601',
            errors='coerce'
        )
        
        return self.transactions, self.customers, self.products
    
    def get_monthly_sales_from_db(self):
        """Direct SQL query for better performance"""
        query = """
        SELECT 
            strftime('%Y-%m', date) as month,
            SUM(total_amount) as revenue,
            COUNT(DISTINCT transaction_id) as transactions,
            COUNT(DISTINCT customer_id) as customers
        FROM transactions 
        GROUP BY strftime('%Y-%m', date)
        ORDER BY month
        """
        return pd.read_sql_query(query, self.conn)
    
    def sales_performance_analysis(self):
        """Core sales metrics and trends"""
        df = self.transactions.copy()
        
        # Basic metrics
        total_revenue = df['total_amount'].sum()
        total_transactions = len(df)
        avg_order_value = df.groupby('transaction_id')['total_amount'].sum().mean()
        
        # Monthly trends - using optimized query
        monthly_sales = self.get_monthly_sales_from_db()
        monthly_sales['aov'] = monthly_sales['revenue'] / monthly_sales['transactions']
        
        # Growth rates
        monthly_sales['revenue_growth'] = monthly_sales['revenue'].pct_change() * 100
        monthly_sales['transaction_growth'] = monthly_sales['transactions'].pct_change() * 100
        
        # Day of week analysis
        df['day_of_week'] = df['date'].dt.day_name()
        dow_sales = df.groupby('day_of_week')['total_amount'].sum().reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ])
        
        # Top products
        product_performance = df.groupby(['product_id', 'product_name']).agg({
            'total_amount': 'sum',
            'quantity': 'sum',
            'profit': 'sum'
        }).reset_index().sort_values('total_amount', ascending=False)
        
        return {
            'summary_metrics': {
                'total_revenue': total_revenue,
                'total_transactions': total_transactions,
                'avg_order_value': avg_order_value,
                'unique_customers': df['customer_id'].nunique()
            },
            'monthly_trends': monthly_sales,
            'dow_analysis': dow_sales,
            'top_products': product_performance.head(10)
        }
    
    def customer_analytics(self):
        """Customer behavior and segmentation analysis"""
        df = self.transactions.copy()
        
        # RFM Analysis
        current_date = df['date'].max()
        
        rfm = df.groupby('customer_id').agg({
            'date': lambda x: (current_date - x.max()).days,  # Recency
            'transaction_id': 'nunique',  # Frequency
            'total_amount': 'sum'  # Monetary
        }).reset_index()
        
        rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']
        
        # RFM scoring (1-5 scale)
        rfm['r_score'] = pd.qcut(rfm['recency'].rank(method='first'), 5, labels=[5,4,3,2,1])
        rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
        rfm['m_score'] = pd.qcut(rfm['monetary'].rank(method='first'), 5, labels=[1,2,3,4,5])
        
        rfm['rfm_score'] = rfm['r_score'].astype(str) + rfm['f_score'].astype(str) + rfm['m_score'].astype(str)
        
        # Customer segmentation based on RFM
        def segment_customers(row):
            if row['rfm_score'] in ['555', '554', '544', '545', '454', '455', '445']:
                return 'Champions'
            elif row['rfm_score'] in ['543', '444', '435', '355', '354', '345', '344', '335']:
                return 'Loyal Customers'
            elif row['rfm_score'] in ['512', '511', '422', '421', '412', '411']:
                return 'Potential Loyalists'
            elif row['rfm_score'] in ['344', '343', '334', '343', '333']:
                return 'New Customers'
            elif row['rfm_score'] in ['155', '154', '144', '214', '215', '115', '114']:
                return 'At Risk'
            elif row['rfm_score'] in ['155', '254', '245']:
                return 'Cannot Lose Them'
            else:
                return 'Others'
        
        rfm['segment'] = rfm.apply(segment_customers, axis=1)
        
        # Customer Lifetime Value
        clv_data = df.groupby('customer_id').agg({
            'total_amount': ['sum', 'mean', 'count'],
            'date': ['min', 'max']
        }).reset_index()
        
        clv_data.columns = ['customer_id', 'total_spent', 'avg_order_value', 'order_count', 'first_purchase', 'last_purchase']
        clv_data['lifespan_days'] = (clv_data['last_purchase'] - clv_data['first_purchase']).dt.days
        clv_data['purchase_frequency'] = clv_data['order_count'] / (clv_data['lifespan_days'] + 1)
        
        # Simple CLV calculation
        clv_data['clv'] = clv_data['avg_order_value'] * clv_data['order_count']
        
        # Merge RFM with CLV
        customer_analysis = rfm.merge(clv_data, on='customer_id', how='left')
        
        # Segment summary
        segment_summary = customer_analysis.groupby('segment').agg({
            'customer_id': 'count',
            'monetary': 'mean',
            'frequency': 'mean',
            'recency': 'mean',
            'clv': 'mean'
        }).reset_index()
        
        return {
            'rfm_data': rfm,
            'customer_segments': customer_analysis,
            'segment_summary': segment_summary,
            'clv_distribution': clv_data
        }
    
    def churn_prediction(self):
        """Predict customer churn using simple logistic regression"""
        df = self.transactions.copy()
        current_date = df['date'].max()
        
        # Define churn (no purchase in last 90 days)
        churn_threshold = 90
        
        customer_features = df.groupby('customer_id').agg({
            'date': ['min', 'max', 'count'],
            'total_amount': ['sum', 'mean'],
            'quantity': 'sum'
        }).reset_index()
        
        customer_features.columns = ['customer_id', 'first_purchase', 'last_purchase', 'purchase_count', 
                                   'total_spent', 'avg_order_value', 'total_quantity']
        
        # Calculate features
        customer_features['days_since_last_purchase'] = (current_date - customer_features['last_purchase']).dt.days
        customer_features['customer_lifespan'] = (customer_features['last_purchase'] - customer_features['first_purchase']).dt.days
        customer_features['purchase_frequency'] = customer_features['purchase_count'] / (customer_features['customer_lifespan'] + 1)
        
        # Define churn target
        customer_features['churned'] = (customer_features['days_since_last_purchase'] > churn_threshold).astype(int)
        
        # Prepare features for modeling
        feature_cols = ['purchase_count', 'total_spent', 'avg_order_value', 'total_quantity', 
                       'customer_lifespan', 'purchase_frequency']
        
        X = customer_features[feature_cols].fillna(0)
        y = customer_features['churned']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = LogisticRegression(random_state=42)
        model.fit(X_scaled, y)
        
        # Predict churn probability
        churn_proba = model.predict_proba(X_scaled)[:, 1]
        customer_features['churn_probability'] = churn_proba
        
        # Risk categories
        def risk_category(prob):
            if prob >= 0.7:
                return 'High Risk'
            elif prob >= 0.4:
                return 'Medium Risk'
            else:
                return 'Low Risk'
        
        customer_features['risk_category'] = customer_features['churn_probability'].apply(risk_category)
        
        return {
            'churn_predictions': customer_features,
            'model_features': feature_cols,
            'feature_importance': dict(zip(feature_cols, model.coef_[0])),
            'churn_rate': y.mean()
        }
    
    def product_analysis(self):
        """Product performance and cross-selling analysis"""
        df = self.transactions.copy()
        
        # Product performance
        product_metrics = df.groupby(['product_id', 'product_name', 'category']).agg({
            'total_amount': ['sum', 'mean'],
            'quantity': 'sum',
            'profit': ['sum', 'mean'],
            'transaction_id': 'nunique'
        }).reset_index()
        
        product_metrics.columns = ['product_id', 'product_name', 'category', 'total_revenue', 
                                 'avg_revenue_per_transaction', 'total_quantity', 'total_profit', 
                                 'avg_profit_per_transaction', 'unique_transactions']
        
        # Profit margin
        product_metrics['profit_margin'] = (product_metrics['total_profit'] / product_metrics['total_revenue']) * 100
        
        # ABC Analysis (Pareto analysis)
        product_metrics = product_metrics.sort_values('total_revenue', ascending=False)
        product_metrics['revenue_cumulative'] = product_metrics['total_revenue'].cumsum()
        product_metrics['revenue_cumulative_pct'] = (product_metrics['revenue_cumulative'] / product_metrics['total_revenue'].sum()) * 100
        
        def abc_category(pct):
            if pct <= 80:
                return 'A'
            elif pct <= 95:
                return 'B'
            else:
                return 'C'
        
        product_metrics['abc_category'] = product_metrics['revenue_cumulative_pct'].apply(abc_category)
        
        # Market basket analysis (simplified)
        baskets = df.groupby('transaction_id')['product_name'].apply(list).reset_index()
        
        # Find most common product pairs
        from itertools import combinations
        
        pairs = []
        for basket in baskets['product_name']:
            if len(basket) > 1:
                for pair in combinations(basket, 2):
                    pairs.append(sorted(pair))
        
        pair_counts = pd.DataFrame(pairs, columns=['product_1', 'product_2'])
        pair_frequency = pair_counts.value_counts().reset_index()
        pair_frequency.columns = ['product_1', 'product_2', 'frequency']
        
        return {
            'product_performance': product_metrics,
            'abc_analysis': product_metrics.groupby('abc_category').agg({
                'product_id': 'count',
                'total_revenue': 'sum'
            }).reset_index(),
            'cross_selling_opportunities': pair_frequency.head(10),
            'category_performance': df.groupby('category').agg({
                'total_amount': 'sum',
                'profit': 'sum',
                'quantity': 'sum'
            }).reset_index()
        }
    
    def cohort_analysis(self):
        """Cohort analysis for customer retention"""
        df = self.transactions.copy()
        
        # Create acquisition cohorts
        df['acquisition_month'] = df.groupby('customer_id')['date'].transform('min').dt.to_period('M')
        df['order_month'] = df['date'].dt.to_period('M')
        
        # Calculate cohort periods
        df['cohort_period'] = (df['order_month'] - df['acquisition_month']).apply(lambda x: x.n)
        
        # Cohort retention matrix
        cohort_data = df.groupby(['acquisition_month', 'cohort_period']).agg({
            'customer_id': 'nunique',
            'total_amount': 'sum'
        }).reset_index()
        
        # Pivot for retention matrix
        cohort_pivot = cohort_data.pivot_table(
            index='acquisition_month',
            columns='cohort_period',
            values='customer_id',
            aggfunc='sum'
        )
        
        # Calculate retention rates
        cohort_sizes = cohort_pivot.iloc[:, 0]
        retention_matrix = cohort_pivot.divide(cohort_sizes, axis=0)
        
        return {
            'cohort_data': cohort_data,
            'retention_matrix': retention_matrix,
            'cohort_sizes': cohort_sizes
        }
    
    def price_elasticity_analysis(self):
        """Analyze price sensitivity"""
        df = self.transactions.copy()
        
        # Group by product and calculate demand at different price points
        price_elasticity = df.groupby(['product_id', 'product_name']).agg({
            'unit_price': ['mean', 'std'],
            'quantity': 'sum',
            'transaction_id': 'count'
        }).reset_index()
        
        price_elasticity.columns = ['product_id', 'product_name', 'avg_price', 'price_std', 'total_quantity', 'transactions']
        
        # Simple elasticity calculation
        price_elasticity['elasticity'] = -price_elasticity['price_std'] / price_elasticity['avg_price'] * \
                                        (price_elasticity['total_quantity'] / price_elasticity['transactions'])
        
        return price_elasticity
    
    def generate_insights(self):
        """Generate automated business insights"""
        insights = []
        
        # Load all analyses
        sales_data = self.sales_performance_analysis()
        customer_data = self.customer_analytics()
        product_data = self.product_analysis()
        churn_data = self.churn_prediction()
        
        # Sales insights
        monthly_trends = sales_data['monthly_trends']
        if len(monthly_trends) > 1:
            latest_growth = monthly_trends['revenue_growth'].iloc[-1]
            if not pd.isna(latest_growth):
                if latest_growth > 10:
                    insights.append(f"📈 Strong growth: Revenue increased by {latest_growth:.1f}% last month")
                elif latest_growth < -10:
                    insights.append(f"📉 Revenue declined by {abs(latest_growth):.1f}% last month - investigate causes")
        
        # Customer insights
        high_value_customers = len(customer_data['customer_segments'][customer_data['customer_segments']['segment'] == 'Champions'])
        at_risk_customers = len(customer_data['customer_segments'][customer_data['customer_segments']['segment'] == 'At Risk'])
        
        insights.append(f"👑 You have {high_value_customers} champion customers generating premium revenue")
        if at_risk_customers > 0:
            insights.append(f"⚠️ {at_risk_customers} customers are at risk of churning - consider retention campaigns")
        
        # Product insights
        abc_analysis = product_data['abc_analysis']
        a_products = abc_analysis[abc_analysis['abc_category'] == 'A']['product_id'].sum()
        total_products = abc_analysis['product_id'].sum()
        
        insights.append(f"🎯 {a_products} products ({(a_products/total_products)*100:.1f}%) generate 80% of your revenue")
        
        # Top cross-selling opportunity
        if not product_data['cross_selling_opportunities'].empty:
            top_pair = product_data['cross_selling_opportunities'].iloc[0]
            insights.append(f"🔗 Cross-sell opportunity: {top_pair['product_1']} + {top_pair['product_2']} (bought together {top_pair['frequency']} times)")
        
        # Churn insights
        high_risk_count = len(churn_data['churn_predictions'][churn_data['churn_predictions']['risk_category'] == 'High Risk'])
        insights.append(f"🚨 {high_risk_count} customers are at high risk of churning")
        
        return insights
    
    def close_connection(self):
        """Close database connection"""
        self.conn.close()

# Example usage
if __name__ == "__main__":
    # Initialize analytics
    analytics = BusinessAnalytics("retail_business.db")
    analytics.load_data()
    
    # Run analyses
    print("=== SALES PERFORMANCE ===")
    sales_results = analytics.sales_performance_analysis()
    print(f"Total Revenue: ${sales_results['summary_metrics']['total_revenue']:,.2f}")
    print(f"Average Order Value: ${sales_results['summary_metrics']['avg_order_value']:.2f}")
    
    print("\n=== CUSTOMER ANALYTICS ===")
    customer_results = analytics.customer_analytics()
    print("Customer Segments:")
    print(customer_results['segment_summary'])
    
    print("\n=== AUTOMATED INSIGHTS ===")
    insights = analytics.generate_insights()
    for insight in insights:
        print(insight)
    
    analytics.close_connection()