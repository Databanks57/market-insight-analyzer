import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import sqlite3
from faker import Faker

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)
fake = Faker()
Faker.seed(42)

class BusinessDataGenerator:
    def __init__(self, business_type="retail"):
        self.business_type = business_type
        self.start_date = datetime(2022, 1, 1)
        self.end_date = datetime(2024, 8, 25)
        self.seasonal_patterns = self.generate_seasonal_patterns()
        
    def generate_seasonal_patterns(self):
        """Generate seasonal patterns for more realistic data"""
        seasonal_patterns = {
            'retail': {
                'monthly_multipliers': [0.8, 0.7, 0.9, 1.0, 1.1, 1.3, 1.2, 1.1, 1.0, 1.2, 1.8, 2.0],
                'dow_multipliers': [0.9, 0.8, 0.9, 1.0, 1.2, 1.5, 1.3]
            },
            'restaurant': {
                'monthly_multipliers': [1.0, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.3, 1.2, 1.1, 1.0, 1.2],
                'dow_multipliers': [1.0, 1.0, 1.1, 1.2, 1.5, 1.8, 1.3]
            }
        }
        return seasonal_patterns[self.business_type]
    
    def generate_customers(self, n_customers=1000):
        """Generate customer data"""
        customers = []
        
        for i in range(1, n_customers + 1):
            # Customer acquisition dates weighted towards earlier periods
            days_from_start = np.random.exponential(200)
            acquisition_date = self.start_date + timedelta(days=min(days_from_start, 900))
            
            # Customer segments based on lifetime value potential
            segment_prob = np.random.random()
            if segment_prob < 0.1:  # 10% high value
                segment = "High Value"
                avg_order_multiplier = 2.5
                purchase_frequency = 0.3  # 30% chance to purchase each month
            elif segment_prob < 0.3:  # 20% medium value
                segment = "Medium Value" 
                avg_order_multiplier = 1.5
                purchase_frequency = 0.15
            else:  # 70% low value
                segment = "Low Value"
                avg_order_multiplier = 0.8
                purchase_frequency = 0.05
                
            customers.append({
                'customer_id': i,
                'acquisition_date': acquisition_date,
                'segment': segment,
                'avg_order_multiplier': avg_order_multiplier,
                'purchase_frequency': purchase_frequency,
                'is_active': True
            })
            
        return pd.DataFrame(customers)
    
    def generate_products(self):
        """Generate product catalog based on business type"""
        if self.business_type == "retail":
            products_data = [
                ("T-Shirt", "Clothing", 15, 25, 0.4),
                ("Jeans", "Clothing", 30, 60, 0.5),
                ("Sneakers", "Footwear", 40, 80, 0.5),
                ("Dress", "Clothing", 25, 50, 0.5),
                ("Jacket", "Clothing", 45, 90, 0.5),
                ("Boots", "Footwear", 60, 120, 0.5),
                ("Accessories", "Accessories", 10, 20, 0.5),
                ("Handbag", "Accessories", 35, 70, 0.5),
                ("Sunglasses", "Accessories", 20, 40, 0.5),
                ("Watch", "Accessories", 80, 160, 0.5),
                ("Shorts", "Clothing", 18, 35, 0.486),
                ("Sandals", "Footwear", 25, 45, 0.444),
            ]
        elif self.business_type == "restaurant":
            products_data = [
                ("Burger", "Main Course", 8, 15, 0.467),
                ("Pizza", "Main Course", 12, 22, 0.455),
                ("Salad", "Main Course", 6, 12, 0.5),
                ("Pasta", "Main Course", 10, 18, 0.444),
                ("Steak", "Main Course", 18, 35, 0.486),
                ("Fish", "Main Course", 15, 28, 0.464),
                ("Soup", "Appetizer", 4, 8, 0.5),
                ("Wings", "Appetizer", 7, 12, 0.417),
                ("Fries", "Side", 3, 6, 0.5),
                ("Soda", "Beverage", 1.5, 3, 0.5),
                ("Beer", "Beverage", 3, 6, 0.5),
                ("Dessert", "Dessert", 4, 8, 0.5),
            ]
        
        products = []
        for i, (name, category, cost, price, margin) in enumerate(products_data, 1):
            products.append({
                'product_id': i,
                'product_name': name,
                'category': category,
                'cost': cost,
                'price': price,
                'profit_margin': margin
            })
            
        return pd.DataFrame(products)
    
    def generate_transactions(self, customers_df, products_df, n_transactions=10000):
        """Generate realistic transaction data"""
        transactions = []
        
        for _ in range(n_transactions):
            # Select customer based on segment probability
            customer = customers_df.sample(1).iloc[0]
            
            # Skip if customer churned (stopped buying)
            churn_probability = self.calculate_churn_probability(customer)
            if np.random.random() < churn_probability:
                continue
                
            # Generate transaction date
            days_since_acquisition = (self.end_date - customer['acquisition_date']).days
            transaction_days_back = np.random.randint(0, min(days_since_acquisition, 730))
            transaction_date = self.end_date - timedelta(days=transaction_days_back)
            
            # Enhanced seasonality with monthly patterns
            month = transaction_date.month
            monthly_multiplier = self.seasonal_patterns['monthly_multipliers'][month - 1]
            
            # Enhanced day of week patterns
            day_of_week = transaction_date.weekday()
            dow_multiplier = self.seasonal_patterns['dow_multipliers'][day_of_week]
                
            # Determine number of items (basket size)
            avg_basket_size = 2.5 * customer['avg_order_multiplier']
            basket_size = max(1, int(np.random.poisson(avg_basket_size)))
            
            # Select products for basket
            selected_products = products_df.sample(
                min(basket_size, len(products_df)), 
                replace=False
            )
            
            for _, product in selected_products.iterrows():
                quantity = np.random.choice([1, 1, 1, 2, 2, 3], p=[0.5, 0.2, 0.1, 0.1, 0.05, 0.05])
                
                # Apply price variation with enhanced seasonality
                base_price = product['price']
                price_variation = np.random.normal(1.0, 0.1)  # ±10% price variation
                unit_price = base_price * price_variation * monthly_multiplier * dow_multiplier
                unit_price = round(unit_price, 2)
                
                total_amount = unit_price * quantity
                
                transactions.append({
                    'transaction_id': len(transactions) + 1,
                    'date': transaction_date.date(),
                    'customer_id': customer['customer_id'],
                    'product_id': product['product_id'],
                    'product_name': product['product_name'],
                    'category': product['category'],
                    'quantity': quantity,
                    'unit_price': unit_price,
                    'total_amount': round(total_amount, 2),
                    'cost': product['cost'] * quantity,
                    'profit': round(total_amount - (product['cost'] * quantity), 2)
                })
                
        return pd.DataFrame(transactions)
    
    def calculate_churn_probability(self, customer):
        """Calculate churn probability based on customer segment and time"""
        base_churn_rate = {
            "High Value": 0.05,
            "Medium Value": 0.15, 
            "Low Value": 0.35
        }
        
        days_since_acquisition = (self.end_date - customer['acquisition_date']).days
        
        # Increase churn probability over time
        time_multiplier = 1 + (days_since_acquisition / 365) * 0.1
        
        return base_churn_rate[customer['segment']] * time_multiplier
    
    def create_database(self, db_name="business_data.db"):
        """Create SQLite database with all tables"""
        # Generate data
        print("Generating customers...")
        customers_df = self.generate_customers(1000)
        
        print("Generating products...")
        products_df = self.generate_products()
        
        print("Generating transactions...")
        transactions_df = self.generate_transactions(customers_df, products_df, 15000)
        
        # Create database
        conn = sqlite3.connect(db_name)
        
        # Save to database
        customers_df[['customer_id', 'acquisition_date', 'segment']].to_sql(
            'customers', conn, if_exists='replace', index=False
        )
        
        products_df.to_sql('products', conn, if_exists='replace', index=False)
        transactions_df.to_sql('transactions', conn, if_exists='replace', index=False)
        
        # Create indexes for better performance
        conn.execute("CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_transactions_customer ON transactions(customer_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_transactions_product ON transactions(product_id)")
        
        conn.close()
        print(f"Database '{db_name}' created successfully!")
        
        # Print summary statistics
        print(f"\nData Summary:")
        print(f"Customers: {len(customers_df):,}")
        print(f"Products: {len(products_df):,}")
        print(f"Transactions: {len(transactions_df):,}")
        print(f"Date range: {transactions_df['date'].min()} to {transactions_df['date'].max()}")
        print(f"Total revenue: ${transactions_df['total_amount'].sum():,.2f}")
        
        return customers_df, products_df, transactions_df

# Generate sample data
if __name__ == "__main__":
    # Create retail business data
    retail_generator = BusinessDataGenerator("retail")
    customers, products, transactions = retail_generator.create_database("retail_business.db")
    
    # Show sample data
    print("\n=== SAMPLE DATA ===")
    print("\nCustomers:")
    print(customers.head())
    print("\nProducts:")
    print(products.head())
    print("\nTransactions:")
    print(transactions.head())