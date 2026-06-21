import os
import urllib.request
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

DATA_DIR = "data"
RAW_FILE = os.path.join(DATA_DIR, "Online_Retail.xlsx")
PROCESSED_FILE = os.path.join(DATA_DIR, "cleaned_retail_data.csv")
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"

def setup_directories():
    """Create data directory if it doesn't exist."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")

def download_uci_dataset():
    """Download the dataset from UCI repository."""
    setup_directories()
    if os.path.exists(RAW_FILE):
        print(f"Raw dataset already exists at {RAW_FILE}")
        return True
    
    print(f"Downloading UCI Online Retail dataset from {UCI_URL}...")
    print("Note: The file is ~45MB and might take a couple of minutes to download.")
    try:
        # User-Agent to avoid blocking
        req = urllib.request.Request(
            UCI_URL, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req, timeout=120) as response, open(RAW_FILE, 'wb') as out_file:
            out_file.write(response.read())
        print("Download complete!")
        return True
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False

def clean_and_process_real_data():
    """Load the raw excel file, clean it, and aggregate it to daily sales per product."""
    if not os.path.exists(RAW_FILE):
        print("Raw Excel file not found. Cannot clean real data.")
        return False
    
    print("Loading raw Excel file. This can take up to a minute...")
    try:
        # Load excel file
        df = pd.read_excel(RAW_FILE, engine='openpyxl')
        print(f"Loaded dataset with {len(df)} rows.")
        
        # 1. Basic Cleaning
        # Remove null CustomerIDs or Descriptions
        df = df.dropna(subset=['Description', 'Quantity', 'UnitPrice'])
        
        # Format InvoiceDate to datetime
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df['Date'] = df['InvoiceDate'].dt.date
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Filter negative quantities (cancellations/returns) and price <= 0
        # For demand forecasting, we want to forecast actual demand (positive orders)
        df_clean = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)].copy()
        
        # Total Sales revenue
        df_clean['Revenue'] = df_clean['Quantity'] * df_clean['UnitPrice']
        
        print(f"Cleaned dataset has {len(df_clean)} rows.")
        
        # 2. Extract Top Products
        # We aggregate by product to find the top products
        top_products = df_clean.groupby(['StockCode', 'Description'])['Quantity'].sum().reset_index()
        top_products = top_products.sort_values(by='Quantity', ascending=False).head(100)
        top_stock_codes = top_products['StockCode'].tolist()
        
        print(f"Extracted top {len(top_stock_codes)} products.")
        
        # 3. Filter for top products and aggregate daily
        df_top = df_clean[df_clean['StockCode'].isin(top_stock_codes)].copy()
        
        # Aggregate daily sales per product
        daily_sales = df_top.groupby(['Date', 'StockCode', 'Description', 'Country']).agg({
            'Quantity': 'sum',
            'Revenue': 'sum',
            'UnitPrice': 'mean' # Average unit price on that day
        }).reset_index()
        
        # Rename columns to avoid confusion
        daily_sales.rename(columns={'Quantity': 'QuantitySold'}, inplace=True)
        
        # Sort values
        daily_sales = daily_sales.sort_values(by=['StockCode', 'Date'])
        
        # Save processed data
        daily_sales.to_csv(PROCESSED_FILE, index=False)
        print(f"Processed daily sales data saved to {PROCESSED_FILE} ({len(daily_sales)} rows).")
        return True
    
    except Exception as e:
        print(f"Error cleaning/processing raw data: {e}")
        return False

def generate_synthetic_data(num_days=730):
    """Generate high-quality synthetic retail sales data with seasonality, trends, and holidays."""
    setup_directories()
    print("Generating high-quality synthetic retail data...")
    
    # Start date 2 years ago from today
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=num_days)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Top 5 popular products with names matching UCI retail
    products = [
        {"StockCode": "85123A", "Description": "WHITE HANGING HEART T-LIGHT HOLDER", "BasePrice": 2.55, "BaseVolume": 120},
        {"StockCode": "22423", "Description": "REGENCY CAKESTAND 3 TIER", "BasePrice": 12.75, "BaseVolume": 45},
        {"StockCode": "84879", "Description": "ASSORTED COLOUR BIRD ORNAMENT", "BasePrice": 1.69, "BaseVolume": 85},
        {"StockCode": "47566", "Description": "PARTY BUNTING", "BasePrice": 4.65, "BaseVolume": 70},
        {"StockCode": "20725", "Description": "LUNCH BAG RED RETROSPOT", "BasePrice": 1.65, "BaseVolume": 110},
        {"StockCode": "22720", "Description": "SET OF 3 CAKE TINS PANTRY DESIGN", "BasePrice": 4.95, "BaseVolume": 40},
        {"StockCode": "85099B", "Description": "JUMBO BAG RED RETROSPOT", "BasePrice": 1.79, "BaseVolume": 150},
        {"StockCode": "23084", "Description": "RABBIT NIGHT LIGHT", "BasePrice": 1.79, "BaseVolume": 95}
    ]
    
    data_list = []
    
    for product in products:
        stock_code = product["StockCode"]
        desc = product["Description"]
        base_price = product["BasePrice"]
        base_vol = product["BaseVolume"]
        
        # Trend component (slight upward trend: 2% growth per year)
        trend = np.linspace(0.9, 1.1, len(date_range))
        
        for idx, date in enumerate(date_range):
            # Calendar details
            day_of_week = date.dayofweek # 0=Monday, 6=Sunday
            month = date.month
            
            # 1. Weekly Seasonality: Sales dip on Sunday, peak on Thursday/Friday
            # Sunday has lower volume (0.4), Thursday has highest (1.25)
            weekly_factors = [1.0, 1.05, 1.1, 1.25, 1.15, 0.7, 0.45]
            weekly_factor = weekly_factors[day_of_week]
            
            # 2. Monthly/Yearly Seasonality: Peaking in Nov (1.5) and Dec (1.8) for Christmas, dipping in Jan (0.7)
            monthly_factors = {
                1: 0.65, 2: 0.75, 3: 0.85, 4: 0.90, 5: 0.95, 6: 0.90,
                7: 0.92, 8: 0.98, 9: 1.10, 10: 1.15, 11: 1.55, 12: 1.85
            }
            monthly_factor = monthly_factors[month]
            
            # 3. Holiday / Festival Impact (Spikes)
            holiday_spike = 1.0
            
            # Black Friday (Friday after 4th Thursday of Nov - approx Nov 23-29)
            if month == 11 and date.day >= 23 and date.day <= 30 and day_of_week == 4:
                holiday_spike = 3.5 # Massive spike
            # Pre-Christmas rush (Dec 10 to Dec 22)
            elif month == 12 and date.day >= 10 and date.day <= 22:
                holiday_spike = 1.4 + np.random.uniform(0, 0.3)
            # Christmas closure / quiet days (Dec 24-26)
            elif month == 12 and date.day in [24, 25, 26]:
                holiday_spike = 0.1 # Minimal sales
            # New Year sales (Dec 31 to Jan 2)
            elif (month == 12 and date.day == 31) or (month == 1 and date.day in [1, 2]):
                holiday_spike = 1.3
            
            # 4. Price Fluctuation (Introduce random pricing adjustments)
            price = base_price * np.random.uniform(0.95, 1.05)
            
            # 5. Combined Demand calculation with random noise
            noise = np.random.normal(loc=0.0, scale=0.15) # Noise standard dev
            quantity_sold = int(max(1, base_vol * trend[idx] * weekly_factor * monthly_factor * holiday_spike * (1 + noise)))
            
            # Revenue
            revenue = quantity_sold * price
            
            data_list.append({
                "Date": date.strftime("%Y-%m-%d"),
                "StockCode": stock_code,
                "Description": desc,
                "Country": "United Kingdom",
                "QuantitySold": quantity_sold,
                "Revenue": round(revenue, 2),
                "UnitPrice": round(price, 2)
            })
            
    df_synthetic = pd.DataFrame(data_list)
    df_synthetic.to_csv(PROCESSED_FILE, index=False)
    print(f"Synthetic daily sales data saved to {PROCESSED_FILE} ({len(df_synthetic)} rows).")
    return True

def main():
    setup_directories()
    
    # Try downloading and cleaning real data first
    download_success = download_uci_dataset()
    process_success = False
    
    if download_success:
        process_success = clean_and_process_real_data()
        
    if not process_success:
        print("\nUsing fallback: Generating high-quality synthetic dataset...")
        generate_synthetic_data()
    else:
        print("\nReal retail dataset successfully processed and prepared!")

if __name__ == "__main__":
    main()
