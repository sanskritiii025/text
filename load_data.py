import pandas as pd
from sqlalchemy import create_engine, text
import json
import os
from dotenv import load_dotenv
load_dotenv()
import urllib.parse  

DB_USER = "postgres"
RAW_PASS = os.getenv("DB_PASSWORD") 
DB_PASS = urllib.parse.quote_plus(RAW_PASS) 

DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "postgres"

CSV_FOLDER = r"C:\Users\sansk\OneDrive\Desktop\work\AR_main_folder_antigravity\data_quality_processor\relationship_importer"

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

print(f"Connecting to: {DB_HOST} as {DB_USER}...") 

engine = create_engine(DATABASE_URL)


def load_table(csv_name, table_name, column_mapping, unique_id_col):
    """
    Reads CSV, maps columns, packs extras into metadata, and inserts to SQL.
    """
    file_path = os.path.join(CSV_FOLDER, csv_name)
    if not os.path.exists(file_path):
        print(f"Skipping {csv_name}: File not found.")
        return

    print(f"Processing {csv_name} -> {table_name}...")
    
 
    df = pd.read_csv(file_path)
    
  
    sql_data = pd.DataFrame()
    

    for csv_col, sql_col in column_mapping.items():
        if csv_col in df.columns:
            sql_data[sql_col] = df[csv_col]
        else:
            print(f"  Warning: Column {csv_col} missing in CSV")

 
    mapped_csv_cols = list(column_mapping.keys())
    extra_cols = [c for c in df.columns if c not in mapped_csv_cols]
    
    if extra_cols:
        print(f"  Packing extra columns into metadata: {extra_cols}")
       
        clean_extras = df[extra_cols].where(pd.notnull(df[extra_cols]), None)
        metadata_list = clean_extras.to_dict(orient='records')
        sql_data['metadata'] = [json.dumps(m) for m in metadata_list]
    else:
        sql_data['metadata'] = '{}'


    for col in sql_data.columns:
        if 'date' in col.lower():
            sql_data[col] = pd.to_datetime(sql_data[col], errors='coerce')

    try:
      
        sql_data.to_sql(table_name, engine, if_exists='append', index=False, method='multi', chunksize=1000)
        print(f"  Success! Loaded {len(sql_data)} rows into {table_name}.")
    except Exception as e:
        print(f"  ERROR loading {table_name}: {e}")



# 1. TERRITORIES
"""load_table(
    'territories.csv', 'territories',
    {
        'TerritoryID': 'territory_id',
        'Region': 'region',
        'Manager': 'manager_name'
    }, 'territory_id'
)

"""# 2. SALES REPS
load_table(
    'salesreps.csv', 'sales_reps',
    {
        'SalesRepID': 'sales_rep_id',
        'Name': 'name',
        'Email': 'email',
        'TerritoryID': 'territory_id'
    }, 'sales_rep_id'
)

# 3. CUSTOMERS
load_table(
    'customers.csv', 'customers',
    {
        'CustomerID': 'customer_id',
        'Name': 'name',
        'Email': 'email',
        'Phone': 'phone',
        'Country': 'country'
    }, 'customer_id'
)

# 4. ACCOUNTS (Maps CustomerID to primary_customer_id)
load_table(
    'accounts.csv', 'accounts',
    {
        'AccountID': 'account_id',
        'AccountName': 'account_name',
        'CustomerID': 'primary_customer_id',
        'Type': 'account_type',
        'Region': 'region'
    }, 'account_id'
)

# 5. PRODUCTS
load_table(
    'products.csv', 'products',
    {
        'ProductID': 'product_id',
        'ProductName': 'product_name',
        'Category': 'category',
        'Price': 'list_price', # CSV - Price, DB - list_price
    }, 'product_id'
)

# 6. OPPORTUNITIES
load_table(
    'opportunities.csv', 'opportunities',
    {
        'OpportunityID': 'opportunity_id',
        'AccountID': 'account_id',
        'Stage': 'stage',
        'Amount': 'amount',
        'CloseDate': 'close_date'
    }, 'opportunity_id'
)

# 7. QUOTES
load_table(
    'quotes.csv', 'quotes',
    {
        'QuoteID': 'quote_id',
        'OpportunityID': 'opportunity_id',
        'QuoteAmount': 'quote_amount',
        'Status': 'status'
    }, 'quote_id'
)

# 8. SALES ORDERS
load_table(
    'sales_orders.csv', 'sales_orders',
    {
        'SalesOrderID': 'sales_order_id',
        'OpportunityID': 'opportunity_id',
        'OrderDate': 'order_date',
        'OrderAmount': 'total_amount',
        'Status': 'status'
    }, 'sales_order_id'
)

# 9. INVOICES
load_table(
    'invoices.csv', 'invoices',
    {
        'InvoiceID': 'invoice_id',
        'SalesOrderID': 'sales_order_id',
        'InvoiceDate': 'invoice_date',
        'InvoiceAmount': 'amount',
        'PaymentStatus': 'payment_status'
    }, 'invoice_id' 
)

print(":)")
print("DATA LOAD COMPLETE")




