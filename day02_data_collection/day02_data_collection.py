# ==============================================================
# DAY 2 — DATA SOURCES & COLLECTION
# Course: ML Fundamentals | Week 2
# ==============================================================

# Before any ML project, we need DATA.
# Data can come from many places. Today we learn how to collect it.

import pandas as pd    # For working with data tables
import sqlite3         # Built-in Python library for SQLite databases
import json            # Built-in library for reading JSON files
import os              # Built-in library for file/folder operations

print("=" * 60)
print("DAY 2: DATA SOURCES & COLLECTION")
print("=" * 60)


# ==============================================================
# SECTION 1: TYPES OF DATA
# ==============================================================

print("\n--- TYPES OF DATA ---")

# STRUCTURED DATA:
#   - Organised into rows and columns (like a spreadsheet or table)
#   - Easy to analyse with tools like Pandas
#   - Examples: CSV files, SQL databases, Excel files

# UNSTRUCTURED DATA:
#   - No fixed format — messy and varied
#   - Examples: emails, images, audio, social media posts
#   - Needs extra processing (NLP, computer vision) to use in ML

print("Structured Data   → Tables, CSVs, SQL databases")
print("Unstructured Data → Images, text, audio, video")


# ==============================================================
# SECTION 2: LOADING A CSV FILE (Most Common!)
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 2: LOADING A CSV FILE")
print("=" * 60)

# CSV = Comma-Separated Values
# It's a plain text file where each row is a record,
# and columns are separated by commas.

# pd.read_csv() reads a CSV file and turns it into a DataFrame
# A DataFrame = a table with rows and columns (like Excel in Python)

csv_path = "../data/raw_data.csv"
df_csv = pd.read_csv(csv_path)

print(f"\nLoaded CSV file: {csv_path}")
print(f"Shape: {df_csv.shape[0]} rows × {df_csv.shape[1]} columns")

# .head(n) shows the first n rows. Default is 5.
print("\nFirst 3 rows:")
print(df_csv.head(3))

# .dtypes shows what type of data is in each column
# (int64=whole numbers, float64=decimal numbers, object=text)
print("\nColumn Data Types:")
print(df_csv.dtypes)


# ==============================================================
# SECTION 3: LOADING A JSON FILE
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 3: LOADING A JSON FILE")
print("=" * 60)

# JSON = JavaScript Object Notation
# It's a very common format for web data (APIs return JSON).
# JSON looks like Python dictionaries: {"key": "value"}

# First, let's create a small JSON file to demonstrate
sample_json_data = [
    {"customer_id": 101, "name": "Diana", "city": "London",     "purchase": 320.0},
    {"customer_id": 102, "name": "Evan",  "city": "Bristol",    "purchase": 85.5},
    {"customer_id": 103, "name": "Fiona", "city": "Manchester", "purchase": 550.0},
    {"customer_id": 104, "name": "George","city": "Leeds",      "purchase": 42.0},
    {"customer_id": 105, "name": "Hannah","city": "Glasgow",    "purchase": 198.0},
]

# Save it as a JSON file so we can practice loading it
json_path = "../data/sample_customers.json"
with open(json_path, "w") as f:
    # json.dump() converts Python data (list/dict) → JSON text and writes to file
    json.dump(sample_json_data, f, indent=4)
print(f"Created JSON file: {json_path}")

# Now load it back using pandas
df_json = pd.read_json(json_path)
print("\nJSON data loaded into DataFrame:")
print(df_json)


# ==============================================================
# SECTION 4: SQLITE DATABASE (SQL)
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 4: LOADING FROM A SQLITE DATABASE")
print("=" * 60)

# A database stores data in organised TABLES.
# SQL (Structured Query Language) is used to query (ask) databases.
# SQLite is a simple file-based database — no server needed!

db_path = "../data/customers.db"

# sqlite3.connect() creates or connects to an SQLite database file
conn = sqlite3.connect(db_path)

# cursor = a tool to run SQL commands
cursor = conn.cursor()

# CREATE TABLE: make a new table called 'orders'
# IF NOT EXISTS means: only create it if it doesn't already exist
cursor.execute("""
    CREATE TABLE IF NOT EXISTS orders (
        order_id    INTEGER PRIMARY KEY,
        customer_id INTEGER,
        product     TEXT,
        amount      REAL,
        order_date  TEXT
    )
""")

# Insert some rows of data into the table
orders_data = [
    (1, 1, "Laptop",    999.99, "2024-01-10"),
    (2, 2, "T-Shirt",    29.99, "2024-01-15"),
    (3, 3, "Cookbook",   18.50, "2024-02-03"),
    (4, 4, "Headphones", 75.00, "2024-02-20"),
    (5, 5, "Running Shoes", 110.00, "2024-03-08"),
]

# executemany inserts multiple rows at once
# '?' are placeholders — pandas fills them safely (prevents SQL injection)
cursor.executemany(
    "INSERT OR IGNORE INTO orders VALUES (?, ?, ?, ?, ?)",
    orders_data
)
conn.commit()   # Save the changes to the database
print(f"Created and populated database: {db_path}")

# Now read data FROM the database using SQL
# SELECT * means "get all columns"
# FROM orders means "from the orders table"
sql_query = "SELECT * FROM orders"
df_sql = pd.read_sql_query(sql_query, conn)

print("\nData from SQLite database:")
print(df_sql)

# Close the database connection when done (good practice!)
conn.close()


# ==============================================================
# SECTION 5: DIFFERENT FILE FORMATS AT A GLANCE
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 5: COMMON DATA FORMATS SUMMARY")
print("=" * 60)

formats = [
    ("CSV",     "pd.read_csv('file.csv')",      "Most common, easy to read"),
    ("JSON",    "pd.read_json('file.json')",     "Web APIs, nested data"),
    ("Excel",   "pd.read_excel('file.xlsx')",    "Business spreadsheets"),
    ("SQL",     "pd.read_sql_query(sql, conn)",  "Databases (MySQL, SQLite)"),
    ("Parquet", "pd.read_parquet('file.parq')",  "Big data, very fast"),
]

print(f"\n{'Format':<10} {'How to Load':<38} {'Best For'}")
print("-" * 70)
for fmt, code, use in formats:
    print(f"{fmt:<10} {code:<38} {use}")


# ==============================================================
# SECTION 6: BASIC INSPECTION OF LOADED DATA
# ==============================================================

print("\n" + "=" * 60)
print("SECTION 6: INSPECTING YOUR DATA")
print("=" * 60)

# Always inspect data after loading — check if it looks right!
df = pd.read_csv("../data/raw_data.csv")

print("\n.info() — column names, data types, non-null counts:")
df.info()

print("\n.describe() — statistical summary of numeric columns:")
# count=how many values, mean=average, std=spread, min/max=range
print(df.describe())

print("\nColumn names:")
print(df.columns.tolist())

print("\nNumber of rows:", len(df))       # len() = length
print("Number of columns:", len(df.columns))


# ==============================================================
# SUMMARY
# ==============================================================

print("\n" + "=" * 60)
print("DAY 2 COMPLETE! KEY TAKEAWAYS:")
print("=" * 60)
print("✓ Data can be Structured (tables) or Unstructured (text/images)")
print("✓ CSV is the most common format: pd.read_csv()")
print("✓ JSON is used for web APIs: pd.read_json()")
print("✓ SQL databases: connect with sqlite3, query with pd.read_sql_query()")
print("✓ Always inspect data after loading with .info() and .describe()")
print("\nTomorrow: Pandas Deep Dive!")
