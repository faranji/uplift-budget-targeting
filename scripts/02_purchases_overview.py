"""
Step 02: First look at the X5 `purchases` table (the purchase history BEFORE the campaign).

HOW THIS FILE DIFFERS FROM STEP 01
  Step 01 looked at the EXPERIMENT: who received the communication and who bought afterwards
  (uplift_train.csv.gz, 200,039 rows) plus age and gender (clients.csv.gz).
  This step looks at a THIRD, much bigger file: purchases.csv.gz, about 45.8 million rows, which
  holds what customers bought in the past, before the communication was sent. It is the raw
  material for new customer variables such as "days since the last purchase", "number of
  purchases in the last 90 days" and "average spend" (RFM-style features, built in step 03).
  We do not know its columns yet, so this script only prints what is inside.

WHY DUCKDB
  45.8 million rows do not fit comfortably in a normal pandas DataFrame. DuckDB reads the compressed
  file directly and answers SQL questions about it without loading everything into memory.

HOW TO RUN (from the repository root):
    pip install duckdb
    mkdir -p data
    curl -L -o data/purchases.csv.gz https://sklift.s3.eu-west-2.amazonaws.com/purchases.csv.gz
    python scripts/02_purchases_overview.py
  (The download is large and can take several minutes. data/ is ignored by git.)
"""

import os

import duckdb

PURCHASES_FILE = "data/purchases.csv.gz"
TRAIN_FILE = "data/uplift_train.csv.gz"
CLIENT_COL = "client_id"  # the column that identifies the customer; change it if section 1 shows another name

for path in (PURCHASES_FILE, TRAIN_FILE):
    if not os.path.exists(path):
        raise SystemExit(f"File not found: {path}  (see 'HOW TO RUN' at the top of this file)")

con = duckdb.connect()

# Read client_id as text in both files so that leading zeros are kept and the two tables can be joined.
# A "view" is just a named query: nothing is loaded into memory yet.
for view, path in (("purchases", PURCHASES_FILE), ("train", TRAIN_FILE)):
    try:
        con.execute(f"CREATE VIEW {view} AS SELECT * FROM read_csv_auto('{path}', types={{'{CLIENT_COL}': 'VARCHAR'}})")
    except duckdb.Error:
        # The file has no column called CLIENT_COL: show the real column names so that it can be fixed.
        names = [row[0] for row in con.execute(f"DESCRIBE SELECT * FROM read_csv_auto('{path}')").fetchall()]
        raise SystemExit(f"No column '{CLIENT_COL}' in {path}. Columns found: {names}. Set CLIENT_COL at the top of this file.")

# ----------------------------------------------------------------------------------------
# 1. Columns and first rows
# ----------------------------------------------------------------------------------------
# HOW TO READ: this shows what one row of the table means. We expect one row per purchased item
# (or per receipt), with a customer id, a date and an amount, but the columns below are the truth.
print("=== 1. Columns and first rows ===")
print(f"File size (compressed): {os.path.getsize(PURCHASES_FILE) / 1e6:,.0f} MB")
for column_name, column_type, *_ in con.execute("DESCRIBE purchases").fetchall():
    print(f"  {column_name:30s} {column_type}")
print(con.execute("SELECT * FROM purchases LIMIT 5").fetchdf().to_string())

# ----------------------------------------------------------------------------------------
# 2. Size and a summary of every column
# ----------------------------------------------------------------------------------------
# HOW TO READ: min / max show the range of every column (for date columns this is the period the
# history covers). null_percentage is the share of missing values: a column with many missing
# values is a poor feature. approx_unique is the approximate number of different values.
print("\n=== 2. Size and column summary ===")
rows = con.execute("SELECT count(*) FROM purchases").fetchone()[0]
print(f"Total rows: {rows:,}")
summary = con.execute("SUMMARIZE purchases").fetchdf()
print(summary[["column_name", "column_type", "min", "max", "approx_unique", "null_percentage"]].to_string(index=False))

# ----------------------------------------------------------------------------------------
# 3. How many experiment customers have a purchase history?
# ----------------------------------------------------------------------------------------
# WHY: features built from purchases only exist for customers who appear in this table.
# HOW TO READ: "with_history_%" should be high and, above all, almost equal in the two groups
# (treatment_flg 0 and 1). If one group had clearly more history, the groups would no longer be
# comparable after we add history-based features.
print("\n=== 3. Share of experiment customers with purchase history ===")
query = f"""
    WITH customers_with_history AS (SELECT DISTINCT {CLIENT_COL} AS client_id FROM purchases)
    SELECT train.treatment_flg,
           count(*) AS customers,
           count(customers_with_history.client_id) AS with_history,
           round(100.0 * count(customers_with_history.client_id) / count(*), 2) AS with_history_pct
    FROM train
    LEFT JOIN customers_with_history ON train.client_id = customers_with_history.client_id
    GROUP BY train.treatment_flg
    ORDER BY train.treatment_flg
"""
print(con.execute(query).fetchdf().to_string(index=False))

# ----------------------------------------------------------------------------------------
# 4. How many purchase rows does a typical customer have?
# ----------------------------------------------------------------------------------------
# HOW TO READ: the median is the "typical" customer, the quartiles show the spread. A customer with
# very few rows has a short history, so history-based features will be noisy for that customer.
print("\n=== 4. Purchase rows per customer ===")
query = f"""
    SELECT count(*) AS customers,
           min(rows_per_customer) AS fewest,
           quantile_cont(rows_per_customer, 0.25) AS quartile_1,
           quantile_cont(rows_per_customer, 0.50) AS median,
           quantile_cont(rows_per_customer, 0.75) AS quartile_3,
           max(rows_per_customer) AS most
    FROM (SELECT {CLIENT_COL}, count(*) AS rows_per_customer FROM purchases GROUP BY {CLIENT_COL})
"""
print(con.execute(query).fetchdf().round(1).to_string(index=False))
