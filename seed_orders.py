"""
Create a mock orders database (data/orders.db) for the chatbot's order-status tool.

Usage:
    python seed_orders.py                # 200 orders, uses data/catalog.parquet if present
    python seed_orders.py --n 500

Orders reference real parent_asin values from the catalog, so the chatbot can
say "your order of <product title> is out for delivery". All names and
tracking numbers are fake.
"""
import argparse
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

DATA = Path("data")
DB_PATH = DATA / "orders.db"

FIRST = ["Aisha", "Omar", "Fatima", "Yousef", "Sara", "Khalid", "Maryam", "Ahmed",
         "Layla", "Hassan", "Noor", "Rashid", "Hind", "Tariq", "Mona", "Sravan"]
LAST = ["Al Mansoori", "Khan", "Rahman", "Nair", "Hassan", "Al Falasi", "Patel",
        "Ibrahim", "Sharma", "Al Suwaidi", "Ali", "Thomas"]
EMIRATES = ["Dubai", "Dubai", "Dubai", "Abu Dhabi", "Sharjah", "Ajman", "Ras Al Khaimah"]


def load_asins():
    cat = DATA / "catalog.parquet"
    if cat.exists():
        import pandas as pd
        df = pd.read_parquet(cat, columns=["parent_asin"])
        asins = df["parent_asin"].astype(str).tolist()
        if asins:
            return asins
    print("catalog.parquet not found; using placeholder product IDs")
    return [f"MOCK{i:04d}" for i in range(30)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    DATA.mkdir(exist_ok=True)
    asins = load_asins()
    today = date.today()

    if DB_PATH.exists():
        DB_PATH.unlink()
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE orders (
            order_id TEXT PRIMARY KEY,
            customer_name TEXT,
            parent_asin TEXT,
            quantity INTEGER,
            status TEXT,
            order_date TEXT,
            estimated_delivery TEXT,
            delivered_date TEXT,
            city TEXT,
            payment_method TEXT,
            tracking_number TEXT
        )
    """)

    rows = []
    for i in range(args.n):
        order_id = f"ORD-{10000 + i}"
        age_days = random.randint(0, 20)
        order_date = today - timedelta(days=age_days)
        est = order_date + timedelta(days=random.randint(2, 5))
        city = random.choice(EMIRATES)
        payment = random.choice(["Card", "Card", "Apple Pay", "Cash on delivery"])
        delivered = None

        # Status depends on how old the order is, with a few cancellations and returns
        roll = random.random()
        if roll < 0.06:
            status = "Cancelled"
        elif age_days <= 1:
            status = "Processing"
        elif age_days <= 3:
            status = random.choice(["Processing", "Shipped"])
        elif age_days <= 5:
            status = random.choice(["Shipped", "Out for delivery", "Delivered"])
        else:
            status = random.choices(["Delivered", "Returned", "Shipped"], weights=[85, 8, 7])[0]
        if status in ("Delivered", "Returned"):
            delivered = min(today, est + timedelta(days=random.choice([0, 0, 1])))
        tracking = None if status in ("Processing", "Cancelled") else f"TRK{random.randint(10**9, 10**10 - 1)}"

        rows.append((
            order_id,
            f"{random.choice(FIRST)} {random.choice(LAST)}",
            random.choice(asins),
            random.randint(1, 3),
            status,
            order_date.isoformat(),
            est.isoformat(),
            delivered.isoformat() if delivered else None,
            city,
            payment,
            tracking,
        ))

    con.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
    con.commit()

    print(f"Created {DB_PATH} with {len(rows)} orders")
    for status, n in con.execute("SELECT status, COUNT(*) FROM orders GROUP BY status ORDER BY 2 DESC"):
        print(f"  {status:<17} {n}")
    print("\nSample:")
    for r in con.execute("SELECT order_id, status, order_date, estimated_delivery, tracking_number FROM orders LIMIT 5"):
        print("  ", r)
    con.close()


if __name__ == "__main__":
    main()
