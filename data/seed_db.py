"""Create and seed data/sample.db with a realistic e-commerce schema.

Run once:  python data/seed_db.py
"""

import os
import random
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "sample.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS categories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    description TEXT
);

CREATE TABLE IF NOT EXISTS products (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    name           TEXT    NOT NULL,
    price          REAL    NOT NULL,
    category_id    INTEGER NOT NULL REFERENCES categories(id),
    stock_quantity INTEGER NOT NULL DEFAULT 0,
    created_at     TEXT    DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS customers (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL,
    email      TEXT NOT NULL UNIQUE,
    city       TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS orders (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id  INTEGER NOT NULL REFERENCES customers(id),
    status       TEXT NOT NULL DEFAULT 'pending'
                 CHECK(status IN ('pending','processing','shipped','completed','cancelled')),
    total_amount REAL,
    created_at   TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS order_items (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id   INTEGER NOT NULL REFERENCES orders(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity   INTEGER NOT NULL CHECK(quantity > 0),
    unit_price REAL NOT NULL
);
"""

CATEGORIES = [
    ("Electronics",   "Consumer electronics and gadgets"),
    ("Clothing",      "Apparel and accessories"),
    ("Books",         "Physical and digital books"),
    ("Home & Garden", "Home furnishings and garden supplies"),
    ("Sports",        "Sports equipment and activewear"),
]

PRODUCTS = [
    ("Wireless Headphones", 79.99, 1, 150),
    ("Bluetooth Speaker",   49.99, 1, 200),
    ("USB-C Cable 6ft",     12.99, 1, 500),
    ("Running Shoes",       89.99, 2,  80),
    ("Yoga Mat",            34.99, 5, 120),
    ("Python Crash Course", 29.99, 3, 300),
    ("Clean Code",          39.99, 3, 250),
    ("Coffee Maker",        59.99, 4,  75),
    ("Plant Pot Set",       24.99, 4, 180),
    ("Resistance Bands",    19.99, 5, 220),
]

CUSTOMERS = [
    ("Alice Johnson", "alice@example.com",  "New York"),
    ("Bob Smith",     "bob@example.com",    "Los Angeles"),
    ("Carol White",   "carol@example.com",  "Chicago"),
    ("David Brown",   "david@example.com",  "Houston"),
    ("Emma Davis",    "emma@example.com",   "Phoenix"),
    ("Frank Miller",  "frank@example.com",  "Philadelphia"),
    ("Grace Wilson",  "grace@example.com",  "San Antonio"),
    ("Henry Moore",   "henry@example.com",  "San Diego"),
]

STATUS_OPTIONS = ["pending", "processing", "shipped", "completed", "cancelled"]
STATUS_WEIGHTS = [0.10, 0.15, 0.20, 0.45, 0.10]


def seed():
    if DB_PATH.exists():
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.executescript(SCHEMA_SQL)

    for name, desc in CATEGORIES:
        c.execute("INSERT INTO categories (name, description) VALUES (?, ?)", (name, desc))

    for name, price, cat_id, stock in PRODUCTS:
        c.execute(
            "INSERT INTO products (name, price, category_id, stock_quantity) VALUES (?, ?, ?, ?)",
            (name, price, cat_id, stock),
        )

    for name, email, city in CUSTOMERS:
        c.execute(
            "INSERT INTO customers (name, email, city) VALUES (?, ?, ?)",
            (name, email, city),
        )

    random.seed(42)
    for order_id in range(1, 31):
        customer_id = random.randint(1, len(CUSTOMERS))
        status = random.choices(STATUS_OPTIONS, weights=STATUS_WEIGHTS, k=1)[0]
        c.execute(
            "INSERT INTO orders (customer_id, status, total_amount) VALUES (?, ?, ?)",
            (customer_id, status, 0.0),
        )
        n_items = random.randint(1, 3)
        chosen_products = random.sample(range(1, len(PRODUCTS) + 1), n_items)
        total = 0.0
        for pid in chosen_products:
            c.execute("SELECT price FROM products WHERE id = ?", (pid,))
            price = c.fetchone()[0]
            qty = random.randint(1, 3)
            c.execute(
                "INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES (?, ?, ?, ?)",
                (order_id, pid, qty, price),
            )
            total += price * qty
        c.execute("UPDATE orders SET total_amount = ? WHERE id = ?", (total, order_id))

    conn.commit()
    conn.close()
    print(f"Database seeded at {DB_PATH}")


if __name__ == "__main__":
    seed()
