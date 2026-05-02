# Conversational Text-to-SQL System

[![Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://text-to-sql-rahul.streamlit.app/)

A schema-aware text-to-SQL pipeline powered by GLM-4.7 via the NVIDIA API. Ask questions in plain English and get SQL-backed answers from a SQLite database — with automatic validation, hallucination prevention, and a benchmark harness.

## Key Features

- **Schema-aware prompting** — live `CREATE TABLE` schema injected into every prompt; the model cannot hallucinate table or column names it can see written out
- **Auto-correction loop** — `sqlglot` AST validation + SQLite execution errors are fed back to the LLM for up to 3 self-correction attempts
- **Read-only safety** — `PRAGMA query_only = ON` blocks any destructive SQL the model might generate
- **Benchmark harness** — Exact Match, Execution Match, and Pass@k metrics over a labeled NL→SQL dataset

## Tech Stack

| Layer | Choice |
|---|---|
| LLM | GLM-4.7 via NVIDIA free-tier API (OpenAI-compatible) |
| Database | SQLite + SQLAlchemy schema introspection |
| SQL validation | `sqlglot` AST parser |
| UI | Streamlit (3 tabs: Chat, Schema Browser, Benchmark) |

## Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/rahuljuluru92/Text-to-SQL
cd Text-to-SQL

# 2. Install dependencies
pip3 install -r requirements.txt

# 3. Configure secrets
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml and add your NVIDIA_API_KEY

# 4. Seed the demo database
python3 data/seed_db.py

# 5. Run the app
python3 -m streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Streamlit Community Cloud Deployment

1. Fork or push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select this repo, branch `main`, entry point `app.py`
4. Open **App settings → Secrets** and paste:
   ```toml
   NVIDIA_API_KEY = "nvapi-your-key-here"
   DB_PATH = "data/sample.db"
   ```
5. Deploy — the app will be live at a public URL

## Benchmark

The Benchmark tab ships with 20 labeled NL→SQL pairs covering aggregates, joins, filters, and grouping. You can also upload your own JSON dataset:

```json
[
  {"question": "How many users signed up last month?", "gold_sql": "SELECT COUNT(*) FROM users WHERE ..."},
  ...
]
```


### Questions that can be asked are :

## Simple (single table)
How many customers are there?
List all product names and their prices
Which products cost more than $50?
How many orders are in each status?
What is the most expensive product?
Which customers are from New York?
What is the average order total?
Which product has the most stock?
## Medium (joins + aggregates)
Which products are in the Electronics category?
How many orders does each customer have?
What is the total revenue from completed orders?
Show me the order items for order number 5
How many products are in each category?
Which customers have never placed an order?
What is the total quantity sold for each product?
## Advanced (multi-table + grouping)
Who are the top 3 customers by total spend on completed orders?
What is the most popular product by units sold?
Which category generates the most revenue?
Show me all orders with more than 2 items
What is the average number of items per order?
Which customers placed orders that are still pending or processing?
What percentage of orders are completed?
## Edge cases (tests hallucination prevention)
How many orders were placed in the last 7 days? ← tests date handling with no date column populated
What is the profit margin on each product? ← unanswerable, no cost data
Show me all users ← wrong table name — tests if the model corrects itself to customers



**Metrics reported:**
- **Exact Match (EM)** — normalized string comparison
- **Execution Match** — result-set equality (order-insensitive)
- **Pass@1 / @2 / @3** — correct within k correction attempts

## Project Structure

```
app.py                  # Streamlit entry point
core/
  db.py                 # SQLite + SQLAlchemy schema introspection
  llm.py                # NVIDIA GLM-4.7 streaming client
  validator.py          # sqlglot AST table-name validator
  pipeline.py           # generate → validate → execute → correct loop
  benchmark.py          # EM, Execution Match, Pass@k metrics
prompts/
  templates.py          # System prompt + correction prompt templates
data/
  seed_db.py            # Creates and seeds sample.db
  sample.db             # Demo e-commerce SQLite database
  benchmark_sample.json # 20 labeled NL→SQL pairs
```
