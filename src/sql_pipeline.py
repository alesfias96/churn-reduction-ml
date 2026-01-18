import os
import sqlite3
import pandas as pd

from src.config import Config


def create_sqlite_db_from_csv(csv_path: str, cfg: Config) -> str:
    os.makedirs(cfg.SQL_DIR, exist_ok=True)
    db_path = os.path.join(cfg.SQL_DIR, cfg.SQL_DB_FILENAME)

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_csv(csv_path)
        df.to_sql("subscriptions", conn, if_exists="replace", index=False)
    finally:
        conn.close()

    return db_path


def run_sql_queries(db_path: str, sql_file_path: str) -> dict:
    """Execute each query separated by ';-- name:' blocks. Returns a dict(name -> dataframe)."""

    with open(sql_file_path, "r", encoding="utf-8") as f:
        sql_text = f.read()

    # Convention:
    # -- name: query_name
    # SELECT ...;
    blocks = sql_text.split("-- name:")

    results = {}
    conn = sqlite3.connect(db_path)
    try:
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            lines = block.splitlines()
            name = lines[0].strip()
            query = "\n".join(lines[1:]).strip()
            if not query:
                continue
            df_res = pd.read_sql_query(query, conn)
            results[name] = df_res
    finally:
        conn.close()

    return results
