import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

ADMIN_DB_CONFIG = {
    "dbname": "postgres",
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
}

TARGET_DB = os.getenv("DB_NAME")

def full_reset_database():
    conn = psycopg2.connect(**ADMIN_DB_CONFIG)
    conn.autocommit = True  # REQUIRED

    try:
        with conn.cursor() as cur:
            # Kill active connections
            cur.execute("""
                SELECT pg_terminate_backend(pid)
                FROM pg_stat_activity
                WHERE datname = %s
                  AND pid <> pg_backend_pid();
            """, (TARGET_DB,))

            # Drop and recreate
            cur.execute(f'DROP DATABASE IF EXISTS "{TARGET_DB}";')
            cur.execute(f'CREATE DATABASE "{TARGET_DB}";')

        print(f"Database '{TARGET_DB}' fully reset.")

    finally:
        conn.close()
        
if __name__ == "__main__":
    full_reset_database()