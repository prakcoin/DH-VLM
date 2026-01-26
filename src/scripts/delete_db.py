import os
import csv
import psycopg2
import boto3
from psycopg2.extras import execute_batch
from dotenv import load_dotenv
load_dotenv()

# ------------------------
# Configuration
# ------------------------

LOOKS_CSV_PATH = "data/Labels/Image Paths.csv"
PIECES_CSV_PATH = "data/Labels/Clothing Items.csv"

S3_BUCKET = os.getenv("S3_BUCKET_NAME")

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
}

def get_connection():
    return psycopg2.connect(**DB_CONFIG)

def reset_tables(conn):
    with conn.cursor() as cur:
        cur.execute("""
            TRUNCATE TABLE pieces, looks
            RESTART IDENTITY
            CASCADE;
        """)
    conn.commit()

def main():
    conn = get_connection()

    try:

        reset_tables(conn)
        print("Deletion completed successfully.")

    finally:
        conn.close()

if __name__ == "__main__":
    main()
