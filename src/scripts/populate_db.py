import os
import csv
import psycopg2
from psycopg2.extras import execute_batch
from dotenv import load_dotenv
load_dotenv()

LOOKS_CSV_PATH = os.getenv("LOOKS_CSV_PATH")
PIECES_CSV_PATH = os.getenv("PIECES_CSV_PATH")

S3_BUCKET = os.getenv("S3_BUCKET_NAME")

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
}

def get_connection():
    return psycopg2.connect(**DB_CONFIG)

def create_tables(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS looks (
                look_number TEXT PRIMARY KEY,
                image_path   TEXT NOT NULL
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS pieces (
                piece_id    SERIAL PRIMARY KEY,
                look_number TEXT NOT NULL,
                name        TEXT,
                ref_code    TEXT,
                category    TEXT,
                subcategory TEXT,
                notes       TEXT,
                CONSTRAINT fk_look
                    FOREIGN KEY (look_number)
                    REFERENCES looks (look_number)
                    ON DELETE CASCADE
            );
        """)

    conn.commit()

def ingest_looks(conn, csv_path):
    records = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            look_number = row["look_number"].strip()
            filename = row["image_path"].strip()
            image_uri = f"s3://{S3_BUCKET}/{filename}"
            records.append((look_number, image_uri))

    insert_sql = """
        INSERT INTO looks (look_number, image_path)
        VALUES (%s, %s)
        ON CONFLICT (look_number) DO NOTHING;
    """

    with conn.cursor() as cur:
        execute_batch(cur, insert_sql, records)

    conn.commit()

def ingest_pieces(conn, csv_path):
    records = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append((
                row["look_number"].strip(),
                row.get("name"),
                row.get("ref_code"),
                row.get("category"),
                row.get("subcategory"),
                row.get("notes"),
            ))

    insert_sql = """
        INSERT INTO pieces (
            look_number,
            name,
            ref_code,
            category,
            subcategory,
            notes
        )
        VALUES (%s, %s, %s, %s, %s, %s);
    """

    with conn.cursor() as cur:
        execute_batch(cur, insert_sql, records)

    conn.commit()

def main():
    conn = get_connection()

    try:
        create_tables(conn)

        ingest_looks(conn, LOOKS_CSV_PATH)
        ingest_pieces(conn, PIECES_CSV_PATH)

        print("Ingestion completed successfully.")

    finally:
        conn.close()

if __name__ == "__main__":
    main()
