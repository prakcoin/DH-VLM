import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

model = SentenceTransformer("google/embeddinggemma-300m")

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
}

def get_connection():
    return psycopg2.connect(**DB_CONFIG)

def piece_to_text(piece):
    parts = []

    if piece["name"]:
        parts.append(f"Garment: {piece['name']}.")

    if piece["category"]:
        parts.append(f"Category: {piece['category']}.")

    if piece["subcategory"]:
        parts.append(f"Subcategory: {piece['subcategory']}.")

    if piece["notes"]:
        parts.append(f"Notes: {piece['notes']}.")

    parts.append(f"Part of runway look {piece['look_number']}.")

    return " ".join(parts)

def embed_pieces():
    conn = get_connection()

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT
                piece_id,
                name,
                category,
                subcategory,
                notes,
                look_number
            FROM pieces
            WHERE embedding IS NULL;
        """)

        rows = cur.fetchall()
        print(f"Embedding {len(rows)} pieces...")

        for row in rows:
            text = piece_to_text(row)

            embedding = model.encode(
                text,
                normalize_embeddings=True
            ).tolist()

            cur.execute(
                """
                UPDATE pieces
                SET embedding = %s
                WHERE piece_id = %s;
                """,
                (embedding, row["piece_id"])
            )

    conn.commit()
    conn.close()
    print("Embedding complete.")

def create_vector_index():
    conn = get_connection()

    with conn.cursor() as cur:
        cur.execute("""
            CREATE INDEX IF NOT EXISTS pieces_embedding_idx
            ON pieces
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)

    conn.commit()
    conn.close()
    print("Index created.")


if __name__ == "__main__":
    embed_pieces()
    create_vector_index()
