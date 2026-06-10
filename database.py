import sqlite3
import os

# Path to the SQLite database file
DB_PATH = "auditiq.db"

# Get a connection to the SQLite database
def get_connection():
    """
    Get a connection to the SQLite database.
    row_factory makes rows return as dictionaries
    so we can access columns by name e.g. row["client_id"]
    instead of by index e.g. row[0]
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# Function to initialize the database and create tables if they don't exist
def init_db():
    """
    Create database tables if they don't exist.
    Runs once automatically when the app starts.
    Safe to call repeatedly — won't recreate existing tables.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Column mappings table. Stores confirmed mapping per client per file type
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS column_mappings (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            client_id       TEXT NOT NULL,
            file_type       TEXT NOT NULL DEFAULT 'general',
            original_column TEXT NOT NULL,
            mapped_to       TEXT NOT NULL,
            field_type      TEXT NOT NULL DEFAULT 'unknown',
            confirmed_by    TEXT,
            created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(client_id, file_type, original_column)
        )
    """)

    # Upload history table. Tracks every file uploaded per client for audit trail
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS uploads (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id     TEXT NOT NULL UNIQUE,
            client_id   TEXT NOT NULL,
            filename    TEXT NOT NULL,
            file_type   TEXT NOT NULL,
            rows        INTEGER,
            status      TEXT DEFAULT 'uploaded',
            upload_time DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    # Add field_type column to existing database if it doesn't exist yet
    try:
        cursor.execute("ALTER TABLE column_mappings ADD COLUMN field_type TEXT NOT NULL DEFAULT 'unknown'")
        conn.commit()
    except Exception:
        # Column already exists, safe to ignore
        pass
    conn.close()

# Save a mapping to the database
def save_mapping(client_id: str, file_type: str, mapping: dict, confirmed_by: str = None):
    """
    Save confirmed column mapping for a client to the database.
    mapping is now a dict of:
    {
      "original_column": {
        "mapped_to": "amount",
        "field_type": "numeric"
      }
    }
    If mapping already exists for this client + file_type + column
    it will be updated not duplicated.
    """
    conn = get_connection()
    cursor = conn.cursor()
    for original_column, info in mapping.items():
        # Handle both old format (string) and new format (dict). This ensures backwards compatibility
        if isinstance(info, dict):
            mapped_to  = str(info.get("mapped_to", "unknown"))
            field_type = str(info.get("field_type", "unknown"))
        else:
            mapped_to  = str(info)
            field_type = "unknown"
        # Insert or update the mapping for each column
        cursor.execute("""
            INSERT INTO column_mappings
                (client_id, file_type, original_column, mapped_to, field_type, confirmed_by, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(client_id, file_type, original_column)
            DO UPDATE SET
                mapped_to    = excluded.mapped_to,
                field_type   = excluded.field_type,
                confirmed_by = excluded.confirmed_by,
                updated_at   = CURRENT_TIMESTAMP
        """, (client_id, file_type, original_column, mapped_to, field_type, confirmed_by))
    conn.commit()
    conn.close()

# Function to get a mapping from the database
def get_mapping(client_id: str, file_type: str = "general") -> dict:
    """
    Retrieve saved mapping for a client from the database.
    Returns a dictionary of:
    {
      "original_column": {
        "mapped_to": "amount",
        "field_type": "numeric"
      }
    }
    Returns empty dict if no mapping found.
    """
    conn = get_connection()
    cursor = conn.cursor()
    # Query for all mappings for this client and file type
    cursor.execute("""
        SELECT original_column, mapped_to, field_type
        FROM column_mappings
        WHERE client_id = ? AND file_type = ?
    """, (client_id, file_type))
    rows = cursor.fetchall()
    conn.close()

    # Return empty dict if no mapping found
    if not rows:
        return {}
    # Convert rows into dictionary with mapped_to and field_type
    return {
        row["original_column"]: {
            "mapped_to":  row["mapped_to"],
            "field_type": row["field_type"]
        }
        for row in rows
    }

# Function to save an upload record to the database
def save_upload(file_id: str, client_id: str, filename: str, file_type: str, rows: int):
    """
    Save an upload record to the database.
    Called after every successful file upload.
    INSERT OR IGNORE prevents duplicate records.
    """
    conn = get_connection()
    cursor = conn.cursor()
    # Use INSERT OR IGNORE to prevent duplicate records if file_id already exists
    cursor.execute(
        "INSERT OR IGNORE INTO uploads (file_id, client_id, filename, file_type, rows) VALUES (?, ?, ?, ?, ?)",
        (file_id, client_id, filename, file_type, rows)
    )
    conn.commit()
    conn.close()

# Function to get upload history for a client
def get_uploads(client_id: str) -> list:
    """
    Get all upload records for a client ordered by most recent first.
    Returns empty list if client has no uploads yet.
    """
    conn = get_connection()
    cursor = conn.cursor()
    # Query for all uploads for this client ordered by most recent first
    cursor.execute("""
        SELECT * FROM uploads
        WHERE client_id = ?
        ORDER BY upload_time DESC
    """, (client_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


