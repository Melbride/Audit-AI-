import os
import mysql.connector

# Database connection config. Reads from environment variables with fallback defaults for local development
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", "@Wango005"),
    "database": os.getenv("DB_NAME", "ai_audit"),
}

# Open and return a new MySQL connection using the config above
def get_connection():
    return mysql.connector.connect(**DB_CONFIG)

# FastAPI dependency that yields a database connection and closes it after the request is done
def get_db():
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()

# Initialize all database tables on startup. Safe to call repeatedly — uses CREATE TABLE IF NOT EXISTS so existing tables are never recreated or overwritten
def init_db():
    conn = get_connection()
    # Use dictionary=True so fetchone() returns dict instead of tuple — needed for the ALTER TABLE column checks below
    cursor = conn.cursor(dictionary=True)

    # Column mappings table. Stores confirmed AI or manual mapping per client per file type.
    # UNIQUE KEY prevents duplicate mappings for the same client + file_type + column combination
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS column_mappings (
            id              INT AUTO_INCREMENT PRIMARY KEY,
            client_id       VARCHAR(255) NOT NULL,
            file_type       VARCHAR(100) NOT NULL DEFAULT 'general',
            original_column VARCHAR(255) NOT NULL,
            mapped_to       VARCHAR(255) NOT NULL,
            field_type      VARCHAR(100) NOT NULL DEFAULT 'unknown',
            confirmed_by    VARCHAR(255),
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            UNIQUE KEY uq_column_mapping (client_id, file_type, original_column)
        )
    """)
    # Clients table. Stores company information for each audit client
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS clients (
            client_id INT AUTO_INCREMENT PRIMARY KEY,
            company_name VARCHAR(255) NOT NULL,
            contact_person VARCHAR(255),
            email VARCHAR(255),
            phone VARCHAR(50),
            industry VARCHAR(255),
            address VARCHAR(255),
            status VARCHAR(50) DEFAULT 'Active',
            kra_pin BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Users table. Stores system users including auditors, accountants and admins.Email must be unique. Password is stored as a hash, never plain text
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INT AUTO_INCREMENT PRIMARY KEY,
            full_name VARCHAR(255) NOT NULL,
            email VARCHAR(255) NOT NULL UNIQUE,
            password_hash VARCHAR(255) NOT NULL,
            phone VARCHAR(50),
            role VARCHAR(50) NOT NULL,
            assigned_client_id INT,
            status VARCHAR(50) DEFAULT 'Active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Password resets table. Stores temporary tokens for password reset requests.Token must be unique. Expires after 1 hour
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS password_resets (
            reset_id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            token VARCHAR(255) NOT NULL UNIQUE,
            expires_at DATETIME NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Engagements table. Represents an audit engagement for a client for a specific financial year
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS engagements (
            engagement_id INT AUTO_INCREMENT PRIMARY KEY,
            client_id INT NOT NULL,
            engagement_name VARCHAR(255) NOT NULL,
            financial_year VARCHAR(50) NOT NULL,
            status VARCHAR(50) DEFAULT 'Planning',
            start_date DATE,
            end_date DATE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Audit sections table. Each engagement has multiple sections like Revenue, Expenses, Inventory.Sections can be assigned to a specific user and tracked by status
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_sections (
            section_id INT AUTO_INCREMENT PRIMARY KEY,
            engagement_id INT NOT NULL,
            section_name VARCHAR(255) NOT NULL,
            status VARCHAR(50) DEFAULT 'Pending',
            assigned_to INT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Engagement team table. Links users to engagements with a role. UNIQUE KEY prevents the same user being added to the same engagement twice
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS engagement_team (
            team_id INT AUTO_INCREMENT PRIMARY KEY,
            engagement_id INT NOT NULL,
            user_id INT NOT NULL,
            role VARCHAR(100) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY uq_engagement_team (engagement_id, user_id)
        )
    """)
    # Submissions table. Tracks work submitted by auditors for review within an engagement section
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS submissions (
            submission_id INT AUTO_INCREMENT PRIMARY KEY,
            engagement_id INT NOT NULL,
            section_id INT NOT NULL,
            submitted_by INT NOT NULL,
            status VARCHAR(50) DEFAULT 'Draft',
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Notifications table. Stores in-app alerts sent to users when submissions are ready for review
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS notifications (
            notification_id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            message TEXT NOT NULL,
            type VARCHAR(100) DEFAULT 'engagement_alert',
            is_read BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Uploads table. Tracks every file uploaded per client for audit trail. File_id is a UUID generated at upload time and must be unique
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS uploads (
            id          INT AUTO_INCREMENT PRIMARY KEY,
            file_id     VARCHAR(255) NOT NULL UNIQUE,
            client_id   VARCHAR(255) NOT NULL,
            filename    VARCHAR(255) NOT NULL,
            file_name   VARCHAR(255),
            file_type   VARCHAR(100) NOT NULL,
            file_path   VARCHAR(500),
            rows        INT,
            status      VARCHAR(50) DEFAULT 'uploaded',
            upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Safety checks, add columns to uploads table if they were missing from an older version of the schema
    cursor.execute("SELECT COUNT(*) AS count FROM information_schema.columns WHERE table_schema = %s AND table_name = 'uploads' AND column_name = 'file_name'", (DB_CONFIG["database"],))
    if cursor.fetchone()["count"] == 0:
        cursor.execute("ALTER TABLE uploads ADD COLUMN file_name VARCHAR(255) NULL")
    cursor.execute("SELECT COUNT(*) AS count FROM information_schema.columns WHERE table_schema = %s AND table_name = 'uploads' AND column_name = 'file_path'", (DB_CONFIG["database"],))
    if cursor.fetchone()["count"] == 0:
        cursor.execute("ALTER TABLE uploads ADD COLUMN file_path VARCHAR(500) NULL")
    # The upload_date column was added later to track the original upload timestamp separately from upload_time which gets updated on status changes. 
    # This check ensures it gets added to existing tables without affecting new ones.
    cursor.execute("SELECT COUNT(*) AS count FROM information_schema.columns WHERE table_schema = %s AND table_name = 'uploads' AND column_name = 'upload_date'", (DB_CONFIG["database"],))
    if cursor.fetchone()["count"] == 0:
        cursor.execute("ALTER TABLE uploads ADD COLUMN upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP")

    conn.commit()
    conn.close()

# Save a confirmed column mapping for a client to the database.
# Mapping is a dict of { "original_column": { "mapped_to": "amount", "field_type": "numeric" } }
# If a mapping already exists for this client + file_type + column it will be updated not duplicated
def save_mapping(client_id: str, file_type: str, mapping: dict, confirmed_by: str = None):
    conn = get_connection()
    cursor = conn.cursor()
    for original_column, info in mapping.items():
        # Handle both old format (string) and new format (dict) to ensure backwards compatibility
        if isinstance(info, dict):
            mapped_to  = str(info.get("mapped_to", "unknown"))
            field_type = str(info.get("field_type", "unknown"))
        else:
            mapped_to  = str(info)
            field_type = "unknown"
        # Insert new mapping or update existing one if the column already exists for this client and file type
        cursor.execute("""
            INSERT INTO column_mappings
                (client_id, file_type, original_column, mapped_to, field_type, confirmed_by, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            ON DUPLICATE KEY UPDATE
                mapped_to    = VALUES(mapped_to),
                field_type   = VALUES(field_type),
                confirmed_by = VALUES(confirmed_by),
                updated_at   = CURRENT_TIMESTAMP
        """, (client_id, file_type, original_column, mapped_to, field_type, confirmed_by))
    conn.commit()
    conn.close()

# Retrieve the saved column mapping for a client and file type from the database.Returns a dict of { "original_column": { "mapped_to": "amount", "field_type": "numeric" } }
# Returns empty dict if no mapping has been saved for this client yet
def get_mapping(client_id: str, file_type: str = "general") -> dict:
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    # Fetch all saved mappings for this client and file type
    cursor.execute("""
        SELECT original_column, mapped_to, field_type
        FROM column_mappings
        WHERE client_id = %s AND file_type = %s
    """, (client_id, file_type))
    rows = cursor.fetchall()
    conn.close()
    # Return empty dict if no mappings found
    if not rows:
        return {}
    # Convert rows into a nested dict keyed by original column name
    return {
        row["original_column"]: {
            "mapped_to":  row["mapped_to"],
            "field_type": row["field_type"]
        }
        for row in rows
    }

# Save an upload record to the database after a file is successfully uploaded.ON DUPLICATE KEY UPDATE prevents duplicate records if the same file_id is uploaded twice
def save_upload(file_id: str, client_id: str, filename: str, file_type: str, rows: int):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO uploads (file_id, client_id, filename, file_type, rows) VALUES (%s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE file_id = file_id",
        (file_id, client_id, filename, file_type, rows)
    )
    conn.commit()
    conn.close()

# Get all upload records for a client ordered by most recent first.Returns an empty list if the client has no uploads yet
def get_uploads(client_id: str) -> list:
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT * FROM uploads
        WHERE client_id = %s
        ORDER BY upload_time DESC
    """, (client_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


