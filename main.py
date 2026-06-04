from fastapi import FastAPI, HTTPException, UploadFile, File, Form
import pdfplumber
from docx import Document 
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json
import uuid
import os
import shutil
from dotenv import load_dotenv
from detector import detect_columns_with_llm, build_detection_result

load_dotenv()
app = FastAPI()

# CORS Middleware, allow access from any origin, for all methods and headers for frontend-backend communication.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory where uploaded files will be stored
UPLOAD_DIR = "uploads"
# Supported file extensions for upload
ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls", "pdf", "docx"}


# Extract file extension from filename
def get_extension(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


# Extract tables from PDF using pdfplumber, falls back to text extraction if no tables found
def extract_pdf(file_path: str):
    tables = []
    full_text = ""

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            # Try tables first
            page_tables = page.extract_tables()
            for table in page_tables:
                if table:
                    headers = table[0]
                    rows = table[1:]
                    df = pd.DataFrame(rows, columns=headers)
                    tables.append(df)
            # Also grab text as fallback
            full_text += page.extract_text() or ""

    if tables:
        return pd.concat(tables, ignore_index=True), "table"

    # No tables found, return text as single column dataframe
    lines = [line.strip() for line in full_text.splitlines() if line.strip()]
    if lines:
        return pd.DataFrame({"raw_text": lines}), "text"
    return None, None


# Extract tables from DOCX using python-docx, falls back to paragraphs if no tables found
def extract_docx(file_path: str):
    doc = Document(file_path)
    tables = []

    # Try tables first
    for table in doc.tables:
        headers = [cell.text.strip() for cell in table.rows[0].cells]
        rows = []
        for row in table.rows[1:]:
            rows.append([cell.text.strip() for cell in row.cells])
        df = pd.DataFrame(rows, columns=headers)
        tables.append(df)

    if tables:
        return pd.concat(tables, ignore_index=True), "table"

    # No tables found, fall back to paragraphs
    lines = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    if lines:
        return pd.DataFrame({"raw_text": lines}), "text"
    return None, None

# Read any supported file and return a dataframe
def read_file_to_df(save_path: str, ext: str):
    if ext == "csv":
        return pd.read_csv(save_path)
    elif ext in ["xlsx", "xls"]:
        return pd.read_excel(save_path)
    elif ext == "pdf":
        df, _ = extract_pdf(save_path)
        return df
    elif ext == "docx":
        df, _ = extract_docx(save_path)
        return df
    return None

# Root endpoint to test whether the API is running
@app.get("/")
def root():
    return {"message": "Audit AI API is running"}

# Endpoint for uploading files
@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    client_id: str = Form(...)
):
    # Validate file extension
    ext = get_extension(file.filename)
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"File type .{ext} not supported. Upload Excel, CSV, PDF or DOCX file only."
        )

    # Check file size
    MAX_FILE_SIZE = 50
    file_bytes = await file.read()
    file_size_mb = len(file_bytes) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds the maximum limit of {MAX_FILE_SIZE} MB. Uploaded file size: {file_size_mb:.2f} MB."
        )

    # Reset file pointer after reading for size check
    file.file.seek(0)

    # Save file to uploads directory
    file_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{file_id}.{ext}")
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # PDF
    if ext == "pdf":
        df, source = extract_pdf(save_path)
        if df is None:
            raise HTTPException(status_code=400, detail="Could not extract any content from PDF.")
        return {
            "file_id": file_id,
            "client_id": client_id,
            "filename": file.filename,
            "source": source,
            "rows": len(df),
            "columns": list(df.columns),
            "preview": df.head(5).fillna("").to_dict(orient="records"),
            "message": f"PDF uploaded — extracted via {source}"
        }

    # DOCX
    if ext == "docx":
        df, source = extract_docx(save_path)
        if df is None:
            raise HTTPException(status_code=400, detail="Could not extract any content from DOCX.")
        return {
            "file_id": file_id,
            "client_id": client_id,
            "filename": file.filename,
            "source": source,
            "rows": len(df),
            "columns": list(df.columns),
            "preview": df.head(5).fillna("").to_dict(orient="records"),
            "message": f"DOCX uploaded — extracted via {source}"
        }

    # CSV / EXCEL
    try:
        if ext == "csv":
            df = pd.read_csv(save_path)
        else:
            df = pd.read_excel(save_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read file: {str(e)}")
    return {
        "file_id": file_id,
        "client_id": client_id,
        "filename": file.filename,
        "source": "table",
        "rows": len(df),
        "columns": list(df.columns),
        "preview": df.head(5).fillna("").to_dict(orient="records"),
        "message": "File uploaded and processed successfully"
    }

# Endpoint for detecting column meanings using Groq LLM
@app.post("/detect-columns")
async def detect_columns_endpoint(
    file_id: str = Form(...),
    columns: str = Form(...)
):
    # Parse columns list
    try:
        columns_list = json.loads(columns)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid columns format.")

    # Find the saved file using file_id
    save_path = None
    file_ext = None
    for extension in ALLOWED_EXTENSIONS:
        path = os.path.join(UPLOAD_DIR, f"{file_id}.{extension}")
        if os.path.exists(path):
            save_path = path
            file_ext = extension
            break
    if not save_path:
        raise HTTPException(status_code=404, detail="File not found. Please upload the file first.")

    # Read file and extract first non-empty value per column as sample
    try:
        df = read_file_to_df(save_path, file_ext)
        if df is None:
            raise HTTPException(status_code=400, detail="Could not read file.")

        # Get first non-empty value per column
        sample_values = {}
        for col in columns_list:
            if col in df.columns:
                non_empty = df[col].dropna().replace("", float("nan")).dropna()
                sample_values[col] = str(non_empty.iloc[0]) if len(non_empty) > 0 else ""
            else:
                sample_values[col] = ""

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read file: {str(e)}")

    # Run LLM detection with column names + sample values
    try:
        mapping = detect_columns_with_llm(columns_list, sample_values)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

    # Build result
    result = build_detection_result(columns_list, mapping)
    result["file_id"] = file_id
    return result



    