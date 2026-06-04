import json
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def detect_columns_with_llm(columns: list, sample_values: dict = None) -> dict:
    
    # Normalize columns to lowercase for LLM
    # Keep a mapping of lowercase → original so we can restore later
    lower_to_original = {}
    for col in columns:
        lower_to_original[col.lower().strip()] = col

    lowercase_columns = list(lower_to_original.keys())

    # Normalize sample values keys to lowercase too
    lowercase_samples = {}
    if sample_values:
        for col, val in sample_values.items():
            lowercase_samples[col.lower().strip()] = val

    # Build column context for LLM
    if lowercase_samples:
        columns_context = json.dumps([
            {"column": col, "sample": str(lowercase_samples.get(col, ""))}
            for col in lowercase_columns
        ])
        columns_instruction = "Column names and a sample value from the uploaded file:"
    else:
        columns_context = json.dumps(lowercase_columns)
        columns_instruction = "Column names from the uploaded file:"

    prompt = f"""You are a financial data expert helping an audit firm.

Your job is to map Excel column names to their standard financial meaning.
{columns_instruction}
{columns_context}

Instructions:
- Map each column to a simple lowercase snake_case standard financial field name
- Use the sample values as context clues when the column name is ambiguous
- Use names like: date, amount, debit, credit, account, description,
  reference, currency, status, tax, tax_amount, approved_by, vendor,
  net_amount, department, invoice_number, po_number, transaction_id,
  transaction_type, balance, running_balance, dr_cr_indicator etc.
- Stay strictly within financial and accounting context
- Use your financial knowledge to map every column confidently
- Only use "unknown" as a last resort when the column has absolutely 
  no financial meaning — like "Unnamed: 15" or random codes
- Empty sample values are fine — use the column name itself to decide
- You already know financial terminology — trust your knowledge
- Do not be overly conservative
- Return ONLY a valid JSON object — no text before or after

Example output format:
{{
  "date": "date",
  "amt": "amount",
  "vendor name": "vendor",
  "trans. type": "transaction_type"
}}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a financial data mapping assistant for an audit firm. You only return valid JSON. No explanations. No markdown. No backticks."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        max_tokens=1000,
    )

    # Parse LLM response
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        lowercase_mapping = json.loads(raw)
    except json.JSONDecodeError:
        lowercase_mapping = {col: "unknown" for col in lowercase_columns}

    # Map results back to original column names
    final_mapping = {}
    for lower_col, original_col in lower_to_original.items():
        mapped_field = lowercase_mapping.get(lower_col, "unknown")
        if not isinstance(mapped_field, str):
            mapped_field = "unknown"
        final_mapping[original_col] = mapped_field

    return final_mapping


def build_detection_result(columns: list, final_mapping: dict) -> dict:
    """
    Build the full detection result with warnings.
    """
    unknown_columns = [col for col, field in final_mapping.items() if field == "unknown"]

    warnings = []
    if unknown_columns:
        warnings.append({
            "type": "unknown_columns",
            "message": f"{len(unknown_columns)} column(s) could not be detected: {unknown_columns}",
            "action": "Please map these manually in the next step."
        })

    return {
        "total_columns": len(columns),
        "mapping": final_mapping,
        "unknown_count": len(unknown_columns),
        "warnings": warnings,
        "requires_manual_mapping": len(unknown_columns) > 0
    }