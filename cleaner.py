import pandas as pd
import re
from datetime import datetime


# Main function to clean the dataframe based on the confirmed mapping
def clean_dataframe(df: pd.DataFrame, mapping: dict) -> tuple:
    """
    Main cleaning function, takes a raw dataframe and confirmed column mapping.
    Mapping structure: {original_col: {"mapped_to": "amount", "field_type": "numeric"}}
    Returns cleaned dataframe and a validation report.
    """
    # Keep a copy of original dataframe for comparison
    original_df = df.copy()

    # Track all issues found
    issues = []

    #Rename columns using confirmed mapping.Must happen first so cleaning functions can find columns by standard names
    df = rename_columns(df, mapping)
    # Clean date columns
    df = clean_dates(df, mapping, issues)
    # Clean amount columns
    df = clean_amounts(df, mapping, issues)
    # Standardize text column casing
    df = standardize_casing(df, mapping, issues)
    # Handle null values
    df = handle_nulls(df, mapping, issues)
    # Handle duplicates
    df = handle_duplicates(df, mapping, issues)
    # Build the validation report
    report = build_validation_report(df, original_df, issues)
    return df, report

# Function to rename columns based on the confirmed mapping
def rename_columns(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """
    Rename original columns to their mapped standard names.Extracts mapped_to from the new mapping structure.
    Must be called before any cleaning step so cleaning functions. Can find columns by their standard names.
    """
    # Build a rename dictionary from the new mapping structure.{original_col: mapped_to}
    rename_dict = {
        original_col: info["mapped_to"]
        for original_col, info in mapping.items()
        if isinstance(info, dict) and "mapped_to" in info
    }
    df = df.rename(columns=rename_dict)
    return df

# Function to clean and standardize date columns
def clean_dates(df: pd.DataFrame, mapping: dict, issues: list) -> pd.DataFrame:
    """
    Find all columns with field_type 'date' using the confirmed mapping.Standardizes all date values to YYYY-MM-DD format.
    Uses DD/MM/YYYY as the standard input format.Flags any dates that cannot be parsed for auditor review.
    """
    # Find all columns whose field_type is date using the mapping. Look at mapped_to names because columns were already renamed.
    date_columns = [
        info["mapped_to"]
        for info in mapping.values()
        if isinstance(info, dict) and info.get("field_type") == "date"
        and info.get("mapped_to") in df.columns
    ]
    for col in date_columns:
        for idx, value in df[col].items():
            # Skip empty values.
            if pd.isna(value) or str(value).strip() == "":
                continue
            original_value = value
            cleaned = None
            # Parse using day first, standard format (DD/MM/YYYY)
            try:
                cleaned = pd.to_datetime(str(value), dayfirst=True).strftime("%Y-%m-%d")
            except Exception:
                pass

            if cleaned:
                # Validate year is reasonable, if year is before 1900 it means pandas could not find the year in the value
                year = int(cleaned[:4])
                if year < 1900:
                    cleaned = None
            # Parse using month first, standard format (MM/DD/YYYY)
            if cleaned:
                # Successfully parsed, replace with standardized date
                df.at[idx, col] = cleaned
            else:
                # Could not parse or year missing, flag for auditor
                df.at[idx, col] = original_value
                issues.append({
                    "row": int(idx) + 2,
                    "column": col,
                    "row_index": idx,
                    "original_value": str(original_value),
                    "issue": "Could not parse date or year is missing, please correct to DD/MM/YYYY format",
                    "severity": "high"
                })
    return df

# Function to clean and standardize amount columns
def clean_amounts(df: pd.DataFrame, mapping: dict, issues: list) -> pd.DataFrame:
    """
    Find all columns with field_type 'numeric' using the confirmed mapping.Standardizes values to float numbers.
    Removes commas, currency symbols and whitespace.Converts accounting negatives e.g.(1,500) to -1500.0.
    Flags values that cannot be converted to a number.
    """
    # Find all columns whose field_type is numeric using the mapping. Look at mapped_to names because columns were already renamed.
    amount_columns = [
        info["mapped_to"]
        for info in mapping.values()
        if isinstance(info, dict) and info.get("field_type") == "numeric"
        and info.get("mapped_to") in df.columns
    ]
    for col in amount_columns:
        for idx, value in df[col].items():
            # Skip empty values.
            if pd.isna(value) or str(value).strip() == "":
                continue
            original_value = value
            # If already a number no cleaning needed, convert to float
            if isinstance(value, (int, float)):
                df.at[idx, col] = float(value)
                continue
            cleaned_str = str(value).strip()
            # Check for accounting negative notation e.g. (1,500)
            is_negative = cleaned_str.startswith("(") and cleaned_str.endswith(")")
            # Remove currency symbols, letters, commas and whitespace. Keeps only digits, dots and minus signs
            cleaned_str = cleaned_str.strip("()")
            cleaned_str = re.sub(r"[^\d.-]", "", cleaned_str)

            # Try converting to float
            try:
                amount = float(cleaned_str)
                # Apply negative if value was in brackets
                if is_negative:
                    amount = -amount
                df.at[idx, col] = amount
            except ValueError:
                # Could not convert, flag for auditor
                df.at[idx, col] = original_value
                issues.append({
                    "row": int(idx) + 2,
                    "column": col,
                    "row_index": idx,
                    "original_value": str(original_value),
                    "issue": "Could not parse amount, please correct to a valid number",
                    "severity": "high"
                })
    return df

# Function to standardize text column casing 
def standardize_casing(df: pd.DataFrame, mapping: dict, issues: list) -> pd.DataFrame:
    """
    Find all columns with field_type 'title' using confirmed mapping.
    Standardizes all text values to Title Case.
    Skips empty values and numeric-looking values.
    """
    # Find all columns whose field_type is text using the mapping.
    text_columns = [
        info["mapped_to"]
        for info in mapping.values()
        if isinstance(info, dict) and info.get("field_type") == "text"
        and info.get("mapped_to") in df.columns
    ]
    # Loop through text columns and standardize to title case, skipping empty and numeric values
    for col in text_columns:
        for idx, value in df[col].items():
            # Skip empty values
            if pd.isna(value) or str(value).strip() == "":
                continue
            # Skip if value is already a number
            if isinstance(value, (int, float)):
                continue
            # Convert to title case
            df.at[idx, col] = str(value).strip().title()
    return df

# Function to handle null values
def handle_nulls(df: pd.DataFrame, mapping: dict, issues: list) -> pd.DataFrame:
    """
    Flags two types of issues:
    1. Missing values in all confirmed mapped columns
    2. Columns that could not be mapped — flagged once per column not per row
    Does not drop or fill any values — auditor decides.
    """
    # Flag unknown columns once — not per row
    for original_col, info in mapping.items():
        if isinstance(info, dict) and info.get("mapped_to") == "unknown":
            issues.append({
                "row": "N/A",
                "column": original_col,
                "row_index": "N/A",
                "original_value": "N/A",
                "issue": f"Column '{original_col}' could not be mapped to a financial field, please confirm its meaning",
                "severity": "medium"
            })

    # Flag missing values in all confirmed columns
    confirmed_columns = [
        info["mapped_to"]
        for info in mapping.values()
        if isinstance(info, dict)
        and info.get("mapped_to") != "unknown"
        and info.get("mapped_to") in df.columns
    ]

    for col in confirmed_columns:
        if col == "_is_duplicate":
            continue
        for idx, value in df[col].items():
            if pd.isna(value) or str(value).strip() == "" or value == "":
                issues.append({
                    "row": int(idx) + 2,
                    "column": col,
                    "row_index": idx,
                    "original_value": "",
                    "issue": f"Missing value in column '{col}',please review the data",
                    "severity": "medium"
                })

    return df
    
# Function to detect and flag duplicate rows
def handle_duplicates(df: pd.DataFrame, mapping: dict, issues: list) -> pd.DataFrame:
    """
    Detect two issues:
    1. Exact duplicates, all column values identical. Flags and marks for removal.
    2. Suspicious duplicates, same date + amount + vendor but different ID. Flagged for auditor review.
    Does not remove any rows, auditor decides.
    """
    # Find all rows that are completely identical to another row
    exact_duplicates = df[df.duplicated(keep=False)]
    # Flag exact duplicates for auditor review
    for idx in exact_duplicates.index:
        issues.append({
            "row": int(idx) + 2,
            "column": "all columns",
            "row_index": idx,
            "original_value": str(df.loc[idx].to_dict()),
            "issue": "Exact duplicate row, identical to a previous row, please review and consider removing duplicates",
            "severity": "high"
        })
        # Mark duplicate rows in the dataframe so auditor can see them
        df["_is_duplicate"] = df.duplicated(keep="first")

        # Find all date and numeric columns using the mapping
        date_cols = [
            info["mapped_to"] for info in mapping.values()
            if isinstance(info, dict) and info.get("field_type") == "date"
            and info.get("mapped_to") in df.columns
        ]
        amount_cols = [
            info["mapped_to"] for info in mapping.values()
            if isinstance(info, dict) and info.get("field_type") == "numeric"
            and info.get("mapped_to") in df.columns
        ]
        # Combine date and numeric columns for suspicious duplicate check
        check_cols = date_cols + amount_cols
        # Only check if we have at least one date and one numeric column
        if date_cols and amount_cols:
            suspicious = df[
                df.duplicated(subset=check_cols, keep=False)
                # Exclude already flagged exact duplicates
                & ~df["_is_duplicate"]  
            ]
            # Flag suspicious duplicates for auditor review
            for idx in suspicious.index:
                issues.append({
                    "row": int(idx) + 2,
                    "column": str(check_cols),
                    "row_index": idx,
                    "original_value": str(df.loc[idx][check_cols].to_dict()),
                    "issue": "Suspicious duplicate, same date and amount values but not an exact duplicate, please review the issues.",
                    "severity": "medium"
                })
    return df

# Function to build the final validation report
def build_validation_report(df: pd.DataFrame, original_df: pd.DataFrame, issues: list) -> dict:
    """
    Build a summary validation report from all issues found during cleaning.
    Shows total rows, clean rows, flagged rows and a breakdown of issues by type and severity.
    """
    total_rows = len(original_df)
    flagged_rows = len(set(issue["row_index"] for issue in issues if "row_index" in issue))
    clean_rows = total_rows - flagged_rows

    # Count issues by severity
    high_issues = [i for i in issues if i.get("severity") == "high"]
    medium_issues = [i for i in issues if i.get("severity") == "medium"]
    # Build report dictionary
    return {
        "total_rows": total_rows,
        "clean_rows": clean_rows,
        "flagged_rows": flagged_rows,
        "total_issues": len(issues),
        "high_issues": len(high_issues),
        "medium_issues": len(medium_issues),
        "issues": issues
    }

            



