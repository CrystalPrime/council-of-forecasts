"""
Shared data loader — supports CSV, TSV, XLSX, XLS.
Used by Tester tools and all forecast MCP servers so file-format handling stays in one place.
"""
from pathlib import Path
import pandas as pd


SUPPORTED_EXTS = {".csv", ".tsv", ".xlsx", ".xls"}


def _normalize_path(file_path: str) -> Path:
    """Strip surrounding quotes and whitespace users often paste with paths."""
    s = file_path.strip().strip('"').strip("'").strip()
    return Path(s)


def load_dataframe(file_path: str, sheet_name: str | int | None = 0) -> pd.DataFrame:
    """
    Load a tabular file by extension. Supports .csv, .tsv, .xlsx, .xls.

    Args:
        file_path: Path to the file. Surrounding quotes/whitespace are stripped automatically.
        sheet_name: For Excel files only. 0 = first sheet (default). Pass a string name
                    or int index to pick a specific sheet. Ignored for CSV/TSV.

    Returns:
        pandas DataFrame.

    Raises:
        FileNotFoundError: if path doesn't exist.
        ValueError: if extension is not supported.
    """
    p = _normalize_path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    ext = p.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(p)
    if ext == ".tsv":
        return pd.read_csv(p, sep="\t")
    if ext == ".xlsx":
        return pd.read_excel(p, sheet_name=sheet_name, engine="openpyxl")
    if ext == ".xls":
        return pd.read_excel(p, sheet_name=sheet_name, engine="xlrd")
    raise ValueError(
        f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTS)}"
    )


def list_sheets(file_path: str) -> list[str]:
    """List sheet names in an Excel file. Returns empty list for CSV/TSV."""
    p = _normalize_path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    ext = p.suffix.lower()
    if ext not in {".xlsx", ".xls"}:
        return []
    engine = "openpyxl" if ext == ".xlsx" else "xlrd"
    xl = pd.ExcelFile(p, engine=engine)
    return list(xl.sheet_names)
