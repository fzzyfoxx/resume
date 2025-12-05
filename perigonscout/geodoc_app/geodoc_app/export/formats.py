import csv
import io

def export_to_csv(data, columns, filename_base):
    """
    Exports a list of dictionaries to a CSV file and returns the CSV content and headers.
    Args:
        data (list): List of dictionaries containing the data to export.
        columns (list): List of column names for the CSV header.
        filename_base (str): Base name for the CSV file (without extension).
    Returns:
        tuple: (csv_content (str), headers (dict))
    """
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    for row in data:
        writer.writerow(row)
    csv_content = output.getvalue()
    output.close()

    safe_name = ''.join(ch for ch in filename_base if ch.isalnum() or ch in ('_', '-'))[:40] or 'feature'
    headers = {
        "Content-Disposition": f"attachment; filename={safe_name}.csv",
        "Content-Type": "text/csv; charset=utf-8",
        "Cache-Control": "no-store"
    }

    return csv_content, headers