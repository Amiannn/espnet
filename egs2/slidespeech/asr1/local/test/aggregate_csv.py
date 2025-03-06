import os
import csv
import re
import argparse

parser = argparse.ArgumentParser(description="Compute WER, also track U-WER/B-WER.")
parser.add_argument("--root_dir", type=str, help="Reference text input path")
args = parser.parse_args()

# ---------------------------------------------------------------------
# Adjust this to be your root directory where you want to start scanning
ROOT_DIR = args.root_dir
# ---------------------------------------------------------------------

# We'll merge domain_result_*.csv into one aggregate,
# and final_result_*.csv into another.

# We'll keep the original columns, plus we add 'decode_dir' and 'context_type'.
# domain CSV has columns: [system, session, WER, N, C, S, D, I, U-WER, B-WER]
# final CSV has columns:  [system, WER, N, C, S, D, I, U-WER, B-WER, 
#                          HotwordRecall, TP, TN, FP, FN]

domain_rows = []  # aggregated domain-level rows
final_rows = []   # aggregated final-level rows

def get_context_type_from_filename(filename):
    """
    Extract the context portion from filenames like:
      domain_result_keywords_fix.csv -> 'keywords_fix'
      final_result_ocr_fix.csv -> 'ocr_fix'
    """
    m = re.search(r'result_(.*?)\.csv', filename)
    if m:
        return m.group(1)  # e.g. 'keywords_fix' or 'ocr_fix'
    else:
        return "unknown"

# Traverse the entire ROOT_DIR tree
for root, dirs, files in os.walk(ROOT_DIR):
    for file in files:
        # Check for domain-level CSV
        if file.startswith("domain_result_") and file.endswith("_fix.csv"):
            filepath = os.path.join(root, file)
            # We'll store the relative path as decode_dir
            decode_dir = os.path.relpath(root, ROOT_DIR)
            context_type = get_context_type_from_filename(file)

            with open(filepath, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)  # skip header row
                for row in reader:
                    # row format: [system, session, WER, N, C, S, D, I, U-WER, B-WER]
                    # We'll prepend decode_dir, context_type:
                    out_row = [decode_dir, context_type] + row
                    domain_rows.append(out_row)

        # Check for final-level CSV
        elif file.startswith("final_result_") and file.endswith("_fix.csv"):
            filepath = os.path.join(root, file)
            decode_dir = os.path.relpath(root, ROOT_DIR)
            context_type = get_context_type_from_filename(file)

            with open(filepath, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)  # skip header
                for row in reader:
                    # row format: [system, WER, N, C, S, D, I, U-WER, B-WER,
                    #             HotwordRecall, TP, TN, FP, FN]
                    out_row = [decode_dir, context_type] + row
                    final_rows.append(out_row)

# Now define headers for the aggregated CSVs.
domain_header = [
    "decode_dir",
    "context_type",
    "system",
    "session",
    "WER",
    "N",
    "C",
    "S",
    "D",
    "I",
    "U-WER",
    "B-WER",
]
final_header = [
    "decode_dir",
    "context_type",
    "system",
    "WER",
    "N",
    "C",
    "S",
    "D",
    "I",
    "U-WER",
    "B-WER",
    "HotwordRecall",
    "TP",
    "TN",
    "FP",
    "FN",
]

exp_folder = './exp/results'
if not os.path.exists(exp_folder):
    os.makedirs(exp_folder)

# Write out aggregated domain CSV
output_path = os.path.join(exp_folder, "domain_result_aggregate.csv")
with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(domain_header)
    writer.writerows(domain_rows)

# Write out aggregated final CSV
output_path = os.path.join(exp_folder, "final_result_aggregate.csv")
with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(final_header)
    writer.writerows(final_rows)

print("Done. Created domain_result_aggregate.csv and final_result_aggregate.csv.")
