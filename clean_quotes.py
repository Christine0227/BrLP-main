#!/usr/bin/env python3
import sys
import csv

if len(sys.argv) != 2:
    print("Usage: python clean_quotes.py <input.csv>")
    sys.exit(1)

input_file = sys.argv[1]
output_file = input_file.replace('.csv', '_clean.csv')

with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8', newline='') as f_out:
    reader = csv.reader(f_in)
    writer = csv.writer(f_out)
    for row in reader:
        cleaned = [cell.replace("'", '').replace('"', '') for cell in row]
        writer.writerow(cleaned)

print(f'✅ Cleaned CSV saved to {output_file}')
