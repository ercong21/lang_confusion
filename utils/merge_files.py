# merge generation files into one file

import os
import pandas as pd
import sys

# model_name = "Llama3.1-8b-instruct"
# model_name = "Llama3-8b-instruct"
# model_name = "Mistral-7b-multilingual"
# output_path = f'./outputs'

if len(sys.argv) != 3:
    print("Usage: python merge_files.py <output_folder_path> <model_id>")
    sys.exit(1)

output_path = sys.argv[1]
model_name = sys.argv[2]

folder_path = f'{output_path}/{model_name}'


# Get a list of all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Read and concatenate all CSV files
merged_df = pd.concat([pd.read_csv(os.path.join(folder_path, file)) for file in csv_files])

# Save the merged DataFrame to a new CSV file
output_file = os.path.join(output_path, f'{model_name}.csv')
merged_df.to_csv(output_file, index=False)

print(f"Merged CSV saved to {output_file}")