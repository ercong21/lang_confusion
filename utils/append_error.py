import csv
import sys

if len(sys.argv) != 2:
    print("Usage: python append_error.py <model_id>")
    sys.exit(1)

model_id = sys.argv[1]

models_dict = {
    "Llama3-8b-instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "Llama3.1-8b-instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Llama3-8b-multilingual": "lightblue/suzume-llama-3-8B-multilingual",
    "Mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.2",
    "Mistral-7b-multilingual": "nthakur/mistral-7b-v0.2-multilingual-full-sft-27th-mar-basilisk"
}

model2name = {value: key for key, value in models_dict.items()}

with open(f"outputs/{model_id}_errors.csv", 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)  # Skip the header row
    num = 0
    for row in list(reader):
        num += 1
        task, model, source, language, acc_ids,full_error_ids,partial_error_ids = row
        model = model2name[model]
        output_path = f"outputs/{model}/{task}-{source}-{language}.csv"
        print(f"Processing {output_path}, {num}")
        with open(output_path, 'r') as f_out:
            reader_out = csv.reader(f_out)
            data = list(reader_out)
            data[0].append("error_type")
            for i in range(1, len(data)):
                if int(data[i][0]) in eval(full_error_ids):
                    data[i].append("full_error")
                elif int(data[i][0]) in eval(partial_error_ids):
                    data[i].append("partial_error")
                elif int(data[i][0]) in eval(acc_ids):
                    data[i].append("correct")
        with open(output_path, 'w') as f_write:
            writer = csv.writer(f_write)
            writer.writerows(data)
