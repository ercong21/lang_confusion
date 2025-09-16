import csv
import argparse
import os
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import fasttext
import string


punc_list = list(string.punctuation+"–")

# load the fasttext language detector
lid_path = "./utils/lid.176.bin"
lid_model = fasttext.load_model(lid_path)

sources_dict = {
    "okapi": 'okapi',
    "native": 'native_prompts',
    "aya": 'aya_human_annotated',
    "dolly": 'dolly_human_edited',
    "complex_prompts": 'complex_prompts',
    "sharegpt": 'sharegpt'}

models_dict = {
    "Llama3-8b-instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "Llama3.1-8b-instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Llama3-8b-multilingual": "lightblue/suzume-llama-3-8B-multilingual",
    "Mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.2",
    "Mistral-7b-multilingual": "nthakur/mistral-7b-v0.2-multilingual-full-sft-27th-mar-basilisk"
}

def langid(line: str) -> str:
    (label,), score = lid_model.predict(line)
    return label.removeprefix('__label__') if score > 0.3 else 'unknown'

# generate answer tokens
def generate_cp_samples(model, tokenizer, text, args, error_type):
    token_ids = tokenizer(text, return_tensors='pt').input_ids.to(model.device)
    prompt_length = token_ids.shape[1]
    
    # outputs = model(input_ids = token_ids, output_hidden_states=True, output_attentions=False, return_dict=True)
    
    output_ids = model.generate(token_ids,
                                max_new_tokens=100,
                                do_sample=False,
                                temperature=None,
                                top_p=None)
    
    new_token_num = len(output_ids[0]) - prompt_length
    cp_idx = new_token_num
    new_token_ids = output_ids[0][prompt_length:]
    new_tokens = [tokenizer.decode(token).strip() for token in new_token_ids]
    for i, token in enumerate(new_tokens):
        if error_type == "correct":
            if langid(token) == args.language:
                cp_idx = i
                break
        else:
            if token in punc_list or token == "":
                continue
            elif args.language == "zh":
                if token[0] in string.ascii_letters:
                    cp_idx = i 
                    break
            else:
                if langid(token) == 'en' and token[0] in string.ascii_letters:
                    cp_idx = i 
                    break
    cp_token = new_tokens[cp_idx] if cp_idx != new_token_num else "no_cp_tokens"
    
    new_input_ids = output_ids[0][:prompt_length+cp_idx]
    new_input = tokenizer.decode(new_input_ids, skip_special_tokens=True)
    new_output_ids = output_ids[0][prompt_length+cp_idx:]
    new_output = tokenizer.decode(new_output_ids, skip_special_tokens=True)

    return new_input, new_output, cp_token
    
def main():
    parser = argparse.ArgumentParser(description="Generate outputs using Llama models.")
    parser.add_argument(
        "--model_id",
        type=str,
        choices=["Llama3-8b-instruct", "Llama3.1-8b-instruct", "Llama3-8b-multilingual", "Mistral-7b-instruct", "Mistral-7b-multilingual"],
        required=True,
        help="Specify the model to use for generation."
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["monolingual", "crosslingual"],
        help="Specify the task type."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        choices=list(sources_dict.keys()),
        help="Specify the source of the prompts."
    )
    parser.add_argument(
        "--language",
        type=str,
        required=True, 
        choices=['ar','de','en','es','fr','id','it','pt','vi','zh','ja','ko','tr','hi','ru'],
        help="Specify the language of the prompts."
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="outputs",
        help="Specify the path to the prompt file."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs_cp",
        help="Specify the path to save the generated outputs."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "0", "1", "2", "3", "4", "5", "6", "7"],
        help="Specify the device to use for generation."
    )
    args = parser.parse_args()
    
    # read csv file and get prompts
    csv_file_path = f"{args.prompt_path}/{args.model_id}/{args.task}-{args.source}-{args.language}.csv"
    if not os.path.exists(csv_file_path):
        print(f"File {csv_file_path} does not exist.")
        return
    with open(csv_file_path, mode="r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        # Skip header
        next(reader)
        # Read prompts
        rows = [row for row in reader]
        prompts = [row[2] for row in rows]
        error_types = [row[-1] for row in rows]

    # Check if GPU is available and set device accordingly
    if args.device == "auto":
        device_map = "auto"
    else:
        device_map = {"": int(args.device)}
    
    # Load the model and tokenizer
    model_name = models_dict[args.model_id]
    tokenizer_name = "mistralai/Mistral-7B-Instruct-v0.2" if args.model_id=="Mistral-7b-multilingual" else model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, trust_remote_code=True)
    
    # Filepath for the output CSV
    output_dir = f"{args.output_path}/{args.model_id}"
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Define the output file path
    output_filepath = f"{output_dir}/{args.task}-{args.source}-{args.language}.csv"

    # Generate and save results
    with open(output_filepath, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(["id", "model", "error_type", "new_prompt", "cp_token", "new_output", "task", "source", "language"])
        
        # Generate outputs for each prompt using both models
        id_counter = 0
        for i, prompt in tqdm(enumerate(prompts), desc="Generating outputs", unit="prompt", 
                        total=len(prompts)):
            new_input, new_output, cp_token = generate_cp_samples(model, tokenizer, prompt, args, error_types[i])
            # print(f"Prompt: \n{prompt}")
            # print(f"Completion: \n{completion}")
            writer.writerow([
                id_counter,
                model_name,
                error_types[i],
                new_input,
                cp_token,
                new_output,
                args.task,
                args.source,
                args.language
            ])
            id_counter += 1


if __name__ == "__main__":
    main()