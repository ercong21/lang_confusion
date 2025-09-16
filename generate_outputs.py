import csv
import argparse
import os
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM


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

# generate answer tokens
def generate_answer(model, tokenizer, text):
    token_ids = tokenizer(text, return_tensors='pt').input_ids.to(model.device)
    prompt_length = token_ids.shape[1]
    
    output_ids = model.generate(token_ids,
                                max_new_tokens=100,
                                do_sample=False,
                                temperature=None,
                                top_p=None)
    
    new_token_ids = output_ids[0][prompt_length:]
    
    generated_text = tokenizer.decode(new_token_ids)
    return generated_text
    
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
        default="test_sets",
        help="Specify the path to the prompt file."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs",
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
    csv_file_path = f"{args.prompt_path}/{args.task}/{args.source}/{args.language}.csv"
    if not os.path.exists(csv_file_path):
        print(f"File {csv_file_path} does not exist.")
        return
    with open(csv_file_path, mode="r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        # Skip header
        next(reader)
        # Read prompts
        prompts = [row[0] for row in reader]

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
        writer.writerow(["id", "model", "prompt", "completion", "task", "source", "language"])
        
        # Generate outputs for each prompt using both models
        id_counter = 0
        for prompt in tqdm(prompts, desc="Generating outputs", unit="prompt", 
                        total=len(prompts)):
            completion = generate_answer(model, tokenizer, prompt)
            # print(f"Prompt: \n{prompt}")
            # print(f"Completion: \n{completion}")
            writer.writerow([
                id_counter,
                model_name,
                prompt,
                completion,
                args.task,
                args.source,
                args.language
            ])
            id_counter += 1


if __name__ == "__main__":
    main()