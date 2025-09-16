import argparse
import csv
import os
from tqdm import tqdm
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model2id = {
    "Llama3-8b-instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "Llama3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "Llama3-8b-multilingual": "lightblue/suzume-llama-3-8B-multilingual",
    "Mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.2",
    "Mistral-7b-multilingual": "nthakur/mistral-7b-v0.2-multilingual-full-sft-27th-mar-basilisk"
    }

# neuron_selection_strategies = ["CNA", "score", "freq"]
neuron_selection_strategies = ["CNA"]

# load the neurons
def load_neuron_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row
        data = list(reader)
    
    return data



def generation(text, model):
    token_ids = tokenizer(text, return_tensors='pt').input_ids.to(model.device)

    prompt_length = token_ids.shape[1]
    
    # Forward pass with the hook
    with torch.no_grad():
        output_ids = model.generate(token_ids,
                                max_new_tokens=100,
                                do_sample=False,
                                temperature=None,
                                top_p=None)
    
    new_token_ids = output_ids[0][prompt_length:]
    new_tokens = [tokenizer.decode(token) for token in new_token_ids]
    
    generated_text = tokenizer.decode(new_token_ids)
    
    return token_ids, output_ids, new_token_ids, generated_text, new_tokens


def generate_top10_tokens(text, model):
    token_ids = tokenizer(text, return_tensors='pt').input_ids.to(model.device)

    with torch.no_grad():
        outputs = model(input_ids=token_ids, output_hidden_states=True, output_attentions=False, return_dict=True)

    last_token_logits = outputs.logits[0, -1, :]
    last_token_probs = torch.softmax(last_token_logits, dim=-1)         
    top_k = torch.topk(last_token_probs, 10) # can be set to other topk values
    top_k_tokens = top_k.indices.tolist()
    top_k_probs = top_k.values.tolist()
    top_k_tokens = {tokenizer.decode(token): prob for token, prob in zip(top_k_tokens, top_k_probs)}
    
    return top_k_tokens

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find neurons in Llama models.")
    parser.add_argument(
        "--model_id",
        type=str,
        choices=["Llama3-8b-instruct", "Llama3.1-8b-instruct", "Llama3-8b-multilingual", "Mistral-7b-instruct", "Mistral-7b-multilingual"],
        default="Llama3-8b-instruct",
        help="Specify the model to use for generation."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="monolingual",
        choices=["monolingual", "crosslingual"],
        help="Specify the task type."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="aya",
        help="Specify the source of the prompts."
    )
    parser.add_argument(
        "--language",
        type=str,
        default="zh",
        help="Specify the language of the prompts."
    )
    
    parser.add_argument(
        "--file_dir",
        type=str,
        default="outputs_cp",
        help="Path to the prompt file."
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs_intervene_mis",
        help="Directory to save the outputs."
    )
    
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results_intervene",
        help="Directory to save the results."
    )
    
    parser.add_argument(
        "--device",
        type=int,
        default=3,
        help="CUDA device number to use."
    )
    
    args = parser.parse_args()
    
    # load the model and tokenizer
    model_name = model2id[args.model_id]
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=args.device, trust_remote_code=True, use_cache=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # load the test data (in-domain)
    # e.g. test data path: "outputs_cp/Llama3-8b-instruct/monolingual-aya-zh.csv"
    # test_data_path = f"{args.file_dir}/{args.model_id}/{args.task}-{args.source}-{args.language}.csv"
    test_data_path = f"outputs_cp/Llama3-8b-instruct/monolingual-aya-zh.csv"
    with open(test_data_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        data = list(reader)
        prompts = [row[3] for row in data[1:]]
    
    # load the neurons
    # e.g. neuron path: "results_neuron/Llama3-8b-instruct/find_neurons_confusion.csv"
    neuron_path_inst = f"results_neuron/{args.model_id}/find_neurons_{args.task}-{args.source}-{args.language}-confusion.csv"
    # neuron_path_mult = f"results_neuron/Llama3-8b-multilingual/find_neurons_{args.task}-{args.source}-{args.language}-confusion.csv"
    neuron_path_mult = f"results_neuron/Mistral-7b-multilingual/find_neurons_{args.task}-{args.source}-{args.language}-confusion.csv"
    data_inst = load_neuron_file(neuron_path_inst)
    data_mult = load_neuron_file(neuron_path_mult)
    neurons_inst = {row[0]: rank for rank, row in enumerate(data_inst)}
    neurons_mult = {row[0]: rank for rank, row in enumerate(data_mult)}
    neuron_score_inst = {row[0]: float(row[1]) for row in data_inst}
    neuron_score_mult = {row[0]: float(row[1]) for row in data_mult}
    
    sorted_neurons = sorted(neurons_inst.items(), key=lambda x: x[1], reverse=True)
    sorted_scores = sorted(neuron_score_inst.items(), key=lambda x: x[1], reverse=True)
    
    score_change = dict()
    for neuron in neurons_inst:
        if neuron not in neuron_score_mult:
            score_change[neuron] = neuron_score_inst[neuron]
        else:
            score_change[neuron] = neuron_score_inst[neuron] - neuron_score_mult[neuron]
    # sort the dictionary score_change by value
    sorted_score_change = sorted(score_change.items(), key=lambda x: x[1], reverse=True)
    
    
    # neuron selection critrion
    for neuron_strategy in neuron_selection_strategies:
        
        if neuron_strategy == "CNA":  
            # CNA analysis top100: Select the top 100 neurons based on score change
            selected_neurons = [(int(neuron.split("_")[0]), int(neuron.split("_")[1])) \
                for neuron,_ in sorted_score_change[:100]]
        elif neuron_strategy == "score":
            # Importance score rank: Select the top 100 neurons based on score
            selected_neurons = [(int(neuron.split("_")[0]), int(neuron.split("_")[1])) \
                for neuron,_ in sorted_scores[:100]]
        elif neuron_strategy == "freq":
            # Frequency rank: Select the top 100 neurons based on frequency
            selected_neurons = [(int(neuron.split("_")[0]), int(neuron.split("_")[1])) \
                for neuron,_ in sorted_neurons[:100]]
        
        selected_neurons_dict = defaultdict(list)
        for layer_idx, neuron_idx  in selected_neurons:
            selected_neurons_dict[layer_idx].append(neuron_idx)
        
        print(f"{'*'*30}\nSelected neurons using {neuron_strategy} strategy\n")
    
        # neuron intervention
        original_weight = None  # Variable to store the original weights
        
        # Save the original weights if not already saved
        if original_weight is None:
            original_weight = {layer_idx: model.model.layers[layer_idx].mlp.down_proj.weight.clone() for layer_idx in selected_neurons_dict.keys()}
            
        # Set the specified input neurons (columns) to zero
        for layer_idx, neuron_ids in selected_neurons_dict.items():
            model.model.layers[layer_idx].mlp.down_proj.weight.data[:, neuron_ids] = 0
        
        # generate outputs with intervened models and save the outputs
        output_dir = f"{args.output_dir}/{neuron_strategy}"
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/intervene_{args.task}_{args.source}_{args.language}.csv"
        with open(output_file, "w", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "strategy", "prompt", "generated_text", "top_k_tokens", "top_k_prob", "task", "source", "language"])
            for idx, prompt in enumerate(tqdm(prompts, desc="Generating outputs", unit="prompt")):
                # Generate text with the intervened neurons
                token_ids, output_ids, new_token_ids, generated_text, new_tokens = generation(prompt, model)
                top10_tokens = generate_top10_tokens(prompt, model)
                top_k_tokens = list(top10_tokens.keys())
                top_k_prob = list(top10_tokens.values())
                # Write the results to the CSV file
                writer.writerow([idx, neuron_strategy, prompt, generated_text, top_k_tokens, top_k_prob, args.task, args.source, args.language])
                
        
        # Restore the original weights
        for layer_idx in selected_neurons_dict.keys():
            model.model.layers[layer_idx].mlp.down_proj.weight.data = original_weight[layer_idx]
    
    # generate outputs with original model and save the outputs
    print(f"Generated outputs with original model")
    output_dir = f"{args.output_dir}/original"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/original_{args.task}_{args.source}_{args.language}.csv"
    with open(output_file, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "strategy", "prompt", "generated_text", "top_k_tokens", "top_k_prob", "task", "source", "language"])
        for idx, prompt in enumerate(tqdm(prompts, desc="Generating outputs", unit="prompt")):
            # Generate text with the original model
            token_ids, output_ids, new_token_ids, generated_text, new_tokens = generation(prompt, model)
            top10_tokens = generate_top10_tokens(prompt, model)
            top_k_tokens = list(top10_tokens.keys())
            top_k_prob = list(top10_tokens.values())
            # Write the results to the CSV file
            writer.writerow([idx, "original", prompt, generated_text, top_k_tokens, top_k_prob, args.task, args.source, args.language])