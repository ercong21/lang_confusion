import argparse
import csv
import os
from collections import defaultdict
from tqdm import tqdm

from transformers import AutoTokenizer, LlamaForCausalLM
import torch

LAYER_NUM = 32
HEAD_NUM = 32
HEAD_DIM = 128
HIDDEN_DIM = HEAD_NUM * HEAD_DIM

torch.set_default_device("cuda:3")

error2group = {
    "confusion": "full_error",
    "correct": "correct",
}

model2id = {
    "Llama3-8b-instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "Llama3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "Llama3-8b-multilingual": "lightblue/suzume-llama-3-8B-multilingual",
    "Mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.2",
    "Mistral-7b-multilingual": "nthakur/mistral-7b-v0.2-multilingual-full-sft-27th-mar-basilisk"
    }

from collections import defaultdict

def transfer_output(model_output):
    all_pos_layer_input = []
    all_pos_attn_output = []
    all_pos_residual_output = []
    all_pos_ffn_output = []
    all_pos_layer_output = []
    all_last_attn_subvalues = []
    all_pos_coefficient_scores = []
    all_attn_scores = []
    for layer_i in range(LAYER_NUM):
        cur_layer_input = model_output[layer_i][0]
        cur_attn_output = model_output[layer_i][1]
        cur_residual_output = model_output[layer_i][2]
        cur_ffn_output = model_output[layer_i][3]
        cur_layer_output = model_output[layer_i][4]
        cur_last_attn_subvalues = model_output[layer_i][5]
        cur_coefficient_scores = model_output[layer_i][6]
        cur_attn_weights = model_output[layer_i][7]
        all_pos_layer_input.append(cur_layer_input[0].tolist())
        all_pos_attn_output.append(cur_attn_output[0].tolist())
        all_pos_residual_output.append(cur_residual_output[0].tolist())
        all_pos_ffn_output.append(cur_ffn_output[0].tolist())
        all_pos_layer_output.append(cur_layer_output[0].tolist())
        all_last_attn_subvalues.append(cur_last_attn_subvalues[0].tolist())
        all_pos_coefficient_scores.append(cur_coefficient_scores[0].tolist())
        all_attn_scores.append(cur_attn_weights)
    return all_pos_layer_input, all_pos_attn_output, all_pos_residual_output, all_pos_ffn_output, \
        all_pos_layer_output, all_last_attn_subvalues, all_pos_coefficient_scores, all_attn_scores

def get_fc2_params(model, layer_num):
    return model.model.layers[layer_num].mlp.down_proj.weight.data

def get_bsvalues(vector, model, final_var):
    vector = vector * torch.rsqrt(final_var + 1e-6)
    vector = vector.to("cuda:2")
    vector_rmsn = vector * model.model.norm.weight.data
    vector_bsvalues = model.lm_head(vector_rmsn).data
    return vector_bsvalues.to("cuda:3")

def get_prob(vector):
    prob = torch.nn.Softmax(-1)(vector)
    return prob


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
        "--error_type",
        type=str,
        default="confusion",
        choices=["confusion", "correct"],
        help="Specify the error type."
    )
    
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results_neuron",
        help="Directory to save the results."
    )
    
    args = parser.parse_args()
    
    modelname = model2id[args.model_id]
    if args.model_id == "Llama3-8b-multilingual":
        tokenizer = AutoTokenizer.from_pretrained(model2id["Llama3-8b-instruct"], trust_remote_code=True)
    elif args.model_id == "Mistral-7b-multilingual":
        tokenizer = AutoTokenizer.from_pretrained(model2id["Mistral-7b-instruct"], trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(modelname, trust_remote_code=True)
    model = LlamaForCausalLM.from_pretrained(modelname, trust_remote_code=True, device_map={"": "cuda:2"}, use_cache=True)
    model.eval()
    
    # load the prompts
    if args.model_id == "Llama3-8b-multilingual":
        file_path = f"{args.file_dir}/Llama3-8b-instruct/{args.task}-{args.source}-{args.language}.csv"
    elif args.model_id == "Mistral-7b-multilingual":
        file_path = f"{args.file_dir}/Mistral-7b-instruct/{args.task}-{args.source}-{args.language}.csv"
    else:
        file_path = f"{args.file_dir}/{args.model_id}/{args.task}-{args.source}-{args.language}.csv"
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        data = list(reader)
        prompts = [row[3] for row in data[1:] if row[2] == error2group[args.error_type]]
    
    print(f"Loaded {len(prompts)} prompts of type {args.error_type} from {file_path}")
    
    all_ffn_neuron_scores = defaultdict(float)
    all_ffn_neuron_freq = defaultdict(int)
    # find the value neurons for each prompt
    for prompt in tqdm(prompts, desc="Finding neurons", total=len(prompts)):
    # for prompt in tqdm(prompts[:10], desc="Finding neurons", total=len(prompts[:10])):  # test
        indexed_tokens = tokenizer.encode(prompt)
        tokens = [tokenizer.decode(x) for x in indexed_tokens]
        tokens_tensor = torch.tensor([indexed_tokens]).to(model.device)
        with torch.no_grad():
            outputs = model(tokens_tensor)
            predictions = outputs[0]
        predicted_top10 = torch.argsort(predictions[0][-1], descending=True)[:10]
        # predicted_text = [tokenizer.decode(x) for x in predicted_top10]
        # print(prompt, "=>", predicted_text)
        all_pos_layer_input, all_pos_attn_output, all_pos_residual_output, all_pos_ffn_output, all_pos_layer_output, \
        all_last_attn_subvalues, all_pos_coefficient_scores, all_attn_scores = transfer_output(outputs.past_key_values)
        final_var = torch.tensor(all_pos_layer_output[-1][-1]).pow(2).mean(-1, keepdim=True)
        pos_len = len(tokens)
        # print(tokens)
        predict_index = predicted_top10[0].item()
        
        #FFN neuron increase (value FFN neuron)
        all_ffn_subvalues = []
        for layer_i in range(LAYER_NUM):
            coefficient_scores = torch.tensor(all_pos_coefficient_scores[layer_i][-1])
            fc2_vectors = get_fc2_params(model, layer_i).to("cuda:3")
            ffn_subvalues = (coefficient_scores * fc2_vectors).T
            all_ffn_subvalues.append(ffn_subvalues)
        ffn_subvalue_list = []
        for layer_i in range(LAYER_NUM):
            cur_ffn_subvalues = all_ffn_subvalues[layer_i]
            cur_residual = torch.tensor(all_pos_residual_output[layer_i][-1])
            origin_prob_log = torch.log(get_prob(get_bsvalues(cur_residual, model, final_var))[predict_index])
            cur_ffn_subvalues_plus = cur_ffn_subvalues + cur_residual
            cur_ffn_subvalues_bsvalues = get_bsvalues(cur_ffn_subvalues_plus, model, final_var)
            cur_ffn_subvalues_probs = get_prob(cur_ffn_subvalues_bsvalues)
            cur_ffn_subvalues_probs = cur_ffn_subvalues_probs[:, predict_index]
            cur_ffn_subvalues_probs_log = torch.log(cur_ffn_subvalues_probs)
            cur_ffn_subvalues_probs_log_increase = cur_ffn_subvalues_probs_log - origin_prob_log
            for index, ffn_increase in enumerate(cur_ffn_subvalues_probs_log_increase):
                ffn_subvalue_list.append([str(layer_i)+"_"+str(index), ffn_increase.item()])
        ffn_subvalue_list_sort = sorted(ffn_subvalue_list, key=lambda x: x[-1])[::-1]
        # for x in ffn_subvalue_list_sort[:10]:
        #     print(x[0], round(x[1], 4))
        #     layer = int(x[0].split("_")[0])
        #     neuron = int(x[0].split("_")[1])
        #     cur_vector = get_fc2_params(model, layer).T[neuron]
        #     cur_vector = cur_vector.to("cuda:3")
        #     cur_vector_bsvalue = get_bsvalues(cur_vector, model, final_var)
        #     cur_vector_bsvalue_sort = torch.argsort(cur_vector_bsvalue, descending=True)
        #     print("top10: ", [tokenizer.decode(a) for a in cur_vector_bsvalue_sort[:10]])
        #     print("last10: ", [tokenizer.decode(a) for a in cur_vector_bsvalue_sort[-10:].tolist()[::-1]])
        FFN_value_neurons = [x for x in ffn_subvalue_list_sort[:300]]
        
        for neuron in FFN_value_neurons:
            all_ffn_neuron_scores[neuron[0]] += neuron[1]
            all_ffn_neuron_freq[neuron[0]] += 1

    # sort the all_ffn_neuron_scores
    all_ffn_neuron_scores = sorted(all_ffn_neuron_scores.items(), key=lambda x: x[1], reverse=True)

    # Save the results
    output_dir = f"{args.results_dir}/{args.model_id}/"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"find_neurons_{args.task}-{args.source}-{args.language}-{args.error_type}.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Neuron", "Score", "Frequency"])
        for neuron, score in all_ffn_neuron_scores:
            freq = all_ffn_neuron_freq[neuron]
            writer.writerow([neuron, score, freq])

    