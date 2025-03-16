import openai
import time
import statistics
import csv
import ast
import json
from typing import Union, List, Tuple, Optional
import torch.nn as nn


def init_weights(m):
    r"""
    Performs weight initialization

    Args:
        m (nn.Module): PyTorch module

    """
    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data = nn.init.xavier_uniform_(
            m.weight.data, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None:
            m.bias.data.zero_()
def llm_generation(sys_prompt: str, user_prompt: str, response_num: int, config: dict) -> Tuple[Union[str,List[str]], Optional[int]]:
    openai.api_base = config["api_base"]
    openai.api_key = config["api_key"]
    llm_model = config["llm_model"]
    reset_time = config["reset_time"]
    
    max_trials = config["max_trials"]
    while max_trials:
        max_trials -= 1
        try:
            completion = openai.ChatCompletion.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=config["temperature"],
                max_tokens=config["max_tokens"],
                seed = config["random_seed"] if response_num == 1 else None,
                n = response_num 
            )
            break
        except openai.error.RateLimitError as e:
            print(f"Rate limit exceeded. Sleeping for {reset_time} seconds.")
            time.sleep(reset_time)
        except Exception as e:
            print(e)
            time.sleep(reset_time)
    time.sleep(reset_time)
    
    total_tokens = None
    if 'usage' in completion:
        total_tokens = completion['usage']['total_tokens']
    
    if response_num > 1:
        return [choice.message.content for choice in completion.choices if choice.message.content], total_tokens
    else:
        return completion['choices'][0]['message']['content'], total_tokens

# used for obtaining viewpoint node embeddings    
def mean_pooling(token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

# Process and load datasets with .json, .jsonl, and .tsv extensions.
def load_dataset(path: str) -> List[dict]:
    # Used to map the ground_truth_label in the dataset to a number.
    # Note that due to the scarcity of Accept (Oral) and Accept (Spotlight) labels in the AI Researcher dataset, 
    # we combine them in to a single label, thereby transforming the task into a three-class classification problem.
    decision_mapping = {
        "Reject": 0,
        "Accept (Poster)": 1,
        "Accept (Spotlight)": 2,
        "Accept (Oral)": 3
    }
    processed_dataset = []
    
    if path.endswith(".jsonl"):
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line)
                ratings = ast.literal_eval(data["ratings"]) if isinstance(data["ratings"], str) else data["ratings"]
                processed_dataset.append({
                    "paper_id": data["paper_id"],
                    "title": data["title"],
                    "abstract": data["abstract"],
                    "rating": statistics.mean(ratings),
                    "decision": decision_mapping.get(data['decision'], -1)
                })
    
    elif path.endswith(".json"):
        with open(path, 'r', encoding='utf-8') as file:
            dataset = json.load(file)
            for data in dataset:
                ratings = ast.literal_eval(data["ratings"]) if isinstance(data["ratings"], str) else data["ratings"]
                processed_dataset.append({
                    "paper_id": data["paper_id"],
                    "title": data["title"],
                    "abstract": data["abstract"],
                    "rating": statistics.mean(ratings),
                    "decision": decision_mapping.get(data['decision'], -1)
                })
    
    elif path.endswith(".tsv"):
        with open(path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file, delimiter='\t')
            for data in reader:
                ratings = ast.literal_eval(data["ratings"]) if isinstance(data["ratings"], str) else data["ratings"]
                processed_dataset.append({
                    "paper_id": data["paper_id"],
                    "title": data["title"],
                    "abstract": data["abstract"],
                    "rating": statistics.mean(ratings),
                    "decision": decision_mapping.get(data['decision'], -1)
                })
    else:
        raise ValueError("Unsupported file format. Please provide a .jsonl, .json, or .tsv file.")
    
    return processed_dataset

def load_paper_data(paper_file: str, task_name: str) -> dict:
    papers = {}
    if task_name=="AI Researcher":
        decision_map = {"Reject": 0, "Accept (Poster)": 1, "Accept (Spotlight)": 2}
    else:
        decision_map = {"Reject": 0, "Accept (Poster)": 1, "Accept (Spotlight)": 2, "Accept (Oral)": 3}
    with open(paper_file, 'r',encoding='utf-8') as f:
        for line in f:
            paper_info = json.loads(line)
            papers[paper_info['paper_id']] = decision_map[paper_info['decision']]
    return papers



def load_node_data(node_files: List[str]) -> List[dict]:
    nodes = []
    for node_file in node_files:
        with open(node_file, 'r',encoding='utf-8') as f:
            for line in f:
                node_info = json.loads(line)
                paper_id = node_info['node_id'].split('[')[0]
                embedding=node_info['embedding']
                nodes.append({'node_id': node_info['node_id'], 'paper_id': paper_id,'node_embedding':embedding})
    return nodes

def load_edge_data(edge_files: List[str]) -> List[dict]:
    edges = []
    for edge_file in edge_files:
        with open(edge_file, 'r',encoding='utf-8') as f:
            for line in f:
                edge_info = json.loads(line)
                edges.append({
                    'node_id1': edge_info['node_id1'],
                    'node_id2': edge_info['node_id2'],
                    'weight': edge_info['weight']
                })
    return edges