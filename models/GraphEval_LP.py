import json
import yaml
import networkx as nx
import numpy as np
from typing import List
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from utils import load_paper_data, load_node_data, load_edge_data


# Build the viewpoint-graph
def build_graph(nodes: List[dict], edges: List[dict]) -> nx.Graph:
    G = nx.Graph()
    node_map = {node['node_id']: i for i, node in enumerate(nodes)}
    for node in nodes:
        G.add_node(node_map[node['node_id']], paper_id=node['paper_id'])
    
    for edge in edges:
        node1, node2 = node_map[edge['node_id1']], node_map[edge['node_id2']]
        G.add_edge(node1, node2, weight=edge['weight'])
    
    return G

# Implement the label propagation algorithm for GraphEval-LP
def label_propagation(G: nx.Graph, training_set: dict, test_set: dict, config: dict) -> dict:
    max_iters = config["LP_max_iters"]
    label_weights = config["LP_label_weights"]
    decision_map = {"Reject": 0, "Accept (Poster)": 1, "Accept (Spotlight)": 2, "Accept (Oral)": 3} # AI Researcher: decision_map = {"Reject": 0, "Accept (Poster)": 1, "Accept (Spotlight)": 2}
    scores = {i: np.zeros(len(decision_map)) for i in G.nodes()}
    for idx in G.nodes():
        paper_id = G.nodes[idx]['paper_id']
        if paper_id in training_set:
            decision = training_set[paper_id]
            scores[idx][decision] = 1.0

    for u, v, data in G.edges(data=True):
        data['weight'] /= sum(data['weight'] for _, _, data in G.edges(u, data=True))

    for iteration in range(max_iters):
        new_scores = scores.copy()
        for i in G.nodes():
            neighbors = G.neighbors(i)
            for neighbor in neighbors:
                weight = G[i][neighbor]['weight']
                for decision in range(len(decision_map)): 
                    new_scores[i][decision] += weight * scores[neighbor][decision] * label_weights[decision]
            new_scores[i] = new_scores[i] / np.sum(new_scores[i]) if np.sum(new_scores[i]) > 0 else new_scores[i]
        
        scores = new_scores

    predictions = {}
    for paper_id in test_set.keys():
        nodes_in_paper = [i for i in G.nodes() if G.nodes[i]['paper_id'] == paper_id]
        if nodes_in_paper:
            aggregated_scores = np.sum([scores[i] for i in nodes_in_paper], axis=0)
            final_decision = np.argmax(aggregated_scores)
            
            predictions[paper_id] = final_decision
    return predictions

def evaluate_results(predictions: dict, papers: dict) -> dict:
    y_gt = [papers[paper_id] for paper_id in predictions.keys()]
    y_pred = [predictions[paper_id] for paper_id in predictions.keys()]
    accuracy = accuracy_score(y_gt, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_gt, y_pred, average='macro')
    
    result_dict = {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }

    return result_dict

# This function implements the GraphEval-LP algorithm and evaluation.
# After setting up the config.yaml file, run the following command:
# python -c "from GraphEval_LP import GraphEval_LP; GraphEval_LP()"
def GraphEval_LP():
    with open("config.yaml", 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    node_list = load_node_data(config['node_list_paths'])
    edge_list = load_edge_data(config['edge_list_paths'])
    training_set = load_paper_data(config["training_set_path"],config["Task_name"])
    test_set = load_paper_data(config["test_set_path"],config["Task_name"])
    
    G = build_graph(node_list, edge_list)
    
    predictions = label_propagation(G, training_set, test_set, config)
    
    result_dict = evaluate_results(predictions, test_set)
    if config["save_baseline_results"]:
        with open(config["baseline_results_save_path"], 'w',encoding='utf-8') as f:
            json.dump(result_dict, f)

    print(result_dict)
    return result_dict
    
if __name__ == "__main__":
    # GraphEval_LP function
    GraphEval_LP()