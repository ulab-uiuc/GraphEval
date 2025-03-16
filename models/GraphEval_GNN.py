import pandas as pd
import torch
import yaml
from collections import defaultdict
import wandb
import numpy as np
from network_for_node import form_data, GNN_train
import random
from utils import load_paper_data, load_node_data, load_edge_data



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def GraphEval_GNN():
    # Load all data
    with open("config.yaml", 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    set_seed(config["seed"])
    wandb.login(key=config["wandb_key"])
    wandb.init(project="GraphEval", name="Experiment")
    node_list = load_node_data(config['node_list_paths'])
    edge_list = load_edge_data(config['edge_list_paths'])
    train_set = load_paper_data(config["training_set_path"],config["Task_name"])
    test_set = load_paper_data(config["test_set_path"],config["Task_name"])

    ## Load node,edge,paper list
    node_df = pd.DataFrame(node_list)
    edge_df = pd.DataFrame(edge_list)

    paper_data=([{'paper_id': key, 'decision':value} for key, value in train_set.items()]+
                [{'paper_id': key, 'decision':value} for key, value in test_set.items()])
    # Initialize a dictionary to hold subgraph nodes grouped by prefix
    subgraphs_by_new_prefix = defaultdict(list)
    for line in node_list:
        node = line['node_id']
        # Extract prefix by removing the square bracket and following content
        prefix = node.split('[')[0]
        # Add nodes to the corresponding subgraph by prefix
        subgraphs_by_new_prefix[prefix].append(node)
    count=0
    iner_count = 0
    prefix_to_value_list={}
    cluster_dict_out = {}
    cluster_decision=[]
    for node in subgraphs_by_new_prefix:
        prefix_to_value_list[node]={}
        prefix_to_value_list[node]["node_list"]=subgraphs_by_new_prefix[node]
        cluster_decision.append(paper_data[count]["decision"])
        node_length = len(subgraphs_by_new_prefix[node])
        cluster_dict_out[count] = list(range(iner_count, iner_count + node_length))
        iner_count += node_length
        count += 1
    node_mapping = {node_id: i for i, node_id in enumerate(node_df['node_id'].unique())}
    edge_df['node_id1_idx'] = edge_df['node_id1'].map(node_mapping)
    edge_df['node_id2_idx'] = edge_df['node_id2'].map(node_mapping)
    edge_index = torch.tensor(edge_df[['node_id1_idx', 'node_id2_idx']].values.T, dtype=torch.long)
    # Assuming all nodes have the same number of features or none at all
    node_features = torch.tensor(node_df['node_embedding'].tolist(), dtype=torch.float).squeeze(1)

    numbers = np.arange(len(paper_data))

    # data_ratio
    train_ratio = 0.70
    val_ratio = 0.15
    test_ratio = 0.25

    n_total = len(paper_data)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    n_test = n_total - n_train - n_val

    # shuffle data
    np.random.shuffle(numbers)
    # split data
    train_set = numbers[:n_train]
    val_set = numbers[n_train:-n_test]
    test_set = numbers[-n_test:]
    cluster_length=len(numbers)

    # Create the graph data
    form_data_all = form_data()

    mask_train=torch.zeros(cluster_length)
    mask_train[train_set]=1
    mask_validate=torch.zeros(cluster_length)
    mask_validate[val_set]=1
    mask_test=torch.zeros(cluster_length)
    mask_test[test_set]=1

    data_all = form_data_all.formulation(node_features=node_features, edge_index=edge_index,edge_weight=edge_df['weight'],
                                         mask_train=mask_train,mask_validate=mask_validate,mask_test=mask_test,cluster_dict_out=cluster_dict_out,cluster_decision=np.array(cluster_decision))

    gnn_train=GNN_train(input_dim=node_features.shape[1], hidden_features=config["GNN_hidden_features_dim"],in_edges=config["GNN_edge_dim"],output_dim=config["GNN_output_dim"],wandb=wandb)

    gnn_train.train_validate(data=data_all,iter_num=config["GNN_max_iters"],mask_rate=config["mask_rate"],batch_size=config["GNN_batch_size"],
                             model_path=config["model_path"])
    test_accuracy,test_f1=gnn_train.model_test(data=data_all,load_with_model=True)


if __name__ == "__main__":
    # GraphEval_GNN function
    GraphEval_GNN()