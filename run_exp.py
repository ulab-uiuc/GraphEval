import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Baselines'))
from Baselines.baselines import run_baseline
from models.GraphEval_LP import GraphEval_LP
from models.GraphEval_GNN import GraphEval_GNN
import argparse


def run_experiment(args):
    if args.method_name == 'GraphEval_GNN':
        GraphEval_GNN()
    elif args.method_name == 'GraphEval_LP':
        GraphEval_LP()
    else:
        run_baseline()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method_name", type=str, default="GraphEval_GNN")
    args = parser.parse_args()
    run_experiment(args)