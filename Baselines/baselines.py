import re
import json
import statistics
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from typing import Tuple, Optional, List
from statistics import multimode
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy.stats import spearmanr
from utils import llm_generation, load_dataset

# Used to map the labels predicted by baselines to a number.
pred_decision_mapping = {
    "Reject": 0,
    "Poster": 1,
    "Spotlight": 2,
    "Oral": 3,
}

# Prompted LLM Baseline: We provide several criteria for assessing research ideas in the prompt. 
# Additionally, we present specific standards for the four review decisions and include one example for each as few-shot examples for in-context learning.
# Moreover, we include the label distribution of the dataset to help the LLMs understand the frequency of each review decision.
def prompted_llm(title: str, abstract: str, config: dict) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    with open('Baselines/Prompt/basic_prompt.txt', 'r', encoding='utf-8') as file:
        sys_prompt = file.read()
        
    user_prompt = f"Title - {title}; Abstract - {abstract};"
    
    response, token_usage = llm_generation(sys_prompt, user_prompt, response_num=1, config=config)
    
    score = None
    decision = None
    
    score_match = re.search(r"Overall Score\s*\(0-100\)\s*=\s*(\d+)", response)
    if score_match:
        score = int(score_match.group(1))
    else:
        print("Vanilla LLM Baseline: No valid score found in the response:")
        print(response)
    
    decision_match = re.search(r"(Reject|Poster|Spotlight|Oral)", response)
    if decision_match:
        decision = pred_decision_mapping[decision_match.group(1)]
    else:
        print("Vanilla LLM Baseline: No valid decision found in the response:")
        print(response)
    
    return score, decision, token_usage

# CoT/CoT-SC Baselines: We modify the prompt used for prompted LLM to adopt a CoT format, guiding it to complete the idea evaluation step by step.
# CoT-SC is an ensemble approach that samples k i.i.d. CoT, then returns the most frequent output
def CoT_SC(title: str, abstract: str, CoT_SC:bool, config: dict) -> tuple[Optional[float], Optional[int], list[int], list[int], Optional[int]]:
    with open('Baselines/Prompt/CoT_prompt.txt', 'r', encoding='utf-8') as file:
        sys_prompt = file.read()
        
    user_prompt = f"Title - {title}; Abstract - {abstract};"
    
    if CoT_SC:
        response_num = config["CoT_SC_k"]
        response_list, token_usage = llm_generation(sys_prompt, user_prompt, response_num, config)
    else:
        response, token_usage = llm_generation(sys_prompt, user_prompt, 1, config)
        response_list = [response]    
    
    scores = []
    decisions = []
    for idx, response in enumerate(response_list):   
        score_match = re.search(r"Overall Score\s*\(0-100\)\s*:\s*(\d+)", response)
        if score_match:
            scores.append(int(score_match.group(1)))
        else:
            print(f"CoT_SC Baseline: No valid score found in response {idx}:")
            print(response)
    
        decision_match = re.search(r"(Reject|Poster|Spotlight|Oral)", response)
        if decision_match:
            decisions.append(pred_decision_mapping[decision_match.group(1)])
        else:
            print(f"CoT_SC Baseline: No valid decision found in the response {idx}:")
            print(response)
            
    CoT_SC_score = None
    CoT_SC_decision = None
    if scores:
        score_modes = multimode(scores) 
        CoT_SC_score = sum(score_modes) / len(score_modes) 
    if decisions:
        decision_modes = multimode(decisions)
        CoT_SC_decision = round(statistics.mean(decision_modes))
        
    return CoT_SC_score, CoT_SC_decision, scores, decisions, token_usage

# ToT Prompt Baseline: Tree of Thoughts (ToT) is an extension of CoT. Similar to CoT, we divide the evaluation process into multiple steps. 
# At each step, we sample k i.i.d. CoTs, and pass the most frequent output as the intermediate result to the next step. 
def ToT_BFS(title: str, abstract: str, config: dict) -> Tuple[Optional[float], Optional[int], int]:
    user_prompt = f"Title: {title}\nAbstract: {abstract}\n"
    metric_list = [
        "Novelty Rating (1-10):",
        "Validity Rating (1-10):",
        "Significance Rating (1-10):",
        "Rigorousness Rating (1-10):",
        "Clarity Rating (1-10):",
        "Ethical Considerations Rating (1-10):",
        "Overall Score (0-100):",
        "Decision:"
    ]
    
    final_score = None
    final_decision = None
    token_usage_list = []
    
    for step in range(8):
        with open(f'Baselines/Prompt/ToT_prompts/ToT_prompt_{step}.txt', 'r', encoding='utf-8') as file:
            sys_prompt = file.read()
        
        step_scores = []
        decisions = []
        response_list = []
        
        response_list, token_usage = llm_generation(sys_prompt, user_prompt, config["ToT_branch_num"], config)
        token_usage_list.append(token_usage)
            
        if step < 7:  # For metric extraction
            for idx, response in enumerate(response_list):
                match = re.search(rf"{re.escape(metric_list[step])}\s*(\d+)", response)
                if match:
                    step_scores.append(int(match.group(1)))
                else:
                    print(f"ToT Baseline: No valid score found for {metric_list[step]} in response {idx}:")
                    print(response)
            if step_scores:
                modes = multimode(step_scores)
                average_score = sum(modes) / len(modes)
                user_prompt += f"{metric_list[step]} {average_score}\n"
                if step == 6:  # Store the overall score
                    final_score = average_score
            else:
                return None, None, sum(token_usage_list)
        elif step == 7:  # For decision extraction
            for idx, response in enumerate(response_list):
                decision_match = re.search(r"(Reject|Poster|Spotlight|Oral)", response)
                if decision_match:
                    decisions.append(pred_decision_mapping[decision_match.group(1)])
                else:
                    print(f"ToT Baseline: No valid decision found in response {idx}:")
                    print(response)
            if decisions:
                modes = multimode(decisions)
                final_decision = round(statistics.mean(decisions))
        
    return final_score, final_decision, sum(token_usage_list)

# Research Agent Baseline: We adopt the idea evaluation method from Research Agent as one of our baselines, 
# where the research problem, scientific method, and experiment design of an idea are each scored based on five criteria. 
# Building on this, we further introduce a final decision step that synthesizes the above evaluation results to provide a comprehensive review decision.
def Research_Agent(title: str, abstract: str, config: dict) -> Tuple[Optional[int], Optional[int], int]:
    step_list = ["problem", "method", "experiment"]
    match_list = ["Research\s*Problem", "Scientific\s*Method", "Experiment\s*Design"]
    component_list = ["research_problem","scientific_method","experiment_design"]
    review_list = []
    feedback_list = []
    rating_list = []
    replacements = {"abstract": abstract, "title": title}
    
    token_usage_list = []
    
    for i in range(len(step_list)):
        with open(f'Baselines/Prompt/ResearchAgent/{step_list[i]}_valid_sys_prompt.txt', 'r', encoding='utf-8') as file:
            sys_prompt = file.read()
        
        with open(f'Baselines/Prompt/ResearchAgent/{step_list[i]}_valid_user_prompt.txt', 'r', encoding='utf-8') as file:
            user_prompt = file.read()
            for key, value in replacements.items():
                placeholder = f"{{{key}}}"
                user_prompt = user_prompt.replace(placeholder, value)

        response, token_usage = llm_generation(sys_prompt, user_prompt, 1, config)
        token_usage_list.append(token_usage)
        
        component_match = re.search(rf"{match_list[i]}\s*:\s*(.+?)(?=\s*Rationale)", response)
        rationale_match = re.search(r"Rationale\s*:\s*(.+?)(?=\s*Review)", response)
        review_match = re.search(r"Review\s*:\s*(.+?)(?=\s*Feedback)", response)
        feedback_match = re.search(r"Feedback\s*:\s*(.+?)(?=\s*Rating)", response)
        rating_match = re.search(r"Rating\s*\(1-5\)\s*:\s*(.+)", response)
        
        if component_match:
            replacements[f"{component_list[i]}"] = component_match.group(1)
        else:
            print(f"Research Agent: No valid {match_list[i]} found in the response:")
            print(response)
            replacements[f"{component_list[i]}"] = "UNKNOWN"
        if rationale_match:
            replacements[f"{component_list[i]}_rationale"] = rationale_match.group(1)
        else:
            print(f"Research Agent: No valid {match_list[i]} Rationale found in the response:")
            print(response)
            replacements[f"{component_list[i]}_rationale"] = "UNKNOWN"
        if review_match:
            review_list.append(review_match.group(1))
        else:
            print(f"Research Agent: No valid {match_list[i]} Review found in the response.")
            review_list.append(" ")
        if feedback_match:
            feedback_list.append(feedback_match.group(1))
        else:
            print(f"Research Agent: No valid {match_list[i]} Feedback found in the response.")
            feedback_list.append(" ")
        if rating_match:
            rating_list.append(rating_match.group(1))
        else:
            print(f"Research Agent: No valid {match_list[i]} Rating found in the response.")
            rating_list.append(" ")
    
    with open('Baselines/Prompt/ResearchAgent/final_eval_prompt.txt', 'r', encoding='utf-8') as file:
            sys_prompt = file.read()
    
    user_prompt = f"""Title - {title}
    Abstract - {abstract}
    Research Problem - {replacements["research_problem"]}
    Research Problem Rationale - {replacements["research_problem_rationale"]}
    Research Problem Review - {review_list[0]}
    Research Problem Feedback - {feedback_list[0]}
    Research Problem Rating - {rating_list[0]}
    Scientific Method - {replacements["scientific_method"]}
    Scientific Method Rationale - {replacements["scientific_method_rationale"]}
    Scientific Method Review - {review_list[1]}
    Scientific Method Feedback - {feedback_list[1]}
    Scientific Method Rating - {rating_list[1]}
    Experiment Design - {replacements["experiment_design"]}
    Experiment Design Rationale - {replacements["experiment_design_rationale"]}
    Experiment Design Review - {review_list[2]}
    Experiment Design Feedback - {feedback_list[2]}
    Experiment Design Rating - {rating_list[2]}
    """    
    response, token_usage = llm_generation(sys_prompt, user_prompt, 1, config)
    token_usage_list.append(token_usage)
    
    score = None
    decision = None
    
    score_match = re.search(r"Overall Score\s*\(0-100\)\s*=\s*(\d+)", response)
    if score_match:
        score = int(score_match.group(1))
    else:
        print("Research Agent: No valid evaluation score found in the response:")
        print(response)
    
    decision_match = re.search(r"(Reject|Poster|Oral|Spotlight)", response)
    if decision_match:
        decision = pred_decision_mapping[decision_match.group(1)]
    else:
        print("No valid decision found in the response.")
        print(response)
    
    return score, decision, sum(token_usage_list)

# Finetuned Bert Baseline: We fine-tune a BERT model using collected paper abstracts and review decisions as a baseline.
def finetuned_Bert(training_set: List[dict], test_set: List[dict], config: dict):
    training_data = pd.DataFrame(training_set)
    test_data = pd.DataFrame(test_set)

    training_data['label'] = training_data['decision']
    test_data['label'] = test_data['decision']

    training_data['text'] = training_data.apply(lambda row: f"Title: {row['title']}; Abstract: {row['abstract']}", axis=1)
    test_data['text'] = test_data.apply(lambda row: f"Title: {row['title']}; Abstract: {row['abstract']}", axis=1)

    train_dataset = Dataset.from_pandas(training_data[['text', 'label']])
    test_dataset = Dataset.from_pandas(test_data[['text', 'label']])

    tokenizer = AutoTokenizer.from_pretrained(config["finetuned_Bert_model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(config["finetuned_Bert_model_name"], num_labels=4)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True)
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)

        accuracy = accuracy_score(labels, predictions)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            labels, predictions, average='macro', zero_division=0
        )

        return {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
        }

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy=config["Bert_evaluation_strategy"],
        per_device_train_batch_size=config["Bert_train_batch_size"],
        per_device_eval_batch_size=config["Bert_eval_batch_size"],
        num_train_epochs=config["Bert_num_train_epochs"],
        learning_rate=float(config["Bert_learning_rate"]),
        weight_decay=config["Bert_weight_decay"],
        logging_dir='./logs',
        logging_steps=config["Bert_logging_steps"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    results = trainer.evaluate()

    return results

# Promted LLM Baseline used for FactScore dataset.
def prompted_llm_for_factscore(statement: str, config:dict) -> Tuple[int, Optional[int]]:
    with open('Baselines/Prompt/factscore_prompt.txt', 'r', encoding='utf-8') as file:
        sys_prompt = file.read()
    user_prompt = f"The statement you need to evaluate is:\n{statement}"
    
    response, token_usage = llm_generation(sys_prompt, user_prompt, response_num=1, config=config)
    words = response.lower().strip()
    
    if "incorrect" in words:
        return 0, token_usage
    elif "correct" in words:
        return 1, token_usage
    else:
        print(f"Factscore Baseline: Unexpected LLM response: {response}")
        return -1, token_usage   

# Compute the metric results of LLM-based baselines.
def evaluate_results(results: List[dict]) -> dict:
    gt_decisions = [result['gt_decision'] for result in results]
    pred_decisions = [result['pred_decision'] for result in results]

    gt_ratings = [result['gt_rating'] for result in results]
    pred_ratings = [result['pred_rating'] for result in results]

    total_tokens_list = [result['total_tokens'] for result in results]

    accuracy = accuracy_score(gt_decisions, pred_decisions)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        gt_decisions, pred_decisions, average='macro', zero_division=0
    )
    spearman_corr, _ = spearmanr(gt_ratings, pred_ratings)
    avg_total_tokens = np.mean(total_tokens_list)

    result_dict = {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "average_total_tokens": avg_total_tokens,
        "spearman_corr": spearman_corr
    }

    return result_dict

# Save the baselines' predictions on the test set (excluding finetuned BERT) and the final evaluation results for each metric as a file in the specified path.
def save_results_to_jsonl(results: List[dict], evaluation: dict, file_path: str):
    with open(file_path, 'w', encoding='utf-8') as file:        
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False) + '\n')
        evaluation_entry = {"evaluation_results": evaluation}
        file.write(json.dumps(evaluation_entry, ensure_ascii=False) + '\n') 

# After setting up the config.yaml file, run the baseline using the following command:
# python -c "from Baselines.baselines import run_baseline; run_baseline()"
def run_baseline():
    with open("config.yaml", 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    training_set_path = config["training_set_path"]
    test_set_path = config["test_set_path"]
    
    training_set = load_dataset(training_set_path)
    test_set = load_dataset(test_set_path)
    
    baseline = config["baseline"] #["Prompted LLM", "CoT", "CoT-SC", "ToT", "Research Agent", "Finetuned Bert"]
    
    if baseline == "Finetuned Bert":
       eval_results = finetuned_Bert(training_set, test_set, config)
       
    results = [] 
    if baseline != "Finetuned Bert":
        for data in tqdm(test_set, desc="Baseline Eval"):
            if baseline == "Prompted LLM":
                pred_score, pred_decision, total_tokens = prompted_llm(data['title'], data['abstract'], config)
            elif baseline == "CoT":
                pred_score, pred_decision, scores, decisions, total_tokens = CoT_SC(data['title'], data['abstract'], False, config)
            elif baseline == "CoT-SC":
                pred_score, pred_decision, scores, decisions, total_tokens = CoT_SC(data['title'], data['abstract'], True, config)
            elif baseline == "ToT":
                pred_score, pred_decision, total_tokens = ToT_BFS(data['title'], data['abstract'], config)
            elif baseline == "Research Agent":
                pred_score, pred_decision, total_tokens = Research_Agent(data['title'], data['abstract'], config)
            else:
                print("Invalid baseline name. Please choose from vanilla_llm, CoT_SC, ToT, or Research_Agent.")
                exit()
            result = {
            "paper_id": data['paper_id'],
            "gt_decision": data['decision'],
            "gt_rating": data["rating"],
            "pred_decision": pred_decision,
            "pred_rating": pred_score,
            "total_tokens": total_tokens}   
            results.append(result)
    
        eval_results = evaluate_results(results)
    
    if config["save_baseline_results"]:
        save_results_to_jsonl(results, eval_results, config["baseline_results_save_path"])
    
    print(eval_results)
    return eval_results
    
