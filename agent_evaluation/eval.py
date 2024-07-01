import re, string, os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "agent_prompts")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../agent_prompts")))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import json
from tqdm import tqdm
import argparse
import pandas as pd
import numpy as np
import statistics


dict_binary2first = {
    "mediation": ['01', '02', '03', '04', '05', '06', '07', '08'],
    "conflict": ['09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
}
dict_first2binary = {item: key for key, value in dict_binary2first.items() for item in value}

dict_quad2first = {
    "verbal cooperation": ['01', '02', '03', '04'],
    "material cooperation": ['05', '06', '07', '08'],
    "verbal conflict": ['09', '10', '11', '12', '13', '14', '15', '16'],
    "material conflict": ['17', '18', '19', '20']
}
dict_first2quad = {item: key for key, value in dict_quad2first.items() for item in value}

dict_code2relation = json.load(open("../data/info/dict_code2relation.json", 'r'))
codes = list(dict_code2relation.keys())
first_level_codes = [code for code in codes if len(code) == 2]
second_level_codes = [code for code in codes if len(code) == 3]

# calculate micro precision, recall, f1
def calculate_metrics(preds, golds):
    tp, fp, fn = 0, 0, 0
    for pred, gold in zip(preds, golds):
        for p in pred:
            if p in gold:
                tp += 1
            else:
                fp += 1
        for g in gold:
            if g not in pred:
                fn += 1
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1


def calculate_macro_metrics(preds, golds):
    precisions, recalls, f1s = [], [], []
    for pred, gold in zip(preds, golds):
        tp, fp, fn = 0, 0, 0
        for p in pred:
            if p in gold:
                tp += 1
            else:
                fp += 1
        for g in gold:
            if g not in pred:
                fn += 1
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        precision *= 100
        recall *= 100
        f1 *= 100
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    precision = sum(precisions) / len(precisions) if len(precisions) > 0 else 0
    recall = sum(recalls) / len(recalls) if len(recalls) > 0 else 0
    f1 = sum(f1s) / len(f1s) if len(f1s) > 0 else 0
    # print(precisions)


    return precision, recall, f1

# calculate discrete KL divergence between two list of distributions
def calculate_kl_divergence(preds, golds, dict_all_items):
    kl_divergences = []
    epsilon = 1e-10  # Small constant to prevent log(0)
    items = dict_all_items.keys()  # All possible items

    for pred, gold in zip(preds, golds):
        # Calculate distributions with all items included, adding epsilon to prevent division by zero
        pred_dist = {item: pred.count(item) / len(pred) if len(pred) > 0 else 0 for item in items}
        gold_dist = {item: gold.count(item) / len(gold) if len(gold) > 0 else 0 for item in items}

        # Adding epsilon to all probabilities to handle cases where count might be zero
        pred_dist = {item: pred_dist[item] + epsilon for item in items}
        gold_dist = {item: gold_dist[item] + epsilon for item in items}

        # Calculate KL divergence
        kl_divergence = sum(gold_dist[item] * np.log(gold_dist[item] / pred_dist[item]) for item in items)
        kl_divergences.append(kl_divergence)

    # Calculate average KL divergence
    avg_kl_divergence = sum(kl_divergences) / len(kl_divergences)
    return avg_kl_divergence

# load predictions
def load_predictions(pred_file, task="relation"):
    logs = json.load(open(pred_file))
    answer_str = logs[-1]["answer"]
    try:
        answer = eval(answer_str)
        if task == "relation":
            first_level = list(answer.keys())
            first_level = list(set([item for item in first_level if item in first_level_codes]))

            second_level = []
            for first in first_level:
                value = answer[first]
                for item in value:
                    if item[:2] == first and item in second_level_codes:
                        second_level.append(item)
            second_level = list(set([item for item in second_level]))
            return first_level, second_level
        else:
            return answer
    except:
        if task == "relation":
            return [], []
        else:
            return []

def load_end_state(pred_file):
    logs = json.load(open(pred_file))
    end_state = logs[-1]["end_state"]
    n_steps = logs[-1]["n_steps"]
    return end_state, n_steps

# eval relation predictions
def eval_relation(data_query, setting_output_dir, args):
    # load gold relation predictions
    golds_first_level, golds_second_level = [], []
    gold_binary_level, gold_quad_level = [], []
    gold_binary_level_dedup, gold_quad_level_dedup = [], []
    for rowid, row in tqdm(data_query.iterrows(), total=len(data_query)):
        answer = eval(row["AnswerDict"])
        first_level = list(answer.keys())
        second_level = [item for sublist in answer.values() for item in sublist]
        golds_first_level.append(first_level)
        golds_second_level.append(second_level)

        binary_level = [dict_first2binary[item] for item in first_level]
        quad_level = [dict_first2quad[item] for item in first_level]
        gold_binary_level.append(binary_level)
        gold_quad_level.append(quad_level)

        binary_level_dedup = list(set(binary_level))
        quad_level_dedup = list(set(quad_level))
        gold_binary_level_dedup.append(binary_level_dedup)
        gold_quad_level_dedup.append(quad_level_dedup)

    # load all round predictions and calculate metrics
    round_metrics = {}
    dict_end_states_count = {}
    n_steps_total = 0
    for curr_round in range(args.rounds):
        preds_first_level, preds_second_level = [], []
        preds_binary_level, preds_quad_level = [], []
        preds_binary_level_dedup, preds_quad_level_dedup = [], []

        print(f"Round {curr_round + 1}")
        curr_round_output_dir = os.path.join(setting_output_dir, f"round{curr_round + 1}")

        for rowid, row in tqdm(data_query.iterrows(), total=len(data_query)):
            query_id = row['QueryId']
            pred_file = os.path.join(curr_round_output_dir, f"{query_id}.json")
            first_level, second_level = load_predictions(pred_file, task="relation")
            preds_first_level.append(first_level)
            preds_second_level.append(second_level)

            binary_level = [dict_first2binary[item] for item in first_level]
            quad_level = [dict_first2quad[item] for item in first_level]
            preds_binary_level.append(binary_level)
            preds_quad_level.append(quad_level)

            binary_level_dedup = list(set(binary_level))
            quad_level_dedup = list(set(quad_level))
            preds_binary_level_dedup.append(binary_level_dedup)
            preds_quad_level_dedup.append(quad_level_dedup)

            end_state, n_steps = load_end_state(pred_file)
            n_steps_total += n_steps
            if end_state not in dict_end_states_count:
                dict_end_states_count[end_state] = 0
            dict_end_states_count[end_state] += 1

        # calculate micro metrics
        precision_first_level, recall_first_level, f1_first_level = calculate_macro_metrics(preds_first_level, golds_first_level)
        precision_second_level, recall_second_level, f1_second_level = calculate_macro_metrics(preds_second_level, golds_second_level)
        precision_binary_level, recall_binary_level, f1_binary_level = calculate_macro_metrics(preds_binary_level_dedup, gold_binary_level_dedup)
        precision_quad_level, recall_quad_level, f1_quad_level = calculate_macro_metrics(preds_quad_level_dedup, gold_quad_level_dedup)

        # calculate KL divergence
        kl_binary_level = calculate_kl_divergence(preds_binary_level, gold_binary_level, dict_binary2first)
        kl_quad_level = calculate_kl_divergence(preds_quad_level, gold_quad_level, dict_quad2first)

        round_metrics[curr_round+1] = {
            "binary_level": {
                "precision": precision_binary_level,
                "recall": recall_binary_level,
                "f1": f1_binary_level,
                "kl": kl_binary_level},
            "quad_level": {
                "precision": precision_quad_level,
                "recall": recall_quad_level,
                "f1": f1_quad_level,
                "kl": kl_quad_level},
            "first_level": {
                "precision": precision_first_level,
                "recall": recall_first_level,
                "f1": f1_first_level},
            "second_level": {
                "precision": precision_second_level,
                "recall": recall_second_level,
                "f1": f1_second_level}
        }

    # calculate average metrics
    round_metrics["average"] = {}
    for level in ["binary_level", "quad_level", "first_level", "second_level"]:
        round_metrics["average"][level] = {}
        for metric in round_metrics[1][level]:
            round_metrics["average"][level][metric] = sum([round_metrics[round][level][metric] for round in range(1, args.rounds+1)]) / args.rounds

    # calculate max metrics: for each single query, calculate each metric for all rounds and keep the max
    max_preds_first_level, max_preds_second_level = [], []
    max_preds_binary_level, max_preds_quad_level = [], []
    max_preds_binary_level_dedup, max_preds_quad_level_dedup = [], []
    for rowid, row in tqdm(data_query.iterrows(), total=len(data_query)):
        query_id = row['QueryId']
        answer = eval(row["AnswerDict"])
        gold_second_level = [item for sublist in answer.values() for item in sublist]

        max_second_level_f1 = 0
        max_first_level, max_second_level = None, None

        for curr_round in range(args.rounds):
            curr_round_output_dir = os.path.join(setting_output_dir, f"round{curr_round + 1}")
            pred_file = os.path.join(curr_round_output_dir, f"{query_id}.json")
            first_level, second_level = load_predictions(pred_file, task="relation")
            precision_second_level, recall_second_level, f1_second_level = calculate_macro_metrics([second_level], [gold_second_level])
            if f1_second_level >= max_second_level_f1:
                max_second_level_f1 = f1_second_level
                max_first_level, max_second_level = first_level, second_level
        max_preds_first_level.append(max_first_level)
        max_preds_second_level.append(max_second_level)

        max_preds_binary_level.append([dict_first2binary[item] for item in max_first_level])
        max_preds_quad_level.append([dict_first2quad[item] for item in max_first_level])

        max_preds_binary_level_dedup.append(list(set([dict_first2binary[item] for item in max_first_level])))
        max_preds_quad_level_dedup.append(list(set([dict_first2quad[item] for item in max_first_level])))

    # calculate micro metrics
    precision_binary_level, recall_binary_level, f1_binary_level = calculate_macro_metrics(max_preds_binary_level_dedup, gold_binary_level_dedup)
    precision_quad_level, recall_quad_level, f1_quad_level = calculate_macro_metrics(max_preds_quad_level_dedup, gold_quad_level_dedup)
    precision_first_level, recall_first_level, f1_first_level = calculate_macro_metrics(max_preds_first_level, golds_first_level)
    precision_second_level, recall_second_level, f1_second_level = calculate_macro_metrics(max_preds_second_level, golds_second_level)

    # calculate KL divergence
    kl_binary_level = calculate_kl_divergence(max_preds_binary_level, gold_binary_level, dict_binary2first)
    kl_quad_level = calculate_kl_divergence(max_preds_quad_level, gold_quad_level, dict_quad2first)

    round_metrics["max"] = {
        "binary_level": {
            "precision": precision_binary_level,
            "recall": recall_binary_level,
            "f1": f1_binary_level,
            "kl": kl_binary_level},
        "quad_level": {
            "precision": precision_quad_level,
            "recall": recall_quad_level,
            "f1": f1_quad_level,
            "kl": kl_quad_level},
        "first_level": {
            "precision": precision_first_level,
            "recall": recall_first_level,
            "f1": f1_first_level},
        "second_level": {
            "precision": precision_second_level,
            "recall": recall_second_level,
            "f1": f1_second_level}
    }


    # keeps answer item only if the same item is given in the answer of at least 2 rounds
    if args.rounds >= 2:
        repeated_preds_first_level, repeated_preds_second_level = [], []
        repeated_preds_binary_level, repeated_preds_quad_level = [], []
        repeated_preds_binary_level_dedup, repeated_preds_quad_level_dedup = [], []
        for rowid, row in tqdm(data_query.iterrows(), total=len(data_query)):
            query_id = row['QueryId']
            pred_files = [os.path.join(setting_output_dir, f"round{round + 1}", f"{query_id}.json") for round in range(args.rounds)]
            answers = [load_predictions(pred_file, task="relation") for pred_file in pred_files]
            first_level = [answer[0] for answer in answers]

            first_level = [item for sublist in first_level for item in sublist]
            second_level = [answer[1] for answer in answers]
            second_level = [item for sublist in second_level for item in sublist]
            first_level = [item for item in first_level if first_level.count(item) >= 2]
            first_level = list(set(first_level))
            second_level = [item for item in second_level if second_level.count(item) >= 2]
            second_level = list(set(second_level))
            repeated_preds_first_level.append(first_level)
            repeated_preds_second_level.append(second_level)

            repeated_preds_binary_level.append([dict_first2binary[item] for item in first_level])
            repeated_preds_quad_level.append([dict_first2quad[item] for item in first_level])

            repeated_preds_binary_level_dedup.append(list(set([dict_first2binary[item] for item in first_level])))
            repeated_preds_quad_level_dedup.append(list(set([dict_first2quad[item] for item in first_level])))

        # calculate micro metrics
        precision_binary_level, recall_binary_level, f1_binary_level = calculate_macro_metrics(repeated_preds_binary_level_dedup, gold_binary_level)
        precision_quad_level, recall_quad_level, f1_quad_level = calculate_macro_metrics(repeated_preds_quad_level_dedup, gold_quad_level)
        precision_first_level, recall_first_level, f1_first_level = calculate_macro_metrics(repeated_preds_first_level, golds_first_level)
        precision_second_level, recall_second_level, f1_second_level = calculate_macro_metrics(repeated_preds_second_level, golds_second_level)

        # calculate KL divergence
        kl_binary_level = calculate_kl_divergence(repeated_preds_binary_level, gold_binary_level, dict_binary2first)
        kl_quad_level = calculate_kl_divergence(repeated_preds_quad_level, gold_quad_level, dict_quad2first)

        round_metrics["repeated"] = {
            "binary_level": {
                "precision": precision_binary_level,
                "recall": recall_binary_level,
                "f1": f1_binary_level,
                "kl": kl_binary_level},
            "quad_level": {
                "precision": precision_quad_level,
                "recall": recall_quad_level,
                "f1": f1_quad_level,
                "kl": kl_quad_level},
            "first_level": {
                "precision": precision_first_level,
                "recall": recall_first_level,
                "f1": f1_first_level},
            "second_level": {
                "precision": precision_second_level,
                "recall": recall_second_level,
                "f1": f1_second_level}
        }

    round_metrics["end_states"] = dict_end_states_count
    round_metrics["n_steps_avg"] = n_steps_total / len(data_query) / args.rounds

    # create a dataframe of the max round predictions and repeated round predictions
    data_max_round = []
    data_repeated_round = []
    column_names = []
    for level in ["binary_level", "quad_level", "first_level", "second_level"]:
        for metric in round_metrics["max"][level]:
            data_max_round.append(round_metrics["max"][level][metric])
            column_names.append(f"{level}_{metric}")
            if args.rounds >= 2:
                data_repeated_round.append(round_metrics["repeated"][level][metric])
    df_max_round = pd.DataFrame([data_max_round], columns=column_names)
    if args.rounds >= 2:
        df_repeated_round = pd.DataFrame([data_repeated_round], columns=column_names)

    # concatenate the two dataframes to two rows
    if args.rounds >= 2:
        df_max_round = pd.concat([df_max_round, df_repeated_round], axis=0)

    print(round_metrics['max']['second_level']['f1'])
    if args.rounds >= 2:
        print(round_metrics['repeated']['second_level']['f1'])

    return round_metrics, df_max_round



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="test_subset", choices=["test", "test_subset"])
    parser.add_argument("--timediff", type=int, default=1, help="date difference from the query date to the current date")

    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-0125",
                        choices=["gpt-3.5-turbo-0125", # latest GPT-3.5 turbo model (Sep 2021)
                                 "gpt-4-turbo-2024-04-09", # latest GPT-4 turbo model (Apr 2024)
                                 "gpt-4-1106-preview", # previous GPT-4 turbo preview model (Apr 2023)
                                 "gpt-4o-2024-05-13", # most advanced GPT-4o model (Oct 2023)
                                 "Meta-Llama-3-8B-Instruct", # Meta-Llama 3 model (March 2023)
                                 "Mistral-7B-Instruct-v0.2" # Mistral 7B model (?)
                                 ])
    parser.add_argument("--temperature", type=float, default=0.4, help="temperature of the model")
    parser.add_argument("--rounds", type=int, default=1, help="number of rounds")

    parser.add_argument("--plan", type=str, default="react", choices=["direct", "cot", "react"], help="planning strategy")
    parser.add_argument("--action", type=str, default="func", choices=["func", "block"], help="action type")
    parser.add_argument("--api", type=str, default="full", choices=["full", "kg", "news"], help="api type")
    parser.add_argument("--max_steps", type=int, default=20, help="maximum action steps")

    parser.add_argument("--output_dir", type=str, default="./../output")
    parser.add_argument("--data_dir", type=str, default="./../data/MIRAI")
    parser.add_argument("--api_dir", type=str, default="./../APIs/api_description_full.py")

    parser.add_argument("--alias", type=str, default="", help="alias for the output file")

    parser.add_argument("--output_eval_dir", type=str, default="./../output_eval")

    args = parser.parse_args()

    # load query dataset
    data_query = pd.read_csv(os.path.join(args.data_dir, args.dataset, 'relation_query.csv'), sep='\t', dtype=str)

    if ('llama' in args.model_name.lower()) or ('mistral' in args.model_name.lower()):
        args.max_steps = 0
        args.action = "none"
        args.api = "none"

    setting_output_dir = os.path.join(args.output_dir, args.dataset, args.model_name,"timediff{}-maxsteps{}-{}-{}-{}-temp{}".format(args.timediff, args.max_steps, args.plan, args.action, args.api, args.temperature))
    if args.alias != "":
        setting_output_dir = setting_output_dir + '-' + args.alias

    # eval relation prediction
    eval_results, eval_df = eval_relation(data_query, setting_output_dir, args)

    # save evaluation results
    eval_dir = os.path.join(args.output_eval_dir, args.dataset)
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    eval_file = "round{}-{}-timediff{}-maxsteps{}-{}-{}-{}-temp{}".format(args.rounds, args.model_name, args.timediff, args.max_steps, args.plan, args.action, args.api, args.temperature)
    if args.alias != "":
        eval_file = eval_file + '-' + args.alias

    json.dump(eval_results, open(os.path.join(eval_dir, eval_file + '.json'), 'w'), indent=4)
    eval_df.to_csv(os.path.join(eval_dir, eval_file + '.csv'), index=False, sep='\t')