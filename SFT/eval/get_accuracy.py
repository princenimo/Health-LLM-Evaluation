import pandas as pd
import json
import argparse
import os

def read_jsonl_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_data = json.loads(line)
            data.append(json_data)
    return data

def read_json_to_dict(extracted_answers_filepath):
    with open(extracted_answers_filepath, 'r', encoding='utf-8') as file:
        data_dict = json.load(file)
    return data_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, help='Path to the benchmark ')
    parser.add_argument('--results', type=str, help='Path to the results')
    parser.add_argument('--report_dir', type=str, help='Path to the metric report')
    args = parser.parse_args()

    print(f"\033[32mComputing Accuracy for Benchmark {args.benchmark} with Inferenced Results: {args.results}\033[0m")

    benchmark = read_jsonl_file(args.benchmark)
    results = read_json_to_dict(args.results)


    count = 0
    for key in results:
        int_key = int(key)
        if results[key] == benchmark[int_key]["answer"]:
            count += 1

    # Write the Accuracy string to the file
    metric_report_filename = "metric_report.txt"
    metric_report_path = os.path.join(args.report_dir, metric_report_filename)

    with open(metric_report_path, 'w') as file:
        accuracy_metric =f'# Accuracy: {count/len(benchmark)}'
        print(accuracy_metric)
        file.write(accuracy_metric)