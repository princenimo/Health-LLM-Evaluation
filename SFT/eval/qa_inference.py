'''
CUDA_VISIBLE_DEVICES=4,5,6,7 python qa_inference.py \
    --model-name-or-path path/to/model \
    --data-path /path/to/test.jsonl \
    --answers-dir /path/to/inferenced_result_dir
    --output-dir /path/to/output_results
    --prompt prompt type selection
    --num-shots number of shots
'''

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

from typing import Dict, Optional, Sequence
import argparse
import jsonlines
from tqdm import tqdm
from functools import partial
import os
from vllm import LLM, SamplingParams
import time
import csv
import random



PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\nYou're a doctor, kindly address the medical queries. Answer with the best option directly, followed by the rationale for the selected option.\n\n### Input:\n{question}\n\n### Response: The correct answer is"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
        "prompt_saq": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\nGiven your profession as a doctor, please provide a response to the medical query.\n\n### Input:\n{question}\n\n### Response:"
    ),
        "prompt_consumer_query": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\nYou are a skilled doctor with years of experience in the medical field in Africa, working in a hospital setting. Your expertise spans a wide range of conditions from common ailments to complex diseases. You are now addressing a set of open-ended questions designed to explore your medical insights and experiences. You should answer each question freely, drawing upon your clinical knowledge to provide thorough, informed responses that reflect your understanding of the topics discussed.\n\n### Input:\n{question}\n\n### Response:"
    ),
        "prompt_mcq_base": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\nThe following are multiple choice questions (MCQs).You should directly answer the question by choosing the correct option and then provide a rationale for your answer. \n\n### Input:\n{question}\n\n### Response: The correct answer is"
    ),
        "prompt_mcq_instruct": (
        "<|start_header_id|>system<|end_header_id|> Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request. "
        "Instruction: You are a skilled doctor with years of experience in the medical field in Africa, working in a hospital setting. Your expertise spans a range of conditions from common ailments to complex diseases. As part of your commitment to ongoing medical education, you are evaluating a set of multiple choice questions (MCQs) designed for medical students. Carefully select the most appropriate answer based on your clinical knowledge. You should directly answer the question by choosing the correct option and then provide a rationale for your answer. <|eot_id|><|start_header_id|>user<|end_header_id|> Question:\n{question}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>Response: The correct answer is"
    ),
}

def format_question(d): 
    question = "###Question: "
    question += d["question"]
    options = d["options"]
    question += "\n###Options:\n"
    for k, v in options.items(): 
        question += f"{k}. {v}\n"
    return question

def add_few_shots(input_data: list, seed: int): 
    random.seed(seed)
    random.shuffle(input_data)
    demonstrations = input_data[:args.num_shots]
    few_shot_prompt = '\n'.join([
                '{} {}.<|eot_id|>\n'.format(
                    demo['pmc_input'],
                    demo['answer_idx']) for demo in demonstrations])
    return few_shot_prompt

def format_saq_question(d): 
    question = "###Question: "
    question += d["question"]
    return question

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name-or-path', type=str)
    parser.add_argument('--data-path', type=str)
    parser.add_argument('--answers-dir', type=str)
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--prompt', type=str)
    parser.add_argument('--num-shots', type=int)
    args = parser.parse_args()
    return args


def construct_spedical_tokens_dict() -> dict:
    DEFAULT_PAD_TOKEN = "[PAD]"
    DEFAULT_EOS_TOKEN = "</s>"
    DEFAULT_BOS_TOKEN = "<s>"
    DEFAULT_UNK_TOKEN = "<unk>"

    special_tokens_dict = dict()
    if tokenizer.pad_token is None or tokenizer.pad_token == '':
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None or tokenizer.eos_token == '':
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None or tokenizer.bos_token == '':
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None or tokenizer.unk_token == '':
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    return special_tokens_dict


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
# enddef



def inference_on_one(input_str: str, use_stop_tokens: bool, model) -> str:
    """
    model_inputs = tokenizer(
      input_str,
      return_tensors='pt',
      padding=True,
    )

    topk_output = model.generate(
        model_inputs.input_ids.cuda(),
        max_new_tokens=1000,
        pad_token_id=tokenizer.eos_token_id,
        top_k=50
    )
    """
    
    tokenizer = model.get_tokenizer()
    #terminators = ["###"]
    if use_stop_tokens:
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
    else:
        terminators = None

    #terminators.append(tokenizer.eos_token_id)
    #stop_seq.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))
    sampling_params = SamplingParams(temperature = 0.0, max_tokens = 1000, top_k=50, stop_token_ids=terminators)
    # topk_output = model.generate(
    #     **model_inputs,
    #     max_new_tokens=1000,
    #     top_k=50
    # )

    start = time.time()
    response = model.generate(input_str, sampling_params)
    end = time.time()

    latency = end-start
    print(f"Latency: {latency} seconds")

    output_tokens = len(response[0].outputs[0].token_ids)
    through_put = output_tokens / latency
    print(f"Throughput: {through_put} tokens/second")

    #output_str = tokenizer.batch_decode(topk_output)  # a list containing just one str
    output_str = response[0].outputs[0].text

    return output_str



def read_jsonl(file_path):
    data_list = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data_list.append(obj)
    return data_list


def prepare_data(data_list: Sequence[dict], model, tokenizer) -> Sequence[dict]:
    prompt_input, prompt_no_input, prompt_saq, prompt_cq = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"], PROMPT_DICT["prompt_saq"], PROMPT_DICT["prompt_consumer_query"]
    prompt_mcq_base, prompt_mcq_instruct = PROMPT_DICT["prompt_mcq_base"], PROMPT_DICT["prompt_mcq_instruct"]
    for _idx in tqdm(range(len(data_list))):
        data_entry = data_list[_idx]
        #sample_id = data_entry['sample_id']

        if args.prompt == "mcqbase":
            full_question = format_question(data_entry)
            data_entry["question"] = full_question
            data_list[_idx]['pmc_input'] = prompt_mcq_base.format_map(data_entry)
        if args.prompt == "mcqinstruct":
            full_question = format_question(data_entry)
            data_entry["question"] = full_question
            data_list[_idx]['pmc_input'] = prompt_mcq_instruct.format_map(data_entry)
        else:
            full_question = format_saq_question(data_entry)
            data_entry["question"] = full_question
            data_list[_idx]['pmc_input'] = prompt_saq.format_map(data_entry)
        """"
        if data_entry.get("question_type", "") == "saq":
            full_question = format_saq_question(data_entry)
            data_entry["question"] = full_question
            data_list[_idx]['pmc_input'] = prompt_saq.format_map(data_entry)
        elif data_entry.get("question_type", "") == "consumer_queries":
            full_question = format_saq_question(data_entry)
            data_entry["question"] = full_question
            data_list[_idx]['pmc_input'] = prompt_cq.format_map(data_entry)
        else:
            full_question = format_question(data_entry)
            data_entry["question"] = full_question
            data_list[_idx]['pmc_input'] = prompt_input.format_map(data_entry) if data_entry.get("question", "") != "" else prompt_no_input.format_map(data_entry)
        #print(f"full prompt: {data_list[_idx]['pmc_input']}")
        """
    # endfor
    return data_list
# enddef


if __name__ == '__main__':
    args = parse_args()

    print(f"\033[32mPrepare Data\033[0m")
    data_list = read_jsonl(args.data_path)
    fn = partial(prepare_data, model=None, tokenizer=None)
    inference_data = fn(data_list)


    print(f"\033[32mLoad Checkpoint\033[0m")
    #model = transformers.LlamaForCausalLM.from_pretrained(args.model_name_or_path)
    #model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map='auto')
    #tokenizer = transformers.LlamaTokenizer.from_pretrained(
    #    args.model_name_or_path,
    #    #cache_dir=training_args.cache_dir,
    #    model_max_length=400,
    #    padding_side="right",
    #    use_fast=False,
    #)
    """"
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=400,
        padding_side="right",
        use_fast=False,
    )
    
    special_tokens_dict = construct_spedical_tokens_dict()
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    #model.cuda()
    """

    tp = torch.cuda.device_count()
        

    #print(f"tensor parallel size = {tp}")
    model = LLM(model=args.model_name_or_path, 
                tokenizer=args.model_name_or_path,
                trust_remote_code=True,
                max_num_seqs=1024,
                tensor_parallel_size=tp
                )
    
    if 'Llama-3' or 'Llama3' in args.model_name_or_path:
        use_stop_tokens = True
    else:
        use_stop_tokens = False

    csv_data = []
    print(f"Num Shots === {args.num_shots}")
    for _idx in tqdm(range(len(data_list))):
        data_entry = data_list[_idx]
        #sample_id = data_entry['sample_id']
        """
        input_str = [
            data_entry['pmc_input']
        ] """
        
        if args.num_shots > 0:
            seed=42
            few_shot_prompt = add_few_shots(data_list, seed)
            input_string = few_shot_prompt + data_entry['pmc_input']
        else:
            input_string = data_entry['pmc_input']
        print(f"Input: {input_string}")
        output_str = inference_on_one(input_string, use_stop_tokens, model)
        print(f"Output: {output_str}")
        #csv_data.append([input_string, data_entry['question'], output_str, data_entry['answer_rationale']])
        csv_data.append([input_string, data_entry['question'], output_str])
        with open(os.path.join(args.answers_dir, f"{_idx}.txt"), 'w', encoding='utf-8') as f:
            f.write(output_str)
    # endfor
    results_table = os.path.join(args.output_dir, 'results.csv')
    with open(results_table, 'w', newline='') as file:
        writer = csv.writer(file)
        #writer.writerow(['Prompt_Input', 'Question', 'Model_Response', 'Answer_Rationale'])
        writer.writerow(['Prompt_Input', 'Question', 'Model_Response'])
        writer.writerows(csv_data)

