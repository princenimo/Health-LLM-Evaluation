#!/bin/bash

usage() { echo "Usage: $0 [-m  <model_name_or_model_path>] [-d <data_path>] [-i <inferenced_result_dir>] [-p <prompt_type>] [-n <num_of_shots>]"; }
OPTSTRING=":m:d:i:p:n:"

MODEL_NAME=""
DATA_PATH=""
INFERENCED_RESULT_DIR=""

while getopts ${OPTSTRING} flag; do
  case ${flag} in
    m)
      if [ -z "${OPTARG}" ] || [[ "${OPTARG}" == -* ]]; then
        usage
        exit
      else
        MODEL_NAME=$OPTARG
      fi
      ;;
    d)
      if [ -z "${OPTARG}" ] || [[ "${OPTARG}" == -* ]]; then
        usage
        exit
      else
        DATA_PATH=$OPTARG
      fi
      ;;
    i)
      if [ -z "${OPTARG}" ] || [[ "${OPTARG}" == -* ]]; then
        usage
        exit
      else
        INFERENCED_RESULT_DIR=$OPTARG
      fi
      ;;
    p)
      if [ -z "${OPTARG}" ] || [[ "${OPTARG}" == -* ]]; then
        usage
        exit
      else
        PROMPT=$OPTARG
      fi
      ;;
    n)
      if [ -z "${OPTARG}" ] || [[ "${OPTARG}" == -* ]]; then
        usage
        exit
      else
        NUM_SHOTS=$OPTARG
      fi
      ;;
    :)
        usage
        ;;
    ?)
        usage
        ;;
  esac
done


echo "Model Name or Path: $MODEL_NAME"
echo "Data Path: $DATA_PATH"
echo "Write Path: $INFERENCED_RESULT_DIR"
echo "Prompt Type: $PROMPT"
echo "Number of Shots: $NUM_SHOTS"

OUTPUT_DIR="$INFERENCED_RESULT_DIR/output"
ANSWER_DIR="$INFERENCED_RESULT_DIR/answers"

mkdir -p $OUTPUT_DIR
mkdir -p $ANSWER_DIR

INFER_ARGS="--model-name-or-path $MODEL_NAME \
        --data-path $DATA_PATH \
        --answers-dir $ANSWER_DIR \
        --output-dir $OUTPUT_DIR \
        --prompt $PROMPT \
        --num-shots $NUM_SHOTS" 


EXTRACTED_ANSWERS="$OUTPUT_DIR/extracted_answers.json"

EVAL_ARGS="--result-dir $ANSWER_DIR \
        --write-path $EXTRACTED_ANSWERS"


REPORT_ARGS="--benchmark $DATA_PATH \
        --results $EXTRACTED_ANSWERS \
        --report_dir $OUTPUT_DIR"



echo "Running Evaluation"
python3 /srv/essa-lab/flash3/cnimo3/work/reproduction/PMC-LLaMA/SFT/eval/qa_inference.py $INFER_ARGS


#echo "Extracting MCQ Answers"
#python3 /srv/essa-lab/flash3/cnimo3/work/reproduction/PMC-LLaMA/SFT/eval/eval_medqa.py $EVAL_ARGS

#echo "Calculate Accuracy"
#python3 /srv/essa-lab/flash3/cnimo3/work/reproduction/PMC-LLaMA/SFT/eval/get_accuracy.py $REPORT_ARGS