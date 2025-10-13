#!/bin/bash

set -x

# Get alpha and GPU ID from command line arguments
export CUDA_VISIBLE_DEVICES=3
alpha=0.0

export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=$PYTHONPATH:/home/qrh/data/code/ArtQuantization

ALPHA=("$alpha")
MODEL_SIZES=("14B")
BASELINES=("shapley")
DATASETS=("gsm8k")
# PRESET_SCHEMES=("W4A16" "W8A8" "FP8")
PRESET_SCHEMES=("W4A16")
NUM_CALIBRATION_SAMPLES=("512")

BASE_MODEL_PATH="/data/Qwen/Qwen2.5"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="result-time-${TIMESTAMP}.log"

run_quantization() {
    local model_size=$1
    local scheme=$2
    local dataset=$3
    local baseline=$4
    local num_calibration_samples=$5
    local alpha=$6
    local model_path="${BASE_MODEL_PATH}-${model_size}-Instruct"
    local script="quantize_by_${baseline}.py"
    
    echo "Running ${script} with:"
    echo "Model: ${model_path}"
    echo "Scheme: ${scheme}"
    echo "Dataset: ${dataset}"
    echo "Num Calibration Samples: ${num_calibration_samples}"
    echo "Alpha: ${alpha}"
    echo "----------------------------------------"
    
    python ${script} \
        --model_path "${model_path}" \
        --scheme "${scheme}" \
        --dataset "${dataset}" \
        --num_calibration_samples "${num_calibration_samples}" \
        --alpha "${alpha}"
}

for model_size in "${MODEL_SIZES[@]}"; do
    for scheme in "${PRESET_SCHEMES[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            for baseline in "${BASELINES[@]}"; do
                for num_calibration_samples in "${NUM_CALIBRATION_SAMPLES[@]}"; do
                    for alpha in "${ALPHA[@]}"; do
                        run_quantization "${model_size}" "${scheme}" "${dataset}" "${baseline}" "${num_calibration_samples}" "${alpha}"
                    done
                done
            done
        done
    done
done