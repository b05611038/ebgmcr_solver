#!/bin/bash
OUTPUT_DIR="component_number_evaluations"
DEVICE="${1:--1}"
REPEAT_TIME=5
MAX_EPOCH=100000

echo "Use device = $DEVICE, can set other device by following command:"
echo "bash run_component_number_evaluation.sh 0 # -1 indicate cpu, and 0 indicate CUDA:0"
echo ""
# Dense4N dataset, SNR=20dB
python3 batch_evaluate_ebgmcr.py --search_argument component_number \
        --search_argument_values 16 32 48 64 96 128 160 192 224 256 \
        --base_name "Dense04N_SNR20dB_" --repeat_time $REPEAT_TIME --magnification_of_data_number 4 \
        --component_dim 512 --min_mixing_component 1 --max_mixing_component 4 \
        --min_concentration 1 --max_concentration 10 --signal_to_nosie_ratio 20 \
        --result_dir $OUTPUT_DIR --device $DEVICE --max_epoch $MAX_EPOCH \
        --non_negative_component --save_ebgmcr --save_dataset_components

echo "Use device = $DEVICE, can set other device by following command:"
echo "bash BatchEvaluateEBgMCRSolver.sh 0 # -1 indicate cpu, and 0 indicate CUDA:0"
echo ""
# Dense4N dataset, SNR=30dB
python3 batch_evaluate_ebgmcr.py --search_argument component_number \
        --search_argument_values 16 32 48 64 96 128 160 192 224 256 \
        --base_name "Dense04N_SNR20dB_" --repeat_time $REPEAT_TIME --magnification_of_data_number 4 \
        --component_dim 512 --min_mixing_component 1 --max_mixing_component 4 \
        --min_concentration 1 --max_concentration 10 --signal_to_nosie_ratio 30 \
        --result_dir $OUTPUT_DIR --device $DEVICE --max_epoch $MAX_EPOCH \
        --non_negative_component --save_ebgmcr --save_dataset_components

# Dense8N dataset, SNR=20dB
python3 batch_evaluate_ebgmcr.py --search_argument component_number \
        --search_argument_values 16 32 48 64 96 128 160 192 224 256 \
        --base_name "Dense08N_SNR20dB_" --repeat_time $REPEAT_TIME --magnification_of_data_number 4 \
        --component_dim 512 --min_mixing_component 1 --max_mixing_component 8 \
        --min_concentration 1 --max_concentration 10 --signal_to_nosie_ratio 20 \
        --result_dir $OUTPUT_DIR --device $DEVICE --max_epoch $MAX_EPOCH \
        --non_negative_component --save_ebgmcr --save_dataset_components

echo "Use device = $DEVICE, can set other device by following command:"
echo "bash BatchEvaluateEBgMCRSolver.sh 0 # -1 indicate cpu, and 0 indicate CUDA:0"
echo ""
# Dense8N dataset, SNR=30dB
python3 batch_evaluate_ebgmcr.py --search_argument component_number \
        --search_argument_values 16 32 48 64 96 128 160 192 224 256 \
        --base_name "Dense08N_SNR20dB_" --repeat_time $REPEAT_TIME --magnification_of_data_number 4 \
        --component_dim 512 --min_mixing_component 1 --max_mixing_component 8 \
        --min_concentration 1 --max_concentration 10 --signal_to_nosie_ratio 30 \
        --result_dir $OUTPUT_DIR --device $DEVICE --max_epoch $MAX_EPOCH \
        --non_negative_component --save_ebgmcr --save_dataset_components
