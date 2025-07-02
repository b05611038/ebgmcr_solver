#!/bin/bash
# Example script for evaluating Langevin dynamic mechanism in EB-gMCR

OUTPUT_DIR="langevin_dynamic_evaluations"
DEVICE="${1:--1}"
REPEAT_TIME=5
MAX_GLOBAL_EPOCH=100000
MAX_LOCAL_EPOCH=100000

echo "Use device = $DEVICE, can set other device by following command:"
echo "bash run_langevin_dynamic_evaluation.sh 0 # -1 indicate cpu, and 0 indicate CUDA:0"
echo ""
# Dense4N dataset, SNR=20dB
python3 batch_evaluate_ebgmcr.py --search_argument langevin_steps --search_argument_values 0 1 2 4 8 16 32 64 \
        --base_name "Dense04N_SNR20dB_" --repeat_time $REPEAT_TIME --magnification_of_data_number 4 \
        --component_number 256 --min_concentration 1 --max_concentration 10 --signal_to_nosie_ratio 20 \
        --result_dir $OUTPUT_DIR --device $DEVICE \
        --max_global_epoch $MAX_GLOBAL_EPOCH --max_local_epoch $MAX_LOCAL_EPOCH \
        --non_negative_component --save_ebgmcr --save_dataset_components

# Dense4N dataset, SNR=30dB
python3 batch_evaluate_ebgmcr.py --search_argument langevin_steps --search_argument_values 0 1 2 4 8 16 32 64 \
        --base_name "Dense04N_SNR30dB_" --repeat_time $REPEAT_TIME --magnification_of_data_number 4 \
        --component_number 256 --min_concentration 1 --max_concentration 10 --signal_to_nosie_ratio 30 \
        --result_dir $OUTPUT_DIR --device $DEVICE \
        --max_global_epoch $MAX_GLOBAL_EPOCH --max_local_epoch $MAX_LOCAL_EPOCH \
        --non_negative_component --save_ebgmcr --save_dataset_components
