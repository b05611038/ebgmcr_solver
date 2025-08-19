import os
import sys
import argparse
from pathlib import Path

import numpy as np

from ebgmcr.eval import StreamingMetrics
from ebgmcr.utils import init_directory

from run_real_ebgmcr import prepare_dataset
from evaluate_synthesized_benchmarks import (NMF_baseline,
                                             SparseNMF_baseline,
                                             BayesNMF_baseline,
                                             ICA_baseline,
                                             MCR_ALS_baseline)

NMF_max_iter = 2000
ICA_max_iter = 2000
MCR_ALS_max_iter = 5000

def search_baseline(data, start_component = 1, max_component = 10):
    assert start_component >= 1
    assert max_component >= start_component
    baseline_result = {'NMF': {}, 'Sparse-NMF': {}, 'Bayes-NMF': {}, 'ICA': {}, 'MCR-ALS': {}}
    for c_num in range(start_component, max_component + 1):
        baseline_result['NMF'][c_num] = NMF_baseline(data, c_num, max_iter = NMF_max_iter)
        baseline_result['Sparse-NMF'][c_num] = SparseNMF_baseline(data, c_num, max_iter = NMF_max_iter)
        baseline_result['Bayes-NMF'][c_num] = BayesNMF_baseline(data, c_num, max_iter = NMF_max_iter)
        baseline_result['ICA'][c_num] = ICA_baseline(data, c_num, max_iter = ICA_max_iter)
        baseline_result['MCR-ALS'][c_num] = MCR_ALS_baseline(data, c_num, max_iter = MCR_ALS_max_iter)

    return baseline_result

def main():
    display_text = 'Apply baseline solvers on real dataset (carbs and nir). ' + \
            'Please use R run real_dataset/prepare_real_data.R to download the dataset.'
    parser = argparse.ArgumentParser(description = display_text)

    parser.add_argument('--dataset_name', type = str, required = True,
            help = 'The dataset ("carbs" and "nir") to be analyzed by EB-gMCR.')
    parser.add_argument('--force_replace', action = 'store_true',
            help = 'To force replace the directory if old file exists.')
    parser.add_argument('--result_dir', type = str, default = 'outputs',
            help = 'The saved directory of the searched results.')

    parser.add_argument('--max_component', type = int, default = 10,
            help = 'The maximum component of the searching algorithm.')

    args = parser.parse_args()
    init_directory(args.result_dir)
    force_replace = args.force_replace
    if not force_replace:
        if any(Path(args.result_dir).iterdir()):
            sys.exit(f'Directory: {args.result_dir} containing files triggers overwriting protection,' + \
                    ' use --force_replace to force overwrite the result.')

    dataset = prepare_dataset(args.dataset_name)
    baseline_result = search_baseline(dataset['D'].numpy(), max_component = args.max_component)
    for method in baseline_result:
        for c_num in baseline_result[method]:
            save_dir = os.path.join(args.result_dir, method + f'_C{c_num}')
            init_directory(save_dir)
            for matrix_type in baseline_result[method][c_num]:
                save_path = os.path.join(save_dir, matrix_type)
                matrix = baseline_result[method][c_num][matrix_type]
                np.save(save_path, matrix)

            print(f'Saving baseline: {method} (component={c_num}) done.')
        
    return None

if __name__ == '__main__':
    main()


