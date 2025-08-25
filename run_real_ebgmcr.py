import os
import sys
import copy
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import torch

from ebgmcr import DenseEBgMCR 
from ebgmcr.utils import init_directory

DefaultEBgMCRArgument = {
        'num_components': 1024,
        'sparse_component': False,
        'non_negative_component': True,
        'component_norm': 2,
        'hidden_size_extension': 2,
        'hidden_layers': 2,
        'optimizer_class': torch.optim.AdamW,
        'optimizer_args': {'lr': 5e-4,
                           'betas': (0.9, 0.995),
                           'weight_decay': 0.001,
                           'eps': 1e-8},
        'checkpoint_criterion': 'R2',
        'checkpoint_threshold': 0.97,
        'checkpoint_interval': 0.005,
        'mini_batch_size': 64,
        'max_epoch': -1,
        'tol_pattern_nmse': 0.01,
        'tol_energy_ratio': 0.25,
        'select_prob_threshld': 0.9999994,
        'min_component_coef': 1.,
        'ambiguity_coef': 1e-10,
        'energy_coef': 1e-10,
        'ambiguity_prob_threshold': 0.5,
        'temperature_max': 1.,
        'temperature_min': 0.4,
        'temperature_drop_value': 0.01,
        'temperature_drop_epochs': 100,
        'temperature_eval': 0.01,
        'langevin_step_size': 0.05,
        'langevin_noise': 0.1,
        'langevin_steps': 32}

def prepare_dataset(dataset_name, data_dir = 'real_dataset'):
    available_datasets = ['carbs', 'nir']
    assert dataset_name in available_datasets

    if not os.path.isdir(data_dir):
        raise OSError(f'Dir: {data_dir} not exist.')

    files = {'carbs': {'C': 'carbs_C_concentrations.csv',
                       'S': 'carbs_S_endmembers.csv',
                       'D': 'carbs_D_spectra.csv'},
             'nir': {'C': 'nir_C_concentrations.csv',
                     'S': None,
                     'D': 'nir_D_spectra.csv'},
             }

    dataset = {}
    for key in files[dataset_name]:
        if files[dataset_name][key] is not None:
            file_path = os.path.join(data_dir, files[dataset_name][key])
            df = pd.read_csv(file_path)
            array = df.to_numpy()
            if key == 'D' and dataset_name =='carbs':
                array = np.flip(array, axis = 1)
            elif key == 'S':
                if dataset_name == 'carbs':
                    array = array[:, 1:] # skip the wavelegnth header line

                array = array.transpose(1, 0)

            array = torch.from_numpy(np.ascontiguousarray(array)).to(torch.float32)
        else:
            array = None

        dataset[key] = array

    dataset['dim_component'] = dataset['D'].shape[-1]
    if dataset_name == 'nir':
        dataset['non_negative_component'] = False
    else:
        dataset['non_negative_component'] = True

    return dataset

def main():
    display_text = 'Apply EBgMCR solver on real dataset (carbs and nir). ' + \
            'Please use R run real_dataset/prepare_real_data.R to download the dataset.'
    parser = argparse.ArgumentParser(description = display_text)

    parser.add_argument('--dataset_name', type = str, required = True,
            help = 'The dataset ("carbs" and "nir") to be analyzed by EB-gMCR.')
    parser.add_argument('--device', type = int, default = -1,
            help = 'The index of the CUDA device, -1 is CPU.')
    parser.add_argument('--force_replace', action = 'store_true',
            help = 'To force replace the directory if old file exists.')
    parser.add_argument('--result_dir', type = str, default = 'outputs',
            help = 'The saved directory of the searched results.')

    # EB-gMCR solver parameters
    parser.add_argument('--num_components', type = int, default = 128,
            help = 'Set candidate component number to the solver.')
    parser.add_argument('--hidden_size_extension', type = int, default = 2,
            help = 'The extension level of the non-linear hidden space (0 will be linear).')
    parser.add_argument('--hidden_layers', type = int, default = 2,
            help = 'The layer number of the concentration and energy evaluator.')
    parser.add_argument('--max_epoch', type = int, default = -1,
            help = 'The maximum epoch of the EB-gMCR solver.')
    parser.add_argument('--base_lr', type = float, default = 5e-4,
            help = 'The default learning rate of the AdamW optimizer.')
    parser.add_argument('--adam_beta1', type = float, default = 0.9,
            help = 'The default beta1 of the AdamW optimizer.')
    parser.add_argument('--adam_beta2', type = float, default = 0.995,
            help = 'The default beta2 of the AdamW optimizer.')
    parser.add_argument('--weight_decay', type = float, default = 0.001,
            help = 'The weight_deacy of the AdamW optimizer.')
    parser.add_argument('--adam_eps', type = float, default = 1e-8,
            help = 'The eps of the AdamW optimizer.')
    parser.add_argument('--checkpoint_criterion', type = str, default = 'R2',
            help = 'To save the acceptable checkpoint state by assigned criterion.')
    parser.add_argument('--checkpoint_threshold', type = float, default = 0.95,
            help = 'The performance boundary of saving the checkpoint.')
    parser.add_argument('--checkpoint_interval', type = float, default = 0.005,
            help = 'The performance interval of saving the checkpoint.')
    parser.add_argument('--mini_batch_size', type = int, default = 64,
            help = 'The mini-batch size of the EBg-MCR solver.')
    parser.add_argument('--tol_pattern_nmse', type = float, default = 0.01,
            help = 'The tolerance of pattern reconstruction to evaluate training converge.')
    parser.add_argument('--tol_energy_ratio', type = float, default = 0.25,
            help = 'The tolerance of selecting probability difference to evaluate training converge.')
    parser.add_argument('--select_prob_threshld', type = float, default = 0.9999994,
            help = 'The threshold of evaluate the effective component.')
    parser.add_argument('--min_component_coef', type = float, default = 1.,
            help = 'The coefficient of minimizing component usage.')
    parser.add_argument('--ambiguity_coef', type = float, default = 1e-10,
            help = 'The coefficient of the ambiguity regularization.')
    parser.add_argument('--energy_coef', type = float, default = 1e-10,
            help = 'The coefficient of the total energy field regularizaiton.')
    parser.add_argument('--ambiguity_prob_threshold', type = float, default = 0.5,
            help = 'The selected threshold of component ambiguity.')
    parser.add_argument('--temperature_max', type = float, default = 1.,
            help = 'The started temperature of the EB-select module.')
    parser.add_argument('--temperature_min', type = float, default = 0.4,
            help = 'The stop temperature of the EB-select module.')
    parser.add_argument('--temperature_drop_value', type = float, default = 0.01,
            help = 'The drop value of a single temperature decay.')
    parser.add_argument('--temperature_drop_epochs', type = int, default = 100,
            help = 'The interval of dropping the temperature for EB-select.')
    parser.add_argument('--temperature_eval', type = float, default = 0.01,
            help = 'The temperature when EB-select is in eval mode.')
    parser.add_argument('--langevin_step_size', type = float, default = 0.05,
            help = 'The single step size of the approximated SGLD.')
    parser.add_argument('--langevin_noise', type = float, default = 0.1,
            help = 'The nosie level of the approximated SGLD.')
    parser.add_argument('--langevin_steps', type = int, default = 32,
            help = 'The default sampling step of the approximated SGLD.')

    args = parser.parse_args()
    init_directory(args.result_dir)
    force_replace = args.force_replace
    if not force_replace:
        if any(Path(args.result_dir).iterdir()):
            sys.exit(f'Directory: {args.result_dir} containing files triggers overwriting protection,' + \
                    ' use --force_replace to force overwrite the result.')

    if args.device < 0:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{0}'.format(args.device))

    dataset = prepare_dataset(args.dataset_name)

    ebgmcr_config = copy.deepcopy(DefaultEBgMCRArgument)
    ebgmcr_config['num_components'] = args.num_components
    ebgmcr_config['dim_components'] = dataset['dim_component']
    ebgmcr_config['non_negative_component'] = dataset['non_negative_component']
    ebgmcr_config['hidden_size_extension'] = args.hidden_size_extension
    ebgmcr_config['optimizer_args'] = {'lr': args.base_lr,
                                       'betas': (args.adam_beta1, args.adam_beta2),
                                       'weight_decay': args.weight_decay,
                                       'eps': args.adam_eps}

    ebgmcr_config['checkpoint_criterion'] = args.checkpoint_criterion
    ebgmcr_config['checkpoint_threshold'] = args.checkpoint_threshold
    ebgmcr_config['checkpoint_interval'] = args.checkpoint_interval
    ebgmcr_config['device'] = device
    ebgmcr_config['mini_batch_size'] = args.mini_batch_size
    ebgmcr_config['max_epoch'] = args.max_epoch
    ebgmcr_config['tol_pattern_nmse'] = args.tol_pattern_nmse
    ebgmcr_config['tol_energy_ratio'] = args.tol_energy_ratio
    ebgmcr_config['select_prob_threshld'] = args.select_prob_threshld
    ebgmcr_config['min_component_coef'] = args.min_component_coef
    ebgmcr_config['ambiguity_coef'] = args.ambiguity_coef
    ebgmcr_config['energy_coef'] = args.energy_coef
    ebgmcr_config['ambiguity_prob_threshold'] = args.ambiguity_prob_threshold
    ebgmcr_config['temperature_max'] = args.temperature_max
    ebgmcr_config['temperature_min'] = args.temperature_min
    ebgmcr_config['temperature_drop_value'] = args.temperature_drop_value
    ebgmcr_config['temperature_drop_epochs'] = args.temperature_drop_epochs
    ebgmcr_config['temperature_eval'] = args.temperature_eval
    ebgmcr_config['langevin_step_size'] = args.langevin_step_size
    ebgmcr_config['langevin_noise'] = args.langevin_noise
    ebgmcr_config['langevin_steps'] = args.langevin_steps

    analyzer = DenseEBgMCR(**ebgmcr_config)
    analyzer.fit(dataset['D'],
                 show_train_info = False,
                 show_eval_info = True,
                 log_dir = args.result_dir,
                 full_result_filename = os.path.join(args.result_dir, 'full_progress.csv'),
                 final_result_filename = os.path.join(args.result_dir, 'final_result.csv'))

    analyzer.save_mcrs(args.result_dir)
    print('Saving EBg-MCR solver (dataset={0}) done.'.format(args.dataset_name))

    return None

if __name__ == '__main__':
    main()


