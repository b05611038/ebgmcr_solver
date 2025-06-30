import os
import sys
import copy
import argparse

from itertools import product

import torch

from ebgmcr import (RandomComponentMixtureSynthesizer,
                    DenseEBgMCR)

from ebgmcr.utils import init_directory


DefaultDatasetArgument = {
        'component_number': 32,
        'component_dim': 512,
        'component_sparsity': 0.,
        'non_negative_component': True,
        'orthogonal_component': True,
        'component_norm': 2,
        'min_mixing_component': 1,
        'max_mixing_component': 4,
        'min_concentration': 1.,
        'max_concentration': 10.,
        'signal_to_nosie_ratio': 20.}

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

AvailableTestArguments = ['magnification_of_data_number'] + \
        list(DefaultDatasetArgument.keys()) + \
        list(DefaultEBgMCRArgument.keys())

DatasetArguments = ['magnification_of_data_number'] + list(DefaultDatasetArgument.keys())
EBgMCRArguments = ['magnification_of_data_number'] + list(DefaultEBgMCRArgument.keys())

def get_arg_type(parser, arg_name):
    arg_name = '--' + arg_name
    for action in parser._actions:
        if arg_name in action.option_strings:
            return action.type

    return None 

def get_combinations(search_values, all_dataset_args, all_ebgmcr_args):
    dataset_search_space = {k: search_values[k] for k in search_values if k in all_dataset_args}
    ebgmcr_search_space = {k: search_values[k] for k in search_values if k in all_ebgmcr_args}

    combination_dataset = [dict(zip(dataset_search_space.keys(), values))
                           for values in product(*dataset_search_space.values())] if dataset_search_space else [{}]

    combination_ebgmcr = [dict(zip(ebgmcr_search_space.keys(), values))
                          for values in product(*ebgmcr_search_space.values())] if ebgmcr_search_space else [{}]

    all_tests, test_idx = {}, 0
    for i, dataset_config in enumerate(combination_dataset):
        for j, ebgmcr_config in enumerate(combination_ebgmcr):
            all_tests[test_idx] = {'dataset': i, 'ebgmcr': j}
            test_idx += 1

    return all_tests, combination_dataset, combination_ebgmcr

def single_round_test(test_code, save_dir, ebgmcr_args, data_sampler, data_number, 
        data = None, save_dataset_components = False, save_dataset_data = False, 
        save_ebgmcr = False, force_replace = False):

    if os.path.isdir(save_dir):
        if len(os.listdir(save_dir)) > 0:
            if not force_replace:
                print('Directory: {0} already have save files,'.format(force_replace) + \
                        ' please use force_replace=True to force run the single round test')

                return None

    print('Start test: {0}'.format(test_code))

    target_data = None
    if data is None:
        target_data = data_sampler(data_number)
    else:
        target_data = data.detach().clone()

    if target_data.shape[0] < 4 * ebgmcr_args['mini_batch_size']:
        if ebgmcr_args['mini_batch_size'] % 4 == 0:
            ebgmcr_args['mini_batch_size'] = target_data.shape[0] // 4
        else:
            ebgmcr_args['mini_batch_size'] = target_data.shape[0] // 4 + 1

    analyzer = DenseEBgMCR(**ebgmcr_args)
    analyzer.fit(target_data, 
                 show_train_info = False, 
                 show_eval_info = False, 
                 log_dir = save_dir,
                 full_result_filename = os.path.join(save_dir, 'full_progress.csv'),
                 final_result_filename = os.path.join(save_dir, 'final_result.csv'))

    if save_dataset_components:
        dataset_components = data_sampler.components
        dataset_components = dataset_components.cpu()
        torch.save(dataset_components, os.path.join(save_dir, 'ground_truth_components.pth'))
        print('Saving ground truth components done.')

    if save_dataset_data:
        target_data = target_data.cpu()
        torch.save(target_data, os.path.join(save_dir, 'sampled_data.pth'))
        print('Saving sampled data done.')

    if save_ebgmcr:
        analyzer.save_mcrs(save_dir)
        print('Saving EBg-MCR solver done.')

    print('Test: {0} done.'.format(test_code))

    return None

def main():
    display_text = "EBgMCR solver evaluation. Setting --fixed_sampled_data flag can become"+ \
            " hyperparameter searcher for EBgMCR. Four flags (arguments) needed to be set in the" + \
            " program: --search_argument --search_argument_values"

    removed_args = ['orthogonal_component', 'sparse_component', 'non_negative_component', 
                    'optimizer_class', 'optimizer_args', 'checkpoint_criterion']

    for args_name in removed_args:
        AvailableTestArguments.remove(args_name)
        if args_name in DatasetArguments:
            DatasetArguments.remove(args_name)

    added_args = ['base_lr', 'adam_beta1', 'adam_beta2', 'weight_decay', 'adam_eps']
    for args_name in added_args:
        AvailableTestArguments.append(args_name)
        EBgMCRArguments.append(args_name)

    display_text += 'AvailableArguments: {0}'.format(AvailableTestArguments)

    parser = argparse.ArgumentParser(description = display_text)

    # search objective
    parser.add_argument('--search_argument', type = str,
            help = 'The argument you want to search')
    parser.add_argument('--search_argument_values', nargs = '+', type = float,
            help = 'To input the searched values of the search_argument.')
    parser.add_argument('--save_dataset_components', action = 'store_true',
            help = 'To save the components in the result directory')
    parser.add_argument('--save_dataset_data', action = 'store_true',
            help = 'To save the target data in the result directory.')
    parser.add_argument('--save_ebgmcr', action = 'store_true',
            help = 'To save the trained EBg-MCR solver in the directory.')

    # search details
    parser.add_argument('--repeat_time', type = int, default = 3,
            help = 'The repeated time of each config.')
    parser.add_argument('--result_dir', type = str, default = 'outputs',
            help = 'The saved directory of the searched results.')
    parser.add_argument('--base_name', type = str, default = '',
            help = 'The basic name of the testing code.')
    parser.add_argument('--device', type = int, default = -1,
            help = 'The index of the CUDA device, -1 is CPU.')
    parser.add_argument('--fixed_sampled_data', action = 'store_true',
            help = 'Not re-generate data every round (use in hyperparameter searaching).')
    parser.add_argument('--force_replace', action = 'store_true',
            help = 'To force replace the directory if old file exists.')

    # dataset details
    parser.add_argument('--component_number', type = int, default = 32,
            help = 'The total component number in the data generator.')
    parser.add_argument('--component_dim', type = int, default = 512,
            help = 'The dimension of a single component.')
    parser.add_argument('--magnification_of_data_number', type = int, default = 4,
            help = 'Determine how many data be sampled in a round.')
    parser.add_argument('--component_sparsity', type = float, default = 0.,
            help = 'The sparsity of the generated components.')
    parser.add_argument('--non_negative_component', action = 'store_true',
            help = 'To generate all positive components.')
    parser.add_argument('--component_norm', type = int, default = 2,
            help = 'Set the norm function of the components.')
    parser.add_argument('--min_mixing_component', type = int, default = 1,
            help = 'The minimum component number in a sampled data.')
    parser.add_argument('--max_mixing_component', type = int, default = 4,
            help = 'The maximum component number in a sampled data.')
    parser.add_argument('--min_concentration', type = float, default = 1.,
            help = 'The minimum concentration of the sampled component.')
    parser.add_argument('--max_concentration', type = float, default = 10.,
            help = 'The maximum concentration of the sampled component.')
    parser.add_argument('--signal_to_nosie_ratio', type = float, default = 20.,
            help = 'The SNR of the generated data (larger than 50 will not add noise).')

    # EB-gMCR solver parameters
    parser.add_argument('--num_components', type = int, default = 1024,
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
    parser.add_argument('--checkpoint_threshold', type = float, default = 0.97,
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
    target_argument = args.search_argument

    if target_argument not in AvailableTestArguments:
        sys.exit('search_argument: {0} is not valid for this program.'.format(args_name))

    search_argument_values = args.search_argument_values
    if len(search_argument_values) == 0:
        sys.exit('--search_argument_values need at least set one number in this program')

    if args.repeat_time < 1:
        sys.exit('Repeated times (now: {0}) must at least set to one.'.format(args.repeat_time))

    if args.device < 0:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{0}'.format(args.device))

    save_dataset_components = False
    if args.save_dataset_components:
        save_dataset_components = True

    save_dataset_data = False
    if args.save_dataset_data:
        save_dataset_data = True

    save_ebgmcr = False
    if args.save_ebgmcr:
        save_ebgmcr = True

    search_values = {target_argument: search_argument_values}
    for args_name in search_values:
        type_function = get_arg_type(parser, args_name)
        for element_idx in range(len(search_values[args_name])):
            if type_function == int:
                try:
                    search_values[args_name][element_idx] = round(float(search_values[args_name][element_idx]))
                except ValueError:
                    search_values[args_name][element_idx] = type_function(search_values[args_name][element_idx])
            else:
                search_values[args_name][element_idx] = type_function(search_values[args_name][element_idx])

        print('Searching argument: {0}, search_values: {1}'.format(args_name, search_values[args_name]))

    (searched_combinations,
     combination_dataset,
     combination_ebgmcr) = get_combinations(search_values, 
                                            DatasetArguments, 
                                            EBgMCRArguments)

    test_codes, dataset_codes, ebgmcr_codes = {}, {}, {}
    for test_idx in searched_combinations:
        single_code = copy.copy(args.base_name)
        dataset_idx = searched_combinations[test_idx]['dataset']
        ebgmcr_idx = searched_combinations[test_idx]['ebgmcr']

        for args_name in combination_dataset[dataset_idx]:
            type_function = get_arg_type(parser, args_name)
            if type_function == int:
                single_code += '{0}_{1}_'.format(args_name, combination_dataset[dataset_idx][args_name])
            elif type_function == float:
                single_code += f"{args_name}_{combination_dataset[dataset_idx][args_name]:.2e}_"

        for args_name in combination_ebgmcr[ebgmcr_idx]:
            type_function = get_arg_type(parser, args_name)
            if type_function == int:
                single_code += '{0}_{1}_'.format(args_name, combination_ebgmcr[ebgmcr_idx][args_name])
            elif type_function == float:
                single_code += f"{args_name}_{combination_ebgmcr[ebgmcr_idx][args_name]:.2e}_"

        if len(single_code) > 0:
            single_code = single_code[: -1]

        test_codes[test_idx] = single_code
        dataset_codes[test_idx] = dataset_idx
        ebgmcr_codes[test_idx] = ebgmcr_idx

    non_negative_component = False
    if args.non_negative_component:
        non_negative_component = True

    ebgmcr_configs = []
    for ebgmcr_idx in range(len(combination_ebgmcr)):
        sparse_component = False
        if args.component_sparsity > 0.:
            sparse_component = True

        single_ebgmcr_config = copy.deepcopy(DefaultEBgMCRArgument)
        single_ebgmcr_config['num_components'] = args.num_components
        single_ebgmcr_config['component_norm'] = args.component_norm
        single_ebgmcr_config['sparse_component'] = sparse_component
        single_ebgmcr_config['non_negative_component'] = non_negative_component
        single_ebgmcr_config['hidden_size_extension'] = args.hidden_size_extension
        single_ebgmcr_config['optimizer_args'] = {'lr': args.base_lr,
                                                  'betas': (args.adam_beta1, args.adam_beta2),
                                                  'weight_decay': args.weight_decay,
                                                  'eps': args.adam_eps}

        single_ebgmcr_config['checkpoint_criterion'] = args.checkpoint_criterion
        single_ebgmcr_config['checkpoint_threshold'] = args.checkpoint_threshold
        single_ebgmcr_config['checkpoint_interval'] = args.checkpoint_interval
        single_ebgmcr_config['device'] = device
        single_ebgmcr_config['mini_batch_size'] = args.mini_batch_size
        single_ebgmcr_config['max_epoch'] = args.max_epoch
        single_ebgmcr_config['tol_pattern_nmse'] = args.tol_pattern_nmse
        single_ebgmcr_config['tol_energy_ratio'] = args.tol_energy_ratio
        single_ebgmcr_config['select_prob_threshld'] = args.select_prob_threshld
        single_ebgmcr_config['min_component_coef'] = args.min_component_coef
        single_ebgmcr_config['ambiguity_coef'] = args.ambiguity_coef
        single_ebgmcr_config['energy_coef'] = args.energy_coef
        single_ebgmcr_config['ambiguity_prob_threshold'] = args.ambiguity_prob_threshold
        single_ebgmcr_config['temperature_max'] = args.temperature_max
        single_ebgmcr_config['temperature_min'] = args.temperature_min
        single_ebgmcr_config['temperature_drop_value'] = args.temperature_drop_value
        single_ebgmcr_config['temperature_drop_epochs'] = args.temperature_drop_epochs
        single_ebgmcr_config['temperature_eval'] = args.temperature_eval
        single_ebgmcr_config['langevin_step_size'] = args.langevin_step_size
        single_ebgmcr_config['langevin_noise'] = args.langevin_noise
        single_ebgmcr_config['langevin_steps'] = args.langevin_steps

        for args_name in combination_ebgmcr[ebgmcr_idx]:
            if args_name in single_ebgmcr_config.keys():
                single_ebgmcr_config[args_name] = combination_ebgmcr[ebgmcr_idx][args_name]

        ebgmcr_configs.append(single_ebgmcr_config)

    mapped_data_samplers, mapped_data_numbers = {}, {}
    mapped_data = None
    if args.fixed_sampled_data:
        mapped_data = {}

    for dataset_idx in range(len(combination_dataset)):
        if args.signal_to_nosie_ratio >= 50.:
            signal_to_nosie_ratio = None

        sampler_config = {'component_number': args.component_number,
                          'component_dim': args.component_dim,
                          'component_sparsity': args.component_sparsity,
                          'non_negative_component': non_negative_component,
                          'orthogonal_component': True,
                          'component_norm': args.component_norm,
                          'min_mixing_component': args.min_mixing_component,
                          'max_mixing_component': args.max_mixing_component,
                          'min_concentration': args.min_concentration,
                          'max_concentration': args.max_concentration,
                          'signal_to_nosie_ratio': args.signal_to_nosie_ratio}

        magnification_of_data_number = args.magnification_of_data_number
        for args_name in combination_dataset[dataset_idx]:
            if args_name in sampler_config.keys():
                sampler_config[args_name] = combination_dataset[dataset_idx][args_name]

            if args_name == 'magnification_of_data_number':
                magnification_of_data_number = combination_dataset[dataset_idx][args_name]

        single_data_sampler = RandomComponentMixtureSynthesizer(**sampler_config)
        single_data_number = magnification_of_data_number * sampler_config['component_number']

        for round_idx in range(args.repeat_time):
            fetch_name = '{0}_{1}'.format(dataset_idx, round_idx)
            mapped_data_samplers[fetch_name] = single_data_sampler
            mapped_data_numbers[fetch_name] = single_data_number
            if mapped_data is not None:
                mapped_data[fetch_name] = single_data_sampler(single_data_number, 
                                                              separation_mode = False)

    round_names, save_dirs, ebgmcr_args = [], [], []
    data_samplers, data_numbers, sampled_data = [], [], []
    for round_idx in range(args.repeat_time):
        complement = ''
        if args.repeat_time > 1:
            complement = '_{0}'.format(round_idx + 1)

        for single_test_code in test_codes:
            round_name = test_codes[single_test_code] + complement
            round_names.append(round_name)

            saving_path = os.path.join(args.result_dir, round_name)
            save_dirs.append(saving_path)

            data_fetch_name = '{0}_{1}'.format(dataset_codes[single_test_code], round_idx)
            single_sampler = copy.deepcopy(mapped_data_samplers[data_fetch_name])
            data_samplers.append(single_sampler)
            single_data_number = copy.deepcopy(mapped_data_numbers[data_fetch_name])
            data_numbers.append(single_data_number)
            if mapped_data is not None:
                sampled_data.append(copy.deepcopy(mapped_data[data_fetch_name]))
            else:
                sampled_data.append(None)

            ebgmcr_config = copy.deepcopy(ebgmcr_configs[ebgmcr_codes[single_test_code]])
            ebgmcr_config['dim_components'] = single_sampler.component_dim
            sparse_component = False
            if single_sampler.component_sparsity > 0.:
                sparse_component = True

            ebgmcr_config['sparse_component'] = sparse_component
            ebgmcr_args.append(ebgmcr_config)

    for idx in range(len(round_names)):
        single_round_test(round_names[idx],
                          save_dirs[idx],
                          ebgmcr_args[idx],
                          data_samplers[idx],
                          data_numbers[idx],
                          data = sampled_data[idx],
                          save_dataset_components = save_dataset_components,
                          save_dataset_data = save_dataset_data,
                          save_ebgmcr = save_ebgmcr)

    return None

if __name__ == '__main__':
    main()


