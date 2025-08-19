import copy
import torch
import numpy as np

from sklearn.decomposition import TruncatedSVD, NMF, FastICA
from scipy.optimize import lsq_linear

import chemometrics as cm
from chemometrics.mcr import McrAR
import chemometrics.mcr.constraint as constraint

from ebgmcr import RandomComponentMixtureSynthesizer
from ebgmcr.eval import StreamingMetrics

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

NMF_max_iter = 2000
ICA_max_iter = 2000
MCR_ALS_max_iter = 5000

def criterions(data, reconstructed_data):
    criterion_func = StreamingMetrics(dim = 512)
    criterion_func.update(torch.tensor(data), torch.tensor(reconstructed_data))
    nmse, r2 = criterion_func.compute()
    return {'nMSE': nmse, 'R2': r2}

def NMF_baseline(data, n_components, l1_ratio = 0.0, alpha = 0.0, max_iter = NMF_max_iter):
    nmf = NMF(n_components = n_components,
              init = "nndsvda",
              solver = "mu",            # multiplicative updates
              beta_loss = "frobenius",
              l1_ratio = l1_ratio,      # 0.0 → no sparsity
              alpha_W = alpha,
              alpha_H = alpha,
              max_iter = max_iter,
              random_state = 0)

    concentration = nmf.fit_transform(data)
    estimated_components = nmf.components_
    reconstructed_data = concentration @ estimated_components

    return {'reconstruction': reconstructed_data,
            'concentration': concentration,
            'compoent': estimated_components}

def SparseNMF_baseline(data, n_components, sparsity = 0.8, max_iter = NMF_max_iter, alpha = 1e-2):
    nmf = NMF(n_components = n_components,
              init = "nndsvda",
              solver = "mu",            # multiplicative updates
              beta_loss = "frobenius",
              l1_ratio = sparsity,      # 0.0 → no sparsity
              alpha_W = alpha,
              alpha_H = alpha,
              max_iter = max_iter,
              random_state = 0)

    concentration = nmf.fit_transform(data)
    estimated_components = nmf.components_
    reconstructed_data = concentration @ estimated_components

    return {'reconstruction': reconstructed_data,
            'concentration': concentration,
            'compoent': estimated_components}

def BayesNMF_baseline(data, n_components, max_iter = NMF_max_iter):
    nmf = NMF(n_components = n_components,
              init = "nndsvda",
              solver = "mu",
              beta_loss = "kullback-leibler",
              l1_ratio = 0.0,
              max_iter = max_iter,
              random_state = 0)

    concentration = nmf.fit_transform(data)
    estimated_components = nmf.components_
    reconstructed_data = concentration @ estimated_components

    return {'reconstruction': reconstructed_data,
            'concentration': concentration,
            'compoent': estimated_components}

def ICA_baseline(data, n_components, max_iter = ICA_max_iter):
    ica = FastICA(n_components = n_components,
                  whiten = "unit-variance",
                  fun = "logcosh",
                  max_iter = max_iter,
                  tol = 1e-3,
                  random_state = 0)

    _ = ica.fit_transform(data)

    estimated_components = ica.components_
    # orient each component so its max peak is positive
    sign_flip = np.sign(estimated_components.max(axis = 1, keepdims = True))
    estimated_components *= sign_flip
    S_T = estimated_components.T.astype(np.double)

    concentration = []
    for x in data:
        x = x.astype(np.double)
        sol = lsq_linear(S_T, x, bounds = (0, np.inf), max_iter = 2000)
        if sol.success:
            concentration.append(sol.x.astype(np.float64))
        else:
            c_ls, *_ = np.linalg.lstsq(S_T, x, rcond=None)
            concentration.append(np.clip(c_ls, 0, None).astype(np.float64))

    concentration = np.vstack(concentration)

    estimated_components[estimated_components < 0.] = 0.
    reconstructed_data = concentration @ estimated_components

    return {'reconstruction': reconstructed_data,
            'concentration': concentration,
            'compoent': estimated_components}

def MCR_ALS_baseline(data, n_components, max_iter = MCR_ALS_max_iter):
    c_init = np.abs(TruncatedSVD(n_components = n_components).fit_transform(data))
    c_init /= np.max(c_init)

    mcr = McrAR(max_iter = int(max_iter),
                tol_increase = 0.5,
                tol_n_increase = 10,
                tol_err_change = 1e-8,
                tol_n_above_min = 100)

    mcr.fit(data, C = c_init)
    concentration = mcr.transform(data)
    estimated_components = mcr.ST_

    reconstructed_data = concentration @ estimated_components

    return {'reconstruction': reconstructed_data,
            'concentration': concentration,
            'compoent': estimated_components}

def search_baselines(data, true_component_number,
                     search_ratios = [1.2, 1.15, 1.1, 1.05, 1., 0.95, 0.9, 0.85, 0.8],
                     success_band = (0.98, 0.985)):

    baseline_result = {'NMF': None, 'Sparse-NMF': None, 'Bayes-NMF': None, 'ICA': None, 'MCR-ALS': None}
    for ratio in search_ratios:
        search_component_number = int(true_component_number * ratio)
        nmf_reconstruction = NMF_baseline(data, search_component_number)['reconstruction']
        nmf_result = criterions(data, nmf_reconstruction)
        nmf_result['component_number'] = search_component_number
        nmf_success = False
        if nmf_result['R2'] > success_band[0] and nmf_result['R2'] < success_band[1]:
            nmf_success = True

        nmf_result['success'] = nmf_success
        if baseline_result['NMF'] is None:
            baseline_result['NMF'] = nmf_result
        else:
            if nmf_success:
                if nmf_result['component_number'] < baseline_result['NMF']['component_number']:
                    baseline_result['NMF'] = nmf_result
            else:
                if not baseline_result['NMF']['success']:
                    if nmf_result['R2'] > baseline_result['NMF']['R2']:
                         baseline_result['NMF'] = nmf_result

        sparse_nmf_reconstruction = SparseNMF_baseline(data, search_component_number)['reconstruction']
        sparse_nmf_result = criterions(data, sparse_nmf_reconstruction)
        sparse_nmf_result['component_number'] = search_component_number
        sparse_nmf_success = False
        if sparse_nmf_result['R2'] >= success_band[0] and sparse_nmf_result['R2'] < success_band[1]:
            sparse_nmf_success = True

        sparse_nmf_result['success'] = sparse_nmf_success
        if baseline_result['Sparse-NMF'] is None:
            baseline_result['Sparse-NMF'] = sparse_nmf_result
        else:
            if sparse_nmf_success:
                if sparse_nmf_result['component_number'] < baseline_result['Sparse-NMF']['component_number']:
                    baseline_result['Sparse-NMF'] = sparse_nmf_result
            else:
                if not baseline_result['Sparse-NMF']['success']:
                    if sparse_nmf_result['R2'] > baseline_result['Sparse-NMF']['R2']:
                        baseline_result['Sparse-NMF'] = sparse_nmf_result

        bayes_nmf_reconstruction = BayesNMF_baseline(data, search_component_number)['reconstruction']
        bayes_nmf_result = criterions(data, bayes_nmf_reconstruction)
        bayes_nmf_result['component_number'] = search_component_number
        bayes_nmf_success = False
        if bayes_nmf_result['R2'] >= success_band[0] and bayes_nmf_result['R2'] < success_band[1]:
            bayes_nmf_success = True

        bayes_nmf_result['success'] = bayes_nmf_success
        if baseline_result['Bayes-NMF'] is None:
            baseline_result['Bayes-NMF'] = bayes_nmf_result
        else:
            if bayes_nmf_success:
                if bayes_nmf_result['component_number'] < baseline_result['Bayes-NMF']['component_number']:
                    baseline_result['Bayes-NMF'] = bayes_nmf_result
            else:
                if not baseline_result['Bayes-NMF']['success']:
                    if bayes_nmf_result['R2'] > baseline_result['Bayes-NMF']['R2']:
                        baseline_result['Bayes-NMF'] = bayes_nmf_result

        ica_reconstruction = ICA_baseline(data, search_component_number)['reconstruction']
        ica_result = criterions(data, ica_reconstruction)
        ica_result['component_number'] = search_component_number
        ica_success = False
        if ica_result['R2'] >= success_band[0] and ica_result['R2'] < success_band[1]:
            ica_success = True

        ica_result['success'] = ica_success
        if baseline_result['ICA'] is None:
            baseline_result['ICA'] = ica_result
        else:
            if ica_success:
                if ica_result['component_number'] < baseline_result['ICA']['component_number']:
                    baseline_result['ICA'] = ica_result
            else:
                if not baseline_result['ICA']['success']:
                    if ica_result['R2'] > baseline_result['ICA']['R2']:
                        baseline_result['ICA'] = ica_result

        mcr_als_reconstruction = MCR_ALS_baseline(data, search_component_number)['reconstruction']
        mcr_als_result = criterions(data, mcr_als_reconstruction)
        mcr_als_result['component_number'] = search_component_number
        mcr_als_success = False
        if mcr_als_result['R2'] >= success_band[0] and mcr_als_result['R2'] < success_band[1]:
            mcr_als_success = True

        mcr_als_result['success'] = mcr_als_success
        if baseline_result['MCR-ALS'] is None:
            baseline_result['MCR-ALS'] = mcr_als_result
        else:
            if mcr_als_success:
                if mcr_als_result['component_number'] < baseline_result['MCR-ALS']['component_number']:
                    baseline_result['MCR-ALS'] = mcr_als_result
            else:
                if not baseline_result['MCR-ALS']['success']:
                    if mcr_als_result['R2'] > baseline_result['MCR-ALS']['R2']:
                        baseline_result['MCR-ALS'] = mcr_als_result

    return baseline_result

def repeat_and_collect_baselines(component_numbers = [16, 32],
                                 datafold = 4,
                                 repeat_time = 100,
                                 search_ratios = [1.2, 1.15, 1.1, 1.05, 1., 0.95, 0.9, 0.85, 0.8],
                                 signal_to_nosie_ratio = 20,
                                 success_band = (0.98, 0.985)):

    total_results = {}
    for c_number in component_numbers:
        print('Start testing component number = ', c_number)
        total_results[c_number] = []
        for _ in range(repeat_time):
            dataset_config = copy.deepcopy(DefaultDatasetArgument)
            dataset_config['component_number'] = c_number
            dataset_config['signal_to_nosie_ratio'] = signal_to_nosie_ratio


            train_data_number = c_number * datafold
            data_sampler = RandomComponentMixtureSynthesizer(**dataset_config)
            generation = data_sampler(train_data_number, separation_mode = True)
            data = np.array(generation[0])

            single_run_result = search_baselines(data, c_number,
                                                 search_ratios = search_ratios,
                                                 success_band = success_band)

            total_results[c_number].append(single_run_result)

    return total_results


def layout_full_result(base_filename, total_results):
    for c_number in total_results:
        filename = base_filename + '_c{0}.csv'.format(c_number)
        lines = []
        results = total_results[c_number]
        running_idx = 0
        for single_run_result in results:
            if len(lines) == 0:
                # header
                header_line = 'index,'
                for model_name in single_run_result:
                    for item in single_run_result[model_name]:
                        header_line += '{0}_{1},'.format(model_name, item)

                header_line += '\n'
                lines.append(header_line)

            single_line = '{0},'.format(running_idx)
            for model_name in single_run_result:
                for item in single_run_result[model_name]:
                    single_line += '{0},'.format(single_run_result[model_name][item])

            single_line += '\n'
            lines.append(single_line)
            running_idx += 1

        with open(filename, 'w') as F:
            F.writelines(lines)
            F.close()

        print('Successfully layout data file: {0}'.format(filename))

    return None

def main():
    component_numbers = [16, 32, 48, 64, 96, 128, 160]
    search_ratios = [1.2, 1.15, 1.1, 1.05, 1., 0.95, 0.9, 0.85, 0.8]
    repeat_time = 100

    # Dense04N, SNR = 20dB, based on EB-gMCR solver, use band range = (0.98, 0.985)
    # Can run in non-for loop, but the large compoent really spend too many time
    for component_number in component_numbers:
        Dense4NSNR20dB_result = repeat_and_collect_baselines(component_numbers = [component_number],
                                                             datafold = 4,
                                                             search_ratios = search_ratios,
                                                             repeat_time = repeat_time,
                                                             signal_to_nosie_ratio = 20.,
                                                             success_band = (0.98, 0.985))

        layout_full_result('Dense4NSNR20dB_benchmark', Dense4NSNR20dB_result)

    # Dense04N, SNR = 30dB, based on EB-gMCR solver, use band range = (0.99, 0.995)
    for component_number in component_numbers:
        Dense4NSNR30dB_result = repeat_and_collect_baselines(component_numbers = [component_number],
                                                             datafold = 4,
                                                             search_ratios = search_ratios,
                                                             repeat_time = repeat_time,
                                                             signal_to_nosie_ratio = 30.,
                                                             success_band = (0.99, 0.995))

        layout_full_result('Dense4NSNR30dB_benchmark', Dense4NSNR30dB_result)

    # Dense08N, SNR = 20dB, based on EB-gMCR solver, use band range = (0.975, 0.98)
    for component_number in component_numbers:
        Dense8NSNR20dB_result = repeat_and_collect_baselines(component_numbers = [component_number],
                                                             datafold = 8,
                                                             search_ratios = search_ratios,
                                                             repeat_time = repeat_time,
                                                             signal_to_nosie_ratio = 20.,
                                                             success_band = (0.975, 0.98))

        layout_full_result('Dense8NSNR20dB_benchmark', Dense8NSNR20dB_result)

    # Dense04N, SNR = 30dB, based on EB-gMCR solver, use band range = (0.99, 0.995)
    for component_number in component_numbers:
        Dense8NSNR30dB_result = repeat_and_collect_baselines(component_numbers = [component_number],
                                                             datafold = 8,
                                                             search_ratios = search_ratios,
                                                             repeat_time = repeat_time,
                                                             signal_to_nosie_ratio = 30.,
                                                             success_band = (0.99, 0.995))

        layout_full_result('Dense8NSNR30dB_benchmark', Dense8NSNR30dB_result)

    print('Program done.')

    return None

if __name__ == '__main__':
    main()


