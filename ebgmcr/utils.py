import os
import json

import torch
import torch.nn as nn

from safetensors import safe_open
from safetensors.torch import save_file

#--------------------------------------------------------------------------------
# utils.py contains the function that help denoise module work
#--------------------------------------------------------------------------------

__all__ = ['init_directory', 'save_as_json', 'load_json', 'save_as_safetensors', 
        'load_safetensors', 'init_layer_weight', 'EnergyL1Scheduler']

def init_directory(path):
    assert isinstance(path, str)

    if len(path) > 0:
        if not os.path.isdir(path):
            os.makedirs(path)

    return path

def save_as_json(obj, filename):
    assert isinstance(filename, str)

    if not filename.endswith('.json'):
        filename += '.json'

    obj = json.dumps(obj)
    with open(filename, 'w') as out_file:
        out_file.write(obj)
        out_file.close()

    return None

def load_json(filename, extension_check = True):
    assert isinstance(filename, str)
    assert isinstance(extension_check, bool)

    content = None
    if extension_check:
        if not filename.endswith('.json'):
            raise RuntimeError('File: {0} is not a .json file.'.format(filename))

    with open(filename, 'r') as in_file:
        content = json.loads(in_file.read())

    return content

def save_as_safetensors(tensors, filename):
    assert isinstance(tensors, dict)
    assert isinstance(filename, str)

    if not filename.endswith('.safetensors'):
        filename += '.safetensors'

    save_file(tensors, filename)
    return None

def load_safetensors(filename, device = 'cpu', extension_check = True):
    assert isinstance(filename, str)
    assert isinstance(device, (str, torch.device))
    assert isinstance(extension_check, bool)

    if extension_check:
        if not filename.endswith('.safetensors'):
            raise RuntimeError('File: {0} is not a .json file.'.format(filename))

    tensors = {}
    with safe_open(filename, framework = 'pt', device = device) as in_files:
        for key in in_files.keys():
            tensors[key] = in_files.get_tensor(key)

    return tensors

def init_layer_weight(layer, initializer_range = 1e-2):
    assert isinstance(layer, (nn.Linear, nn.Parameter))
    assert isinstance(initializer_range, float)
    
    if isinstance(layer, nn.Parameter):
        nn.init.normal_(layer.data, mean = 0., std = initializer_range)
    elif isinstance(layer, nn.Linear):
        nn.init.normal_(layer.weight, mean = 0., std = initializer_range)
        nn.init.normal_(layer.bias, mean = 0., std = initializer_range)

    return layer

def checkpoint_regions(criterion, threshold, interval):
    assert criterion in ['R2', 'nMSE']
    assert isinstance(threshold, float)
    assert threshold > 0. and threshold < 1.
    assert isinstance(interval, float)
    assert interval > 0.

    regions = []
    if criterion == 'R2':
        start_value, end_value = threshold, threshold + interval
        while end_value <= 1.:
            context = '{0}[->]{1}'.format(start_value, end_value)
            regions.append(context)
            start_value = end_value
            end_value = start_value + interval

    elif criterion == 'nMSE':
        start_value, end_value = threshold, threshold - interval
        while end_value >= 0.:
            context = '{0}[->]{1}'.format(start_value, end_value)
            regions.append(context)
            start_value = end_value
            end_value = start_value - interval

    return regions

def specify_checkpoint_region(criterion_value, criterion, effective_regions):
    assert criterion in ['R2', 'nMSE']
    specified_region = None
    for region_context in effective_regions:
        split_context = region_context.split('[->]')
        start_value = float(split_context[0])
        end_value = float(split_context[1])
        if criterion == 'R2':
            if criterion_value >= start_value and criterion_value < end_value:
                specified_region = region_context
                break

        elif criterion == 'nMSE':
            if criterion_value <= start_value and criterion_value > end_value:
                specified_region = region_context
                break

    return specified_region


class EnergyL1Scheduler:
    """
    Keeps a smoothed λ1 that turns on once nMSE < margin, and stays near
    (MSE / (used_N / total_N)) - eps, but smoothed to avoid oscillations.
    """

    def __init__(self,
                 total_components: int,
                 target_ratio: float = 1.,
                 nmse_margin: float = 0.005,
                 init_coef: float = 0.0,
                 smooth_factor: float = 0.95):
        """
        Args:
            total_components: total number of available components (N).
            target_ratio:  the ratio of regularizor targeting
            margin:        the nMSE threshold at which λ1 starts to become nonzero.
            init_lambda1:  initial λ1 at epoch 0.
            smooth_factor: EMA factor (β) in [0,1). Higher ≈ more smoothing.
                           New λ1 = β * old λ1 + (1-β) * target.
        """
        self.N_total = float(total_components)
        self.target_ratio = float(target_ratio)
        self.nmse_margin = float(nmse_margin)
        self.beta = float(smooth_factor)
        self.coef = float(init_coef)

    def step(self, nMSE: float, MSE: float, used_components: int) -> float:
        """
        Call once per epoch after computing nMSE and MSE on validation set.

        Args:
            nMSE:              normalized MSE (e.g., mse / var(data) or whatever)
            MSE:               the true (unnormalized) MSE for the same batch.
            used_components:   how many components were active in the last epoch.

        Returns:
            The updated, smoothed λ1 to use for the next training epoch.
        """
        # If nMSE is above the margin, we zero out λ1 (no sparsity pressure yet).
        if nMSE >= self.nmse_margin or used_components == 0:
            target_coef = 0.0
        else:
            # Compute the “raw” target so that:
            #   λ1 * (used/total) ≈ MSE * ratio
            frac = used_components / self.N_total
            # Avoid division by zero; we know used_components>0 here.
            raw = MSE * self.target_ratio / frac
            # We never allow λ1 to go negative.
            target_coef = max(0.0, raw)

        # Smooth by EMA: new_lambda1 = β * old + (1-β) * target
        self.coef = self.beta * self.coef + (1.0 - self.beta) * target_coef

        return self.coef


