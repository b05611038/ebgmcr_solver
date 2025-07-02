import os
import abc
import copy
import inspect

from collections import deque

import torch
import torch.nn as nn

import torch.optim as optim
from torch.utils.data import (TensorDataset, 
                              DataLoader)

from torch.utils.tensorboard import SummaryWriter

from .utils import (init_directory,
                    checkpoint_regions,
                    specify_checkpoint_region,
                    EnergyL1Scheduler)

from .ambiguity_penalty import (linear_penalty,
                                polynomial_penalty,
                                laplacian_penalty,
                                rbf_penalty,
                                matern_penalty)

from .mcr import (EBSelectiveMCR,
                  default_aggreate_function)

from .eval import StreamingMetrics

#--------------------------------------------------------------------------------
# searcher.py contains the core algorithm of filtering polluted pattern
#--------------------------------------------------------------------------------


__all__ = ['BaseEBgMCR', 'DenseEBgMCR']


class BaseEBgMCR(abc.ABC):
    def __init__(self, 
            num_components,
            dim_components = None,
            sparse_component = False,
            non_negative_component = True,
            component_norm = 2,
            hidden_size_extension = 2,
            hidden_layers = 2,
            optimizer_class = optim.AdamW,
            optimizer_args = {'lr': 5e-4, 'betas': (0.9, 0.995), weight_decay: 1e-3},
            checkpoint_criterion = 'R2',
            checkpoint_threshold = 0.97,
            checkpoint_interval = 0.005,
            device = torch.device('cpu'),
            mini_batch_size = 64,
            max_epoch = -1,
            tol_pattern_nmse = 0.01,
            tol_energy_ratio = 0.25,
            select_prob_threshld = 0.9999994,
            min_component_coef = 1.,
            ambiguity_coef = 1e-10,
            energy_coef = 1e-10,
            ambiguity_penalty = 'rbf',
            ambiguity_prob_threshold = 0.5,
            poly_dim = 2,
            matern_nu = 1.5,
            temperature_max = 1., 
            temperature_min = 0.4,
            temperature_drop_value = 0.01,
            temperature_drop_epochs = 100,
            temperature_eval = 0.01,
            langevin_step_size = 0.05, 
            langevin_noise = 0.1, 
            langevin_steps = 32): 

        self.stored_mcrs = nn.ModuleList()

        self.num_components = num_components
        self.dim_components = dim_components
        self.sparse_component = sparse_component
        self.non_negative_component = non_negative_component
        self.component_norm = component_norm
        self.hidden_size_extension = hidden_size_extension
        self.hidden_layers = hidden_layers

        self.langevin_step_size = langevin_step_size
        self.langevin_noise = langevin_noise
        self.langevin_steps = langevin_steps

        self.optimizer_class = optimizer_class
        self.optimizer_args = optimizer_args
        self.optimizer = None

        self.checkpoint_criterion = checkpoint_criterion
        self.checkpoint_threshold = checkpoint_threshold
        self.checkpoint_interval = checkpoint_interval

        self.mini_batch_size = mini_batch_size
        self.device = device

        self.max_epoch = max_epoch
        self.tol_pattern_nmse = tol_pattern_nmse
        self.tol_energy_ratio = tol_energy_ratio
        self.select_prob_threshld = select_prob_threshld

        self.min_component_coef = min_component_coef
        self.ambiguity_coef = ambiguity_coef
        self.energy_coef = energy_coef
        self.ambiguity_penalty = ambiguity_penalty
        self.ambiguity_prob_threshold = ambiguity_prob_threshold
        self.poly_dim = poly_dim
        self.matern_nu = matern_nu

        self._temperature_max = None
        self._temperature_min = None
        self.temperature_max = temperature_max
        self.temperature_min = temperature_min

        self.temperature_drop_value = temperature_drop_value
        self.temperature_drop_epochs = temperature_drop_epochs
        self.temperature_eval = temperature_eval

        self.__finish_analysis = False

    @property
    def num_components(self):
        if len(self.stored_mcrs) > 0:
             num_components = []
             for mcr in self.stored_mcrs:
                 num_components.append(mcr.num_components)

             self._num_components = tuple(num_components)
        else:
             num_components = self._num_components

        return num_components

    @num_components.setter
    def num_components(self, num_components):
        if len(self.stored_mcrs) == 0:
            assert isinstance(num_components, (int, tuple, list))
            if isinstance(num_components, int):
                num_components = (num_components, )

            for number in num_components:
                assert isinstance(number, int)
                assert number > 0

            self._num_components = tuple(num_components)
        else:
            raise RuntimeError('Please directly set num_components in target MCR module' + \
                    ' in {0}.stored_mcrs'.format(self.__class__.__name__))

        return None

    @property
    def dim_components(self):
        return self._dim_components

    @dim_components.setter
    def dim_components(self, dim_components):
        if dim_components is not None:
            assert isinstance(dim_components, int)
            assert dim_components > 0

        self._dim_components = dim_components
        if self.stored_mcrs is not None and dim_components is not None:
            for mcr in self.stored_mcrs:
                mcr.dim_components = dim_components

        return None

    @property
    def sparse_component(self):
        return self._sparse_component

    @sparse_component.setter
    def sparse_component(self, sparse_component):
        assert isinstance(sparse_component, bool)
        self._sparse_component = sparse_component
        if self.stored_mcrs is not None:
            for mcr in self.stored_mcrs:
                mcr.sparse_component = sparse_component

        return None

    @property
    def non_negative_component(self):
        return self._non_negative_component

    @non_negative_component.setter
    def non_negative_component(self, non_negative_component):
        assert isinstance(non_negative_component, bool)
        self._non_negative_component = non_negative_component
        if self.stored_mcrs is not None:
            for mcr in self.stored_mcrs:
                mcr.non_negative_component = non_negative_component

        return None

    @property
    def component_norm(self):
        return self._component_norm

    @component_norm.setter
    def component_norm(self, component_norm):
        if component_norm is not None:
            assert isinstance(component_norm, int)
            assert component_norm >= 1

        self._component_norm = component_norm
        return None

    @property
    def hidden_size_extension(self):
        return self._hidden_size_extension

    @hidden_size_extension.setter
    def hidden_size_extension(self, hidden_size_extension):
        if hidden_size_extension is not None:
            assert isinstance(hidden_size_extension, int)
            assert hidden_size_extension >= 1

        self._hidden_size_extension = hidden_size_extension
        return None

    @property
    def hidden_layers(self):
        return self._hidden_layers

    @hidden_layers.setter
    def hidden_layers(self, hidden_layers):
        assert isinstance(hidden_layers, int)
        assert hidden_layers >= 1

        self._hidden_layers = hidden_layers
        return None

    @property
    def langevin_step_size(self):
        return self._langevin_step_size

    @langevin_step_size.setter
    def langevin_step_size(self, langevin_step_size):
        assert isinstance(langevin_step_size, (int, float))
        assert langevin_step_size > 0.
        langevin_step_size = float(langevin_step_size)
        self._langevin_step_size = langevin_step_size
        if self.stored_mcrs is not None:
            for mcr in self.stored_mcrs:
                mcr.langevin_step_size = langevin_step_size

        return None

    @property
    def langevin_noise(self):
        return self._langevin_noise

    @langevin_noise.setter
    def langevin_noise(self, langevin_noise):
        assert isinstance(langevin_noise, (int, float))
        assert langevin_noise > 0.
        langevin_noise = float(langevin_noise)
        self._langevin_noise = langevin_noise
        if self.stored_mcrs is not None:
            for mcr in self.stored_mcrs:
                mcr.langevin_noise = langevin_noise

        return None

    @property
    def langevin_steps(self):
        return self._langevin_steps

    @langevin_steps.setter
    def langevin_steps(self, langevin_steps):
        assert isinstance(langevin_steps, int)
        assert langevin_steps >= 0
        self._langevin_steps = langevin_steps
        if self.stored_mcrs is not None:
            for mcr in self.stored_mcrs:
                mcr.langevin_steps = langevin_steps

        return None

    @property
    def optimizer_class(self):
        return self._optimizer_class

    @optimizer_class.setter
    def optimizer_class(self, optimizer_class):
        assert inspect.isclass(optimizer_class)
        assert issubclass(optimizer_class, torch.optim.Optimizer)
        self._optimizer_class = optimizer_class
        return None

    @property
    def optimizer_args(self):
        return self._optimizer_args

    @optimizer_args.setter
    def optimizer_args(self, optimizer_args):
        assert isinstance(optimizer_args, dict)
        self._optimizer_args = optimizer_args
        return None

    @property
    def checkpoint_criterion(self):
        return self._checkpoint_criterion

    @checkpoint_criterion.setter
    def checkpoint_criterion(self, checkpoint_criterion):
        if checkpoint_criterion is not None:
            assert isinstance(checkpoint_criterion, str)
            assert checkpoint_criterion in ['nMSE', 'R2']

        self._checkpoint_criterion = checkpoint_criterion
        return None

    @property
    def checkpoint_threshold(self):
        return self._checkpoint_threshold

    @checkpoint_threshold.setter
    def checkpoint_threshold(self, checkpoint_threshold):
        assert isinstance(checkpoint_threshold, float)
        assert checkpoint_threshold >= 0. and checkpoint_threshold <= 1.

        self._checkpoint_threshold = checkpoint_threshold
        return None

    @property
    def checkpoint_interval(self):
        return self._checkpoint_interval

    @checkpoint_interval.setter
    def checkpoint_interval(self, checkpoint_interval):
        assert isinstance(checkpoint_interval, float)
        assert checkpoint_interval < 1. and checkpoint_interval > 0.
        self._checkpoint_interval = checkpoint_interval
        return None

    @property
    def mini_batch_size(self):
        return self._mini_batch_size

    @mini_batch_size.setter
    def mini_batch_size(self, mini_batch_size):
        assert isinstance(mini_batch_size, int)
        assert mini_batch_size > 0
        self._mini_batch_size = mini_batch_size
        return None

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        assert isinstance(device, (str, torch.device))
        if isinstance(device, str):
            device = torch.device(device)

        self._device = device
        self.stored_mcrs = self.stored_mcrs.to(device)
        return None

    @property
    def max_epoch(self):
        return self._max_epoch

    @max_epoch.setter
    def max_epoch(self, max_epoch):
        assert isinstance(max_epoch, int)
        self._max_epoch = max_epoch
        return None

    @property
    def tol_pattern_nmse(self):
        return self._tol_pattern_nmse

    @tol_pattern_nmse.setter
    def tol_pattern_nmse(self, tol_pattern_nmse):
        assert isinstance(tol_pattern_nmse, float)
        assert tol_pattern_nmse >= 0. and tol_pattern_nmse <= 1.
        self._tol_pattern_nmse = tol_pattern_nmse
        return None

    @property
    def tol_energy_ratio(self):
        return self._tol_energy_ratio

    @tol_energy_ratio.setter
    def tol_energy_ratio(self, tol_energy_ratio):
        assert isinstance(tol_energy_ratio, (int, float))
        assert tol_energy_ratio >= 0. and tol_energy_ratio <= 1.
        self._tol_energy_ratio = tol_energy_ratio
        return None

    @property
    def select_prob_threshld(self):
        return self._select_prob_threshld

    @select_prob_threshld.setter
    def select_prob_threshld(self, select_prob_threshld):
        assert isinstance(select_prob_threshld, (int, float))
        assert select_prob_threshld > 0. and select_prob_threshld <= 1.
        self._select_prob_threshld = float(select_prob_threshld)
        return None

    @property
    def min_component_coef(self):
        return self._min_component_coef

    @min_component_coef.setter
    def min_component_coef(self, min_component_coef):
        assert isinstance(min_component_coef, (int, float))
        assert min_component_coef >= 0.
        self._min_component_coef = float(min_component_coef)
        return None

    @property
    def ambiguity_coef(self):
        return self._ambiguity_coef

    @ambiguity_coef.setter
    def ambiguity_coef(self, ambiguity_coef):
        assert isinstance(ambiguity_coef, (int, float))
        assert ambiguity_coef >= 0.
        self._ambiguity_coef = float(ambiguity_coef)
        return None

    @property
    def energy_coef(self):
        return self._energy_coef

    @energy_coef.setter
    def energy_coef(self, energy_coef):
        assert isinstance(energy_coef, (int, float))
        assert energy_coef >= 0.
        self._energy_coef = energy_coef
        return None

    @property
    def ambiguity_penalty(self):
        return self._ambiguity_penalty

    @ambiguity_penalty.setter
    def ambiguity_penalty(self, ambiguity_penalty):
        assert isinstance(ambiguity_penalty, str)
        assert ambiguity_penalty.lower() in ['linear', 'poly', 'laplacian', 'rbf', 'matern']

        self._ambiguity_penalty = ambiguity_penalty.lower()
        self.ambiguity_penalty_func = None
        if ambiguity_penalty.lower() == 'linear':
            self.ambiguity_penalty_func = linear_penalty 
        elif ambiguity_penalty.lower() == 'poly':
            self.ambiguity_penalty_func = polynomial_penalty
        elif ambiguity_penalty.lower() == 'laplacian':
            self.ambiguity_penalty_func = laplacian_penalty
        elif ambiguity_penalty.lower() == 'rbf':
            self.ambiguity_penalty_func = rbf_penalty
        elif ambiguity_penalty.lower() == 'matern_penalty':
            self.ambiguity_penalty_func = matern_penalty 

        return None

    @property
    def ambiguity_prob_threshold(self):
        return self._ambiguity_prob_threshold

    @ambiguity_prob_threshold.setter
    def ambiguity_prob_threshold(self, ambiguity_prob_threshold):
        assert isinstance(ambiguity_prob_threshold, (int, float))
        assert ambiguity_prob_threshold > 0.
        assert ambiguity_prob_threshold <= 1.
        self._ambiguity_prob_threshold = float(ambiguity_prob_threshold)
        return None

    @property
    def poly_dim(self):
        return self._poly_dim

    @poly_dim.setter
    def poly_dim(self, poly_dim):
        assert isinstance(poly_dim, int)
        assert poly_dim >= 1
        self._poly_dim = poly_dim
        return None

    @property
    def matern_nu(self):
        return self._matern_nu

    @matern_nu.setter
    def matern_nu(self, matern_nu):
        assert isinstance(matern_nu, float)
        self._matern_nu = matern_nu
        return None

    @property
    def temperature_max(self):
        return self._temperature_max

    @temperature_max.setter
    def temperature_max(self, temperature_max):
        assert isinstance(temperature_max, (int, float))
        assert temperature_max > 0.
        if self.temperature_min is not None:
            assert temperature_max >= self.temperature_min

        self._temperature_max = temperature_max
        return None

    @property
    def temperature_min(self):
        return self._temperature_min

    @temperature_min.setter
    def temperature_min(self, temperature_min):
        assert isinstance(temperature_min, (int, float))
        assert temperature_min > 0.
        if self.temperature_max is not None:
            assert temperature_min <= self.temperature_max

        self._temperature_min = temperature_min
        return None

    @property
    def temperature_drop_value(self):
        return self._temperature_drop_value

    @temperature_drop_value.setter
    def temperature_drop_value(self, temperature_drop_value):
        assert isinstance(temperature_drop_value, (int, float))
        assert temperature_drop_value > 0.
        assert temperature_drop_value <= (self.temperature_max - self.temperature_min)
        temperature_drop_value = float(temperature_drop_value)
        self._temperature_drop_value = temperature_drop_value
        return None

    @property
    def temperature_drop_epochs(self):
        return self._temperature_drop_epochs

    @temperature_drop_epochs.setter
    def temperature_drop_epochs(self, temperature_drop_epochs):
        assert isinstance(temperature_drop_epochs, int)
        self._temperature_drop_epochs = temperature_drop_epochs
        return None

    @property
    def temperature_eval(self):
        return self._temperature_eval

    @temperature_eval.setter
    def temperature_eval(self, temperature_eval):
        assert isinstance(temperature_eval, (int, float))
        assert temperature_eval > 0.
        temperature_eval = float(temperature_eval)
        self._temperature_eval = temperature_eval
        if self.stored_mcrs is not None:
            for mcr in self.stored_mcrs:
                mcr.temperature_eval = temperature_eval

        return None

    def __repr__(self):
        return '{0}(eval_temperature={1})'.format(self.__class__.__name__, self.temperature_eval)

    def summary(self, layout_func = print):
        content = '{0}(\nMCR Modules: \n'.format(self.__class__.__name__)
        content += (str(self.stored_mcrs) + '\n')
        content += 'Fitting Args:\n'
        args = {'mini_batch_size': self.mini_batch_size,
                'checkpoint_criterion': self.checkpoint_criterion,
                'checkpoint_threshold': self.checkpoint_threshold,
                'checkpoint_interval': self.checkpoint_interval,
                'max_epoch': self.max_epoch,
                'tol_pattern_nmse': self.tol_pattern_nmse,
                'tol_energy_ratio': self.tol_energy_ratio,
                'select_prob_threshld': self.select_prob_threshld,
                'min_component_coef': self.min_component_coef,
                'ambiguity_coef': self.ambiguity_coef,
                'energy_coef': self.energy_coef,
                'ambiguity_penalty': self.ambiguity_penalty,
                'ambiguity_prob_threshold': self.ambiguity_prob_threshold,
                'temperature_start': self.temperature_max,
                'temperature_end': self.temperature_min,
                'temperature_drop_value': self.temperature_drop_value,
                'temperature_drop_epochs': self.temperature_drop_epochs,
                'temperature_eval': self.temperature_eval}

        args_content = str(args).replace(',', ',\n ')
        content += (args_content + '\n')
        layout_func(content)
        return None

    def _init_mcrs(self, num_components, dim_components, components = None, learnable = None):
        self.stored_mcrs = nn.ModuleList()
        if components is not None:
            assert isinstance(components, list)
            num_components = []
            for matrix in components:
                assert matrix.shape[-1] == dim_components
                num_components.append(matrix.shape[0])

            self.num_components = num_components
            if learnable is not None:
                assert isinstance(learnable, list)
                assert len(learnable) == len(components)
            else:
                learnable = [True for _ in range(len(components))]
        else:
            self.num_components = num_components

        self.stored_mcrs = nn.ModuleList()
        for n_component in self.num_components:
            single_mcr = EBSelectiveMCR(num_components = n_component,
                                        dim_components = dim_components,
                                        sparse_component = self.sparse_component,
                                        non_negative_component = self.non_negative_component,
                                        hidden_size_extension = self.hidden_size_extension,
                                        hidden_layers = self.hidden_layers,
                                        component_norm = self.component_norm,
                                        learnable_component = True,
                                        component_post_processing = self.component_post_processing,
                                        aggregate_function = self.aggregate_function,
                                        select_prob_threshld = self.select_prob_threshld,
                                        temperature_train = self.temperature_max,
                                        temperature_eval = self.temperature_eval,
                                        langevin_step_size = self.langevin_step_size,
                                        langevin_noise = self.langevin_noise,
                                        langevin_steps = self.langevin_steps)

            self.stored_mcrs.append(single_mcr)

        if components is not None:
            for idx in range(len(components)):
                component_matrix = components[idx]
                self.stored_mcrs[idx].load_components(component_matrix, learnable[idx])

        return None
 
    def fit(self, data, 
            eval_data = None, 
            show_args = False, 
            show_train_info = False, 
            show_eval_info = False,
            train_info_display_interval = 1,
            eval_info_display_interval = 100,
            log_dir = None,
            log_info_interval = 10,
            full_result_filename = None,
            final_result_filename = None,
            force_write_file = False):

        assert isinstance(data, torch.Tensor)
        assert data.dim() >= 2
        if eval_data is not None:
            assert isinstance(eval_data, torch.Tensor)
            assert eval_data.dim() >= 2
            assert data.shape[1: ] == eval_data.shape[1: ]

        assert isinstance(show_args, bool)
        assert isinstance(show_train_info, bool)
        assert isinstance(show_eval_info, bool)

        assert isinstance(train_info_display_interval, int)
        assert train_info_display_interval >= 1
        assert isinstance(eval_info_display_interval, int)
        assert eval_info_display_interval >=1

        writer = None
        if log_dir is not None:
            assert isinstance(log_dir, str)
            init_directory(log_dir)
            writer = SummaryWriter(log_dir = log_dir)

        assert isinstance(log_info_interval, int)
        assert log_info_interval > 0
        assert isinstance(force_write_file, bool)

        if full_result_filename is not None:
            assert isinstance(full_result_filename, str)
            if os.path.isfile(full_result_filename):
                if not force_write_file:
                    raise RuntimeError('Old result file ({0}) detected.'.format(full_result_filename) + \
                            ' Please set force_write_file=True to overwrite the result file.')

        if final_result_filename is not None:
            assert isinstance(final_result_filename, str)
            if os.path.isfile(final_result_filename):
                if not force_write_file:
                    raise RuntimeError('Old result file ({0}) detected.'.format(final_result_filename) + \
                            ' Please set force_write_file=True to overwrite the result file.')

        self.__finish_analysis = False

        dataset = TensorDataset(data)
        if eval_data is not None:
            eval_dataset = TensorDataset(eval_data)
        else:
            eval_dataset = None

        result_file_recording = None
        if (full_result_filename is not None) or (final_result_filename is not None):
            result_file_recording = {'epoch': [],
                                     'nMSE': [],
                                     'R2': [],
                                     'mean_energy': [],
                                     'effective_components': []}

        if len(self.stored_mcrs) == 0:
            dim_components = data.shape[-1]
            self._init_mcrs(self.num_components, dim_components)

        self.stored_mcrs = self.stored_mcrs.to(self.device)
        if show_args:
            self.summary()

        checkpoint_mcrs, checkpoint_records = None, None
        if self.checkpoint_criterion is not None:
            if log_dir is None:
                raise ValueError('Checkpoint saving need to assign the argument:log_dir in the fit function.')

            checkpoint_mcrs = {region: None for region in checkpoint_regions(self.checkpoint_criterion, 
                                                                             self.checkpoint_threshold, 
                                                                             self.checkpoint_interval)}

            checkpoint_records = {region: None for region in checkpoint_regions(self.checkpoint_criterion, 
                                                                                self.checkpoint_threshold,
                                                                                self.checkpoint_interval)}

        train_loader = DataLoader(dataset, batch_size = self.mini_batch_size, shuffle = True)
        per_epoch_step = len(train_loader)
        if eval_data is not None:
            eval_loader = DataLoader(eval_dataset, batch_size = self.mini_batch_size, shuffle = False)
        else:
            eval_loader = DataLoader(dataset, batch_size = self.mini_batch_size, shuffle = False)

        criterion = nn.MSELoss()
        optimizer = self.optimizer_class(filter(lambda p: p.requires_grad, self.stored_mcrs.parameters()), 
                                         **self.optimizer_args)

        component_number_history, init_components = [], []
        for mcr in self.stored_mcrs:
            init_components.append(mcr.num_components)

        component_number_history.append(init_components)

        energy_l1_coef = 0.
        energy_coef_sched = EnergyL1Scheduler(sum(init_components),
                                              target_ratio = self.min_component_coef)

        converge = False
        reach_converge_region, leave_converge_region, sliding_window_nmse = False, False, deque(maxlen = 100)
        init_energies, init_energy = [], None
        iter_now, epoch_now, temp_now = 0, 0, self.temperature_max
        while not converge:
            for mcr in self.stored_mcrs:
                mcr.temperature_train = temp_now

            self.stored_mcrs = self.stored_mcrs.train()
            for inner_batch_iter, target_data in enumerate(train_loader):
                target_data = target_data[0].to(self.device)
                mcr_outputs = [mcr(target_data) for mcr in self.stored_mcrs]
                parsed_data = torch.stack([output['synthesized_pattern'] for output in mcr_outputs]).sum(0)

                optimizer.zero_grad()
                
                reconstructed_loss = criterion(parsed_data, target_data)

                used_components, component_counts = [], []
                sparsity_energies, select_energies = [], []
                for mcr_idx in range(len(mcr_outputs)):
                    effective_indices = mcr_outputs[mcr_idx]['effective_indices']
                    component_counts.append(effective_indices)

                    used_indices = (effective_indices > self.ambiguity_prob_threshold).any(dim = 0)
                    used_indices = torch.nonzero(used_indices, as_tuple = True)[0]
                    if used_indices.numel() > 0:
                        used_components.append(self.stored_mcrs[mcr_idx].components(used_indices))

                    if mcr_outputs[mcr_idx]['sparsity_energy'].dim() > 0:
                        sparsity_energies.append(mcr_outputs[mcr_idx]['sparsity_energy'])

                    select_energies.append(mcr_outputs[mcr_idx]['select_energy'])

                component_counts = torch.cat(component_counts, dim = 0).sum(-1)
                norm_count_reg = (component_counts / sum(init_components)).mean()
                if len(sparsity_energies) > 0:
                    sparsity_energies = torch.cat(sparsity_energies, dim = 0)
                else:
                    sparsity_energies = None
                
                total_mean_energy, total_energy_values = torch.tensor(0., device = self.device), 0
                energy_reg = torch.tensor(0., device = self.device)
                if sparsity_energies is not None:
                    total_energy_values += sparsity_energies.numel()
                    total_mean_energy = total_mean_energy + sparsity_energies.sum()

                select_energies = torch.cat(select_energies, dim = 0)
                energy_reg = energy_reg + (select_energies ** 2).mean()

                total_energy_values += select_energies.numel()
                total_mean_energy = total_mean_energy + select_energies.sum()
                total_mean_energy = total_mean_energy / total_energy_values

                ambiguity_penalty_args = {}
                if self.ambiguity_penalty == 'poly':
                    ambiguity_penalty_args['d'] = self.poly_dim 
                elif self.ambiguity_penalty == 'matern':
                    ambiguity_penalty_args['nu'] = self.matern_nu

                ambiguity_reg = torch.tensor(0., device = self.device)
                if len(used_components) > 0:
                    used_components = torch.cat(used_components, dim = 0)
                    ambiguity_reg = self.ambiguity_penalty_func(used_components, **ambiguity_penalty_args)

                loss_value = reconstructed_loss +\
                        energy_l1_coef * norm_count_reg +\
                        self.energy_coef * energy_reg +\
                        self.ambiguity_coef * ambiguity_reg 

                loss_value.backward(retain_graph = True)
                optimizer.step()

                iter_now += 1

                transformed, display_training_details, display_sep_line = False, False, False
                if writer is not None and (iter_now % log_info_interval == 0):
                    reconstructed_loss = reconstructed_loss.item()
                    ambiguity_reg = ambiguity_reg.item()
                    transformed = True

                    writer.add_scalar('train/reconstruct_error', reconstructed_loss, iter_now)
                    writer.add_scalar('train/component_similarity', ambiguity_reg, iter_now)

                if show_train_info:
                    if not transformed:
                        reconstructed_loss = reconstructed_loss.item()
                        ambiguity_reg = ambiguity_reg.item()
                        total_mean_energy = total_mean_energy.item()
                        transformed = True

                    if inner_batch_iter == (per_epoch_step - 1):
                        display_training_details = True
                        display_sep_line = True
                    elif (inner_batch_iter + 1) % train_info_display_interval == 0:
                        display_training_details = True

                    if display_training_details:
                        print("Epoch: {0} [iter: {1}/{2}] | Reconstruct Err. (↓): "\
                                .format(epoch_now + 1, inner_batch_iter + 1, per_epoch_step) +\
                                "{0:.6f} | Components' Similarity (↓): {1:.6f} | Energy (↓): {2:.6f}"\
                                .format(reconstructed_loss, ambiguity_reg, total_mean_energy))

                    if display_sep_line:
                        print('')

            total_eval_step = 0
            total_mse = torch.tensor(0., device = self.device)
            metric_mode = 'dense'
            if self.sparse_component:
                metric_mode = 'threshold'
                                
            metrics = StreamingMetrics(dim = self.dim_components,
                                       mode = metric_mode)

            total_used_components = []
            total_mean_energy, total_energy_values = torch.tensor(0., device = self.device), 0
            for _, target_data in enumerate(eval_loader):
                target_data = target_data[0].to(self.device)
                single_batch_size = target_data.shape[0]

                self.stored_mcrs = self.stored_mcrs.eval()
                train_outputs = [mcr(target_data) for mcr in self.stored_mcrs]
                train_outputs = [{key: tensor.detach() for key, tensor in output.items()} for output in train_outputs]
                self.stored_mcrs = self.stored_mcrs.eval()
                eval_outputs = [mcr(target_data) for mcr in self.stored_mcrs]
                eval_outputs = [{key: tensor.detach() for key, tensor in output.items()} for output in eval_outputs]

                parsed_data = torch.stack([output['synthesized_pattern'] for output in eval_outputs]).sum(0)

                mse = criterion(target_data, parsed_data)
                metrics.update(target_data, parsed_data)

                used_components, effective_components = [], []
                sparsity_energies, select_energies = [], []
                for mcr_idx in range(len(train_outputs)):
                    single_module_used_components = (eval_outputs[mcr_idx]['effective_indices'] >\
                                                     self.select_prob_threshld).any(dim = 0)

                    effective_component = self.stored_mcrs[mcr_idx].vanilla_components[single_module_used_components, :]

                    used_components.append(single_module_used_components)
                    effective_components.append(effective_component)
                    if mcr_outputs[mcr_idx]['sparsity_energy'].dim() > 0:
                        sparsity_energies.append(mcr_outputs[mcr_idx]['sparsity_energy'])

                    select_energies.append(mcr_outputs[mcr_idx]['select_energy'])

                if len(sparsity_energies) > 0:
                    sparsity_energies = torch.cat(sparsity_energies, dim = 0)
                else:
                    sparsity_energies = None

                if sparsity_energies is not None:
                    total_energy_values += sparsity_energies.numel()
                    total_mean_energy = total_mean_energy + sparsity_energies.sum()

                select_energies = torch.cat(select_energies, dim = 0)
                total_energy_values += select_energies.numel()
                total_mean_energy = total_mean_energy + select_energies.sum()

                used_components = torch.cat(used_components, dim = 0)

                total_mse += mse 
                total_used_components.append(used_components)

                total_eval_step += 1

            if total_mse > 0.:
                total_mse /= total_eval_step

            total_mse = total_mse.cpu().item()
            total_used_components = torch.stack(total_used_components).any(dim = 0).sum().cpu().item()
            total_mean_energy = (total_mean_energy / total_energy_values).cpu().item()
            nmse, r2 = metrics.compute()

            if reach_converge_region:
                sliding_window_nmse.append(nmse)
            else:
                init_energies.append(total_mean_energy)

            if nmse < self.tol_pattern_nmse:
                reach_converge_region = True
                init_energy = sum(init_energies) / len(init_energies)

            if reach_converge_region:
                if len(sliding_window_nmse) == sliding_window_nmse.maxlen:
                    sl_nmse = sum(sliding_window_nmse) / sliding_window_nmse.maxlen
                    if sl_nmse > self.tol_pattern_nmse:
                        leave_converge_region = True

            energy_check = False
            if init_energy is not None:
                if total_mean_energy <= (init_energy * self.tol_energy_ratio):
                    energy_check = True

            if reach_converge_region and leave_converge_region and energy_check:
                print('Epoch: {0} | nMSE: {1:.4f} | E: {2:.4f}'\
                        .format(epoch_now + 1, nmse, total_mean_energy) + \
                        ' reach terminate condition (leave converge region).')
                converge = True

            energy_l1_coef = energy_coef_sched.step(nmse, total_mse, total_mean_energy)

            replace_checkpoint, specified_region = False, None
            if self.checkpoint_criterion is not None:
                if self.checkpoint_criterion == 'nMSE':
                    criterion_value = nmse
                elif self.checkpoint_criterion == 'R2':
                    criterion_value = r2

                specified_region = specify_checkpoint_region(criterion_value,
                                                             self.checkpoint_criterion,
                                                             list(checkpoint_records.keys()))

                if specified_region is not None:
                    if checkpoint_mcrs[specified_region] is None:
                        replace_checkpoint = True
                    else:
                        if total_used_components < checkpoint_records[specified_region]['effective_components']:
                            replace_checkpoint = True
                        elif total_used_components == checkpoint_records[specified_region]['effective_components']:
                            if self.checkpoint_criterion == 'nMSE':
                                if nmse <= checkpoint_records[specified_region]['nMSE']:
                                    replace_checkpoint = True

                            elif self.checkpoint_criterion == 'R2':
                                if r2 >= checkpoint_records[specified_region]['R2']:
                                    replace_checkpoint = True

            if replace_checkpoint:
                checkpoint_mcrs[specified_region] = copy.deepcopy(self.stored_mcrs.cpu())
                self.stored_mcrs = self.stored_mcrs.to(self.device)
                checkpoint_records[specified_region] = {'nMSE': nmse,
                                                        'R2': r2,
                                                        'mean_energy': total_mean_energy,
                                                        'effective_components': total_used_components}

            epoch_now += 1
            if writer is not None:
                writer.add_scalar('eval/nMSE', nmse, iter_now)
                writer.add_scalar('eval/R2', r2, iter_now)
                writer.add_scalar('eval/mean_energy', total_mean_energy, iter_now)
                writer.add_scalar('eval/effective_components', total_used_components, iter_now)

            if result_file_recording is not None:
                result_file_recording['epoch'].append(epoch_now)
                result_file_recording['nMSE'].append(nmse)
                result_file_recording['R2'].append(r2)
                result_file_recording['mean_energy'].append(total_mean_energy)
                result_file_recording['effective_components'].append(total_used_components)

            display_eval_details = False
            if show_eval_info:
                if epoch_now == self.max_epoch:
                    display_eval_details = True
                elif epoch_now % eval_info_display_interval == 0:
                    display_eval_details = True

                if display_eval_details:
                    print('Epoch: {0} | nMSE (pattern): {1:.4f} | R2: {2:.4f}'\
                            .format(epoch_now, nmse, r2) + \
                            ' | E: {0:.4f} | Used components: {1}'\
                            .format(total_mean_energy, total_used_components))

                if display_sep_line:
                    print('')

            del train_outputs
            del eval_outputs
            del parsed_data

            if self.temperature_drop_epochs > 0:
                if epoch_now % self.temperature_drop_epochs == 0:
                    temp_now -= self.temperature_drop_value
                    if temp_now < self.temperature_min:
                        temp_now = self.temperature_min

            del effective_components
            if epoch_now == self.max_epoch:
                print('Reach max epoch: {0}, stop fitting.'.format(self.max_epoch))
                break

        if checkpoint_mcrs is not None and checkpoint_records is not None:
            final_mcrs = copy.deepcopy(self.stored_mcrs.cpu())
            for criterion_region in checkpoint_mcrs:
                if checkpoint_mcrs[criterion_region] is None:
                    continue

                checkpoint_dir = os.path.join(log_dir, criterion_region)
                self.stored_mcrs = checkpoint_mcrs[criterion_region]
                self.save_mcrs(checkpoint_dir)
                print('Successfully save recorded checkpoint (region={0}, effective_components={1})'\
                        .format(criterion_region, checkpoint_records[criterion_region]['effective_components']))

                lines = ['nMSE,R2,mean_energy,effective_components\n',
                         '{0},{1},{2},{3}\n'.format(checkpoint_records[criterion_region]['nMSE'],
                                                    checkpoint_records[criterion_region]['R2'],
                                                    checkpoint_records[criterion_region]['mean_energy'],
                                                    checkpoint_records[criterion_region]['effective_components'])]

                checkpoint_result_filename = os.path.join(checkpoint_dir, 'result.csv')
                with open(checkpoint_result_filename, 'w') as F:
                    F.writelines(lines)
                    F.close()

                print('Successfully checkpoint result to file:{0}.'\
                        .format(checkpoint_result_filename))

            self.stored_mcrs = final_mcrs.to(self.device)

        if writer is not None:
            writer.close()
            writer = None

        if full_result_filename is not None:
            if not full_result_filename.endswith('.csv'):
                full_result_filename += '.csv'

            lines = ['epoch,nMSE,R2,mean_energy,effective_components\n']
            for idx in range(len(result_file_recording['epoch'])):
                single_line = '{0},{1},{2},{3},{4}\n'.format(result_file_recording['epoch'][idx],
                                                             result_file_recording['nMSE'][idx],
                                                             result_file_recording['R2'][idx],
                                                             result_file_recording['mean_energy'][idx],
                                                             result_file_recording['effective_components'][idx])

                lines.append(single_line)

            file_dir = os.path.split(full_result_filename)[0]
            init_directory(file_dir)
            with open(full_result_filename, 'w') as F:
                F.writelines(lines)
                F.close()

            print('Save full result to file: {0}'.format(full_result_filename))

        if final_result_filename is not None:
            if not final_result_filename.endswith('.csv'):
                final_result_filename += '.csv'

            if len(result_file_recording['epoch']) > 0:
                lines = ['nMSE,R2,mean_energy,effective_components\n',
                         '{0},{1},{2},{3}\n'.format(result_file_recording['nMSE'][-1],
                                                    result_file_recording['R2'][-1],
                                                    result_file_recording['mean_energy'][-1],
                                                    result_file_recording['effective_components'][-1])]

                with open(final_result_filename, 'w') as F:
                    F.writelines(lines)
                    F.close()

                print('Save final result to file: {0}'.format(final_result_filename))

        self.__finish_analysis = True

        return None

    def insert_mcr(self, module):
        assert isinstance(module, EBSelectiveMCR)
        if self.__finish_analysis:
            raise RuntimeError('The {0}.insert_mcr function'.format(self.__class__.__name__) + \
                    ' is used to insert user-defined MCR module before fitting data.')

        module = module.to(self.device)
        self.stored_mcrs.append(module)

        return None

    @abc.abstractmethod
    def parse_pattern(self, data):
        raise NotImplementedError()

    @abc.abstractmethod
    def save_mcrs(self, path):
        raise NotImplementedError()

    @abc.abstractmethod
    def load_mcrs(self, path):
        raise NotImplementedError()


class DenseEBgMCR(BaseEBgMCR):
    def __init__(self,
            num_components,
            dim_components = None,
            sparse_component = False,
            non_negative_component = True,
            component_norm = 2,
            hidden_size_extension = 2,
            hidden_layers = 2,
            optimizer_class = optim.AdamW,
            optimizer_args = {'lr': 5e-4, 'betas': (0.9, 0.995), weight_decay: 1e-3},
            checkpoint_criterion = 'R2',
            checkpoint_threshold = 0.97,
            checkpoint_interval = 0.005,
            device = torch.device('cpu'),
            mini_batch_size = 64,
            max_epoch = -1,
            tol_pattern_nmse = 0.01,
            tol_energy_ratio = 0.5,
            select_prob_threshld = 0.9999994,
            min_component_coef = 1.,
            ambiguity_coef = 1e-10,
            energy_coef = 1e-10,
            ambiguity_penalty = 'rbf',
            ambiguity_prob_threshold = 0.5,
            poly_dim = 2,
            matern_nu = 1.5,
            temperature_max = 1.,
            temperature_min = 0.4,
            temperature_drop_value = 0.01,
            temperature_drop_epochs = 100,
            temperature_eval = 0.01,
            langevin_step_size = 0.05,
            langevin_noise = 0.1,
            langevin_steps = 32):

        super(DenseEBgMCR, self).__init__(
                num_components,
                dim_components = dim_components,
                sparse_component = sparse_component,
                non_negative_component = non_negative_component,
                component_norm = component_norm,
                hidden_size_extension = hidden_size_extension,
                hidden_layers = hidden_layers,
                optimizer_class = optimizer_class,
                optimizer_args = optimizer_args,
                checkpoint_criterion = checkpoint_criterion,
                checkpoint_threshold = checkpoint_threshold,
                checkpoint_interval = checkpoint_interval,
                device = device,
                mini_batch_size = mini_batch_size,
                max_epoch = max_epoch,
                tol_pattern_nmse = tol_pattern_nmse,
                tol_energy_ratio = tol_energy_ratio,
                select_prob_threshld = select_prob_threshld,
                min_component_coef = min_component_coef,
                ambiguity_coef = ambiguity_coef,
                energy_coef = energy_coef,
                ambiguity_penalty = ambiguity_penalty,
                ambiguity_prob_threshold = ambiguity_prob_threshold,
                poly_dim = poly_dim,
                matern_nu = matern_nu,
                temperature_max = temperature_max,
                temperature_min = temperature_min,
                temperature_drop_value = temperature_drop_value,
                temperature_drop_epochs = temperature_drop_epochs,
                temperature_eval = temperature_eval,
                langevin_step_size = langevin_step_size,
                langevin_noise = langevin_noise,
                langevin_steps = langevin_steps)

        self.component_post_processing = nn.Identity() 
        self.aggregate_function = default_aggreate_function

    def parse_pattern(self, data, select_threshold = None):
        assert isinstance(data, torch.Tensor)
        if select_threshold is None:
            select_threshold = self.select_prob_threshld
        else:
            assert isinstance(select_threshold, float)
            assert select_threshold > 0. and select_threshold <= 1.

        if len(self.stored_mcrs) == 0:
            raise RuntimeError('No MCR module in {0}, use {0}.fit to acquire the module.'\
                    .format(self.__class__.__name__))

        self.stored_mcrs = self.stored_mcrs.eval()
        data = data.to(self.device)

        mcr_outputs = [mcr(data, select_prob_threshld = select_threshold) for mcr in self.stored_mcrs]
        mcr_idx = 0
        reconstruction, concentration, components, select_prob = [], [], [], []
        for output in mcr_outputs:
            single_reconstruction = output['synthesized_pattern']
            single_concentration = output['concentration']
            single_components = output['components']
            single_select_prob = output['effective_indices']
            single_reconstruction = self.stored_mcrs[mcr_idx].aggregate_function(single_concentration, 
                                                                                 single_components)

            reconstruction.append(single_reconstruction)
            concentration.append(single_concentration)
            components.append(single_components)
            select_prob.append(single_select_prob)

            mcr_idx += 1

        reconstruction = torch.sum(torch.stack(reconstruction), dim = 0)
        concentration = torch.cat(concentration, dim = 1)
        components = torch.cat(components, dim = 1)
        select_prob = torch.cat(select_prob, dim = 1)

        return {'reconstruction': reconstruction.detach(),
                'concentration': concentration.detach(),
                'components': components.detach(),
                'select_prob': select_prob.detach()}

    def save_mcrs(self, path):
        assert isinstance(path, str)
        assert len(path) > 0

        if len(self.stored_mcrs) == 0:
            raise RuntimeError('No MCR module in {0}, use {0}.fit to acquire the module.'\
                    .format(self.__class__.__name__))

        elif len(self.stored_mcrs) == 1:
            self.stored_mcrs[0].save_pretrained(path)
        else:
            init_directory(path)
            for idx in range(len(self.stored_mcrs)):
                single_path = os.path.join(path, str(idx))
                init_directory(single_path)
                self.stored_mcrs[idx].save_pretrained(single_path)

        return None

    def load_mcrs(self, path):
        assert isinstance(path, str)
        assert len(path) > 0

        has_model_file, has_param_file = False, False
        files = os.listdir(path)
        for f in files:
            if f.endswith('.safetensors'):
                has_model_file = True

            if f.endswith('.json'):
                has_param_file = True

        if has_model_file and has_param_file:
            mcr = EBSelectiveMCR(1, 1).from_pretrained(path)
            self.stored_mcrs = nn.ModuleList([mcr])
        else:
            mcrs = []
            for sub_path in files:
                sub_path = os.path.join(path, sub_path)
                if os.path.isdir(sub_path):
                    mcr = EBSelectiveMCR(1, 1).from_pretrained(sub_path)
                    mcrs.append(mcr)

            self.stored_mcrs = nn.ModuleList(mcrs)

        self.stored_mcrs = self.stored_mcrs.to(self.device)

        return None


