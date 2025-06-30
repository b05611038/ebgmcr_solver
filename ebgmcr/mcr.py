import os

import torch
import torch.nn as nn

from .utils import (init_directory,
                    save_as_json, 
                    load_json,
                    save_as_safetensors,
                    load_safetensors,
                    init_layer_weight)

from .core import (FixedEBSelectiveModule,
                   DynamicEBSelectiveModule)

#--------------------------------------------------------------------------------
# mcr.py contains the implementation of the energy-based selective MCR.
#--------------------------------------------------------------------------------


__all__ = ['EBSelectiveMCR']


def default_aggreate_function(concentration, components):
    return (concentration * components).sum(1)

class EBSelectiveMCR(nn.Module):
    def __init__(self, 
            num_components,
            dim_components,
            sparse_component = True,
            non_negative_component = True,
            learnable_component = True,
            component_norm = None,
            hidden_size_extension = None,
            hidden_layers = 1,
            component_post_processing = nn.Identity(),
            aggregate_function = default_aggreate_function,
            temperature_train = 1.,
            temperature_eval = 0.001,
            select_prob_threshld = 0.99994,
            langevin_step_size = 0.05,
            langevin_noise = 0.1,
            langevin_steps = 32):

        super(EBSelectiveMCR, self).__init__()

        self.num_components = num_components
        self.dim_components = dim_components
        self.sparse_component = sparse_component
        self.non_negative_component = non_negative_component
        self.hidden_size_extension = hidden_size_extension
        self.hidden_layers = hidden_layers

        self.temperature_train = temperature_train
        self.temperature_eval = temperature_eval
        self.select_prob_threshld = select_prob_threshld

        init_component = torch.randn(num_components, dim_components)
        init_component.normal_(mean = 0., std = 0.1)
        if non_negative_component:
            init_component = torch.abs(init_component).detach().clone()

        self.vanilla_components = nn.Parameter(init_component)
        self.learnable_component = learnable_component
        self.component_sparsity = FixedEBSelectiveModule(num_components * dim_components,
                                                         (num_components, dim_components),
                                                         tau = self.tau)

        self.component_parser = DynamicEBSelectiveModule(dim_components,
                                                         num_components,
                                                         energy_hidden_extension = hidden_size_extension,
                                                         energy_hidden_layers = hidden_layers,
                                                         tau = self.tau)
 
        if self.hidden_size_extension is None:
            self.concentration_regressor = init_layer_weight(nn.Linear(dim_components, num_components))
        else:
            hidden_size = num_components * self.hidden_size_extension
            layers = [init_layer_weight(nn.Linear(dim_components, hidden_size)), nn.ReLU()]
            for _ in range(self.hidden_layers - 1):
                layers.append(init_layer_weight(nn.Linear(hidden_size, hidden_size)))
                layers.append(nn.ReLU())

            layers.append(init_layer_weight(nn.Linear(hidden_size, num_components)))
            self.concentration_regressor = nn.Sequential(*layers)

        self.component_norm = component_norm
        assert callable(component_post_processing)
        self.component_post_processing = component_post_processing
        assert callable(aggregate_function)
        self.aggregate_function = aggregate_function

        self.langevin_step_size = langevin_step_size
        self.langevin_noise = langevin_noise
        self.langevin_steps = langevin_steps

    @property
    def num_components(self):
        return self._num_components

    @num_components.setter
    def num_components(self, num_components):
        assert isinstance(num_components, int)
        assert num_components > 0
        self._num_components = num_components
        return None

    @property
    def dim_components(self):
        return self._dim_components

    @dim_components.setter
    def dim_components(self, dim_components):
        assert isinstance(dim_components, int)
        assert dim_components > 0
        self._dim_components = dim_components
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
    def learnable_component(self):
        return self._learnable_component

    @learnable_component.setter
    def learnable_component(self, learnable_component):
        assert isinstance(learnable_component, bool)
        self._learnable_component = learnable_component
        self.load_components(self.vanilla_components, learnable = learnable_component)
        return None

    @property
    def sparse_component(self):
        return self._sparse_component

    @sparse_component.setter
    def sparse_component(self, sparse_component):
        assert isinstance(sparse_component, bool)
        self._sparse_component = sparse_component
        return None

    @property
    def non_negative_component(self):
        return self._non_negative_component

    @non_negative_component.setter
    def non_negative_component(self, non_negative_component):
        assert isinstance(non_negative_component, bool)
        self._non_negative_component = non_negative_component
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
    def select_prob_threshld(self):
        return self._select_prob_threshld

    @select_prob_threshld.setter
    def select_prob_threshld(self, select_prob_threshld):
        assert isinstance(select_prob_threshld, float)
        assert select_prob_threshld > 0. and select_prob_threshld <= 1.
        self._select_prob_threshld = select_prob_threshld
        return None

    @property
    def temperature_train(self):
        return self._temperature_train

    @temperature_train.setter
    def temperature_train(self, temperature_train):
        assert isinstance(temperature_train, (int, float))
        assert temperature_train > 0.
        self._temperature_train = float(temperature_train)
        return None

    @property
    def temperature_eval(self):
        return self._temperature_eval

    @temperature_eval.setter
    def temperature_eval(self, temperature_eval):
        assert isinstance(temperature_eval, (int, float))
        assert temperature_eval > 0.
        self._temperature_eval = float(temperature_eval)
        return None

    @property
    def tau(self):
        tau = None
        if self.training:
            tau = self.temperature_train
        else:
            tau = self.temperature_eval

        return tau

    @property
    def langevin_step_size(self):
        return self._langevin_step_size

    @langevin_step_size.setter
    def langevin_step_size(self, langevin_step_size):
        assert isinstance(langevin_step_size, (int, float))
        assert langevin_step_size > 0
        langevin_step_size = float(langevin_step_size)

        self._langevin_step_size = langevin_step_size
        self.component_sparsity.langevin_step_size = langevin_step_size
        self.component_parser.langevin_step_size = langevin_step_size
        return None

    @property
    def langevin_noise(self):
        return self._langevin_noise

    @langevin_noise.setter
    def langevin_noise(self, langevin_noise):
        assert isinstance(langevin_noise, (int, float))
        assert langevin_noise > 0
        langevin_noise = float(langevin_noise)

        self._langevin_noise = langevin_noise
        self.component_sparsity.langevin_noise = langevin_noise
        self.component_parser.langevin_noise = langevin_noise
        return None

    @property
    def langevin_steps(self):
        return self._langevin_steps

    @langevin_steps.setter
    def langevin_steps(self, langevin_steps):
        assert isinstance(langevin_steps, int)
        assert langevin_steps >= 0

        self._langevin_steps = langevin_steps
        self.component_sparsity.langevin_steps = langevin_steps
        self.component_parser.langevin_steps = langevin_steps
        return None

    @property
    def param_dict(self):
        return {'num_components': self.num_components,
                'dim_components': self.dim_components,
                'sparse_component': self.sparse_component,
                'non_negative_component': self.non_negative_component,
                'component_norm': self.component_norm,
                'hidden_size_extension': self.hidden_size_extension,
                'hidden_layers': self.hidden_layers,
                'learnable_component': self.learnable_component,
                'temperature_train': self.temperature_train,
                'temperature_eval': self.temperature_eval,
                'langevin_step_size': self.langevin_step_size,
                'langevin_noise': self.langevin_noise,
                'langevin_steps': self.langevin_steps}

    def components(self, indices = None):
        if indices is not None:
            assert isinstance(indices, (int, list, tuple, torch.Tensor))
            if isinstance(indices, int):
                indices = [indices]

            assert len(indices) > 0

        if self.non_negative_component:
            components = torch.abs(self.vanilla_components.clone())
        else:
            components = self.vanilla_components

        if self.component_norm is not None:
            components = components / torch.norm(components,
                                                 p = self.component_norm,
                                                 dim = 1,
                                                 keepdim = True)

        if indices is not None:
            if isinstance(indices, torch.Tensor):
                components = components[indices]
            else:
                components = components[tuple(indices)]

        return self.component_post_processing(components)

    def load_components(self, components, learnable = None):
        assert isinstance(components, torch.Tensor)
        assert components.shape == (self.num_components, self.dim_components)
        if learnable is None:
            learnable = self.learnable_component
        else:
            assert isinstance(learnable, bool)

        with torch.no_grad():
            components = components.detach().requires_grad_(learnable)
            self.vanilla_components = nn.Parameter(components)

        return None

    def parse_pattern(self, pattern):
        batch_size = pattern.shape[0]
        assert pattern.shape[-1] == self.dim_components

        if self.non_negative_component:
            components = torch.abs(self.vanilla_components)
        else:
            components = self.vanilla_components

        if self.component_norm is not None:
            components = components / torch.norm(components,
                                                 p = self.component_norm,
                                                 dim = 1,
                                                 keepdim = True)

        self.component_post_processing(components)
        c_shape = self.vanilla_components.shape
        components = components.unsqueeze(0).expand(batch_size, *components.shape) 

        sparsity_energy = torch.tensor(0., device = pattern.device)
        if self.sparse_component:
            self.component_sparsity.tau = self.tau
            (sparsity,
             sparsity_energy) = self.component_sparsity(batch_size = batch_size)

            components = components * sparsity

        self.component_parser.tau = self.tau
        (effective_indices,
         select_energy) = self.component_parser(pattern)

        concentration = self.concentration_regressor(pattern)
        concentration = torch.abs(concentration).unsqueeze(-1)

        return concentration, components, effective_indices, sparsity_energy, select_energy

    def forward(self, pattern, select_prob_threshld = None):
        if select_prob_threshld is None:
            select_prob_threshld = self.select_prob_threshld

        (concentration, 
         components,
         effective_indices,
         sparsity_energy,
         select_energy) = self.parse_pattern(pattern)

        if not self.training:
            effective_indices = (effective_indices > select_prob_threshld).float()

        concentration = concentration * effective_indices.unsqueeze(-1)
        components = components * effective_indices.unsqueeze(-1)

        return {'synthesized_pattern': self.aggregate_function(concentration, components),
                'concentration': concentration,
                'components': components,
                'effective_indices': effective_indices,
                'sparsity_energy': sparsity_energy,
                'select_energy': select_energy}

    def from_pretrained(self, 
            path, 
            param_filename = 'param.json', 
            model_filename = 'model.safetensors'):

        assert isinstance(path, str)
        assert isinstance(param_filename, str)
        assert len(param_filename) > 0
        assert isinstance(model_filename, str)
        assert len(model_filename) > 0

        if not os.path.isdir(path):
            raise OSError('{0} is not a directory.'.format(path))

        if not param_filename.endswith('.json'):
            param_filename += '.json'

        if not model_filename.endswith('.safetensors'):
            model_filename += '.safetensors'

        param_filename = os.path.join(path, param_filename)
        model_filename = os.path.join(path, model_filename)

        param_dict = load_json(param_filename)
        self.__init__(**param_dict)

        state_dict = load_safetensors(model_filename)
        self.load_state_dict(state_dict)

        return self

    def save_pretrained(self, 
            path, 
            param_filename = 'param.json', 
            model_filename = 'model.safetensors'):

        assert isinstance(path, str)
        assert isinstance(param_filename, str)
        assert len(param_filename) > 0
        assert isinstance(model_filename, str)
        assert len(model_filename) > 0

        if not param_filename.endswith('.json'):
            param_filename += '.json'

        if not os.path.isdir(path):
            init_directory(path)

        param_filename = os.path.join(path, param_filename)
        save_as_json(self.param_dict, param_filename)

        cpu_model = self.cpu()
        state_dict = cpu_model.state_dict()

        model_filename = os.path.join(path, model_filename)
        save_as_safetensors(state_dict, model_filename)

        return None


