import torch
import torch.nn as nn

#--------------------------------------------------------------------------------
# to synthesize data for evaluating the EB-gMCR
#--------------------------------------------------------------------------------


__all__ = ['RandomComponentMixtureSynthesizer']


def _safe_mask_for_sparse_components(source_components, sparsity):
    assert isinstance(source_components, torch.Tensor)
    assert source_components.dim() == 2
    assert isinstance(sparsity, float)
    assert sparsity >= 0. and sparsity < 1.

    N, d = source_components.shape
    valid_masks = None
    while ((valid_masks is None) or (valid_masks.shape[0] < N)):
        if valid_masks is None:
            remain_N = N
            remain_sparsity = sparsity
        else:
            remain_N = N - valid_masks.shape[0]
            should_contain_zeros = (sparsity * N * d) - (valid_masks == 0).sum().item()
            remain_sparsity = should_contain_zeros / (remain_N * d)
            
        num_zeros = int(remain_N * d * remain_sparsity)
        single_round_mask = torch.ones(remain_N * d, dtype = torch.float32)
        zero_indices = torch.randperm(remain_N * d)[: num_zeros]
        single_round_mask[zero_indices] = 0.
        single_round_mask = single_round_mask.reshape(remain_N, d)
        single_round_mask = single_round_mask[single_round_mask.sum(1) > 0.]
        if valid_masks is None:
            valid_masks = single_round_mask
        else:
            valid_masks = torch.cat((valid_masks, single_round_mask), dim = 0)

    return valid_masks


class RandomComponentMixtureSynthesizer:
    def __init__(self,
            component_number,
            component_dim,
            component_sparsity = 0.,
            non_negative_component = True,
            orthogonal_component = True,
            component_norm = None,
            min_mixing_component = 1,
            max_mixing_component = -1,
            min_concentration = 1.,
            max_concentration = 10.,
            signal_to_nosie_ratio = None,# dB
            components = None,
            device = None):

        self.__finish_init = False

        self.component_number = component_number
        self.component_dim = component_dim
        self.component_sparsity = component_sparsity
        self.non_negative_component = non_negative_component
        self.orthogonal_component = orthogonal_component
        self.component_norm = component_norm
        self.min_mixing_component = min_mixing_component
        self.max_mixing_component = max_mixing_component
        self.min_concentration = min_concentration
        self.max_concentration = max_concentration
        self.signal_to_nosie_ratio = signal_to_nosie_ratio
        self.device = device

        self.components = self._init_components(components).to(self.device)
        self.__finish_init = True

    @property
    def component_number(self):
        return self._component_number

    @component_number.setter
    def component_number(self, component_number):
        assert isinstance(component_number, int)
        assert component_number >= 1
        self._component_number = component_number
        return None

    @property
    def component_dim(self):
        return self._component_dim

    @component_dim.setter
    def component_dim(self, component_dim):
        assert isinstance(component_dim, int)
        assert component_dim >= 1
        self._component_dim = component_dim
        return None

    @property
    def component_sparsity(self):
        return self._component_sparsity

    @component_sparsity.setter
    def component_sparsity(self, component_sparsity):
        assert isinstance(component_sparsity, (int, float))
        assert component_sparsity >= 0. and component_sparsity < 1.
        self._component_sparsity = float(component_sparsity)

    @property
    def non_negative_component(self):
        return self._non_negative_component

    @non_negative_component.setter
    def non_negative_component(self, non_negative_component):
        assert isinstance(non_negative_component, bool)
        self._non_negative_component = non_negative_component
        return None

    @property
    def orthogonal_component(self):
        return self._orthogonal_component

    @orthogonal_component.setter
    def orthogonal_component(self, orthogonal_component):
        assert isinstance(orthogonal_component, bool)
        self._orthogonal_component = orthogonal_component
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
    def min_mixing_component(self):
        return self._min_mixing_component

    @min_mixing_component.setter
    def min_mixing_component(self, min_mixing_component):
        assert isinstance(min_mixing_component, int)
        assert min_mixing_component >= 1
        assert min_mixing_component < self.component_number
        self._min_mixing_component = min_mixing_component
        return None

    @property
    def max_mixing_component(self):
        return self._max_mixing_component

    @max_mixing_component.setter
    def max_mixing_component(self, max_mixing_component):
        assert isinstance(max_mixing_component, int)
        if max_mixing_component > 0:
            assert max_mixing_component > self.min_mixing_component
            assert max_mixing_component <= self.component_number

        self._max_mixing_component = max_mixing_component
        return None

    @property
    def min_concentration(self):
        return self._min_concentration

    @min_concentration.setter
    def min_concentration(self, min_concentration):
        assert isinstance(min_concentration, (int, float))
        assert min_concentration > 0.
        self._min_concentration = float(min_concentration)
        return None

    @property
    def max_concentration(self):
        return self._max_concentration

    @max_concentration.setter
    def max_concentration(self, max_concentration):
        assert isinstance(max_concentration, (int, float))
        assert max_concentration >= self.min_concentration
        self._max_concentration = float(max_concentration)
        return None

    @property
    def signal_to_nosie_ratio(self):
        return self._signal_to_nosie_ratio

    @signal_to_nosie_ratio.setter
    def signal_to_nosie_ratio(self, signal_to_nosie_ratio):
        if signal_to_nosie_ratio is not None:
            assert isinstance(signal_to_nosie_ratio, (int, float))
            assert signal_to_nosie_ratio >= 0.
            signal_to_nosie_ratio = float(signal_to_nosie_ratio)

        self._signal_to_nosie_ratio = signal_to_nosie_ratio

        return None

    @property
    def device(self):
        return self._device
    
    @device.setter
    def device(self, device):
        if device is None:
            device = torch.device('cpu')
        else:                          
            assert isinstance(device, (int, torch.device))
            if isinstance(device, int): 
                if device < 0:
                    device = torch.device('cpu')
                else:
                    device = torch.device('cuda:{0}'.format(device))
    
        self._device = device
        return None

    def _init_components(self, components):
        if components is not None:
            assert isinstance(components, torch.Tensor)
            assert components.dim() == 2
            self.component_number = components.shape[0]
            self.component_dim = components.shape[1]
            self.component_sparsity = ((components == 0.).float().sum() / components.numel()).item()
            self.non_negative_component = (components >= 0.).all().item()
            self.orthogonal_component = False
            self.component_norm = None
        else:
            if self.orthogonal_component:
                components = torch.empty((self.component_number, self.component_dim))
                nn.init.orthogonal_(components, gain = 1.)
            else:
                components = torch.randn(self.component_number, self.component_dim)

            if self.non_negative_component:
                components = torch.abs(components)

            if self.component_sparsity > 0.:
                mask = _safe_mask_for_sparse_components(components, self.component_sparsity)
                components = components * mask

            if self.component_norm is not None:
                components = components / torch.norm(components, 
                                                     p = self.component_norm, 
                                                     dim = 1, 
                                                     keepdim = True)

        return components.detach()

    def __repr__(self):
        line =  '{0}(component_number={1}, component_dim={2}, non_negative={3}'\
                .format(self.__class__.__name__, self.component_number, self.component_dim, 
                self.non_negative_component)

        if self.component_norm is not None:
            line += ', component_norm={0}'.format(self.component_norm)

        line += ')'
        return line

    def generate(self, sample_number, separation_mode = False):
        assert isinstance(sample_number, int)
        assert sample_number > 0
        assert isinstance(separation_mode, bool)

        if self.max_mixing_component < 0:
             max_mixing_component = self.component_number
        else:
             max_mixing_component = self.max_mixing_component

        used_component_number = torch.randint(self.min_mixing_component, max_mixing_component + 1, 
                (sample_number, ), device = self.device)

        rand_indices = torch.rand(sample_number, self.component_number, 
                device = self.device).argsort(dim = -1)
        selected_indices = rand_indices[:, :used_component_number.max()]

        selected_components = self.components.detach().clone()[selected_indices]

        weight_mask = torch.arange(used_component_number.max(), device = self.device).unsqueeze(0)
        weight_mask = weight_mask >= used_component_number.unsqueeze(-1)

        weights = torch.empty(sample_number, used_component_number.max(), 
                device = self.device).uniform_(self.min_concentration, self.max_concentration)
        weights[weight_mask] = 0.

        data = (selected_components * weights.unsqueeze(-1)).sum(dim = 1)
        if self.signal_to_nosie_ratio is not None:
            nonzero_mask = data != 0
            signal_power = (data ** 2).sum(dim = 1) / nonzero_mask.sum(dim = 1).clamp(min = 1)

            noise_variance = signal_power / (10. ** (self.signal_to_nosie_ratio / 10.))
            std_per_sample = noise_variance.sqrt().unsqueeze(1)

            noise = torch.randn_like(data) * std_per_sample
            data = data + noise * nonzero_mask.float()

        if self.non_negative_component:
            if self.component_sparsity > 0.:
                data = torch.clamp(data, min = 0.)
            else:
                data = torch.clamp(data, min = 1e-6)

        if separation_mode:
            selected_indices[weight_mask] = -1
            selected_indices = selected_indices.reshape(-1)
            selected_indices = selected_indices.tolist()
            selected_indices = list(set(selected_indices))
            selected_indices.remove(-1)
            outputs = (data.detach(), weights.detach(), selected_components.detach(), selected_indices)
        else:
            outputs = data.detach()

        return outputs

    def __call__(self, sample_number = None, separation_mode = False, device = None):
        if sample_number is None:
            sample_number = 1

        if device is not None:
            self.device = device

        return self.generate(sample_number, separation_mode = separation_mode)


