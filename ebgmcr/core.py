import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import init_layer_weight

#--------------------------------------------------------------------------------
# core.py contains the core algorithm of filtering polluted pattern
#--------------------------------------------------------------------------------


__all__ = ['FixedEBSelectiveModule', 'DynamicEBSelectiveModule']


ENERGY_STD = 5e-2

class LangevinRefinedEnergyModule(nn.Module):
    def __init__(self, step_size = 0.05, noise_scale = 0.1, num_steps = 32):
        super(LangevinRefinedEnergyModule, self).__init__()

        assert isinstance(step_size, (int, float))
        assert step_size > 0
        assert isinstance(noise_scale, (int, float))
        assert noise_scale > 0
        assert isinstance(num_steps, int)
        assert num_steps >= 0

        self.step_size = float(step_size)
        self.noise_scale = float(noise_scale)
        self.num_steps = int(num_steps)

    def forward(self, energy):
        total_noise = self.noise_scale * torch.randn_like(energy) * \
                (self.num_steps ** 0.5)

        energy.requires_grad_(True)
        gradients = torch.autograd.grad(energy.sum(), energy, create_graph = True)[0]

        if self.training:
            refined_energy = energy - self.step_size * self.num_steps * gradients + total_noise
        else:
            refined_energy = energy

        return refined_energy


class EBSelectiveModule(nn.Module):
    def __init__(self,
            tau = 0.1,
            langevin_step_size = 0.05,
            langevin_nosie = 0.1,
            langevin_steps = 32):

        super(EBSelectiveModule, self).__init__()

        assert isinstance(tau, (int, float))
        assert tau > 0.
        self.tau = float(tau)

        self.langevin_dynamics = LangevinRefinedEnergyModule(
                step_size = langevin_step_size,
                noise_scale = langevin_nosie,
                num_steps = langevin_steps)

    def forward(self, energy):
        refined_energy = -self.langevin_dynamics(energy)
        select_prob = F.gumbel_softmax(refined_energy, 
                                       tau = self.tau,
                                       hard = False,
                                       dim = -1)[..., 0]

        return select_prob, refined_energy


class FixedEBSelectiveModule(EBSelectiveModule):
    def __init__(self,
            gate_number,
            output_shape = None,
            tau = 0.1,
            langevin_step_size = 0.05,
            langevin_nosie = 0.1,
            langevin_steps = 16):

        super(FixedEBSelectiveModule, self).__init__(
                tau = tau,
                langevin_step_size = langevin_step_size,
                langevin_nosie = langevin_nosie,
                langevin_steps = langevin_steps)

        assert isinstance(gate_number, int)
        assert gate_number > 0
        if output_shape is not None:
             assert isinstance(output_shape, (tuple, list))
             total_dim = 1
             for d in output_shape:
                 assert isinstance(d, int)
                 assert d > 0
                 total_dim *= d

             assert total_dim == gate_number

        self.gate_number = int(gate_number)
        if output_shape is None:
            output_shape = (self.gate_number, )

        self.output_shape = output_shape
        self.energy = init_layer_weight(nn.Parameter(torch.randn(self.gate_number, 2)), ENERGY_STD)

    def forward(self, batch_size = 1):
        assert isinstance(batch_size, int)
        assert batch_size > 0

        single_energy = self.energy.view(*self.output_shape, 2)
        repeated_unchange = [1 for _ in range(single_energy.dim())]
        batch_energy = single_energy.unsqueeze(0).repeat(batch_size, *repeated_unchange)

        return super(FixedEBSelectiveModule, self).forward(batch_energy)
        

class DynamicEBSelectiveModule(EBSelectiveModule):
    def __init__(self, 
            source_shape,
            gate_number,
            output_shape = None,
            energy_evaluator = None,
            energy_hidden_extension = None,
            energy_hidden_layers = 1,
            tau = 0.1,
            langevin_step_size = 0.05,
            langevin_nosie = 0.1,
            langevin_steps = 32):

        super(DynamicEBSelectiveModule, self).__init__(
                tau = tau,
                langevin_step_size = langevin_step_size,
                langevin_nosie = langevin_nosie,
                langevin_steps = langevin_steps)

        assert isinstance(source_shape, int)
        assert source_shape > 0
        assert isinstance(gate_number, int)
        assert gate_number > 0
        if output_shape is not None:
             assert isinstance(output_shape, (tuple, list))
             total_dim = 1
             for d in output_shape:
                 assert isinstance(d, int)
                 assert d > 0
                 total_dim *= d

             assert total_dim == gate_number

        if energy_evaluator is not None:
            assert isinstance(energy_evaluator, nn.Module)

        if energy_hidden_extension is not None:
            assert isinstance(energy_hidden_extension, int)
            assert energy_hidden_extension >= 1

        self.source_shape = source_shape
        self.gate_number = int(gate_number)
        if output_shape is None:
            output_shape = (self.gate_number, )

        self.output_shape = output_shape
        if energy_evaluator is None:
            if energy_hidden_extension is None:
                energy_evaluator = init_layer_weight(nn.Linear(source_shape, self.gate_number * 2), ENERGY_STD)
            else:
                hidden_size = self.gate_number * 2 * energy_hidden_extension
                layers = [init_layer_weight(nn.Linear(source_shape, hidden_size), ENERGY_STD), nn.Tanh()]
                for _ in range(energy_hidden_layers - 1):
                    layers.append(init_layer_weight(nn.Linear(hidden_size, hidden_size), ENERGY_STD))
                    layers.append(nn.Tanh())

                layers.append(init_layer_weight(nn.Linear(hidden_size, self.gate_number * 2), ENERGY_STD))
                energy_evaluator = nn.Sequential(*layers)

            self.energy_hidden_extension = energy_hidden_extension
            self.energy_hidden_layers = energy_hidden_layers
        else:
            self.energy_hidden_extension = None
            self.energy_hidden_layers = None

        self.energy_evaluator = energy_evaluator

    def forward(self, x):
        batch_size = x.shape[0]
        evaluated_energy = self.energy_evaluator(x)
        evaluated_energy = evaluated_energy.view(batch_size, *self.output_shape, 2)
        return super(DynamicEBSelectiveModule, self).forward(evaluated_energy)


