import torch

#--------------------------------------------------------------------------------
# ambiguity_panelty implement common kernel penalty for ensuring ambiguity of 
# components
#--------------------------------------------------------------------------------

__all__ = ['linear_penalty', 'polynomial_penalty', 'laplacian_penalty', 
        'rbf_penalty', 'matern_penalty']


def _scale_triu_terms(K):
    num_pairs = K.shape[0] * (K.shape[0] - 1) / 2
    off_diag_sum = K.triu(diagonal = 1).sum()
    return off_diag_sum / num_pairs

def linear_penalty(all_components):
    penalty = torch.tensor(0.)
    if all_components.shape[0] > 1:
        K = all_components @ all_components.T
        penalty = _scale_triu_terms(K)

    return penalty

def polynomial_penalty(all_components, c = 1., d = 2):
    penalty = torch.tensor(0.)
    if all_components.shape[0] > 1:
        dot_matrix = all_components @ all_components.T
        K = (dot_matrix + c) ** d
        penalty = _scale_triu_terms(K)

    return penalty

def laplacian_penalty(all_components, sigma = 1.):
    penalty = torch.tensor(0.)
    if all_components.shape[0] > 1:
        diff = all_components.unsqueeze(1) - all_components.unsqueeze(0)
        dist_l1 = diff.abs().sum(dim = -1)
        K = torch.exp(-dist_l1 / sigma)
        penalty = _scale_triu_terms(K)

    return penalty

def rbf_penalty(all_components, sigma = 1.):
    penalty = torch.tensor(0.)
    if all_components.shape[0] > 1:
        diff = all_components.unsqueeze(1) - all_components.unsqueeze(0)
        dist_sq = torch.sum(diff ** 2, dim = -1)
        K = torch.exp(-dist_sq / (2 * sigma ** 2))
        penalty = _scale_triu_terms(K)

    return penalty

def matern_penalty(all_components, nu = 1.5, lengthscale = 1., sigma = 1.):
    penalty = torch.tensor(0.)
    if all_components.shape[0] > 1:
        diff = all_components.unsqueeze(1) - all_components.unsqueeze(0)
        dist = diff.norm(p = 2, dim = -1)
        r = dist / lengthscale
        if nu == 0.5:
            # K(r) = σ² * exp(-r)
            K = sigma**2 * torch.exp(-r)

        elif nu == 1.5:
            # K(r) = σ² (1 + sqrt(3) r) exp(- sqrt(3) r)
            sqrt3_r = (3. ** 0.5) * r
            K = (1.0 + sqrt3_r) * torch.exp(-sqrt3_r)

        elif nu == 2.5:
            # K(r) = σ² (1 + sqrt(5) r + 5 r^2 / 3) exp(- sqrt(5) r)
            alpha = 5. ** 0.5
            K = sigma ** 2 * (1. + alpha * r + 5. / 3. * r ** 2) * torch.exp(-alpha * r)

        elif nu == 3.5:
            alpha = 7. ** 0.5
            term1 = 1. + alpha * r
            term2 = 21. / 5. * r ** 2
            term3 = (7. * alpha) / 15. * r ** 3
            K = sigma ** 2 * (term1 + term2 + term3) * torch.exp(-alpha * r)

        else:
            raise NotImplementedError('Matern kernel for ν={0} is not explicitly implemented.'\
                    .format(nu))

        penalty = _scale_triu_terms(K)

    return penalty


