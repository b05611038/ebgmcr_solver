# EB-gMCR: Energy-Based Generative Multivariate Curve Resolution

📄 **ArXiv Preprint:** [EB-gMCR: Energy-Based Generative Modeling for Signal Unmixing and Multivariate Curve Resolution](https://arxiv.org/abs/2507.23600v1)

Multivariate Curve Resolution (MCR) aims to decompose mixtures of signals into their latent components and concentrations under non-unique and ill-posed settings. Traditional MCR methods typically rely on matrix factorization techniques such as MCR-ALS or NMF, which may struggle when the number of components becomes very large or data sizes are extremely large. Our proposed **EB-gMCR** framework leverages generative deep learning and variational optimization to provide a flexible and scalable solver for challenging large-scale MCR problems.

## 🔑 Features

* Automatic component selection via differentiable energy-based gating
* Scalable to large pools of candidate spectra (e.g., N≫10)
* Supports noisy and large-scale datasets
* Built on PyTorch to leverage GPU acceleration

## 📦 Installation

```bash
git clone https://github.com/b05611038/ebgmcr_solver
cd ebgmcr_solver
pip install -r requirements.txt
```

## 🚀 How to Use

You can apply the EB-gMCR solver in just a few lines:

```python
>>> import torch
>>> from ebgmcr import RandomComponentMixtureSynthesizer, DenseEBgMCR
>>> data_generator = RandomComponentMixtureSynthesizer(component_number=16,
...                                                    component_dim=512,
...                                                    non_negative_component=True,
...                                                    min_concentration=1.,
...                                                    max_concentration=10.,
...                                                    signal_to_nosie_ratio=20.) # SNR in dB
>>> data = data_generator(64) # 4N dataset
>>> solver = DenseEBgMCR(num_components=1024, dim_components=512, mini_batch_size=16, device = torch.device('cuda:0'))
>>> solver.fit(data, show_eval_info = True, eval_info_display_interval = 100, log_dir = 'outputs')
Epoch: 100 | nMSE (pattern): 0.0355 | R2: 0.8337 | E: 1.5545 | Used components: 429
Epoch: 200 | nMSE (pattern): 0.0325 | R2: 0.8477 | E: 1.5434 | Used components: 408
Epoch: 300 | nMSE (pattern): 0.0279 | R2: 0.8691 | E: 1.5343 | Used components: 378
Epoch: 400 | nMSE (pattern): 0.0191 | R2: 0.9103 | E: 1.5306 | Used components: 372
Epoch: 500 | nMSE (pattern): 0.0140 | R2: 0.9346 | E: 1.5120 | Used components: 372
Epoch: 600 | nMSE (pattern): 0.0104 | R2: 0.9513 | E: 1.5082 | Used components: 374
Epoch: 700 | nMSE (pattern): 0.0070 | R2: 0.9671 | E: 1.5259 | Used components: 378
Epoch: 800 | nMSE (pattern): 0.0083 | R2: 0.9612 | E: 1.5896 | Used components: 136
Epoch: 900 | nMSE (pattern): 0.0064 | R2: 0.9702 | E: 1.5688 | Used components: 83
Epoch: 1000 | nMSE (pattern): 0.0073 | R2: 0.9657 | E: 1.5967 | Used components: 45
...
Epoch: 14300 | nMSE (pattern): 0.0050 | R2: 0.9765 | E: 0.7738 | Used components: 22
Epoch: 14377 | nMSE: 0.0050 | E: 0.7627 reach terminate condition (leave converge region).
Successfully save recorded checkpoint (region=0.97[->]0.975, effective_components=19)
Successfully checkpoint result to file:outputs/0.97[->]0.975/result.csv.
Successfully save recorded checkpoint (region=0.975[->]0.98, effective_components=21)
Successfully checkpoint result to file:outputs/0.975[->]0.98/result.csv.
>>> sovler.load_mcrs('outputs/0.975[->]0.98') # load best checkpoint
>>> result = solver.parse_pattern(data)
>>> result.keys()
dict_keys(['reconstruction', 'concentration', 'components', 'select_prob'])
```

## 📖 Citation

You can use the following BibTeX entry, which follows the formal arXiv recommendation:

```bibtex
@misc{chang2025ebgmcr,
  title        = {EB-gMCR: Energy-Based Generative Modeling for Signal Unmixing and Multivariate Curve Resolution},
  author       = {Chang, Yu-Tang and Others},
  year         = {2025},
  eprint       = {2507.23600v1},
  archivePrefix= {arXiv},
  primaryClass = {cs.LG}
}
```

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check [issues](https://github.com/b05611038/ebgmcr_solver/issues) and [pull requests](https://github.com/b05611038/ebgmcr_solver/pulls).

## 🛡 License

This repository is released under the MIT License. See the [LICENSE](./LICENSE) file for details.

