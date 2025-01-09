
# Data and scripts: optimizing intermediates for free energy calculations used in Bayesian Inference of Conformational Populations (BICePs)

Previously, the BICePs score was used to select optimal forward models for protein
NMR scalar coupling constants. We demonstrate how to select the optimal schedule of \xi_k values to calculate the BICePs score.

### Below is a description of what this repository contains:
- [`fmo_u_kln.npy`](fmo_u_kln.npy): u_kln[k,l,n] matrix of free energies with shape [N_states, N_states, N_samples]. Here, we start with a uninformed choice of eleven equally spaced $\xi_k$ values $\{0.0, 0.1, ... 0.9, 1.0\}$ and sample for 50k steps, cycling through these intermediates in reverse order, progressively decreasing the value of $\xi$  for 5000 MCMC steps each. We saved samples every 10 steps. The matrix is constructed by taking the snapshot n \in 1,...,N_k of state k \in 1,...,K evaluated with the potential for state l. This matrix is then passed to MBAR to obtain the BICePs score.
- [`XiOpt.py`](XiOpt.py): script for loading in the u_kln matrix for optimizing the xi-values.



For more details regarding this data, please refer to:
"Automatic Forward Model Parameterization with Bayesian Inference of
Conformational Populations" [https://arxiv.org/pdf/2405.18532](https://arxiv.org/pdf/2405.18532)
```
@ARTICLE{Raddi2024FMO,
  title={Automatic Forward Model Parameterization with Bayesian Inference of Conformational Populations},
  author={Raddi, Robert M and Marshall, Tim and Voelz, Vincent A},
  journal={arXiv preprint arXiv:2405.18532},
  year={2024}
}
```


