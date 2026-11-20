# Splitting Compositions of Monotone Operators

This repository contains Python scripts for the numerical experiments accompanying *Splitting Compositions of Monotone Operators* by Haoming Qu, Yefan Wu, and Thomas Chaffey.

## Reproduce the experiments

From the repository root, install the dependencies and run the scripts:

```bash
python -m pip install numpy matplotlib
python phase_analysis.py
python rlc_experiment.py
```

- `phase_analysis.py` reproduces the coupling-phase scan and compares observed convergence factors with the theoretical gain bounds.
- `rlc_experiment.py` computes the periodic response of the nonlinear RLC circuit and plots the capacitor voltage, capacitor current, and fixed-point residuals.

Both scripts create `figures/` for the plots and `data/` for the numerical results. The plots are saved to disk for viewing after each run.

## Citation

If you use this code in your research, please cite our paper:

Haoming Qu, Yefan Wu, and Thomas Chaffey. “Splitting Compositions of Monotone Operators.” [arXiv:2610.39480](https://arxiv.org/abs/2610.39480), 2026.
