# Time Series Engression

**Modeling Conditional Distributions with Input-Dependent Noise and Temporal Structure**

Master's Thesis, University of Geneva (2025)  
Author: Yuri Croci  
Supervisor: Prof. Dr. Sebastian Engelke

## About

This repository contains the complete implementation from my master's thesis on extending **Engression** to time series forecasting. Engression is a distributional regression framework that models full conditional distributions P(Y|X) through pre-additive noise injection, enabling reliable uncertainty quantification and extrapolation beyond training support.

### What is Engression?

Unlike traditional regression that predicts point estimates, Engression models entire conditional distributions by:
- Injecting noise at the input level (pre-additive) rather than post-output
- Learning distributions implicitly through sampling
- Providing stable extrapolation near training boundaries
- Avoiding restrictive parametric assumptions

### Key Contributions

This thesis extends Engression to temporal forecasting through:

1. **Temporal Encoding with GRU**: Captures sequential dependencies through recurrent architectures, producing a pooled latent representation $\tilde{h}_t$ using static attention mechanisms

2. **Heteroskedastic Noise Modeling**: Replaces fixed noise with input-dependent variance $\sigma^2(\tilde{h}_t)$, enabling adaptive uncertainty that varies with temporal context

3. **Comprehensive Benchmarking**: Systematic evaluation across synthetic and real-world datasets:
   - **vs Deterministic counterparts**: Competitive mean predictions with superior extreme event forecasting capabilities
   - **vs Baseline Engression**: Context-dependent improvements including enhanced tail quantile estimation in sequential settings and adaptive uncertainty quantification in heteroskedastic environments

## Repository Structure
```
├── Grid_Search_results/    # Hyperparameter optimization across all models and experiments
├── Models/                 # All model architectures (Engression variants + deterministic baselines)
├── Modules/                # Core implementation modules
├── multi_seed_results/     # Cross-seed experimental outputs (10 random initializations)
├── River_Discharge_Study/  # Data preparation and EDA for real-world application
├── Simulation_Study/       # Data generation and EDA for synthetic experiments
├── Thesis_and_Presentation/# Thesis document and presentation slides
└── Thesis_Figures/         # Selected figures used in thesis
```

## Experiments

### Synthetic Time Series
Controlled simulation with known ground truth featuring sequential dependencies and heteroskedastic dynamics. Enables direct evaluation of:
- Conditional quantile accuracy (Q10, Q50, Q90)
- Asymmetric distribution modeling
- Extrapolation stability

### River Discharge Forecasting  
One-day-ahead discharge forecasting on Aare River at Bern-Schönau (Switzerland):
- Historical period: 1930-2014 (31,046 daily observations)
- Features: 1 upstream discharge + 6 precipitation stations
- Test case: August 2005 extreme flood event
- Focus: Extreme event prediction under non-stationary conditions

## Dataset

The Swiss river discharge and precipitation data used in this study are **not included** in this repository due to data sharing restrictions. 

These datasets are available to academics upon request from:
- **River discharge**: [Swiss Federal Office for the Environment (FOEN)](https://www.hydrodaten.admin.ch/)
- **Precipitation**: [MeteoSwiss](https://gate.meteoswiss.ch/idaweb)

## Model Architectures

**Deterministic Baselines:**
- `MLP`: Feedforward network with lagged features
- `Sequential MLP`: GRU encoder with pooling + MLP decoder

**Engression Variants:**
- `Engression`: Original framework (baseline)
- `Heteroskedastic Engression`: Baseline + input-dependent noise
- `Sequential Engression`: Baseline + temporal encoding  
- `Heteroskedastic Sequential Engression`: Full extension (both components)

*All models benchmarked across 10 random initializations to quantify performance variability.*

## Citation
```bibtex
@mastersthesis{croci2025timeseries,
  author = {Croci, Yuri},
  title = {Time Series Engression: Modeling Conditional Distributions 
           with Input-Dependent Noise and Temporal Structure},
  school = {University of Geneva},
  year = {2025},
  type = {Master's Thesis},
  note = {Supervisor: Prof. Dr. Sebastian Engelke}
}
```

**Original Engression framework:**
```bibtex
@article{shen2024engression,
  title={Engression: Extrapolation through the lens of distributional regression},
  author={Shen, Xinwei and Meinshausen, Nicolai},
  journal={Journal of the Royal Statistical Society Series B},
  volume={86},
  number={2},
  pages={305--329},
  year={2024}
}
```

## License

MIT License

---
