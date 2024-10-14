[![Build Status](https://github.com/biaslab/UnboundedBNN/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/biaslab/UnboundedBNN/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fbartvanerp%2Fd12677a8265d5cff93a9737ecd36c305%2Fraw%2FUnboundedBNN__main.json)](https://github.com/biaslab/UnboundedBNN/actions/workflows/CI.yml?query=branch%3Amain)

# Improved Depth Estimation of Bayesian Neural Networks
*By Bart van Erp and Bert de Vries*

**Accepted to the Workshop on Bayesian Decision-making and Uncertainty, 38th Conference on Neural Information Processing Systems (NeurIPS 2024)**

---
**Abstract**

This paper proposes improvements over earlier work by Nazareth and Blei for estimating the depth of Bayesian neural networks. Here, we propose a discrete truncated normal distribution over the network depth to independently learn its mean and variance. Posterior distributions are inferred by minimizing the variational free energy, which balances the model complexity and accuracy. Our method improves test accuracy in the spiral data set and reduces the variance in posterior depth estimates.

---
This repository contains all experiments of the paper.

## Installation instructions
1. Install [Julia](https://julialang.org/)

2. Open Julia
```bash
julia
```

3. activate environment (using `]` and backspace you can switch between the regular prompt and package manager)
```julia
>> ] activate .
```

4. instantiate environment (only required once)
```julia
>> ] instantiate
```

5. activate environments
```julia
>> ] activate experiments
```

6. instantiate environment (only required once)
```julia
>> ] instantiate
```

# License

[MIT License](LICENSE) Copyright (c) 2024-present BIASlab
