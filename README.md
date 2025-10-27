# Dendritic Artificial Neural Networks (dANNs) with Receptive Fields (RFs)

This repository contains code and resources for implementing Dendritic Artificial Neural Networks (dANNs) with Receptive Fields (RFs). dANNs are inspired by the structure and function of biological neurons, particularly their dendritic trees, which play a crucial role in processing synaptic inputs.

This implementation fixes issues from [the original dANN codebase](https://github.com/Poirazi-Lab/dendritic_anns), including:

- [ ] Pytorch version requiring Keras.
- [ ] RAPIDS compatibility issues.
- [ ] Refactorings for better readability and maintainability.
- [ ] Efficiency improvements in applying receptive fields.

## Setup

To set up the environment, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/dann.git
   ```
2. Navigate to the project directory:
   ```bash
   cd dann
   ```
3. Install the environment using conda. Make sure to adjust the CUDA version according to your system:
   ```bash
   conda create -n dann -c rapidsai -c conda-forge -c nvidia  \
   rapids=25.10 python=3.11 'cuda-version>=12.0,<=12.9' \
   jupyterlab 'pytorch=*=*cuda*'
   ```
