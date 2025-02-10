# Run with template

execute the following shell scritps

```shell
git clone https://github.com/Rorschach1989/transformer-for-gmm.git
cd transformer-for-gmm
chmod +x ./run_template.sh  # Or edit it to be any configurations of interest
./run_template.sh
```

# Configuration explanations

## Overview

The configuration is divided into four main sections: `task`, `model`, `train`, and `eval`. Each section controls a different aspect of the experiment.

## 1. `task` Section

This section defines the specific type of GMM task the model will learn.

```yaml
task:
  type: MultiTaskIsotropicGaussianMixture
  n_components:
  - 2
  - 3
  dim: 8
```

*   **`type: MultiTaskIsotropicGaussianMixture`**: This specifies the task type.  `MultiTask` suggests the model will be trained on multiple $K$ specs. `IsotropicGaussianMixture` indicates that the GMMs will have isotropic (spherical) covariance matrices.  This means each Gaussian component has a covariance matrix that is a scalar multiple of the identity matrix (equal variance in all dimensions).
*   **`n_components: [2, 3]`**: This is a *list* defining the number of Gaussian components in the mixtures the model will encounter.  The model will be trained on GMMs with *both* 2 components *and* 3 components.
*   **`dim: 8`**:  This sets the dimensionality of the data points generated from the GMMs. Each data point will be an 8-dimensional vector.

## 2. `model` Section

This section specifies the architecture of the transformer model.

```yaml
model:
  n_positions: 4096
  n_embd: 128
  n_layer: 12
  n_head: 4
```

*   **`n_positions: 4096`**: This is the maximum sequence length the transformer can handle. It represents the maximum number of tokens (or data points, in this case) the model can process in a single input sequence.  This is often related to the positional embeddings used in the transformer.
*   **`n_embd: 128`**:  This is the dimensionality of the embedding space (also known as the hidden size or model dimension).  Each input token (or data point) will be projected into a 128-dimensional vector.
*   **`n_layer: 12`**: This defines the number of transformer layers (or blocks) in the model. A deeper model (more layers) can potentially learn more complex relationships, but also requires more computational resources.
*   **`n_head: 4`**: This is the number of attention heads in each transformer layer. Multi-head attention allows the model to attend to different parts of the input sequence simultaneously.

## 3. `train` Section

This section configures the training process.

```yaml
train:
  verbose: false
  seed: 42
  n_sample: 64
  batch_size: 32
  eval_every: 1000
  learning_rate: 0.001
  num_train_steps: 10001
```

*   **`verbose: false`**: This controls the verbosity of the training log (whether to use ``tqdm`` or not). 
*   **`seed: 42`**:  This sets the random seed for reproducibility. Using the same seed ensures that the random number generator will produce the same sequence of random numbers, leading to consistent results across multiple runs (assuming all other parameters are kept constant).
*   **`n_sample: 64`**: This refers to the number of in-context samples drawn from the GMM. During training, ``n_sample`` is only an upper bound of $N$. Currently, the sampling logic is uniform between $[N/2, N]$
*   **`batch_size: 32`**: This is the number of tasks during each training step.
*   **`eval_every: 1000`**: This specifies how often the model is evaluated on evaluation tasks.
*   **`learning_rate: 0.001`**: Learning rate for Adam/AdamW, currently no weight decay used.
*   **`num_train_steps: 10001`**: This is the total number of training steps the model will undergo. The training process will run for 10001 iterations.

## 4. `eval` Section

This section sets up the evaluation parameters.

```yaml
eval:
  n_sample: 128
  batch_size: 128
```

*   **`n_sample: 128`**: This refers to the number of in-context samples drawn from the GMM. During evaluation, ``n_sample`` is fixed.
*   **`batch_size: 128`**: This is the number of tasks during each evaluation step, shall be large enough for stabilization of results. 
