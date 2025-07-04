# Insider Attack on Backdoor Watermarks in Federated Learning

This repository contains the code accompanying our paper:  
**_Insider Attack on Backdoor Watermarks in Federated Learning_**

We provide scripts / jupyter notebooks to reproduce the main experiments, including:

- Federated training with waffle watermarking  
- Inversion Baseline using Neural Dehydration
- Basic insider inversion attacks using intermediate model states
- Insider Inversion attacks using intermediate model states and proxy clean data
- Visualizing Results (TSNE, Heatmaps of salient neuron activations)

## Setup

This project was developed and tested with the following versions:

- Python 3.10.12
- PyTorch 2.1.2 (built with CUDA 11.8)
- Torchvision 0.16.2
- Flower (with simulation support) 1.6.0

Hardware:
- NVIDIA RTX 3090 GPU with CUDA 11.8 drivers
- Approximate RAM usage during CIFAR-10 training: ~30 GB

To create the conda environment with all necessary dependencies, use the provided `environment.yaml` file.

```bash
conda env create -f environment.yaml
conda activate flenv
```

To run training scripts, set the `PYTHONPATH` to the repo root to ensure imports work:

```bash
export PYTHONPATH=$(pwd)
```

## Logging and Results

This project uses **TensorBoard** for logging training metrics, evaluation results, images, and hyperparameters.

To visualize the logs:

```bash
tensorboard --logdir <path_to_log_dir>

```
Replace <path_to_log_dir> with the directory specified in the script argument `--output_dir` in the training or attack script


- **Training Metrics**: Metrics such as loss and accuracy are logged throughout training and can be viewed in the **Scalars** tab.
- **Hyperparameters and Final Metrics**: At the end of training, hyperparameters and final evaluation metrics are logged separately and shown in the **HParams** tab.

## Trained Models
All models trained in MNIST and their intermediate states required to run the attacks can be found in the folder `trained_models`.
Models trained with CIFAR-10 are available on [Zenodo](https://doi.org/10.5281/zenodo.15682233).
The models are organized in the following folder structure:

`<dataset>/<trigger>/<seed>/round_*.pth`

- `<dataset>`: Dataset name (e.g., `MNIST`, `CIFAR-10`)

- `<trigger>`: Trigger type, one of:
  - `A` (abstract images, no pattern)
  - `AP` (abstract images, with pattern)
  - `RP` (random images, with pattern)

- `<seed>`: Random seed used for training

- `round_*.pth`: Checkpoint files saved at various federated learning rounds

- To run attacks, specify the directory in the script argument `--base_model_path` accordingly ([Jump to Insider Attack](#running-the-insider-inversion-attack)).
## Training with Waffle Watermarking

To train a federated model on **CIFAR-10** using the **VGG16** architecture and pretrained weights with **Waffle watermarking** and the **RP** (Random background + class-specific Pattern) trigger set (corresponding to the model in `trained_models/cifar10/RP/42` ), run the following command:

```bash
python src/training/training_main.py \
  --dataset cifar10 \
  --model vgg16 \
  --model_checkpoint_rounds 0 1 25 50 \
  --client_cpu 1 \
  --client_cuda 0.2 \
  --device cuda \
  --num_rounds 50 \
  --clients_per_round 20 \
  --clients 20 \
  --watermark \
  --seed 42 \
  --background randomN \
  --pattern \
  --pretrain \
  --output_dir experiments/training/cifar10/RP/42
```
To train a model on MNIST:

```bash
python src/training/training_main.py \
  --dataset mnist \
  --model mnist_l5 \
  --model_checkpoint_rounds 0 1 25 50 \
  --client_cpu 1 \
  --client_cuda 0.2 \
  --device cuda \
  --num_rounds 50 \
  --clients_per_round 20 \
  --clients 20 \
  --watermark \
  --seed 42 \
  --background randomN \
  --pattern \
  --output_dir experiments/training/mnist/RP/42
```

Explanation of Key Flags:

- `--watermark`: Enables Waffle watermarking. If omitted, normal federated learning (FL) training takes place without watermarking.
- `--background`: Specifies the trigger background type. Options include:
  - `randomN` — random noise background (used in RP triggers)
  - `abstract` — abstract background patterns
- `--pattern`: When provided, adds a class-specific pattern to the trigger set. If omitted, no pattern is added.
- `--model_checkpoint_rounds 0 1 25 50`: Saves intermediate models at these training rounds for use in subsequent insider attacks.
- `--client_cuda 0.2`: Allocates 0.2 GPU per client (adjust this value according to your system’s resources).
- `--pretrain`: Uses a pre-trained model for initialization before federated training.
- `--seed`: Affects amogn other things:
  - Data generation and division among clients
  - Selection and generation of trigger set images 
- `--output_dir` directory in which logs and model checkpoints will be saved


## Inversion Visualizations
Examples of visualizations of neuron activations for different inversion sets can be found in the notebook:  
[`visualizing_neuron_activations.ipynb`](explore_inversion.ipynb)

> **Note:** Install Jupyter Notebook and use the same conda environment as the main experiments to ensure all dependencies are available.

## Running the Insider Inversion Attack

To perform an inversion attack on saved model checkpoints, run the following command example:

```bash
python src/attack/attack_main.py \
  --model_str vgg16 \
  --dataset_str cifar10 \
  --background randomN \
  --pattern \
  --seed 42 \
  --device cuda \
  --base_model_path experiments/training/cifar10/RP/42 \
  --model_paths round_0.pth round_1.pth round_25.pth round_50.pth \
  --marks_per_class 50 \
  --inversion_epochs 1000 \
  --inversion_lr 0.01 \
  --model_weights 0.25 0.5 0.25 0 \
  --unlearn_epochs 10 \
  --unlearn_lr 0.001 \
  --unlearn_batch_size 128 \
  --unlearn_momentum 0 \
  --tv_weight 0.03 \
  --l2_weight 0.01 \
  --inverse_loss_weight 10 \
  --use_clean \
  --output_dir experiments/attacks/cifar10/RP/42/0.25_0.5_0.25_0_use_clean
```

or  for the MNIST model:

```bash
python src/attack/attack_main.py \
  --model_str mnist_l5 \
  --dataset_str mnist \
  --background randomN \
  --pattern \
  --seed 42 \
  --device cuda \
  --base_model_path experiments/training/mnist/RP/42 \
  --model_paths round_0.pth round_1.pth round_25.pth round_50.pth \
  --marks_per_class 50 \
  --inversion_epochs 1000 \
  --inversion_lr 0.1 \
  --model_weights 0.25 0.5 0.25 0 \
  --unlearn_epochs 10 \
  --unlearn_lr 0.01 \
  --unlearn_batch_size 128 \
  --unlearn_momentum 0 \
  --tv_weight 0.03 \
  --l2_weight 0.01 \
  --inverse_loss_weight 10 \
  --use_clean \
  --output_dir experiments/attacks/mnist/RP/42/0.25_0.5_0.25_0_use_clean
```

Explanation of Key Flags:

- `--base_model_path`: Path to the folder containing multiple saved model checkpoints of a trained model.
- `--model_paths`: Specific checkpoint files to use within `--base_model_path`.
- `--model_weights`: Weights assigned to each model checkpoint for the attack (must match the number of checkpoints).
- `--inverse_loss_weight`: Weight of the inversion loss; relevant when attacker uses clean data (`--use_clean`) or proxy clean data (`--generate_proxy_clean`).
- `--use_clean`: Use one clients data for inversion loss during the attack.
- `--generate_proxy_clean`: Use proxy clean data generated during the attack.
- `--background`: Trigger background type (e.g., `abstract` or `randomN`).
- `--marks_per_class`: Number of watermark triggers per class.
- `--inversion_epochs`: Number of epochs for the inversion optimization.
- `--inversion_lr`: Learning rate for the inversion optimization.
- `--unlearn_epochs`: Number of epochs for the unlearning step.
- `--unlearn_lr`: Learning rate for unlearning.
- `--unlearn_batch_size`: Batch size during unlearning.
- `--unlearn_momentum`: Momentum factor during unlearning.
- `--tv_weight`: Total variation loss for inversion.
- `--l2_weight`: L2 regularization weight for inversion.
- `--output_dir` directory in which logs and model checkpoints will be saved

> **Note:** The flags `--model_str`, `--dataset_str`, `--seed`, and `--wm_size` should match the corresponding settings used during training to ensure consistent and meaningful attack evaluation.

### More examples

You can run several types of insider inversion attacks on saved model checkpoints.

---

#### Neural Dehydration (uses single checkpoint, split by salient neuron activations)

```bash
python src/attack/attack_main.py \
  --model_str vgg16 \
  --dataset_str cifar10 \
  --background randomN \
  --pattern \
  --seed 42 \
  --device cuda \
  --base_model_path experiments/training/cifar10/RP/42 \
  --model_paths round_50.pth \
  --marks_per_class 100 \
  --inversion_epochs 1000 \
  --inversion_lr 0.01 \
  --model_weights 1.0 \
  --unlearn_epochs 10 \
  --unlearn_lr 0.001 \
  --unlearn_batch_size 128 \
  --unlearn_momentum 0 \
  --tv_weight 0.03 \
  --l2_weight 0.01 \
  --split_by_salient_activations \
  --output_dir experiments/attacks/cifar10/RP/42/0.25_0.5_0.25_0_dehydra_no_clean

  
```
> **Note:** This can be run with real clean data or without clean data depending on the --use_clean flag.

#### Our attack without clean data
```bash
python src/attack/attack_main.py \
  --model_str vgg16 \
  --dataset_str cifar10 \
  --background randomN \
  --pattern \
  --seed 42 \
  --device cuda \
  --base_model_path experiments/training/cifar10/RP/42 \
  --model_paths round_0.pth round_1.pth round_25.pth round_50.pth \
  --marks_per_class 50 \
  --inversion_epochs 1000 \
  --inversion_lr 0.01 \
  --model_weights 0.25 0.5 0.25 0 \
  --unlearn_epochs 10 \
  --unlearn_lr 0.001 \
  --unlearn_batch_size 128 \
  --unlearn_momentum 0 \
  --tv_weight 0.03 \
  --l2_weight 0.01 \
  --output_dir experiments/attacks/cifar10/RP/42/0.25_0.5_0.25_0_no_clean

```

#### Our attack with proxy clean data

```bash
python src/attack/attack_main.py \
  --model_str vgg16 \
  --dataset_str cifar10 \
  --background randomN \
  --pattern \
  --seed 42 \
  --device cuda \
  --base_model_path experiments/training/cifar10/RP/42 \
  --model_paths round_0.pth round_1.pth round_25.pth round_50.pth \
  --marks_per_class 50 \
  --inversion_epochs 1000 \
  --inversion_lr 0.01 \
  --model_weights 0.25 0.5 0.25 0 \
  --unlearn_epochs 10 \
  --unlearn_lr 0.001 \
  --unlearn_batch_size 128 \
  --unlearn_momentum 0 \
  --tv_weight 0.03 \
  --l2_weight 0.01 \
  --inverse_loss_weight 1 \
  --generate_proxy_clean \
  --output_dir experiments/attacks/cifar10/RP/42/0.25_0.5_0.25_0_proxy_clean
```

### Resources Used
This project includes code and resources adapted from the following repositories:

- [**Dehydra**](https://github.com/LouisVann/Dehydra):  
  Loss functions were reused and parts of the main inversion procedure were adapted from this repository.
- 
- [**Turning Your Weakness Into a Strength**](https://github.com/adiyoss/WatermarkNN):  
  Abstract trigger images used in this project were sourced from this repository.

We thank the authors of these works for making their code and/or datasets publicly available.
