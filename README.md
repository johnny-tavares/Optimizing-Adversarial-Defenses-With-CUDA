# Adversarial Manager

A lightweight toolkit for evaluating adversarial robustness and inference performance. This project integrates standard PGD attacks with custom CUDA-accelerated defenses and high-precision timing benchmarks.

## Key Features

* **PGD Attacks:** Randomized, iterative Projected Gradient Descent via `torchattacks`.
* **Inference Timing:** Synchronized CUDA benchmarking with warmup cycles and disabled GC for high-precision latency measurement.
* **Feature Squeezing:** Custom **CUDA C++ kernels** (inline JIT compiled) for fused median filtering and bit-depth reduction (posterization).

## Requirements

* **CUDA Toolkit** (Properly configured `nvcc`)
* **Python 3.8+**
* **Dependencies:**
    ```bash
    pip install torch torchvision timm kornia torchattacks Pillow
    ```

## Usage

The manager operates in two primary modes: `attack` (to test robustness) and `timing` (to test latency).

### 1. Adversarial Attack Evaluation
Run a PGD attack against a ResNet50v2 model to see if the defense successfully recovers the correct classification.
```bash
# Defense options: none, baseline, custom
python adversarial_manager.py input.jpg --mode attack --defense custom
