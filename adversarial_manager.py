"""
Adversarial Manager

Combines functionality for:
1. Baseline PGD Attacks (Randomized & Iterative)
2. Inference Timing (Synchronized CUDA Benchmarking)
3. Feature Squeezing using Custom CUDA Kernels

Usage:
    python adversarial_manager.py input.jpg --mode attack --defense custom
    python adversarial_manager.py input.jpg --mode timing --defense custom
"""

import torch, timm, sys, os, time, argparse, json, urllib.request
from PIL import Image

import gc  # Added for disabling garbage collection during timing

# Defense imports
import kornia
import kornia.filters as KF
import kornia.enhance as KE

# Attack imports
import torchattacks

# Custom Kernel Compilation
from torch.utils.cpp_extension import load_inline

# --------------------------
# Attack Configuration & Constants
# --------------------------
EPS = 16/255      # Strong enough for ResNet50v2
ALPHA = 2/255
STEPS = 40

cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void my_fused_kernel(float* input, float* output, int width, int height, float levels) {
    // Global coordinates
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int z = blockIdx.z;

    // Calculating the index considering RGB planes
    int plane_size = width * height;
    int index = (z * plane_size) + (y * width) + x;

    // Spatial smoothing window (3x3)
    float window[9];
    int count = 0;

    // Add values to the window, and provide clamping if the pixel is on an edge
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = max(0, min(width - 1, x + dx));
            int ny = max(0, min(height - 1, y + dy));
            int idx = (z * plane_size) + (ny * width) + nx;
            window[count++] = input[idx];
        }
    }

    // Sort the window to find median
    for (int i = 0; i < 9; ++i) {
        for (int j = i + 1; j < 9; ++j) {
            if (window[i] > window[j]) {
                float temp = window[i];
                window[i] = window[j];
                window[j] = temp;
            }
        }
    }

    // Grab the median
    float median = window[4];

    // Feature Squeezing (Posterization)
    // The formula for reducing bit depth is different due to RGB being between 0.0 and 1.0
    float val = floorf(median * levels) / levels;

    // Write to output array
    output[index] = val;
}

torch::Tensor launch_fuse(torch::Tensor input, float levels) {
    auto output = torch::empty_like(input);
    int h = input.size(2);
    int w = input.size(3);
    dim3 threads(16, 16);
    dim3 blocks(14, 14, 3);

    my_fused_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        w, h, levels
    );
    return output;
}
"""

cpp_source = "torch::Tensor launch_fuse(torch::Tensor input, float levels);"

my_ext = None
def compile_extension():
    global my_ext
    try:
        my_ext = load_inline(
            name='defense_ext_v1',
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            functions=['launch_fuse'],
            verbose=False
        )
    except Exception as e:
        print(f"Compilation Failed: {e}")
        sys.exit(1)

# --------------------------
# Helpers
# --------------------------
def load_imagenet_labels():
    url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    filename = "imagenet_class_index.json"
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)
    with open(filename) as f:
        class_idx = json.load(f)
    return {int(k): v[1] for k, v in class_idx.items()}

def apply_defense(image_batch, defense_type):
    if defense_type == 'none':
        return image_batch
    elif defense_type == 'baseline':
        smoothed = KF.median_blur(image_batch, kernel_size=(3, 3))
        defended = KE.posterize(smoothed, bits=4)
        return defended
    elif defense_type == 'custom':
        if my_ext is None:
            compile_extension()
        return my_ext.launch_fuse(image_batch, 16.0) # 16.0 levels will be the equivalent of 4 bit depth reduction
    else:
        raise ValueError(f"Unknown defense type: {defense_type}")

def load_model_and_image(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    model = timm.create_model('resnetv2_50', pretrained=True)
    model.to(device)
    model.eval()
    
    data_config = timm.data.resolve_model_data_config(model)
    preprocess = timm.data.create_transform(**data_config, is_training=False)
    
    try:
        img = Image.open(image_path).convert("RGB")
        batch = preprocess(img).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)
        
    return model, batch, device

# --------------------------
# Mode: Attack
# --------------------------
def run_attack(image_path, defense_type):
    model, batch, device = load_model_and_image(image_path)
    if defense_type == 'custom' and my_ext is None:
        compile_extension()

    with torch.no_grad():
        clean_logits = model(batch)
    _, label = clean_logits.max(1)

    ITERATIONS = 10 
    print(f"Generating PGD Attacks (EPS={EPS}, Steps={STEPS}) over {ITERATIONS} iterations...")
    print(f"Applying Defense: {defense_type}")
    
    confidences = []
    labels_map = load_imagenet_labels()

    for i in range(ITERATIONS):
        # random_start=True ensures a different attack path each time
        model.train()
        pgd = torchattacks.PGD(model, eps=EPS, alpha=ALPHA, steps=STEPS, random_start=True)
        x_adv = pgd(batch, label)
        model.eval()

        final_input = apply_defense(x_adv, defense_type)

        with torch.no_grad():
            output = model(final_input)
            probs = torch.softmax(output, dim=1)

        top_prob, top_id = probs.max(1)
        confidence = float(top_prob.item() * 100)
        confidences.append(confidence)
        
        class_name = labels_map.get(top_id.item(), "Unknown")
        print(f"Iter {i+1}: {class_name} ({confidence:.2f}%)")

    avg_conf = sum(confidences) / len(confidences)
    print("-" * 30)
    print(f"Average Confidence: {avg_conf:.2f}%")
    print("-" * 30)

# --------------------------
# Mode: Timing
# --------------------------
def run_timing(image_path, defense_type):
    model, batch, device = load_model_and_image(image_path)
    
    # Pre-compile to avoid pollution
    if defense_type == 'custom' and my_ext is None:
        compile_extension()

    print(f"Timing Mode | Defense: {defense_type}")

    # --- Warmup ---
    # Ensures caches are hot and kernels are optimized
    print("Warming up (10 iters)...")
    for _ in range(10):
       with torch.inference_mode():
          defended = apply_defense(batch, defense_type)
          _ = model(defended)
    torch.cuda.synchronize()

    # --- Timed Loop ---
    print("Running benchmark (50 iters)...")
    times = []
    
    # Disable GC to prevent CPU pauses from affecting timing
    gc.disable() 
    
    try:
        for _ in range(50):
            # Synchronize BEFORE start time ensures previous GPU work is done
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            # The workload
            defended = apply_defense(batch, defense_type)
            with torch.inference_mode():
                output = model(defended)
            
            # Synchronize AFTER end time ensures GPU work is actually done
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000) # Convert to ms
            
    finally:
        gc.enable()

    # Result Statistics
    avg_ms = torch.tensor(times).mean().item()
    std_ms = torch.tensor(times).std().item()

    # Get prediction for sanity check
    probs = torch.softmax(output, dim=1)
    top_prob, top_id = probs.max(1)
    class_name = load_imagenet_labels().get(top_id.item(), "Unknown")

    print("-" * 40)
    print(f"Prediction: {class_name}")
    print(f"Confidence: {top_prob.item()*100:.2f}%")
    print("-" * 40)
    print(f"Average Time: {avg_ms:.4f} ms")
    print(f"Std Deviation: {std_ms:.4f} ms")
    print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--mode", choices=['attack', 'timing'], required=True)
    parser.add_argument("--defense", choices=['none', 'baseline', 'custom'], default='none')
    args = parser.parse_args()

    if args.mode == 'attack':
        run_attack(args.image, args.defense)
    elif args.mode == 'timing':
        run_timing(args.image, args.defense)