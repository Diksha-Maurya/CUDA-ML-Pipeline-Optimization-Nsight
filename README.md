# Cuda-ML-Pipeline-Optimization-Nsight 🚀  
Optimized a real-time video segmentation pipeline using **CUDA**, **TensorRT**, **CV-CUDA**, and **Nsight profiling tools** — achieving full GPU acceleration and reducing end-to-end runtime from **52.14s to 3.89s**.

---

## 🧠 Overview

This project demonstrates how to analyze and optimize a machine learning video processing pipeline using NVIDIA's profiling tools. The focus is on:

- **Profiling and identifying bottlenecks** with Nsight Systems
- **Accelerating model inference** with TensorRT
- **Moving preprocessing and postprocessing to the GPU** using CV-CUDA
- **Using NVDEC/NVENC for GPU-based video decoding and encoding**

---

## 🔧 Pipeline Stages

### 🔹 1. Decoder (NVDEC Hardware Acceleration)
- Original: OpenCV-based decoding on CPU  
- Optimized: Replaced with **NVIDIA NVDEC**, which performs decoding on GPU — eliminating large host-to-device memory transfers.

### 🔹 2. Preprocessing (CV-CUDA)
- Original: OpenCV (CPU) operations like resize, normalize  
- Optimized: All pre-processing moved to GPU using **CV-CUDA** — boosting throughput and reducing latency.

### 🔹 3. Inference (TensorRT)
- Original: PyTorch model executed using standard CUDA kernels  
- Optimized: Converted model to **TensorRT engine** for GPU-specific optimizations, kernel fusion, and FP16 support.

### 🔹 4. Postprocessing (CV-CUDA)
- Applied operations like blurring directly on GPU using **CV-CUDA**, avoiding host-device syncs.

### 🔹 5. Encoder (NVENC Hardware Acceleration)
- Original: OpenCV-based encoder  
- Optimized: Switched to **NVENC**, leveraging dedicated GPU silicon for real-time video encoding.

---

## 📈 Profiling & Results

### Baseline Model Profiling  
Command:
```bash
nsys profile --trace cuda,nvtx,osrt \
--output baseline \
--force-overwrite=true \
python video_segmentation/main_nvtx.py
````

**Pipeline Execution Time:** 52.142 seconds

---

### Optimized Model Profiling

Command:

```bash
nsys profile --trace cuda,nvtx,osrt \
--output optimized \
--force-overwrite=true \
python video_segmentation/main_optimized.py
```

**Pipeline Execution Time:** ⏱️ 3.890 seconds

✅ Achieved a **\~13.4× speedup** by removing CPU-GPU bottlenecks and using GPU-native libraries throughout the pipeline.

---

## 🔍 Tools & Technologies

* 🧠 **TensorRT** — GPU-optimized deep learning inference
* 🎞 **NVDEC / NVENC** — GPU-based video decoding & encoding
* 🧰 **CV-CUDA** — GPU-native image pre/post-processing
* 🔬 **Nsight Systems / Nsight Compute** — CUDA profiling & bottleneck analysis
* 🔥 **NVTX Markers** — For annotated timeline visualizations
