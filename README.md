# Cuda-ml-pipeline-optimization-nsight
Optimized a real-time video segmentation pipeline using CUDA, TensorRT, CV-CUDA, and Nsight tools — achieving end-to-end GPU acceleration and identifying bottlenecks with fine-grained profiling.

#Command to profile baseline modal

```
!nsys profile --trace cuda,nvtx,osrt \
--output baseline \
--force-overwrite=true \
python video_segmentation/main_nvtx.py
```

#Command to profile optimized modal

```
!nsys profile --trace cuda,nvtx,osrt \
--output optimized \
--force-overwrite=true \
python video_segmentation/main_optimized.py
```
