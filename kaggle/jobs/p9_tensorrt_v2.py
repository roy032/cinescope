# extends: p9_tensorrt
# The TensorRT run again, after the fix that keeps variable-length models (Light-ASD, CRNN) on CUDA
# instead of building a TensorRT engine per input shape. Writes results/tensorrt_v2 beside the first run.
OUT_TAG = "tensorrt_v2"
