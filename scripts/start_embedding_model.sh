conda activate vllm
export CUDA_VISIBLE_DEVICES=5
vllm serve /data2/home/lijidong/models/bge-m3 \
  --dtype auto \
  --api-key EMPTY \
  --port 7979 \
  --gpu-memory-utilization 0.3