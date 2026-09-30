mkdir -p logs

RUN_ID=$(date +%Y%m%d_%H%M%S)

# Set the maximum number of open file descriptors to 65535 to avoid "too many open files" errors during training
ulimit -n 65535

# nohup - run the command in the background and redirect output to a log file
# flock - acquire a lock on the specified file to prevent multiple instances from running simultaneously
# PYTHONUNBUFFERED=1 - ensures that the output is not buffered and is written to the log file immediately
# HF_HUB_DISABLE_XET=1 - disables the use of the Hugging Face Hub's XET feature
# HF_HOME=/media/mbed/T7/hf_cache - sets the cache directory for Hugging Face models and datasets to a specific location
# CUDA_VISIBLE_DEVICES=0 - specifies which GPU to use for training
# systemd-run - limits training RAM so the workstation stays responsive
nohup flock -n /tmp/train_smolvla_${RUN_ID}.lock \
  systemd-run --scope \
      -p MemoryMax=20G \
      -p MemorySwapMax=8G \
      env PYTHONUNBUFFERED=1 \
          HF_HUB_DISABLE_XET=1 \
          HF_HOME=/media/mbed/T7/hf_cache \
          CUDA_VISIBLE_DEVICES=0 \
          pixi run TrainSmolVLATrainedPolicy \
              --steps 50000 \
              --chunk-size 50 \
              --n-action-steps 50 \
              --batch-size 1 \
              --save-freq 10000 \
              --num-workers 1 \
              --output-dir=aic-dagger-data-outputs/train/smolvla/${RUN_ID} \
  > logs/train_smolvla_${RUN_ID}.log 2>&1 &

echo $! > logs/train_smolvla_${RUN_ID}.pid
