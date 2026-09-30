mkdir -p logs

RUN_ID=$(date +%Y%m%d_%H%M%S)

# Set the maximum number of open file descriptors to 65535 to avoid "too many open files" errors during training
ulimit -n 65535

# nohup - run the command in the background and redirect output to a log file
# flock - acquire a lock on the specified file to prevent multiple instances from running simultaneously
# PYTHONUNBUFFERED=1 - ensures that the output is not buffered and is written to the log file immediately
# HF_HOME=~/hf_cache - sets the cache directory for Hugging Face models and datasets to a specific location
# CUDA_VISIBLE_DEVICES=0 - specifies which GPU to use for training (in this case, GPU 0)
# systemd-run - limits training RAM so the workstation stays responsive
nohup flock -n /tmp/train.lock \
  systemd-run --scope --user \
      -p MemoryMax=20G \
      -p MemorySwapMax=8G \
      env PYTHONUNBUFFERED=1 \
          HF_HOME=~/hf_cache \
          HF_HUB_OFFLINE=0 \
          LD_PRELOAD="$PWD/.pixi/envs/default/lib/libjpeg.so.8:$PWD/.pixi/envs/default/lib/libpng16.so.16" \
          CUDA_VISIBLE_DEVICES=0 \
          pixi run TrainCustomACTMulti \
              --steps 50000 \
              --chunk-size 100 \
              --n-action-steps 1 \
              --temporal-ensemble-coeff=0.01 \
              --batch-size 16 \
              --save-freq 5000 \
              --num-workers 4 \
              --dataset.video_backend=pyav \
              --output-dir=aic-dagger-data-outputs/train/act/${RUN_ID} \
  > logs/train_custom_act_multi_${RUN_ID}.log 2>&1 &

echo $! > logs/train_custom_act_multi_${RUN_ID}.pid
