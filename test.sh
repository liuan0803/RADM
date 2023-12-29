python3 train_net.py --num-gpus 1 \
    --config-file configs/radm.yaml \
    --eval-only --resume
python3 metrics.py
