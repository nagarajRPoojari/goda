### nanochat like compute optimal training pipeline for Gemma models

python scripts/train_with_streaming_download.py --num-train-shards 1 --min-shards 1 --train-script example.py --train-args="--config config_d12.yaml"

python sfttrain.py --config config_d12.yaml