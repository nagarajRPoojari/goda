### nanochat like compute optimal training pipeline for Gemma models


uv run python scripts/download_eval_bundle.py

uv run python scripts/train_with_streaming_download.py --num-train-shards 1 --min-shards 1 --train-script pretrain.py --train-args="--config configs/config_d12_pretrain.toml"

uv run python sfttrain.py --config configs/config_d12_sft.toml

uv run scripts/prepare_dpo_dataset.py
uv run python dpo.py --config configs/config_d12_dpo.toml --ref-model checkpoints/checkpoint_step_10.pt

uv run python dpo_v2.py --model-name "Qwen/Qwen2-0.5B" --config configs/config_hf_dpo.toml 