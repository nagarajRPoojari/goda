import torch
import torch.nn as nn
import argparse

from goda.logger import logger
from goda.sfttrain import SFTTrainer
from mint.config.base import Config
from mint.data.datasets.arc import ARC
from mint.data.datasets.base import SFTEvalDataset
from mint.data.datasets.customjsonl import CustomJSON
from mint.data.datasets.gsm8k import GSM8K
from mint.data.datasets.humaneval import HumanEval
from mint.data.datasets.mmlu import MMLU
from mint.data.datasets.smoltalk import SmolTalk
from mint.data.datasets.spellingbee import SimpleSpelling, SpellingBee
from mint.data.dist.sft import DistributedSFTDataloader
from mint.nn.models import Gemma, configure_optimizer
from mint.tokenizer import Tokenizer
from mint.utils.device import Device


def parse_dataset_spec(spec: str, dataset_map: dict):
    if '*' in spec:
        count_str, name = spec.split('*', 1)
        count = int(count_str)
        name = name.strip()
    else:
        count = 1
        name = spec.strip()
    
    if name in dataset_map:
        return [dataset_map[name]() for _ in range(count)]
    
    try:
        logger.info(f"Loading CustomJSON dataset from {name} ({count} instance(s))")
        return [CustomJSON(filepath=name) for _ in range(count)]
    except Exception as e:
        logger.error(f"Failed to load dataset '{name}': {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="SFT training for Gemma model")
    parser.add_argument("--config", type=str, default="config_d12.yaml", help="Path to config YAML file")
    parser.add_argument("--datasets", type=str, nargs="+", default=["gsm8k", "smoltalk", "arc", "mmlu", "humaneval", "spellingbee", "simplespelling"],
                        help="Datasets to train on (e.g., 'arc', '3*arc' for 3 instances)")
    parser.add_argument("--shuffle", action="store_true", default=True, help="Shuffle dataset mixture")
    args = parser.parse_args()
    
    tokenizer = Tokenizer()
    config = Config.from_yaml(yaml_path=args.config)
    
    # Validate vocab size matches tokenizer
    if config.vocab_size != tokenizer.vocab_size:
        logger.error(f"Config vocab_size ({config.vocab_size}) does not match tokenizer vocab_size ({tokenizer.vocab_size})")
        logger.error(f"Please update the config file to set vocab_size: {tokenizer.vocab_size}")
        raise ValueError(f"Vocab size mismatch: config={config.vocab_size}, tokenizer={tokenizer.vocab_size}")
    
    device = Device(config)
    
    dataset_map = {
        "arc": lambda: ARC(subset="ARC-Challenge", split="train"),
        "mmlu": lambda: MMLU(subset="all", split="auxiliary_train"),
        "gsm8k": lambda: GSM8K(subset="main", split="train"),
        "humaneval": lambda: HumanEval(),
        "smoltalk": lambda: SmolTalk(split="train"),
        "spellingbee": lambda: SpellingBee(size=1000, split="train"),
        "simplespelling": lambda: SimpleSpelling(size=1000, split="train"),
    }
    
    train_datasets = []
    for spec in args.datasets:
        train_datasets.extend(parse_dataset_spec(spec, dataset_map))
    
    eval_dataset_map = {
        "arc": lambda: ARC(subset="ARC-Challenge", split="test"),
        "mmlu": lambda: MMLU(subset="all", split="test"),
        "gsm8k": lambda: GSM8K(subset="main", split="test"),
        "humaneval": lambda: HumanEval(),
        "spellingbee": lambda: SpellingBee(size=100, split="test"),
        "simplespelling": lambda: SimpleSpelling(size=100, split="test"),
    }
    
    unique_dataset_names = set()
    for ds in train_datasets:
        ds_name = type(ds).__name__.lower()
        unique_dataset_names.add(ds_name)
    
    eval_datasets = []
    for name in unique_dataset_names:
        if name in eval_dataset_map:
            ds = eval_dataset_map[name]()
            if isinstance(ds, SFTEvalDataset):
                eval_datasets.append(ds)
    
    logger.info(f"Training datasets: {args.datasets}")
    logger.info(f"Total training examples: {sum(len(ds) for ds in train_datasets):,}")
    
    model: nn.Module
    if config.use_meta_device:
        logger.info("Initializing model on meta device...")
        with torch.device('meta'):
            model = Gemma(config)
        logger.info(f"Model parameters: {model.get_num_params():,}")
        model = device.move_to_device(model, from_meta=True)
    else:
        model = Gemma(config)
        logger.info(f"Model parameters: {model.get_num_params():,}")
        model = device.move_to_device(model, from_meta=False)
    
    if config.compile_model:
        model = device.compile_model(model, enabled=True)  # type: ignore[assignment]
    
    optimizer = configure_optimizer(model, config)
    
    dataloader = DistributedSFTDataloader(
        device=device,
        config=config,
        tokenizer=tokenizer,
        datasets=train_datasets,
        shuffle=args.shuffle
    )
    
    logger.info(f"Device: {device}")
    logger.info(f"Training on {device.type} with AMP={device.use_amp}")
    
    trainer = SFTTrainer(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        device=device,
        config=config,
        tokenizer=tokenizer,
        eval_datasets=eval_datasets
    )
    trainer.train()


if __name__ == "__main__":
    main()
