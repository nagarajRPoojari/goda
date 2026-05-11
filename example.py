import torch
import torch.nn as nn
import argparse
from torch.optim import Optimizer
from goda.dataloader import DistributedDataloader, DistributedPretrainDataloader
from goda.device import Device
from goda.config import DEFAULT_CONFIG, Config
from goda.tokenizer import Tokenizer
from goda.gemma import Gemma, configure_optimizer
from goda.logger import logger
from goda.pretrain import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train Gemma model with distributed dataloader")
    parser.add_argument("--config", type=str, default="config_d12.yaml", help="Path to config YAML file")
    parser.add_argument("--min-shards", type=int, default=2, help="Minimum shards required before starting training")
    parser.add_argument("--max-shards-to-wait", type=int, default=-1, help="Maximum shards to wait for (-1 = wait indefinitely)")
    args = parser.parse_args()
    
    tokenizer = Tokenizer()
    config = Config.from_yaml(yaml_path=args.config)
    
    device = Device(config)
    
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
    
    dataloader: DistributedDataloader = DistributedPretrainDataloader(
        device=device,
        config=config,
        tokenizer=tokenizer,
        min_shards_required=args.min_shards,
        max_shards_to_wait=args.max_shards_to_wait
    )
    
    logger.info(f"Device: {device}")
    logger.info(f"Training on {device.type} with AMP={device.use_amp}")
    logger.info(f"Dataloader min_shards: {args.min_shards}")
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        device=device,
        config=config,
        tokenizer=tokenizer
    )
    trainer.train()


if __name__ == "__main__":
    main()
