import torch
import torch.nn as nn
from torch.optim import Optimizer
from goda.dataloader import DistributedDataloader, DistributedPretrainDataloader
from goda.device import Device
from goda.config import Config
from goda.tokenizer import Tokenizer
from goda.gemma import Gemma, configure_optimizer
from goda.logger import logger
from goda.pretrain import Trainer

tokenizer = Tokenizer()

config = Config(
    embed_dim=768,
    hidden_dim=3072,
    seq_length=512,
    vocab_size=tokenizer.vocab_size,
    n_layers=12,
    mixed_precision=True,
    compile_model=True,
    use_meta_device=True
)

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
    tokenizer=tokenizer
)

if __name__ == "__main__":
    logger.info(f"\n{'='*50}")
    logger.info(f"Device: {device}")
    logger.info(f"Training on {device.type} with AMP={device.use_amp}")
    logger.info(f"{'='*50}\n")
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        device=device,
        config=config
    )

    trainer.train()


