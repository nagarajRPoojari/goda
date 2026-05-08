import argparse
import os
from pathlib import Path
from typing import Optional
import sys
import ssl
import urllib.request
from goda.logger import logger

try:
    from huggingface_hub import hf_hub_download
    USE_HF_HUB = True
except ImportError:
    USE_HF_HUB = False
    logger.info("Warning: huggingface_hub not installed. Using urllib with SSL context.")
    logger.info("For better reliability, install with: pip install huggingface-hub")

def download_file_urllib(url: str, destination: Path, show_progress: bool = True) -> None:
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    def progress_hook(block_num, block_size, total_size):
        if not show_progress or total_size <= 0:
            return
        downloaded = block_num * block_size
        percent = min(downloaded * 100.0 / total_size, 100.0)
        sys.stdout.write(f"\r  Progress: {percent:.1f}% ({downloaded / (1024**2):.1f}MB / {total_size / (1024**2):.1f}MB)")
        sys.stdout.flush()
    
    try:
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, destination, progress_hook if show_progress else None)
        if show_progress:
            logger.info("")  # New line after progress
    except Exception as e:
        raise RuntimeError(f"Failed to download {url}: {e}")


def download_file_hf(repo_id: str, filename: str, destination: Path) -> None:
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=destination.parent,
            local_dir_use_symlinks=False
        )

        if Path(downloaded_path) != destination:
            Path(downloaded_path).rename(destination)
    except Exception as e:
        raise RuntimeError(f"Failed to download {filename}: {e}")


def download_climbmix_dataset(
    output_dir: str = "data/climbmix",
    num_train_shards: int = 10,
    repo_id: str = "karpathy/climbmix-400b-shuffle",
    base_url: str = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main",
    shard_pattern: str = "shard_{:05d}.parquet",
) -> None:
    output_path = Path(output_dir)
    train_dir = output_path / "train"
    val_dir = output_path / "val"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading ClimbMix dataset to: {output_path}")
    logger.info(f"Training shards: {num_train_shards}")
    logger.info(f"Validation shards: 1 (last shard)")
    logger.info(f"Using: {'huggingface_hub' if USE_HF_HUB else 'urllib with SSL context'}")
    logger.info("-" * 60)
    
    logger.info(f"\nDownloading {num_train_shards} training shards...")
    for shard_idx in range(num_train_shards):
        shard_name = shard_pattern.format(shard_idx)
        destination = train_dir / shard_name
        
        if destination.exists():
            logger.info(f"[{shard_idx + 1}/{num_train_shards}] Skipping {shard_name} (already exists)")
            continue
        
        logger.info(f"[{shard_idx + 1}/{num_train_shards}] Downloading {shard_name}...")
        try:
            if USE_HF_HUB:
                download_file_hf(repo_id, shard_name, destination)
            else:
                url = f"{base_url}/{shard_name}"
                download_file_urllib(url, destination)
            logger.info(f"  ✓ Saved to {destination}")
        except RuntimeError as e:
            logger.info(f"  ✗ Error: {e}")
            continue
    
    logger.info(f"\nDownloading validation shard...")
    val_shard_idx = num_train_shards  
    val_shard_name = shard_pattern.format(val_shard_idx)
    val_destination = val_dir / val_shard_name
    
    if val_destination.exists():
        logger.info(f"Skipping {val_shard_name} (already exists)")
    else:
        logger.info(f"Downloading {val_shard_name}...")
        try:
            if USE_HF_HUB:
                download_file_hf(repo_id, val_shard_name, val_destination)
            else:
                val_url = f"{base_url}/{val_shard_name}"
                download_file_urllib(val_url, val_destination)
            logger.info(f"  ✓ Saved to {val_destination}")
        except RuntimeError as e:
            logger.info(f"  ✗ Error: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Download complete!")
    logger.info(f"Training data: {train_dir}")
    logger.info(f"Validation data: {val_dir}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Download ClimbMix dataset shards for training and validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--output-dir", type=str, default="data/climbmix", help="Directory to save downloaded files")
    parser.add_argument("--num-train-shards", type=int, default=10, help="Number of shards to download for training")
    parser.add_argument("--repo-id", type=str, default="karpathy/climbmix-400b-shuffle",help="Hugging Face repository ID")
    parser.add_argument( "--base-url", type=str, default="https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main", help="Base URL for the dataset (used with urllib)")
    parser.add_argument("--shard-pattern", type=str, default="shard_{:05d}.parquet", help="Pattern for shard filenames (use Python format string)")
    
    args = parser.parse_args()
    
    download_climbmix_dataset(output_dir=args.output_dir, num_train_shards=args.num_train_shards, repo_id=args.repo_id, base_url=args.base_url, shard_pattern=args.shard_pattern)


if __name__ == "__main__":
    main()

