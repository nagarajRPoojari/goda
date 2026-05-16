from datasets import load_dataset

ds = load_dataset("HuggingFaceTB/smol-smoltalk", split="test").shuffle(seed=42)
print(f"✓ Dataset loaded successfully with {len(ds)} examples")


for c in ds:
    print(c)
    print("*" * 50)

