import h5py
import math
import random
from pathlib import Path

if __name__ == '__main__':
    valid = []

    train_dir = Path(f'data/train')
    val_dir = Path(f'data/val')

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1,8000 + 1):
        with h5py.File(f'demos/demo{i}.data') as f:
            if f.attrs['success']:
                valid.append(i)

    train = set(random.sample(valid, math.ceil(len(valid) * 0.8)))
    val = set(valid) - train

    for i in train:
        top_dir = train_dir.joinpath(f'demo{i}')
        top_dir.mkdir(parents=True, exist_ok=True)
        top_dir.joinpath('states.data').symlink_to(Path('demos').joinpath(f'demo{i}.data').resolve())
        top_dir.joinpath('imgs').symlink_to(Path('demos').joinpath(f'demo{i}_imgs').resolve())

    for i in val:
        top_dir = val_dir.joinpath(f'demo{i}')
        top_dir.mkdir(parents=True, exist_ok=True)
        top_dir.joinpath('states.data').symlink_to(Path('demos').joinpath(f'demo{i}.data').resolve())
        top_dir.joinpath('imgs').symlink_to(Path('demos').joinpath(f'demo{i}_imgs').resolve())
