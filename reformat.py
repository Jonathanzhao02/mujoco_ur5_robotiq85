import h5py
from pathlib import Path

if __name__ == '__main__':
    valid = 0

    for i in range(1,2000 + 1):
        top_dir = Path(f'data/demo{i}')
        top_dir.mkdir(parents=True, exist_ok=True)
        top_dir.joinpath('states.data').symlink_to(Path('demos').joinpath(f'demo{i}.data').resolve())
        top_dir.joinpath('imgs').symlink_to(Path('demos').joinpath(f'demo{i}_imgs').resolve())
