import h5py
from pathlib import Path

if __name__ == '__main__':
    valid = 0

    for i in range(1,11):
        top_dir = Path(f'chunked/demos{i}')
        top_dir.mkdir(parents=True, exist_ok=True)

        for j in range((i-1) * 200 + 1, i * 200 + 1):
            top_dir.joinpath(f'demo{j}.data').symlink_to(Path('demos').joinpath(f'demo{j}.data').resolve())
            top_dir.joinpath(f'demo{j}_imgs').symlink_to(Path('demos').joinpath(f'demo{j}_imgs').resolve())
