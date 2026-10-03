from pathlib import Path

import webdataset as wds


def iter_samples(path: Path = Path("data/webdataset/train.tar")):
    dataset = wds.WebDataset(str(path), shardshuffle=False).decode()
    for sample in dataset:
        yield sample["json"]


if __name__ == "__main__":
    count = 0
    first = None
    for sample in iter_samples():
        if first is None:
            first = sample
        count += 1
    print(count, first)
