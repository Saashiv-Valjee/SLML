import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

DATA_DIR = Path(".")


def count_frames(rel_path):
    path = DATA_DIR / rel_path

    df = pd.read_parquet(path, columns=["frame"])

    return df["frame"].nunique()


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")

    paths = train["path"].tolist()

    n_workers = max(1, cpu_count() - 1)

    print(f"Using {n_workers} workers")
    print(f"Counting frames for {len(paths)} parquet files")

    with Pool(n_workers) as pool:
        frame_counts = list(
            tqdm(
                pool.imap_unordered(count_frames, paths, chunksize=128),
                total=len(paths),
            )
        )

    frame_counts = np.array(frame_counts, dtype=np.int32)

    np.save("frame_counts.npy", frame_counts)

    print("\nFrame statistics")
    print("----------------")
    print("min   :", frame_counts.min())
    print("mean  :", frame_counts.mean())
    print("median:", np.median(frame_counts))
    print("max   :", frame_counts.max())

    print("\nPercentiles")
    print("-----------")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"{p:2d}% :", np.percentile(frame_counts, p))

    print("\nRecommended N_FRAMES candidates")
    print("-------------------------------")
    print("Conservative :", int(np.percentile(frame_counts, 50)))
    print("Balanced     :", int(np.percentile(frame_counts, 75)))
    print("High coverage:", int(np.percentile(frame_counts, 90)))


if __name__ == "__main__":
    main()