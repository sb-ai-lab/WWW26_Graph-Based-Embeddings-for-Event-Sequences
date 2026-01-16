import os

import pandas as pd
from tqdm import tqdm

PARQUET_DF_PATH = './data/original_data/competition_data_final.parquet'
OUTPUT_PATH = "data/urls.txt"


if __name__ == "__main__":
    parquet_parts = list(set(os.listdir(PARQUET_DF_PATH)) - set(['_SUCCESS']))

    urls = set()
    for parquet_part in tqdm(parquet_parts):
        df = pd.read_parquet(os.path.join(PARQUET_DF_PATH, parquet_part))
        urls.update(set(df['url_host']))
        
    print(f"{len(urls) = }")

    with open(OUTPUT_PATH, "w") as file:
        file.write("\n".join(urls) + "\n")
