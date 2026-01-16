import argparse

from tqdm import tqdm
import polars as pl
import os

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default="data/original_format_data/competition_data_final_preprocessed.parquet")
    parser.add_argument('--save_path', type=str, default="data/competition_data_edges.parquet")
    args = parser.parse_args(args)
    return args


def drop_duplicates_and_sum(df: pl.DataFrame) -> pl.DataFrame:
    # Group by c1 and c2, and aggregate the sum of cnt
    result = df.group_by(['user_id', 'url_host_preprocesed']).agg(pl.sum('request_cnt'))
    return result


def preprocess_dfs(dir):
    res = []
    for file in tqdm(os.listdir(dir)):
        if not file.endswith('.parquet'):
            continue
        path = os.path.join(dir, file)
        df_tr = pl.read_parquet(path)
        df_tr_no_duplicates = drop_duplicates_and_sum(df_tr)
        res.append(df_tr_no_duplicates)
    df_final = pl.concat(res, how="vertical")
    df_final = drop_duplicates_and_sum(df_final)
    return df_final


if __name__ == '__main__':
    args = parse_args()
    df_final = preprocess_dfs(args.dataset_path)
    df_final.write_parquet(args.save_path)
