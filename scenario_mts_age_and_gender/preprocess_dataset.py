import argparse
import os
import re
from typing import Tuple, List

import pandas as pd
import polars as pl
from tqdm import tqdm


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default="data/competition_data_final_pqt")
    parser.add_argument('--save_path', type=str, default="data/competition_data_final_pqt_preprocessed")
    args = parser.parse_args(args)
    return args

def convert_punycode(puny_domain: str) -> str:
    """
    Converts punycode to unicode

    Example:
    >>> convert_punycode('xn--22-glcqfm3bya1b.xn--p1ai')
    <<< 'грузчик22.рф'
    """
    return puny_domain.encode().decode('idna')


def remove_extension(url: str) -> str:
    # I removed `ru-an` specifically, because it was the only
    # subdomain present in mixed cyrillic-latin domains.
    # Without `ru-an` there are no mixed domains like коронавирус.ru-an.info

    #  Нужно ли удалять:
    # livejournal.com
    # turbopages.org

    url = ('.').join(url.split('.')[:-1])
    url = url[:-6] if url.endswith('ru-an') else url
    return url


def preprocess_url(url: str) -> str:
    if is_punycode(url):
        url = convert_punycode(url)
    url = url.lower()
    url = remove_extension(url)
    return url


def is_punycode(s: str) -> bool:
    return s.startswith('xn--')


def is_url_number(url: str) -> bool:
    return set(url) <= set('-0123456789')


def preprocess_df(df: pl.DataFrame) -> pl.DataFrame:
    df = df.filter(pl.col("url_host").map_elements(lambda url: not is_url_number(url), return_dtype=pl.Boolean))
    df = df.with_columns(pl.col("url_host").map_elements(preprocess_url, return_dtype=pl.String).alias("url_host_preprocesed"))
    df = df.drop("url_host")
    return df


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    for file in tqdm(os.listdir(args.dataset_path)):
        if not file.endswith('.parquet'):
            continue
        path = os.path.join(args.dataset_path, file)
        df = pl.read_parquet(path)
        df = preprocess_df(df)
        df.write_parquet(os.path.join(args.save_path, file))