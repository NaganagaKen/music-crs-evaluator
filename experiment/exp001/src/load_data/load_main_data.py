from __future__ import annotations

from typing import Any

import polars as pl

def load_user_embeddings() -> pl.DataFrame:
    splits = {'train': 'data/train-00000-of-00001.parquet', 'test_warm': 'data/test_warm-00000-of-00001.parquet', 'test_cold': 'data/test_cold-00000-of-00001.parquet'}
    df_train = pl.read_parquet("hf://datasets/talkpl-ai/TalkPlayData-Challenge-User-Embeddings/" + splits["train"])
    df_test_warm = pl.read_parquet("hf://datasets/talkpl-ai/TalkPlayData-Challenge-User-Embeddings/" + splits["test_warm"])
    df_test_cold = pl.read_parquet("hf://datasets/talkpl-ai/TalkPlayData-Challenge-User-Embeddings/" + splits["test_cold"])
    
    df_train = df_train.with_columns(pl.lit("train").alias("split"))
    df_test_warm = df_test_warm.with_columns(pl.lit("test_warm").alias("split"))
    df_test_cold = df_test_cold.with_columns(pl.lit("test_cold").alias("split"))

    return pl.concat([df_train, df_test_warm, df_test_cold], how="vertical")


def load_dev_data() -> pl.DataFrame:
    splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
    df_train = pl.read_parquet("hf://datasets/talkpl-ai/TalkPlayData-Challenge-Dataset/" + splits["train"])
    df_test = pl.read_parquet("hf://datasets/talkpl-ai/TalkPlayData-Challenge-Dataset/" + splits["test"])
    df_train = df_train.with_columns(pl.lit("train").alias("split_dev"))
    df_test = df_test.with_columns(pl.lit("test").alias("split_dev"))
    return pl.concat([df_train, df_test], how="vertical")


def load_user_metadata() -> pl.DataFrame:
    df = pl.read_parquet("hf://datasets/talkpl-ai/TalkPlayData-Challenge-User-Metadata/data/all_users-00000-of-00001.parquet")
    return df
