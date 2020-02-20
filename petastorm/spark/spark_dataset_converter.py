from petastorm import make_batch_reader
from petastorm.tf_utils import make_petastorm_dataset
from pyspark.sql.dataframe import DataFrame

import numpy as np
import os
import tensorflow as tf

assert(tf.version.VERSION == '1.15.0')

# Config
CACHE_DIR = "/tmp/tf"

# primitive type mapping
type_map = {
    "boolean": np.bool_,
    "byte": np.int8,  # 8-bit signed
    "double": np.float64,
    "float": np.float32,
    "integer": np.int32,
    "long": np.int64,
    "short": np.int16
}


def save_df_as_parquet(df: DataFrame, dir_path: str) -> str:
    """
    Save a SparkDataFrame in parquet format in dbfs (Liang: during dev: the local file system).
    :param df: SparkDataFrame
    :param dir_path: the directory for the saved parquet file, could be local, hdfs, dbfs, ...
    :return: the path of the saved parquet file
    """
    df.write.mode("overwrite") \
      .option("parquet.block.size", 1024 * 1024) \
      .parquet(CACHE_DIR)

    # remove _xxx files, which will break `pyarrow.parquet` loading
    underscore_files = [f for f in os.listdir(dir_path) if f.startswith("_")]
    for f in underscore_files:
        os.remove(os.path.join(dir_path, f))
    return CACHE_DIR


def spark_df_to_tf_dataset(df: DataFrame) -> tf.data.Dataset:
    # save as parquet
    saved_path = save_df_as_parquet(df, CACHE_DIR)
    # load parquet into petastorm reader
    reader = make_batch_reader("file://"+saved_path)
    dataset = make_petastorm_dataset(reader)
    return dataset
