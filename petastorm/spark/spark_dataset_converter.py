from petastorm import make_batch_reader
from petastorm.tf_utils import make_petastorm_dataset
from petastorm.unischema import Unischema
from pyspark.sql.dataframe import DataFrame

import numpy as np
import os
import shutil
import tensorflow as tf
import uuid

assert(tf.version.VERSION == '1.15.0')

# Config TODO: we should use DBFS FUSE dir: /dbfs/ml/...
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


class SparkDatasetConverter:
    """
    The SparkDatasetConverter class manages the intermediate files when converting a SparkDataFrame to a
    Tensorflow Dataset or PyTorch DataLoader.
    """
    def __init__(self, cache_file_path: str, unischema: Unischema = None):
        """
        :param cache_file_path: The path to store the cache files.
        :param unischema: Unischema that contains metadata of the dataset. TODO use this param.
        """
        self.cache_file_path = cache_file_path
        self.unischema = unischema
        self.dataset = None

    def make_tf_dataset(self):
        reader = make_batch_reader("file://" + self.cache_file_path)
        self.dataset = make_petastorm_dataset(reader)
        return tf_dataset_context_manager(self)

    def close(self):
        """
        Close the underlying reader. It would be called by the context manager at exit.
        :return:
        """
        pass

    def delete(self):
        """
        Delete cache files at self.cache_file_path.
        :return:
        """
        shutil.rmtree(self.cache_file_path, ignore_errors=True)


class tf_dataset_context_manager:

    def __init__(self, converter: SparkDatasetConverter):
        self.converter = converter

    def __enter__(self) -> tf.data.Dataset:
        return self.converter.dataset

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.converter.close()


def _get_uuid():
    """
    Generate a UUID from a host ID, sequence number, and the current time.
    :return: a string of UUID.
    """
    return str(uuid.uuid1())


def _cache_df_or_retrieve_cache_path(df: DataFrame, root_dir: str) -> str:
    """
    Check whether the df is cached.
    If so, return the existing cache file path.
    If not, cache the df into the configured cache dir in parquet format and return the cache file path.
    :param df: SparkDataFrame
    :param root_dir: the directory for the saved parquet file, could be local, hdfs, dbfs, ...
    :return: the path of the saved parquet file
    """
    # Todo: add cache management, currently always use a new dir.
    uuid_str = _get_uuid()
    save_to_dir = os.path.join(root_dir, uuid_str)
    df.write.mode("overwrite") \
        .option("parquet.block.size", 1024 * 1024) \
        .parquet(save_to_dir)

    # remove _xxx files, which will break `pyarrow.parquet` loading
    underscore_files = [f for f in os.listdir(save_to_dir) if f.startswith("_")]
    for f in underscore_files:
        os.remove(os.path.join(save_to_dir, f))
    return save_to_dir


def make_spark_converter(df: DataFrame, unischema: Unischema = None, root_dir=CACHE_DIR) -> SparkDatasetConverter:
    """
    This class will check whether the df is cached.
    If so, it will use the existing cache file path to construct a SparkDatasetConverter.
    If not, Materialize the df into the configured cache dir in parquet format and use the cache file path to
    construct a SparkDatasetConverter.
    :param df: The DataFrame to materialize.
    :param unischema: An instance of Unischema that contains metadata such as type, shape and codec.
    :param root_dir: The root dir to store cache files.
    :return: a SparkDatasetConverter
    """
    cache_file_path = _cache_df_or_retrieve_cache_path(df, root_dir)
    return SparkDatasetConverter(cache_file_path, unischema)
