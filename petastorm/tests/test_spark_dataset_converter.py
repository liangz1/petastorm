#  Copyright (c) 2020 Databricks, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager

import numpy as np
import pytest
import tensorflow as tf
from pyspark.sql import SparkSession
from pyspark.sql.types import (BinaryType, BooleanType, ByteType, DoubleType,
                               FloatType, IntegerType, LongType, ShortType,
                               StringType, StructField, StructType)
from six.moves.urllib.parse import urlparse

from petastorm import make_batch_reader
from petastorm.fs_utils import FilesystemResolver
from petastorm.spark import (SparkDatasetConverter, make_spark_converter,
                             spark_dataset_converter)
from petastorm.spark.spark_dataset_converter import (
    _check_url, _get_parent_cache_dir_url, _make_sub_dir_url,
    register_delete_dir_handler)


class TestContext(object):

    def __init__(self):
        self.spark = SparkSession.builder \
            .master("local[2]") \
            .appName("petastorm.spark tests") \
            .getOrCreate()
        self.tempdir = tempfile.mkdtemp('_spark_converter_test')
        self.temp_url = 'file://' + self.tempdir.replace(os.sep, '/')
        self.spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF,
                            self.temp_url)

    def tear_down(self):
        self.spark.stop()


@pytest.fixture(scope='module')
def test_ctx():
    ctx = TestContext()
    yield ctx
    ctx.tear_down()


def test_primitive(test_ctx):
    schema = StructType([
        StructField("bool_col", BooleanType(), False),
        StructField("float_col", FloatType(), False),
        StructField("double_col", DoubleType(), False),
        StructField("short_col", ShortType(), False),
        StructField("int_col", IntegerType(), False),
        StructField("long_col", LongType(), False),
        StructField("str_col", StringType(), False),
        StructField("bin_col", BinaryType(), False),
        StructField("byte_col", ByteType(), False),
    ])
    df = test_ctx.spark.createDataFrame(
        [(True, 0.12, 432.1, 5, 5, 0, "hello",
          bytearray(b"spark\x01\x02"), -128),
         (False, 123.45, 0.987, 9, 908, 765, "petastorm",
          bytearray(b"\x0012345"), 127)],
        schema=schema).coalesce(1)
    # If we use numPartition > 1, the order of the loaded dataset would
    # be non-deterministic.
    expected_df = df.collect()

    converter = make_spark_converter(df)
    with converter.make_tf_dataset() as dataset:
        iterator = dataset.make_one_shot_iterator()
        tensor = iterator.get_next()
        with tf.Session() as sess:
            ts = sess.run(tensor)
            # TODO: we will improve the test once the batch_size argument
            #  added.
            # Now we only have one batch.
        for i in range(converter.dataset_size):
            for col in df.schema.names:
                actual_ele = getattr(ts, col)[i]
                expected_ele = expected_df[i][col]
                if col == "str_col":
                    actual_ele = actual_ele.decode()
                if col == "bin_col":
                    actual_ele = bytearray(actual_ele)
                assert expected_ele == actual_ele

        assert len(expected_df) == len(converter)

    assert np.bool_ == ts.bool_col.dtype.type
    assert np.float32 == ts.float_col.dtype.type
    assert np.float64 == ts.double_col.dtype.type
    assert np.int16 == ts.short_col.dtype.type
    assert np.int32 == ts.int_col.dtype.type
    assert np.int64 == ts.long_col.dtype.type
    assert np.object_ == ts.str_col.dtype.type
    assert np.object_ == ts.bin_col.dtype.type


def test_delete(test_ctx):
    df = test_ctx.spark.createDataFrame([(1, 2), (4, 5)], ["col1", "col2"])
    # TODO add test for hdfs url
    converter = make_spark_converter(df)
    local_path = urlparse(converter.cache_dir_url).path
    assert os.path.exists(local_path)
    converter.delete()
    assert not os.path.exists(local_path)


def test_atexit(test_ctx):
    lines = """
    from petastorm.spark import SparkDatasetConverter, make_spark_converter
    from pyspark.sql import SparkSession
    import os
    spark = SparkSession.builder.getOrCreate()
    spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, '{temp_url}')
    df = spark.createDataFrame([(1, 2),(4, 5)], ["col1", "col2"])
    converter = make_spark_converter(df)
    f = open(os.path.join('{tempdir}', 'test_atexit.out'), "w")
    f.write(converter.cache_dir_url)
    f.close()
    """.format(tempdir=test_ctx.tempdir, temp_url=test_ctx.temp_url)
    code_str = "; ".join(
        line.strip() for line in lines.strip().splitlines())
    ret_code = subprocess.call([sys.executable, "-c", code_str])
    assert 0 == ret_code
    with open(os.path.join(test_ctx.tempdir, 'test_atexit.out')) as f:
        cache_dir_url = f.read()

    fs = FilesystemResolver(cache_dir_url).filesystem()
    assert not fs.exists(urlparse(cache_dir_url).path)


def test_set_delete_handler(test_ctx):
    def test_delete_handler(dir_url):
        raise RuntimeError('Not implemented delete handler.')

    register_delete_dir_handler(test_delete_handler)

    with pytest.raises(RuntimeError, match='Not implemented delete handler'):
        spark_dataset_converter._delete_dir_handler(test_ctx.temp_url)

    # Restore default delete handler (other test will use it)
    register_delete_dir_handler(None)


def _get_compression_type(data_url):
    files = os.listdir(urlparse(data_url).path)
    pq_files = list(filter(lambda x: x.endswith('.parquet'), files))
    filename_splits = pq_files[0].split('.')
    if len(filename_splits) == 2:
        return "uncompressed"
    else:
        return filename_splits[1]


def test_compression(test_ctx):
    df1 = test_ctx.spark.range(10)

    converter1 = make_spark_converter(df1)
    assert "uncompressed" == \
           _get_compression_type(converter1.cache_dir_url).lower()

    converter2 = make_spark_converter(df1, compression_codec="snappy")
    assert "snappy" == \
           _get_compression_type(converter2.cache_dir_url).lower()


def test_df_caching(test_ctx):
    df1 = test_ctx.spark.range(10)
    df2 = test_ctx.spark.range(10)
    df3 = test_ctx.spark.range(20)

    converter1 = make_spark_converter(df1)
    converter2 = make_spark_converter(df2)
    assert converter1.cache_dir_url == converter2.cache_dir_url

    converter3 = make_spark_converter(df3)
    assert converter1.cache_dir_url != converter3.cache_dir_url

    converter11 = make_spark_converter(
        df1, parquet_row_group_size_bytes=8 * 1024 * 1024)
    converter21 = make_spark_converter(
        df1, parquet_row_group_size_bytes=16 * 1024 * 1024)
    assert converter11.cache_dir_url != converter21.cache_dir_url

    converter12 = make_spark_converter(df1, compression_codec=None)
    converter22 = make_spark_converter(df1, compression_codec="snappy")
    assert converter12.cache_dir_url != converter22.cache_dir_url


def test_check_url():
    with pytest.raises(ValueError, match='scheme-less'):
        _check_url('/a/b/c')


def test_make_sub_dir_url():
    assert _make_sub_dir_url('file:///a/b', 'c') == 'file:///a/b/c'
    assert _make_sub_dir_url('hdfs:/a/b', 'c') == 'hdfs:/a/b/c'
    assert _make_sub_dir_url('hdfs://nn1:9000/a/b',
                             'c') == 'hdfs://nn1:9000/a/b/c'


def test_pickling_remotely(test_ctx):
    df1 = test_ctx.spark.range(100, 101)
    converter1 = make_spark_converter(df1)

    def map_fn(_):
        with converter1.make_tf_dataset() as dataset:
            iterator = dataset.make_one_shot_iterator()
            tensor = iterator.get_next()
            with tf.Session() as sess:
                ts = sess.run(tensor)
        return getattr(ts, 'id')[0]

    result = test_ctx.spark.sparkContext.parallelize(range(1), 1) \
                                        .map(map_fn).collect()[0]
    assert result == 100


def test_change_cache_dir_raise_error(test_ctx):
    temp_url2 = 'file://' + tempfile.mkdtemp('_spark_converter_test2').replace(
        os.sep, '/')
    test_ctx.spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF,
                            temp_url2)

    with pytest.raises(RuntimeError,
                       match="{} has been set to be".format(
                           SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF)):
        _get_parent_cache_dir_url()

    # restore conf (other test need use it)
    test_ctx.spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF,
                            test_ctx.temp_url)
    assert test_ctx.temp_url == _get_parent_cache_dir_url()


def test_torch_primitive(test_ctx):
    import torch

    schema = StructType([
        StructField("bool_col", BooleanType(), False),
        StructField("float_col", FloatType(), False),
        StructField("double_col", DoubleType(), False),
        StructField("short_col", ShortType(), False),
        StructField("int_col", IntegerType(), False),
        StructField("long_col", LongType(), False),
        StructField("byte_col", ByteType(), False),
    ])
    df = test_ctx.spark.createDataFrame(
        [(True, 0.12, 432.1, 5, 5, 0, -128),
         (False, 123.45, 0.987, 9, 908, 765, 127)],
        schema=schema).coalesce(1)
    # If we use numPartition > 1, the order of the loaded dataset would
    # be non-deterministic.
    expected_df = df.collect()

    converter = make_spark_converter(df)
    batch = None
    with converter.make_torch_dataloader(num_epochs=1) as dataloader:
        for i, batch in enumerate(dataloader):
            # default batch_size = 1
            for col in df.schema.names:
                actual_ele = batch[col][0]
                expected_ele = expected_df[i][col]
                assert expected_ele == actual_ele

        assert len(expected_df) == len(converter)
    assert torch.uint8 == batch["bool_col"].dtype
    assert torch.int8 == batch["byte_col"].dtype
    assert torch.float32 == batch["double_col"].dtype
    assert torch.float32 == batch["float_col"].dtype
    assert torch.int32 == batch["int_col"].dtype
    assert torch.int64 == batch["long_col"].dtype
    assert torch.int16 == batch["short_col"].dtype


def test_torch_pickling_remotely(test_ctx):
    df1 = test_ctx.spark.range(100, 101)
    converter1 = make_spark_converter(df1)

    def map_fn(_):
        with converter1.make_torch_dataloader(num_epochs=1) as dataloader:
            for batch in dataloader:
                ret = batch["id"][0]
        return ret

    result = test_ctx.spark.sparkContext.parallelize(range(1), 1) \
        .map(map_fn).collect()[0]
    assert result == 100


def test_torch_batch_size(test_ctx):
    df = test_ctx.spark.range(8)
    conv = make_spark_converter(df)
    batch_size = 2
    with conv.make_torch_dataloader(batch_size=batch_size,
                                    num_epochs=1) as dataloader:
        for batch in dataloader:
            assert batch_size == batch['id'].shape[0]


def test_torch_transform_spec(test_ctx):
    df = test_ctx.spark.range(8)
    conv = make_spark_converter(df)

    from torchvision import transforms
    from petastorm import TransformSpec

    def _transform_row(df_row):
        scale_tranform = transforms.Compose([
            transforms.Lambda(lambda x: x * 0.1),
        ])
        return scale_tranform(df_row)

    transform = TransformSpec(_transform_row)
    with conv.make_torch_dataloader(transform_spec=transform,
                                    num_epochs=1) as dataloader:
        for batch in dataloader:
            assert min(batch['id']) >= 0 and max(batch['id']) < 1


def test_torch_unexpected_param(test_ctx):
    df = test_ctx.spark.range(8)
    conv = make_spark_converter(df)

    with pytest.raises(TypeError, match="unexpected keyword argument 'xyz'"):
        with conv.make_torch_dataloader(xyz=1) as _:
            pass


@contextmanager
def mock_torch_make_batch_reader():
    captured_args = []
    import petastorm.spark

    def mock_fn(dataset_url, **kwargs):
        reader_args = {'dataset_url': dataset_url}
        reader_args.update(kwargs)
        captured_args.append(reader_args)
        return make_batch_reader(dataset_url, **kwargs)

    petastorm.spark.spark_dataset_converter.make_batch_reader = mock_fn
    try:
        yield captured_args
    finally:
        petastorm.spark.spark_dataset_converter.make_batch_reader = \
            make_batch_reader


def test_torch_advanced_params(test_ctx):
    SHARD_COUNT = 3
    df = test_ctx.spark.range(100).repartition(SHARD_COUNT)
    conv = make_spark_converter(df)

    with mock_torch_make_batch_reader() as captured_args:
        with conv.make_torch_dataloader(reader_pool_type='dummy', cur_shard=1,
                                        shard_count=SHARD_COUNT) as _:
            pass
        peta_args = captured_args[0]
        assert peta_args['reader_pool_type'] == 'dummy' and \
            peta_args['cur_shard'] == 1 and \
            peta_args['shard_count'] == SHARD_COUNT and \
            peta_args['num_epochs'] is None and \
            ('workers_count' not in peta_args)

    # Test default value overridden arguments.
    with mock_torch_make_batch_reader() as captured_args:
        with conv.make_torch_dataloader(num_epochs=1, workers_count=2) as _:
            pass
        peta_args = captured_args[0]
        assert peta_args['num_epochs'] == 1 and peta_args['workers_count'] == 2
