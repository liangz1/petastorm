#
# Copyright 2017 Databricks, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, \
    BooleanType, FloatType, ShortType, IntegerType, LongType, DoubleType

from petastorm.spark.spark_dataset_converter import spark_df_to_tf_dataset

import tensorflow as tf
import numpy as np
import unittest


class TfConverterTest(unittest.TestCase):

    def setUp(self) -> None:
        self.spark = SparkSession.builder \
            .master("local[2]") \
            .appName("petastorm.spark tests") \
            .getOrCreate()

    def test_primitive(self):
        # test primitive columns
        schema = StructType([
            StructField("bool_col", BooleanType(), False),
            StructField("float_col", FloatType(), False),
            StructField("double_col", DoubleType(), False),
            StructField("short_col", ShortType(), False),
            StructField("int_col", IntegerType(), False),
            StructField("long_col", LongType(), False)
        ])
        df = self.spark.createDataFrame([
            (True, 0.12, 432.1, 5, 5, 0),
            (False, 123.45, 0.987, 9, 908, 765)],
            schema=schema)

        dataset = spark_df_to_tf_dataset(df)

        iterator = dataset.make_one_shot_iterator()
        tensor = iterator.get_next()
        with tf.Session() as sess:
            ts = sess.run(tensor)

        assert (ts.bool_col.dtype.type == np.bool_)
        assert (ts.float_col.dtype.type == np.float32)
        assert (ts.double_col.dtype.type == np.float64)
        assert (ts.short_col.dtype.type == np.int16)
        assert (ts.int_col.dtype.type == np.int32)
        assert (ts.long_col.dtype.type == np.int64)
