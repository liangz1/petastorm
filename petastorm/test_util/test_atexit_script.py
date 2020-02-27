from pyspark.sql import SparkSession

from petastorm.spark.spark_dataset_converter import make_spark_converter

spark = SparkSession.builder.getOrCreate()
df = spark.range(10)
converter = make_spark_converter(df, 'file:///tmp/spark_converter_test_atexit')
f = open("/tmp/spark_converter_test_atexit/output", "w")
f.write(converter.cache_file_path)
f.close()
