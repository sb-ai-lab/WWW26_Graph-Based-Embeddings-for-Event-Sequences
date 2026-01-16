import argparse
import datetime
import logging
import os
from ptls.preprocessing import PysparkDataPreprocessor
from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)

class DatasetConverter:
    def __init__(self):
        self.config = None

    def parse_args(self, args=None):
        parser = argparse.ArgumentParser()

        parser.add_argument('--data_path', type=os.path.abspath)
        parser.add_argument('--trx_files', nargs='+')
        parser.add_argument('--target_files', nargs='*', default=[])
        parser.add_argument('--target_as_array', action='store_true')

        parser.add_argument('--print_dataset_info', action='store_true')
        parser.add_argument('--sample_fraction', type=float, default=None)
        parser.add_argument('--col_client_id', type=str)
        parser.add_argument('--cols_event_time', nargs='+')

        parser.add_argument('--dict', nargs='*', default=[])
        parser.add_argument('--cols_category', nargs='*', default=[])
        parser.add_argument('--cols_log_norm', nargs='*', default=[])
        parser.add_argument('--col_target', nargs='*', default=[])
        parser.add_argument('--test_size', default='0.1')
        parser.add_argument('--salt', type=int, default=42)
        parser.add_argument('--max_trx_count', type=int, default=5000)

        parser.add_argument('--output_train_path', type=os.path.abspath)
        parser.add_argument('--output_test_path', type=os.path.abspath)
        parser.add_argument('--output_test_ids_path', type=os.path.abspath)
        parser.add_argument('--save_partitioned_data', action='store_true')

        self.config = parser.parse_args(args)

    def process_data(self):
        spark = SparkSession.builder.appName("DatasetConverter").getOrCreate()
        data_path = self.config.data_path
        trx_files = [os.path.join(data_path, file) for file in self.config.trx_files]

        # Read transactions
        df = spark.read.option("header", "true").csv(trx_files)

        # Initialize the PysparkDataPreprocessor
        preprocessor = PysparkDataPreprocessor(
            col_id=self.config.col_client_id,
            col_event_time=self.config.cols_event_time[0],
            event_time_transformation='none',
            cols_category=self.config.cols_category,
            cols_numerical=self.config.cols_log_norm,
        )

        # Fit and transform the data
        features = preprocessor.fit_transform(df)

        # Save the preprocessed data
        features.write.parquet(self.config.output_train_path)

        if self.config.test_size > 0:
            # Split the dataset into training and testing sets
            train, test = features.randomSplit([1 - float(self.config.test_size), float(self.config.test_size)], seed=self.config.salt)
            train.write.parquet(self.config.output_train_path)
            test.write.parquet(self.config.output_test_path)

            test_ids = test.select(self.config.col_client_id).distinct()
            test_ids.write.csv(self.config.output_test_ids_path, header=True)
        else:
            features.write.parquet(self.config.output_train_path)

    def run(self):
        start_time = datetime.datetime.now()
        self.process_data()
        duration = datetime.datetime.now() - start_time
        logger.info(f'Data processed in {duration.seconds} sec ({duration})')

if __name__ == "__main__":
    converter = DatasetConverter()
    converter.parse_args()
    converter.run()
