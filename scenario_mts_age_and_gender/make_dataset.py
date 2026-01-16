import sys; import os; sys.path.insert(1, os.path.dirname(os.getcwd()))

from ptls_extension_2024_research.make_datasets_spark import DatasetConverter


if __name__ == '__main__':
    DatasetConverter().run()
