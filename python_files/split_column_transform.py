from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCols, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.functions import col, concat, lit, split, when


class SplitColumnTransform(
    Transformer,
    HasInputCols,
    HasOutputCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    @keyword_only
    def __init__(self, inputCols=None, outputCol=None):
        super(SplitColumnTransform, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)
        return

    @keyword_only
    def setParams(self, inputCols=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        input_cols = self.getInputCols()
        output_col = self.getOutputCol()

        # Generate a list of columns to append
        concat_list = []
        for count, column in enumerate(input_cols):
            column_func = when(col(column).isNull(), lit("")).otherwise(col(column))
            concat_list.append(column_func)
            if count < len(input_cols) - 1:
                concat_list.append(lit(" "))
        dataset = dataset.withColumn(
            output_col, split(concat(*concat_list), pattern=" ")
        )
        return dataset
