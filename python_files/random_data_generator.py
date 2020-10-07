"""
import itertools
import random
import re
import uuid
from datetime import datetime, timedelta

import numpy as np
"""
import pandas as pd


class RandomDataGenerator(object):
    def __init__(self, data_file=None):
        if data_file:
            self.data = pd.read_csv(data_file)

    def main(self):
        pass
        # data_df = pd.DataFrame('')
        # return data_df


if __name__ == "__main__":
    pass
# data_df = RandomDataGenerator().main()
