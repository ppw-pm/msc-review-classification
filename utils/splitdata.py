import argparse
import random

import pandas as pd

"""
Util script to split processed (clean) data into multiple files in an ubiased way.
"""


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="python %(prog)s [-h] [INPUT_FILE] [--output-path <folder>]",
        description="Utility script that splits clean data into multiple files in an unbiased way",
        epilog="Script will create two distinct files in the specified output path",
    )

    parser.add_argument(
        "input_file", help="Path to the file to process (relative to script)"
    )

    parser.add_argument("--output-path", "-o", help="Path to output processed text")

    return parser


args_parser = init_argparse()
args = args_parser.parse_args()
input_file = args.input_file
output_path = args.output_path

# Load your CSV data into a DataFrame
data = pd.read_csv(input_file, index_col="order")

# Shuffle the DataFrame to randomize the order of rows
data = data.sample(frac=1, random_state=42).reset_index(drop=True, names="order")
# Calculate the index to split at, weight is 33% to generate 3 files.
start = 0
stop = None
step = 3  # number of 'sets' to split data in
first_set = data.iloc[start:stop:step].reset_index(drop=True)
first_set.index = range(1, len(first_set) + 1)
second_set = data.iloc[start + 1 : stop : step].reset_index(drop=True)
second_set.index = range(1, len(second_set) + 1)
third_set = data.iloc[start + 2 : stop : step].reset_index(drop=True)
third_set.index = range(1, len(third_set) + 1)

first_set.to_csv(
    f"{output_path}/processed_text_set_1.csv", index=True, index_label="order"
)
second_set.to_csv(
    f"{output_path}/processed_text_set_2.csv", index=True, index_label="order"
)
third_set.to_csv(
    f"{output_path}/processed_text_set_3.csv", index=True, index_label="order"
)
