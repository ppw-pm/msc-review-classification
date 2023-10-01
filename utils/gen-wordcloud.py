import argparse

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

"""
Generate a wordcloud based on a list of topics (topic reference)
"""


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="python %(prog)s [-h] [INPUT_FILE]",
        description="Utility script that generates a wordcloud image based on a csv file generated from topic modelling",
    )

    parser.add_argument(
        "input_file", help="Path to topic list file (relative to script)"
    )

    return parser


args_parser = init_argparse()
args = args_parser.parse_args()

# Load the topics DataFrame you saved earlier
topics_csv_file_path = args.input_file
topics_df = pd.read_csv(topics_csv_file_path)

# Concatenate all top terms into a single string
all_top_terms = ", ".join(topics_df["Top Terms"])

# Generate and display a word cloud for all top terms
wordcloud = WordCloud(
    width=800, height=400, background_color="white", colormap="viridis"
).generate(all_top_terms)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
