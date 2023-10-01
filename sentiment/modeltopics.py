import argparse
import os
import random

import gensim
import numpy as np
import pandas as pd
from gensim import models
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk

"""
Extract topics from processed text. Each topic will be distributed throughout reviews.
"""


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="python %(prog)s [-h] [INPUT_FILE] [--output <filename>] [--reference-file <filename>] [-n]",
        description="Run topic modelling on a series of sentences that have been extracted from reviews. It will generate two files: one for topics that preserves initial text, and a table with the assignment.",
        epilog="WARNING: script will download nltk data. First run might take longer than expected. nltk data wil be downloaded in current working folder.",
    )
    parser.add_argument(
        "input_file", help="Path to the file to process (relative to script)"
    )

    parser.add_argument("--output-file", "-o", help="Path to output processed text")

    parser.add_argument(
        "--reference-file", help="Path to output reference table for assignment"
    )

    parser.add_argument(
        "--topic-num", "-n", help="Number of topics (default: 10)", default=10
    )
    return parser


args_parser = init_argparse()
args = args_parser.parse_args()

# If it doesn't exist, create an nltk folder to save data
nltk_path = f"{os.getcwd()}/nltk"
if not os.path.exists(nltk_path):
    os.makedirs(nltk_path)

nltk.data.path.append(nltk_path)
nltk.download("stopwords", nltk_path)

# Create a set of English stopwords
stopwords = set(nltk.corpus.stopwords.words("english"))

# Load the CSV file with text data and sentiment scores
input_file_path = args.input_file
df = pd.read_csv(input_file_path)

# Specify the columns
text_column = "text"
sentiment_column = "Aggregated Score"

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words="english")

# Create a Document-Term Matrix (DTM)
tfidf_matrix = tfidf_vectorizer.fit_transform(df[text_column])

# Convert the DTM to a Gensim corpus
corpus = gensim.matutils.Sparse2Corpus(tfidf_matrix.T)

# Define the number of topics
num_topics = args.topic_num

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Apply Latent Dirichlet Allocation (LDA)
lda_model = models.LdaModel(
    corpus,
    num_topics=num_topics,
    id2word=dict(enumerate(tfidf_vectorizer.get_feature_names_out())),
    passes=15,
)

# Map topic assignments to each sentence
topic_assignments = []
for i, text in enumerate(df[text_column]):
    bow = tfidf_vectorizer.transform([text])
    dense_bow = bow.toarray()[0]
    bow = [(i, value) for i, value in enumerate(dense_bow) if value > 0]
    topic_probs = lda_model.get_document_topics(bow)
    dominant_topic = max(topic_probs, key=lambda x: x[1])
    topic_id = dominant_topic[0]
    topic_assignments.append(topic_id)

# Add topic assignments to the DataFrame
df["Topic Assignment"] = topic_assignments

# Save the DataFrame to a CSV file
output_file_path = args.output_file
df.to_csv(output_file_path, index=False)

# Get and save the list of topics and their top terms to a file
topics = []
for i, topic in lda_model.show_topics(
    num_topics=num_topics, num_words=10, formatted=False
):
    topics.append((i + 1, ", ".join([word for word, _ in topic])))

topics_df = pd.DataFrame(topics, columns=["Topic Number", "Top Terms"])
reference_file_path = args.reference_file
topics_df.to_csv(reference_file_path, index=False)
