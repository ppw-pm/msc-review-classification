import argparse
import csv
import os

from nltk.corpus import sentiwordnet as swn

import nltk

"""
Sentiwordnet will classify processed text and identify the sentence by aggregating
score from each word. The summed up score assigns sentiment to the sentence.
"""


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="python %(prog)s [-h] [INPUT_FILE] [--output <filename>]",
        description="Run sentiwordnet on processed data. Sentiwordnet will classify processed text and identify the sentence by aggregating score from each word into a sum that assigns the sentiment to the sentence.",
        epilog="WARNING: script will download nltk data. First run might take longer than expected. nltk data wil be downloaded in current working folder.",
    )
    parser.add_argument(
        "input_file",
        help="Path to the file to process (relative to script)",
    )

    parser.add_argument(
        "--output-file",
        "-o",
        help="Path to output processed text",
    )
    return parser


def get_sentiment_score(word, pos):
    """
    Helper function that will retrieve the sentiment score for a given word
    """
    # Map NLTK POS tags to SentiWordNet POS tags because they use different terms/abbreviation
    if pos.startswith("J"):
        pos = "a"  # Adjective
    elif pos.startswith("V"):
        pos = "v"  # Verb
    elif pos.startswith("N"):
        pos = "n"  # Noun
    elif pos.startswith("R"):
        pos = "r"  # Adverb
    else:
        pos = "n"  # Default to noun if not found

    # Get sentiment scores from SentiWordNet
    synsets = list(swn.senti_synsets(word, pos))

    if not synsets:
        return 0.0, 0.0  # Default to objective if word not found in SentiWordNet

    # Calculate the average positive and negative scores from multiple synsets
    pos_score = sum(s.pos_score() for s in synsets) / len(synsets)
    neg_score = sum(s.neg_score() for s in synsets) / len(synsets)

    return pos_score, neg_score


def get_sentiment_label(aggregated_score):
    """
    Helper function that maps a score to a label (Positive or Negative)
    """
    if aggregated_score > 0.1:
        return "Positive"
    elif aggregated_score < -0.1:
        return "Negative"
    else:
        return "Objective or Neutral"


def analyze_sentiment(text):
    """
    Take a sentence (text) as input, tokenize it using nltk's word lexer and
    then start rating word sentiments.
    """
    words = nltk.word_tokenize(text)

    words_sentiment_scores = []

    for token, pos in nltk.pos_tag(words):
        pos_score, neg_score = get_sentiment_score(token, pos)
        words_sentiment_scores.append((token, pos_score, neg_score))

    aggregated_score = sum(
        pos_score - neg_score for _, pos_score, neg_score in words_sentiment_scores
    )
    sentiment_label = get_sentiment_label(aggregated_score)

    return sentiment_label, words_sentiment_scores


args_parser = init_argparse()
args = args_parser.parse_args()

# If it doesn't exist, create an nltk folder to save data
nltk_path = f"{os.getcwd()}/nltk"
if not os.path.exists(nltk_path):
    os.makedirs(nltk_path)

nltk.data.path.append(nltk_path)
nltk.download("sentiwordnet", nltk_path)
nltk.download("averaged_perceptron_tagger", nltk_path)
nltk.download("wordnet", nltk_path)

# Define the output file path
output_file_path = args.output_file
input_file_path = args.input_file
# Open the input CSV file
with open(
    input_file_path,
    "r",
    newline="",
) as input_csvfile:
    reader = csv.DictReader(input_csvfile)

    # Create an output CSV file to save sentiment values
    with open(output_file_path, "w", newline="") as output_csvfile:
        fieldnames = [
            "Text",
            "Sentiment Label",
            "Positive Words",
            "Negative Words",
            "Objective Words",
            "Aggregated Score",
        ]
        writer = csv.DictWriter(output_csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Read processed text and extract sentences
        for row in reader:
            text = row["text"]
            sentiment_label, words_sentiment_scores = analyze_sentiment(text)

            positive_words = [
                f"{word} ({pos_score:.2f})"
                for word, pos_score, neg_score in words_sentiment_scores
                if pos_score > neg_score
            ]
            negative_words = [
                f"{word} ({neg_score:.2f})"
                for word, pos_score, neg_score in words_sentiment_scores
                if neg_score > pos_score
            ]
            objective_words = [
                f"{word} (Pos: {pos_score:.2f}, Neg: {neg_score:.2f})"
                for word, pos_score, neg_score in words_sentiment_scores
                if abs(pos_score - neg_score) <= 0.1
            ]

            # Calculate aggregated_score here based on the sentiment scores of individual words
            aggregated_score = sum(
                pos_score - neg_score
                for _, pos_score, neg_score in words_sentiment_scores
            )

            writer.writerow(
                {
                    "Text": text,
                    "Sentiment Label": sentiment_label,
                    "Positive Words": ", ".join(positive_words),
                    "Negative Words": ", ".join(negative_words),
                    "Objective Words": ", ".join(objective_words),
                    "Aggregated Score": aggregated_score,
                }
            )

print(f"Saved sentiword output to '{output_file_path}'")
