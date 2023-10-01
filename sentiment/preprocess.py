import argparse
import csv
import os
import string

import spacy

import nltk

"""
Clean and lemmatize text
"""


def init_argparse() -> argparse.ArgumentParser:
    """Helps re-run the script multiple times with different parameters without changing the code over and over"""
    parser = argparse.ArgumentParser(
        usage="python %(prog)s [-h] [INPUT_FILE] [--output <filename>]",
        description="Preprocess data using nltk and spacy's NLP model. Text from reviews will be stripped of punctuation and lemmatized using spacy.",
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


# Function to clean and lemmatize text
def cleanAndGroupText(text):
    """
    Using spaCy, lemmatize the input text. Before we lemmatize, we clean up by
    removing punctuation
    """

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Lemmatization using spaCy
    doc = nlp(text)
    lemmatized_text = " ".join([token.lemma_ for token in doc])

    return lemmatized_text


args_parser = init_argparse()
args = args_parser.parse_args()

"""
Use NLTK modules
 punkt is a sentence tokeniser
 stopwords contains a dictionary of common stop words we want to remove from dataset
 NLTK docs: https://www.nltk.org/
"""
# If it doesn't exist, create an nltk folder to save data
nltk_path = f"{os.getcwd()}/nltk"
if not os.path.exists(nltk_path):
    os.makedirs(nltk_path)

nltk.data.path.append(nltk_path)
print("Downloading nltk modules...")
nltk.download("punkt", nltk_path)
nltk.download("stopwords", nltk_path)

# Load tokenizer to split text later on.
sent_detector = nltk.data.load("tokenizers/punkt/english.pickle")

# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")
input_file = args.input_file
output_file = args.output_file

if os.path.exists(input_file):
    with open(input_file, "r", encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file)

        # Skip the header row, otherwise everything will be mixed up
        next(csv_reader, None)

        processed_text = []
        order = 1
        for row in csv_reader:
            # Input is assumed to be a scrape. The scrape has the following format:
            # ID USERNAME RATING TEXT
            # we use index 3 to get only the text
            text = row[3]
            # Break text into sentences. For each sentence clean and lemmatize text.
            # We simplify text because the simpler the input the easier it is for algorithm
            # to detect. If it is complex (many words, many sentences) error rate is higher.
            sentences = sent_detector.tokenize(text.strip())
            for sentence in sentences:
                cleaned_text = cleanAndGroupText(sentence)
                processed_text.append((order, cleaned_text))
                order = order + 1

        # Save processed text to a new CSV file
        with open(output_file, "w", newline="", encoding="utf-8") as output_file:
            csv_writer = csv.writer(output_file)
            csv_writer.writerow(["order", "text"])  # Header row
            csv_writer.writerows(processed_text)

        print(f"Saved processed text to {output_file}")
else:
    print(f"Error: input file '{input_file}' not found")
