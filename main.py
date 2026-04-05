# Command-line tool: print TF-IDF key points from one or more text files.
import argparse
import sys

from file_handler import read_file, read_multiple
from preprocess import download_nltk_if_needed
from summarizer import summarize_tfidf


def main():
    parser = argparse.ArgumentParser(
        description="Summarize plain text using TF-IDF (extractive bullet points).",
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Input .txt file(s). Multiple files are combined in order.",
    )
    parser.add_argument(
        "-n",
        "--num-points",
        type=int,
        default=5,
        help="How many bullet points to show (default: 5).",
    )
    args = parser.parse_args()

    if args.num_points < 1:
        print("Error: -n must be 1 or more.", file=sys.stderr)
        sys.exit(2)

    download_nltk_if_needed()

    if len(args.files) == 1:
        full_text = read_file(args.files[0])
    else:
        full_text = read_multiple(args.files)

    bullets = summarize_tfidf(full_text, num_sentences=args.num_points)

    print("")
    print("Key points:")
    print("")
    if len(bullets) == 0:
        print("(No sentences found. Is the file empty?)")
    else:
        for sentence in bullets:
            print("- " + sentence.strip())

    sys.exit(0)


if __name__ == "__main__":
    main()
