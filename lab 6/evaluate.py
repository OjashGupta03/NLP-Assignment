import math
from pathlib import Path

from language_model import NGramLanguageModel


LAB6_DIR = Path(__file__).resolve().parent
LAB4_DIR = LAB6_DIR.parent / "lab 4"

DEV_FILE = LAB4_DIR / "split data" / "dev.txt"
TEST_FILE = LAB4_DIR / "split data" / "test.txt"


def calculate_perplexity(log_probability, total_tokens):
    if log_probability == float("-inf"):
        return float("inf")

    return math.exp(
        -log_probability / total_tokens
    )


def evaluate_file(model, filename):

    total_words = 0
    total_sentences = 0

    results = {
        "interpolated_bigram": 0.0,
        "interpolated_trigram": 0.0,
        "interpolated_quadgram": 0.0,

        "good_turing_bigram": 0.0,
        "good_turing_trigram": 0.0,
        "good_turing_quadgram": 0.0,

        "katz_bigram": 0.0,
        "katz_trigram": 0.0,
        "katz_quadgram": 0.0,

        "stupid_backoff_bigram": 0.0,
        "stupid_backoff_trigram": 0.0,
        "stupid_backoff_quadgram": 0.0,

        "kneser_ney_bigram": 0.0,
        "kneser_ney_trigram": 0.0,
        "kneser_ney_quadgram": 0.0
    }

    with open(filename, "r", encoding="utf-8") as f:

        for line in f:

            words = line.strip().split()

            if not words:
                continue

            total_sentences += 1
            total_words += len(words)

            # Interpolated
            results["interpolated_bigram"] += (
                model.interpolated_bigram_log_probability(words)
            )

            results["interpolated_trigram"] += (
                model.interpolated_trigram_log_probability(words)
            )

            results["interpolated_quadgram"] += (
                model.interpolated_quadgram_log_probability(words)
            )

            # Good-Turing
            results["good_turing_bigram"] += (
                model.good_turing_bigram_log_probability(words)
            )

            results["good_turing_trigram"] += (
                model.good_turing_trigram_log_probability(words)
            )

            results["good_turing_quadgram"] += (
                model.good_turing_quadgram_log_probability(words)
            )

            # Katz Backoff
            results["katz_bigram"] += (
                model.katz_bigram_log_probability(words)
            )

            results["katz_trigram"] += (
                model.katz_trigram_log_probability(words)
            )

            results["katz_quadgram"] += (
                model.katz_quadgram_log_probability(words)
            )

            # Stupid Backoff
            results["stupid_backoff_bigram"] += (
                model.stupid_backoff_bigram_log_probability(words)
            )

            results["stupid_backoff_trigram"] += (
                model.stupid_backoff_trigram_log_probability(words)
            )

            results["stupid_backoff_quadgram"] += (
                model.stupid_backoff_quadgram_log_probability(words)
            )

            # Kneser-Ney
            results["kneser_ney_bigram"] += (
                model.kneser_ney_bigram_log_probability(words)
            )

            results["kneser_ney_trigram"] += (
                model.kneser_ney_trigram_log_probability(words)
            )

            results["kneser_ney_quadgram"] += (
                model.kneser_ney_quadgram_log_probability(words)
            )

    higher_order_tokens = (
        total_words + total_sentences
    )

    perplexities = {}

    for name in results:

        perplexities[name] = calculate_perplexity(
            results[name],
            higher_order_tokens
        )

    return perplexities


def print_results(title, results):

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    print("\nInterpolated Smoothing")
    print(
        "Bigram   :",
        results["interpolated_bigram"]
    )
    print(
        "Trigram  :",
        results["interpolated_trigram"]
    )
    print(
        "Quadgram :",
        results["interpolated_quadgram"]
    )

    print("\nGood-Turing Smoothing")
    print(
        "Bigram   :",
        results["good_turing_bigram"]
    )
    print(
        "Trigram  :",
        results["good_turing_trigram"]
    )
    print(
        "Quadgram :",
        results["good_turing_quadgram"]
    )

    print("\nKatz Backoff Smoothing")
    print(
        "Bigram   :",
        results["katz_bigram"]
    )
    print(
        "Trigram  :",
        results["katz_trigram"]
    )
    print(
        "Quadgram :",
        results["katz_quadgram"]
    )

    print("\nStupid Backoff Smoothing")
    print(
        "Bigram   :",
        results["stupid_backoff_bigram"]
    )
    print(
        "Trigram  :",
        results["stupid_backoff_trigram"]
    )
    print(
        "Quadgram :",
        results["stupid_backoff_quadgram"]
    )

    print("\nKneser-Ney Smoothing")
    print(
        "Bigram   :",
        results["kneser_ney_bigram"]
    )
    print(
        "Trigram  :",
        results["kneser_ney_trigram"]
    )
    print(
        "Quadgram :",
        results["kneser_ney_quadgram"]
    )


# =========================================================
# Main
# =========================================================

model = NGramLanguageModel()


print("\nEvaluating development set...")
dev_results = evaluate_file(
    model,
    DEV_FILE
)

print_results(
    "DEV SET",
    dev_results
)


print("\nEvaluating test set...")
test_results = evaluate_file(
    model,
    TEST_FILE
)

print_results(
    "TEST SET",
    test_results
)