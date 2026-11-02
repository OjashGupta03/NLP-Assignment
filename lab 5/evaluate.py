import math
from pathlib import Path

from language_model import NGramLanguageModel


# Lab 5 reuses the development and test sets from Lab 4.
LAB5_DIR = Path(__file__).resolve().parent
LAB4_DIR = LAB5_DIR.parent / "lab 4"

DEV_FILE = LAB4_DIR / "split data" / "dev.txt"
TEST_FILE = LAB4_DIR / "split data" / "test.txt"


def evaluate_file(model, filename):
    total_unigram_log_probability = 0.0
    total_bigram_log_probability = 0.0
    total_trigram_log_probability = 0.0
    total_quadgram_log_probability = 0.0

    total_words = 0
    total_sentences = 0

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            words = line.strip().split()

            if not words:
                continue

            total_sentences += 1
            total_words += len(words)

            # Unigram: score the actual words.
            for word in words:
                total_unigram_log_probability += math.log(
                    model.unigram_probability(word)
                )

            # Higher-order models include <END>, as in Lab 4.
            total_bigram_log_probability += (
                model.bigram_log_probability(words)
            )

            total_trigram_log_probability += (
                model.trigram_log_probability(words)
            )

            total_quadgram_log_probability += (
                model.quadgram_log_probability(words)
            )

    # Unigram log probability scores total_words events.
    unigram_perplexity = math.exp(
        -total_unigram_log_probability / total_words
    )

    # Bigram/trigram/quadgram additionally score one <END>
    # per sentence.
    higher_order_tokens = total_words + total_sentences

    bigram_perplexity = math.exp(
        -total_bigram_log_probability / higher_order_tokens
    )

    trigram_perplexity = math.exp(
        -total_trigram_log_probability / higher_order_tokens
    )

    quadgram_perplexity = math.exp(
        -total_quadgram_log_probability / higher_order_tokens
    )

    return (
        unigram_perplexity,
        bigram_perplexity,
        trigram_perplexity,
        quadgram_perplexity
    )


# =========================================================
# Main
# =========================================================

model = NGramLanguageModel(k=0.3)


print("\n================ DEV SET ================")

dev_results = evaluate_file(model, DEV_FILE)

print("Unigram Perplexity :", dev_results[0])
print("Bigram Perplexity  :", dev_results[1])
print("Trigram Perplexity :", dev_results[2])
print("Quadgram Perplexity:", dev_results[3])


print("\n================ TEST SET ===============")

test_results = evaluate_file(model, TEST_FILE)

print("Unigram Perplexity :", test_results[0])
print("Bigram Perplexity  :", test_results[1])
print("Trigram Perplexity :", test_results[2])
print("Quadgram Perplexity:", test_results[3])
