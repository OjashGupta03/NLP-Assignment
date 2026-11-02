import pickle
import math
from pathlib import Path


# Lab 5 reuses all n-gram counts generated in Lab 4.
LAB5_DIR = Path(__file__).resolve().parent
LAB4_DIR = LAB5_DIR.parent / "lab 4"
NGRAM_DIR = LAB4_DIR / "ngram_counts"

UNIGRAM_FILE = NGRAM_DIR / "unigram.pkl"
BIGRAM_FILE = NGRAM_DIR / "bigram.pkl"
TRIGRAM_FILE = NGRAM_DIR / "trigram.pkl"
QUADGRAM_FILE = NGRAM_DIR / "quadgram.pkl"


class NGramLanguageModel:
    def __init__(self, k=0.3):
        self.k = k

        with open(UNIGRAM_FILE, "rb") as f:
            self.unigram_counts = pickle.load(f)

        with open(BIGRAM_FILE, "rb") as f:
            self.bigram_counts = pickle.load(f)

        with open(TRIGRAM_FILE, "rb") as f:
            self.trigram_counts = pickle.load(f)

        with open(QUADGRAM_FILE, "rb") as f:
            self.quadgram_counts = pickle.load(f)

        self.vocab_size = len(self.unigram_counts)

        print("N-gram model loaded!")
        print("Smoothing: Add-K")
        print("K =", self.k)
        print("Vocabulary size:", self.vocab_size)


    # =========================================================
    # Add-K probabilities
    # =========================================================

    def unigram_probability(self, word):
        # P(w) = (C(w) + k) / (N + kV)
        count = self.unigram_counts[word]
        total_words = sum(self.unigram_counts.values())

        return (count + self.k) / (
            total_words + self.k * self.vocab_size
        )


    def bigram_probability(self, previous_word, word):
        # P(w_i | w_{i-1})
        # = (C(w_{i-1}, w_i) + k)
        #   / (C(w_{i-1}) + kV)

        bigram_count = self.bigram_counts[(previous_word, word)]
        previous_count = self.unigram_counts[previous_word]

        return (bigram_count + self.k) / (
            previous_count + self.k * self.vocab_size
        )


    def trigram_probability(self, previous_word1, previous_word2, word):
        # P(w_i | w_{i-2}, w_{i-1})
        # = (C(w_{i-2}, w_{i-1}, w_i) + k)
        #   / (C(w_{i-2}, w_{i-1}) + kV)

        trigram_count = self.trigram_counts[
            (previous_word1, previous_word2, word)
        ]

        previous_count = self.bigram_counts[
            (previous_word1, previous_word2)
        ]

        return (trigram_count + self.k) / (
            previous_count + self.k * self.vocab_size
        )


    def quadgram_probability(
        self,
        previous_word1,
        previous_word2,
        previous_word3,
        word
    ):
        # P(w_i | w_{i-3}, w_{i-2}, w_{i-1})
        # = (C(w_{i-3}, w_{i-2}, w_{i-1}, w_i) + k)
        #   / (C(w_{i-3}, w_{i-2}, w_{i-1}) + kV)

        quadgram_count = self.quadgram_counts[
            (
                previous_word1,
                previous_word2,
                previous_word3,
                word
            )
        ]

        previous_count = self.trigram_counts[
            (
                previous_word1,
                previous_word2,
                previous_word3
            )
        ]

        return (quadgram_count + self.k) / (
            previous_count + self.k * self.vocab_size
        )


    # =========================================================
    # Sentence log probabilities
    # =========================================================

    def bigram_log_probability(self, words):
        tokens = ["<START>"] + words + ["<END>"]
        log_probability = 0.0

        for i in range(1, len(tokens)):
            p = self.bigram_probability(
                tokens[i - 1],
                tokens[i]
            )
            log_probability += math.log(p)

        return log_probability


    def trigram_log_probability(self, words):
        tokens = ["<START>", "<START>"] + words + ["<END>"]
        log_probability = 0.0

        for i in range(2, len(tokens)):
            p = self.trigram_probability(
                tokens[i - 2],
                tokens[i - 1],
                tokens[i]
            )
            log_probability += math.log(p)

        return log_probability


    def quadgram_log_probability(self, words):
        tokens = ["<START>", "<START>", "<START>"] + words + ["<END>"]
        log_probability = 0.0

        for i in range(3, len(tokens)):
            p = self.quadgram_probability(
                tokens[i - 3],
                tokens[i - 2],
                tokens[i - 1],
                tokens[i]
            )
            log_probability += math.log(p)

        return log_probability


    # =========================================================
    # Next-word prediction
    # =========================================================

    def predict_next_bigram(self, previous_word):
        best_word = None
        best_probability = -1

        for word in self.unigram_counts:
            if word == "<START>":
                continue

            p = self.bigram_probability(previous_word, word)

            if p > best_probability:
                best_probability = p
                best_word = word

        return best_word, best_probability


    def predict_next_trigram(self, previous_word1, previous_word2):
        best_word = None
        best_probability = -1

        for word in self.unigram_counts:
            if word == "<START>":
                continue

            p = self.trigram_probability(
                previous_word1,
                previous_word2,
                word
            )

            if p > best_probability:
                best_probability = p
                best_word = word

        return best_word, best_probability


    def predict_next_quadgram(
        self,
        previous_word1,
        previous_word2,
        previous_word3
    ):
        best_word = None
        best_probability = -1

        for word in self.unigram_counts:
            if word == "<START>":
                continue

            p = self.quadgram_probability(
                previous_word1,
                previous_word2,
                previous_word3,
                word
            )

            if p > best_probability:
                best_probability = p
                best_word = word

        return best_word, best_probability
