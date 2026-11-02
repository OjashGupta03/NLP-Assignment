import pickle
import math
from pathlib import Path


LAB6_DIR = Path(__file__).resolve().parent
LAB4_DIR = LAB6_DIR.parent / "lab 4"
NGRAM_DIR = LAB4_DIR / "ngram_counts"

UNIGRAM_FILE = NGRAM_DIR / "unigram.pkl"
BIGRAM_FILE = NGRAM_DIR / "bigram.pkl"
TRIGRAM_FILE = NGRAM_DIR / "trigram.pkl"
QUADGRAM_FILE = NGRAM_DIR / "quadgram.pkl"


class NGramLanguageModel:

    def __init__(
        self,
        bigram_lambdas=(0.3, 0.7),
        trigram_lambdas=(0.1, 0.3, 0.6),
        quadgram_lambdas=(0.05, 0.15, 0.3, 0.5)
    ):

        with open(UNIGRAM_FILE, "rb") as f:
            self.unigram_counts = pickle.load(f)

        with open(BIGRAM_FILE, "rb") as f:
            self.bigram_counts = pickle.load(f)

        with open(TRIGRAM_FILE, "rb") as f:
            self.trigram_counts = pickle.load(f)

        with open(QUADGRAM_FILE, "rb") as f:
            self.quadgram_counts = pickle.load(f)

        self.vocab_size = len(self.unigram_counts)
        self.total_words = sum(self.unigram_counts.values())

        self.bigram_lambdas = bigram_lambdas
        self.trigram_lambdas = trigram_lambdas
        self.quadgram_lambdas = quadgram_lambdas

        assert math.isclose(sum(bigram_lambdas), 1.0)
        assert math.isclose(sum(trigram_lambdas), 1.0)
        assert math.isclose(sum(quadgram_lambdas), 1.0)

        print("N-gram model loaded!")
        print("Smoothing: Interpolated")
        print("Vocabulary size:", self.vocab_size)

    # =========================================================
    # MLE probabilities
    # =========================================================

    def unigram_probability(self, word):
        count = self.unigram_counts[word]

        if self.total_words == 0:
            return 0.0

        return count / self.total_words

    def bigram_probability(self, previous_word, word):
        count = self.bigram_counts[
            (previous_word, word)
        ]

        context_count = self.unigram_counts[
            previous_word
        ]

        if context_count == 0:
            return 0.0

        return count / context_count

    def trigram_probability(
        self,
        previous_word1,
        previous_word2,
        word
    ):
        count = self.trigram_counts[
            (previous_word1, previous_word2, word)
        ]

        context_count = self.bigram_counts[
            (previous_word1, previous_word2)
        ]

        if context_count == 0:
            return 0.0

        return count / context_count

    def quadgram_probability(
        self,
        previous_word1,
        previous_word2,
        previous_word3,
        word
    ):
        count = self.quadgram_counts[
            (
                previous_word1,
                previous_word2,
                previous_word3,
                word
            )
        ]

        context_count = self.trigram_counts[
            (
                previous_word1,
                previous_word2,
                previous_word3
            )
        ]

        if context_count == 0:
            return 0.0

        return count / context_count

    # =========================================================
    # Interpolated probabilities
    # =========================================================

    def interpolated_bigram_probability(
        self,
        previous_word,
        word
    ):
        lambda1, lambda2 = self.bigram_lambdas

        return (
            lambda1 * self.unigram_probability(word)
            +
            lambda2 * self.bigram_probability(
                previous_word,
                word
            )
        )

    def interpolated_trigram_probability(
        self,
        previous_word1,
        previous_word2,
        word
    ):
        lambda1, lambda2, lambda3 = self.trigram_lambdas

        return (
            lambda1 * self.unigram_probability(word)
            +
            lambda2 * self.bigram_probability(
                previous_word2,
                word
            )
            +
            lambda3 * self.trigram_probability(
                previous_word1,
                previous_word2,
                word
            )
        )

    def interpolated_quadgram_probability(
        self,
        previous_word1,
        previous_word2,
        previous_word3,
        word
    ):
        lambda1, lambda2, lambda3, lambda4 = (
            self.quadgram_lambdas
        )

        return (
            lambda1 * self.unigram_probability(word)
            +
            lambda2 * self.bigram_probability(
                previous_word3,
                word
            )
            +
            lambda3 * self.trigram_probability(
                previous_word2,
                previous_word3,
                word
            )
            +
            lambda4 * self.quadgram_probability(
                previous_word1,
                previous_word2,
                previous_word3,
                word
            )
        )

    # =========================================================
    # Log probabilities
    # =========================================================

    def interpolated_bigram_log_probability(self, words):
        tokens = ["<START>"] + words + ["<END>"]
        log_probability = 0.0

        for i in range(1, len(tokens)):
            p = self.interpolated_bigram_probability(
                tokens[i - 1],
                tokens[i]
            )

            if p <= 0:
                return float("-inf")

            log_probability += math.log(p)

        return log_probability

    def interpolated_trigram_log_probability(self, words):
        tokens = ["<START>", "<START>"] + words + ["<END>"]
        log_probability = 0.0

        for i in range(2, len(tokens)):
            p = self.interpolated_trigram_probability(
                tokens[i - 2],
                tokens[i - 1],
                tokens[i]
            )

            if p <= 0:
                return float("-inf")

            log_probability += math.log(p)

        return log_probability

    def interpolated_quadgram_log_probability(self, words):
        tokens = [
            "<START>",
            "<START>",
            "<START>"
        ] + words + ["<END>"]

        log_probability = 0.0

        for i in range(3, len(tokens)):
            p = self.interpolated_quadgram_probability(
                tokens[i - 3],
                tokens[i - 2],
                tokens[i - 1],
                tokens[i]
            )

            if p <= 0:
                return float("-inf")

            log_probability += math.log(p)

        return log_probability

    # =========================================================
    # Prediction
    # =========================================================

    def predict_next_bigram(self, previous_word):
        best_word = None
        best_probability = -1

        for word in self.unigram_counts:
            if word == "<START>":
                continue

            p = self.interpolated_bigram_probability(
                previous_word,
                word
            )

            if p > best_probability:
                best_probability = p
                best_word = word

        return best_word, best_probability

    def predict_next_trigram(
        self,
        previous_word1,
        previous_word2
    ):
        best_word = None
        best_probability = -1

        for word in self.unigram_counts:
            if word == "<START>":
                continue

            p = self.interpolated_trigram_probability(
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

            p = self.interpolated_quadgram_probability(
                previous_word1,
                previous_word2,
                previous_word3,
                word
            )

            if p > best_probability:
                best_probability = p
                best_word = word

        return best_word, best_probability