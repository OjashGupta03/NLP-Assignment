import pickle
import math

class NGramLanguageModel:

    def __init__(self):
        # Load saved counts
        with open("ngram_counts/unigram.pkl", "rb") as f:
            self.unigram_counts = pickle.load(f)

        with open("ngram_counts/bigram.pkl", "rb") as f:
            self.bigram_counts = pickle.load(f)

        with open("ngram_counts/trigram.pkl", "rb") as f:
            self.trigram_counts = pickle.load(f)

        with open("ngram_counts/quadgram.pkl", "rb") as f:
            self.quadgram_counts = pickle.load(f)

        # Vocabulary size
        self.vocab_size = len(self.unigram_counts)

        print("N-gram model loaded!")
        print("Vocabulary size:", self.vocab_size)

    # -------------------------
    # UNIGRAM
    # -------------------------
    def unigram_probability(self, word):
        count = self.unigram_counts[word]
        total_words = sum(self.unigram_counts.values())
    
        return (count + 1) / (total_words + self.vocab_size)

    # -------------------------
    # BIGRAM + ADD-1 SMOOTHING
    # -------------------------
    def bigram_probability(self, previous_word, word):

        bigram_count = self.bigram_counts[(previous_word, word)]
        previous_count = self.unigram_counts[previous_word]

        probability = (bigram_count + 1) / (
            previous_count + self.vocab_size
        )

        return probability
    
    def trigram_probability(self, previous_word1, previous_word2, word):

        trigram_count = self.trigram_counts[(previous_word1, previous_word2, word)]
        previous_count = self.bigram_counts[(previous_word1, previous_word2)]

        probability = (trigram_count + 1) / ( previous_count + self.vocab_size )

        return probability

    def quadgram_probability(self, previous_word1, previous_word2, previous_word3, word):

        quadgram_count = self.quadgram_counts[(previous_word1, previous_word2, previous_word3, word)]
        previous_count = self.trigram_counts[(previous_word1, previous_word2, previous_word3)]

        probability = (quadgram_count + 1) / ( previous_count + self.vocab_size )

        return probability

    def bigram_sentence_log_probability(self, sentence):

        words = sentence.split()
    
        words = ["<START>"] + words + ["<END>"]
    
        log_probability = 0.0
    
        for i in range(1, len(words)):
            previous_word = words[i - 1]
            current_word = words[i]
    
            probability = self.bigram_probability(
                previous_word,
                current_word
            )
    
            log_probability += math.log(probability)
    
        return log_probability

    def trigram_sentence_log_probability(self, sentence):
    
            words = sentence.split()
    
            # Add sentence boundaries
            words = ["<START>", "<START>"] + words + ["<END>"]
    
            log_probability = 0.0
    
            for i in range(2, len(words)):
                previous_word1 = words[i - 2]
                previous_word2 = words[i - 1]
                current_word = words[i]
    
                probability = self.trigram_probability(
                    previous_word1,
                    previous_word2,
                    current_word
                )
    
                log_probability += math.log(probability)
    
            return log_probability

    def quadgram_sentence_log_probability(self, sentence):
    
            words = sentence.split()
    
            # Add sentence boundaries
            words = ["<START>", "<START>", "<START>"] + words + ["<END>"]
    
            log_probability = 0.0
    
            for i in range(3, len(words)):
                previous_word1 = words[i - 3]
                previous_word2 = words[i - 2]
                previous_word3 = words[i - 1]
                current_word = words[i]
    
                probability = self.quadgram_probability(
                    previous_word1,
                    previous_word2,
                    previous_word3,
                    current_word
                )
    
                log_probability += math.log(probability)
            return log_probability

    def predict_next_word_bigram(self, previous_word):

        best_word = None
        best_probability = 0

        for word in self.unigram_counts:
            if word == "<START>":
                continue

            probability = self.bigram_probability(
                previous_word,
                word
            )

            if probability > best_probability:
                best_probability = probability
                best_word = word

        return best_word, best_probability

    def predict_next_word_trigram(self, word1, word2):

        best_word = None
        best_probability = 0

        for word in self.unigram_counts:
            if word == "<START>":
                continue

            probability = self.trigram_probability(
                word1,
                word2,
                word
            )

            if probability > best_probability:
                best_probability = probability
                best_word = word

        return best_word, best_probability

    def predict_next_word_quadgram(self, word1, word2, word3):

        best_word = None
        best_probability = 0

        for word in self.unigram_counts:
            if word == "<START>":
                continue

            probability = self.quadgram_probability(
                word1,
                word2,
                word3,
                word
            )

            if probability > best_probability:
                best_probability = probability
                best_word = word

        return best_word, best_probability
    
if __name__ == "__main__":

    model = NGramLanguageModel()

    # --------------------------------
    # 1. TEST INDIVIDUAL PROBABILITIES
    # --------------------------------

    print("\n--- Individual Probabilities ---")

    print(
        "P(खाना | मैं) =",
        model.bigram_probability("मैं", "खाना")
    )

    print(
        "P(खाता | मैं, खाना) =",
        model.trigram_probability("मैं", "खाना", "खाता")
    )

    print(
        "P(हूँ | मैं, खाना, खाता) =",
        model.quadgram_probability("मैं", "खाना", "खाता", "हूँ")
    )


    # --------------------------------
    # 2. TEST SENTENCE PROBABILITIES
    # --------------------------------

    sentence = "मैं खाना खाता हूँ ।"

    print("\n--- Sentence Probabilities ---")

    print(
        "Sentence:",
        sentence
    )

    print(
        "Bigram probability =",
        model.bigram_sentence_log_probability(sentence)
    )

    print(
        "Trigram probability =",
        model.trigram_sentence_log_probability(sentence)
    )

    print(
        "Quadgram probability =",
        model.quadgram_sentence_log_probability(sentence)
    )


    # --------------------------------
    # 3. TEST NEXT-WORD PREDICTION
    # --------------------------------

    print("\n--- Next Word Prediction ---")

    # Bigram: मैं ___
    word, probability = model.predict_next_word_bigram("मैं")

    print("\nBigram:")
    print("Context: मैं")
    print("Predicted word:", word)
    print("Probability:", probability)


    # Trigram: मैं खाना ___
    word, probability = model.predict_next_word_trigram(
        "मैं",
        "खाना"
    )

    print("\nTrigram:")
    print("Context: मैं खाना")
    print("Predicted word:", word)
    print("Probability:", probability)


    # Quadgram: मैं खाना खाता ___
    word, probability = model.predict_next_word_quadgram(
        "मैं",
        "खाना",
        "खाता"
    )

    print("\nQuadgram:")
    print("Context: मैं खाना खाता")
    print("Predicted word:", word)
    print("Probability:", probability)

