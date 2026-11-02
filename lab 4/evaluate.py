from language_model import NGramLanguageModel
import math


def calculate_perplexity(model, file_path, ngram_type):

    total_log_probability = 0.0
    total_tokens = 0

    with open(file_path, "r", encoding="utf-8") as f:

        for sentence in f:

            sentence = sentence.strip()

            if not sentence:
                continue

            # Count actual words in the sentence
            words = sentence.split()
            total_tokens += len(words) + 1   # +1 for <END>

            # Get log probability
            if ngram_type == "unigram":
            
                log_probability = 0.0

                for word in words:
                    probability = model.unigram_probability(word)
                    log_probability += math.log(probability)

            elif ngram_type == "bigram":
                log_probability = model.bigram_sentence_log_probability(sentence)

            elif ngram_type == "trigram":
                log_probability = model.trigram_sentence_log_probability(sentence)

            elif ngram_type == "quadgram":
                log_probability = model.quadgram_sentence_log_probability(sentence)

            total_log_probability += log_probability

    # Perplexity
    perplexity = math.exp(
        -total_log_probability / total_tokens
    )

    return perplexity


# -----------------------------------
# Main
# -----------------------------------

model = NGramLanguageModel()

dev_file = "split data/dev.txt"
test_file = "split data/test.txt"


print("\n========== DEV SET ==========")

dev_unigram = calculate_perplexity(model, dev_file, "unigram")
dev_bigram = calculate_perplexity(model, dev_file, "bigram")
dev_trigram = calculate_perplexity(model, dev_file, "trigram")
dev_quadgram = calculate_perplexity(model, dev_file, "quadgram")

print("Unigram Perplexity :", dev_unigram)
print("Bigram Perplexity  :", dev_bigram)
print("Trigram Perplexity :", dev_trigram)
print("Quadgram Perplexity:", dev_quadgram)


print("\n========== TEST SET ==========")

test_unigram = calculate_perplexity(model, test_file, "unigram")
test_bigram = calculate_perplexity(model, test_file, "bigram")
test_trigram = calculate_perplexity(model, test_file, "trigram")
test_quadgram = calculate_perplexity(model, test_file, "quadgram")

print("Unigram Perplexity :", test_unigram)
print("Bigram Perplexity  :", test_bigram)
print("Trigram Perplexity :", test_trigram)
print("Quadgram Perplexity:", test_quadgram)