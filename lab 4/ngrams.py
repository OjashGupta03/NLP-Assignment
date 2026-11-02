from collections import Counter
import pickle
import os

TRAIN_FILE = "split data/train.txt"
os.makedirs("ngram_counts", exist_ok=True)

# Four frequency dictionaries
unigram_counts = Counter()
bigram_counts = Counter()
trigram_counts = Counter()
quadgram_counts = Counter()


# ============================================================
# Read training data
# ============================================================

print("Building n-gram counts...")

with open(TRAIN_FILE, "r", encoding="utf-8") as f:

    for sentence_number, line in enumerate(f, start=1):

        tokens = line.strip().split()
        
        if not tokens:
            continue
        
        # Unigram
        for token in tokens:
            unigram_counts[token] += 1
        
        # Bigram
        bigram_tokens = ["<START>"] + tokens + ["<END>"]
        
        for i in range(len(bigram_tokens) - 1):
            bigram_counts[
                (bigram_tokens[i], bigram_tokens[i + 1])
            ] += 1
        
        # Trigram
        trigram_tokens = ["<START>", "<START>"] + tokens + ["<END>"]
        
        for i in range(len(trigram_tokens) - 2):
            trigram_counts[
                (
                    trigram_tokens[i],
                    trigram_tokens[i + 1],
                    trigram_tokens[i + 2]
                )
            ] += 1
        
        # Quadgram
        quadgram_tokens = (
            ["<START>", "<START>", "<START>"]
            + tokens
            + ["<END>"]
        )
        
        for i in range(len(quadgram_tokens) - 3):
            quadgram_counts[
                (
                    quadgram_tokens[i],
                    quadgram_tokens[i + 1],
                    quadgram_tokens[i + 2],
                    quadgram_tokens[i + 3]
                )
            ] += 1
        
        # Progress
        if sentence_number % 100_000 == 0:
            print(
                f"Processed {sentence_number:,} sentences..."
            )


# ============================================================
# Display statistics
# ============================================================

print("\n" + "=" * 60)
print("N-GRAM COUNTS")
print("=" * 60)

print(f"Unique unigrams  : {len(unigram_counts):,}")

print(f"Unique bigrams   : {len(bigram_counts):,}")

print(f"Unique trigrams  : {len(trigram_counts):,}")

print(f"Unique quadgrams : {len(quadgram_counts):,}")


# ============================================================
# Display most common examples
# ============================================================

print("\nMost common unigrams:")

for gram, count in unigram_counts.most_common(10):
    print(f"{gram} : {count}")


print("\nMost common bigrams:")

for gram, count in bigram_counts.most_common(10):
    print(f"{gram} : {count}")


print("\nMost common trigrams:")

for gram, count in trigram_counts.most_common(10):
    print(f"{gram} : {count}")


print("\nMost common quadgrams:")

for gram, count in quadgram_counts.most_common(10):
    print(f"{gram} : {count}")

# save unigram
with open("ngram_counts/unigram.pkl", "wb") as f:
    pickle.dump(unigram_counts, f)

# Save bigram
with open("ngram_counts/bigram.pkl", "wb") as f:
    pickle.dump(bigram_counts, f)

# Save trigram
with open("ngram_counts/trigram.pkl", "wb") as f:
    pickle.dump(trigram_counts, f)

# Save quadgram
with open("ngram_counts/quadgram.pkl", "wb") as f:
    pickle.dump(quadgram_counts, f)

print("All n-gram counts saved successfully!")
