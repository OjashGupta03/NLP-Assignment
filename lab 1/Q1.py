# ============================================================
# Assignment 1 - Hindi Corpus Processing using IndicCorpV2
# ============================================================

import re
import json
import os
# ============================================================
# 1. CONFIGURATION
# ============================================================

DATASET_REPO = "ai4bharat/IndicCorpV2"

DATASET_FILE = "../data/hi-1.txt"

MAX_SENTENCES = 1_000_000

DATA_DIR = "indiccorp_hindi"

# Output files
OUTPUT_FILE = "hindi_tokenized.txt"
STATS_FILE = "hindi_corpus_statistics.json"


# ============================================================
# 2. CORPUS FILE
# ============================================================

CORPUS_FILE = "../indiccorp_hindi/data/hi-1.txt"

print("=" * 60)
print("HINDI INDICCORPV2 CORPUS")
print("=" * 60)

print("\nCorpus file:")
print(CORPUS_FILE)

if not os.path.exists(CORPUS_FILE):
    raise FileNotFoundError(
        f"Corpus file not found: {CORPUS_FILE}"
    )
# ============================================================
# 3. SENTENCE TOKENIZER
# ============================================================

SENTENCE_PATTERN = re.compile(
    r"(?<=[।॥.!?…])\s+"
)


def sentence_tokenize(paragraph):
    """
    Convert one paragraph into a list of sentences.
    """

    paragraph = paragraph.strip()

    if not paragraph:
        return []

    sentences = SENTENCE_PATTERN.split(paragraph)

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    return sentences


# ============================================================
# 4. WORD TOKENIZER
# ============================================================

TOKEN_PATTERN = re.compile(
    r"""
    https?://[^\s]+
    |
    [A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@
    [A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+
    |
    \d{1,4}[/-]\d{1,2}[/-]\d{1,4}
    |
    \d+(?:[,.]\d+)*
    |
    [\u0900-\u0963\u0966-\u097F]+
    |
    [A-Za-z]+
    |
    [^\w\s]
    """,
    re.UNICODE | re.VERBOSE
)


def word_tokenize(sentence):
    """
    Convert sentence into word/punctuation tokens.
    """

    return TOKEN_PATTERN.findall(sentence)


# ============================================================
# 5. TEST TOKENIZER
# ============================================================

print("\nTesting tokenizer...")

test_text = (
    "यह एक परीक्षण वाक्य है। "
    "आज 10/09/2026 है! "
    "कीमत 12.50 रुपये है। "
    "वेबसाइट https://example.com पर जाएँ। "
    "ईमेल test@example.com पर भेजें?"
)

test_sentences = sentence_tokenize(test_text)

for i, sentence in enumerate(test_sentences, start=1):

    print(f"\nSentence {i}:")
    print(sentence)

    print("Tokens:")
    print(word_tokenize(sentence))

print("\nTokenizer test complete.")


# ============================================================
# 6. PROCESS CORPUS
# ============================================================

total_sentences = 0
total_tokens = 0
total_characters = 0

unique_tokens = set()

print("\n" + "=" * 60)
print("STARTING CORPUS PROCESSING")
print("=" * 60)

print("Input :", CORPUS_FILE)
print("Output:", OUTPUT_FILE)
print("Limit :", MAX_SENTENCES)

# IMPORTANT:
# encoding="utf-8" fixes the Windows cp1252 problem
with open(CORPUS_FILE,"r",encoding="utf-8",errors="replace") as corpus:

    with open(OUTPUT_FILE,"w",encoding="utf-8") as output:

        for line in corpus:

            paragraph = line.strip()

            if not paragraph:
                continue

            # ------------------------------------------------
            # Paragraph -> sentences
            # ------------------------------------------------

            sentences = sentence_tokenize(paragraph)

            for sentence in sentences:

                # ------------------------------------------------
                # Sentence -> tokens
                # ------------------------------------------------

                tokens = word_tokenize(sentence)

                if not tokens:
                    continue

                # ------------------------------------------------
                # Save tokenized sentence
                # ------------------------------------------------

                output.write(
                    " ".join(tokens) + "\n"
                )

                # ------------------------------------------------
                # Statistics
                # ------------------------------------------------

                total_sentences += 1

                total_tokens += len(tokens)

                for token in tokens:

                    total_characters += len(token)

                    unique_tokens.add(token)

                # ------------------------------------------------
                # Stop after required number of sentences
                # ------------------------------------------------

                if (
                    MAX_SENTENCES is not None
                    and total_sentences >= MAX_SENTENCES
                ):
                    break

            # ----------------------------------------------------
            # Stop processing file
            # ----------------------------------------------------

            if (
                MAX_SENTENCES is not None
                and total_sentences >= MAX_SENTENCES
            ):
                break

            # ----------------------------------------------------
            # Progress
            # ----------------------------------------------------

            if (
                total_sentences > 0
                and total_sentences % 100_000 == 0
            ):

                print(
                    f"Processed {total_sentences:,} sentences..."
                )


# ============================================================
# 7. CALCULATE STATISTICS
# ============================================================

unique_token_count = len(unique_tokens)

if total_sentences > 0:

    average_sentence_length = (
        total_tokens / total_sentences
    )

else:

    average_sentence_length = 0.0


if total_tokens > 0:

    average_word_length = (
        total_characters / total_tokens
    )

else:

    average_word_length = 0.0


if total_tokens > 0:

    ttr = (
        unique_token_count / total_tokens
    )

else:

    ttr = 0.0


# ============================================================
# 8. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("HINDI CORPUS STATISTICS")
print("=" * 60)

print(
    f"Total number of sentences : "
    f"{total_sentences:,}"
)

print(
    f"Total number of tokens    : "
    f"{total_tokens:,}"
)

print(
    f"Unique tokens             : "
    f"{unique_token_count:,}"
)

print(
    f"Total characters          : "
    f"{total_characters:,}"
)

print(
    f"Average sentence length   : "
    f"{average_sentence_length:.4f} tokens"
)

print(
    f"Average word length       : "
    f"{average_word_length:.4f} characters"
)

print(
    f"Type/Token Ratio (TTR)    : "
    f"{ttr:.8f}"
)

print("=" * 60)


# ============================================================
# 9. SAVE STATISTICS
# ============================================================

statistics = {

    "language": "Hindi",

    "dataset": DATASET_REPO,

    "source_file": DATASET_FILE,

    "maximum_sentences_requested": MAX_SENTENCES,

    "total_sentences": total_sentences,

    "total_tokens": total_tokens,

    "unique_tokens": unique_token_count,

    "total_characters": total_characters,

    "average_sentence_length":
        average_sentence_length,

    "average_word_length":
        average_word_length,

    "type_token_ratio":
        ttr
}


with open(
    STATS_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        statistics,
        f,
        ensure_ascii=False,
        indent=4
    )


# ============================================================
# 10. SHOW FIRST 10 TOKENIZED SENTENCES
# ============================================================

print(
    f"\nStatistics saved to: {STATS_FILE}"
)

print(
    f"Tokenized corpus saved to: {OUTPUT_FILE}"
)

print("\nFirst 10 tokenized sentences:\n")


with open(
    OUTPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    for i in range(10):

        line = f.readline()

        if not line:
            break

        print(
            f"{i + 1}: {line.strip()}"
        )


print("\nProcessing complete!")