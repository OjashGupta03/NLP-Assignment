INPUT_FILE = "../lab 1/hindi_tokenized.txt"
TRAIN_FILE = "train.txt"
DEV_FILE = "dev.txt"
TEST_FILE = "test.txt"

TRAIN_SIZE = 998_000
DEV_SIZE = 1_000
TEST_SIZE = 1_000


# ============================================================
# Split the corpus
# ============================================================

print("Starting dataset split...")

with open(INPUT_FILE, "r", encoding="utf-8") as input_file:

    with open(TRAIN_FILE, "w", encoding="utf-8") as train_file, \
         open(DEV_FILE, "w", encoding="utf-8") as dev_file, \
         open(TEST_FILE, "w", encoding="utf-8") as test_file:

        for i, line in enumerate(input_file):

            if i < TRAIN_SIZE:
                train_file.write(line)

            elif i < TRAIN_SIZE + DEV_SIZE:
                dev_file.write(line)

            elif i < TRAIN_SIZE + DEV_SIZE + TEST_SIZE:
                test_file.write(line)

            else:
                break


# ============================================================
# Verify
# ============================================================

def count_lines(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


train_count = count_lines(TRAIN_FILE)
dev_count = count_lines(DEV_FILE)
test_count = count_lines(TEST_FILE)


print("\n" + "=" * 50)
print("DATASET SPLIT")
print("=" * 50)

print(f"Training sentences : {train_count:,}")
print(f"Dev sentences      : {dev_count:,}")
print(f"Test sentences     : {test_count:,}")

print("=" * 50)

print("\nFiles created:")
print(TRAIN_FILE)
print(DEV_FILE)
print(TEST_FILE)