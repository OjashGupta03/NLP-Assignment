import os

INPUT_FILE = "brown_nouns.txt"

K_S = "S"
K_ES = "ES"
K_IES = "IES"
K_SINGULAR = "SG"
K_PLURAL = "PL"

NOUN = "N"

STATES = [
    "q0",
    "q_root",
    "q_s",
    "q_es",
    "q_ies",
    "q_final"
]

INPUT_ALPHABET = set("abcdefghijklmnopqrstuvwxyz")
INPUT_ALPHABET.update(["<END>"])

OUTPUT_ALPHABET = {
    "N",
    "SG",
    "PL",
    "+",
    "<eps>"
}


TRANSITION_TABLE = [
    ("q0", "letter", "q_root", "<eps>"),
    ("q_root", "<END>", "q_final", "+N+SG"),
    ("q_root", "s", "q_s", "<eps>"),
    ("q_root", "e", "q_es", "<eps>"),
    ("q_root", "i", "q_ies", "<eps>"),
    ("q_s", "<END>", "q_final", "+N+PL"),
    ("q_es", "s", "q_final", "+N+PL"),
    ("q_ies", "e", "q_ies", "<eps>"),
    ("q_ies", "s", "q_final", "+N+PL")
]


def read_words(filename):
    words = []

    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            word = line.strip().lower()

            if word:
                words.append(word)

    return words


def is_special_ending(root):
    return (
        root.endswith("s")
        or root.endswith("z")
        or root.endswith("x")
        or root.endswith("ch")
        or root.endswith("sh")
    )


def analyze(word, nouns):
    if word in nouns:
        if word.endswith("s"):
            if len(word) > 1:
                possible_root = word[:-1]

                if possible_root in nouns:
                    if not is_special_ending(possible_root):
                        return possible_root + "+N+PL"

        return word + "+N+SG"

    if word.endswith("ies"):
        root = word[:-3] + "y"

        if root in nouns and root.endswith("y"):
            return root + "+N+PL"

        return "Invalid Word"

    if word.endswith("es"):
        root = word[:-2]

        if root in nouns and is_special_ending(root):
            return root + "+N+PL"

        return "Invalid Word"

    if word.endswith("s"):
        root = word[:-1]

        if root in nouns and not is_special_ending(root):
            return root + "+N+PL"

        return "Invalid Word"

    return "Invalid Word"


def generate(root, nouns):
    if root not in nouns:
        return "Invalid Word"

    if root.endswith("y"):
        plural = root[:-1] + "ies"

    elif is_special_ending(root):
        plural = root + "es"

    else:
        plural = root + "s"

    if plural in nouns:
        return plural + " = " + root + "+N+PL"

    return "Invalid Word"


def print_transition_table():
    print()
    print("=" * 70)
    print("FST TRANSITION TABLE")
    print("=" * 70)

    print(
        f"{'Current State':<18}"
        f"{'Input':<15}"
        f"{'Next State':<18}"
        f"{'Output'}"
    )

    print("-" * 70)

    for state, inp, next_state, output in TRANSITION_TABLE:
        print(
            f"{state:<18}"
            f"{inp:<15}"
            f"{next_state:<18}"
            f"{output}"
        )

    print("=" * 70)


def print_alphabets():
    print()
    print("INPUT ALPHABET:")
    print(sorted(INPUT_ALPHABET))

    print()
    print("OUTPUT ALPHABET:")
    print(sorted(OUTPUT_ALPHABET))


def print_states():
    print()
    print("STATES:")
    print("q0       : Start state")
    print("q_root   : Reading noun root")
    print("q_s      : Checking regular plural -s")
    print("q_es     : Checking plural -es")
    print("q_ies    : Checking plural -ies")
    print("q_final  : Final state")


def main():
    if not os.path.exists(INPUT_FILE):
        print("Invalid Word")
        return

    words = read_words(INPUT_FILE)
    nouns = set(words)

    print_states()
    print_alphabets()
    print_transition_table()

    print()
    print("=" * 70)
    print("FST OUTPUT")
    print("=" * 70)

    for word in words:
        result = analyze(word, nouns)
        print(word + " = " + result)

    print()
    print("=" * 70)
    print("TEST CASES")
    print("=" * 70)

    tests = [
        "fox",
        "foxes",
        "foxs",
        "watch",
        "watches",
        "watchs",
        "bag",
        "bags",
        "try",
        "tries",
        "trys"
    ]

    for word in tests:
        print(word + " = " + analyze(word, nouns))


if __name__ == "__main__":
    main()