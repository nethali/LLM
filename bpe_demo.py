from collections import defaultdict


def get_pair_stats(corpus):
    """Count frequency of adjacent symbol pairs across all words."""
    pairs = defaultdict(int)
    for word_tokens, freq in corpus.items():
        for i in range(len(word_tokens) - 1):
            pair = (word_tokens[i], word_tokens[i + 1])
            pairs[pair] += freq
    return pairs


def merge_pair(pair, corpus):
    """Merge all occurrences of a specific pair into a new single token."""
    new_corpus = {}
    bigram = " ".join(pair)
    replacement = "".join(pair)

    for word_tokens, freq in corpus.items():
        # Convert tuple back to string to easily replace adjacent pairs
        word_str = " ".join(word_tokens)
        new_word_str = word_str.replace(bigram, replacement)
        # Store as new token tuple
        new_word_tokens = tuple(new_word_str.split())
        new_corpus[new_word_tokens] = freq

    return new_corpus


def run_bpe_simulation(initial_data, num_merges=4):
    # Represent words as tuples of single characters alongside their corpus frequency
    corpus = {tuple(word): freq for word, freq in initial_data.items()}

    # Initialize vocabulary with unique characters
    vocab = set(
        char for word_tokens in corpus.keys() for char in word_tokens
    )

    print("=== INITIAL CORPUS STATE ===")
    for word_tokens, freq in corpus.items():
        print(f"  {' '.join(word_tokens)}: count = {freq}")
    print(f"\nInitial Base Vocabulary: {sorted(list(vocab))}\n")
    print("=" * 45 + "\n")

    # Run BPE merge iterations
    for step in range(1, num_merges + 1):
        pair_stats = get_pair_stats(corpus)
        if not pair_stats:
            break

        # Find the most frequent adjacent pair
        best_pair = max(pair_stats, key=pair_stats.get)
        best_freq = pair_stats[best_pair]

        new_token = "".join(best_pair)
        vocab.add(new_token)
        corpus = merge_pair(best_pair, corpus)

        print(f"--- Iteration {step} ---")
        print(
            f"Most frequent pair: {best_pair} (Occurrences: {best_freq})"
        )
        print(f"New Merged Token Created: '{new_token}'")
        print("Updated Corpus:")
        for word_tokens, freq in corpus.items():
            print(f"  {' '.join(word_tokens)}: count = {freq}")
        print()

    print("=" * 45)
    print(f"Final Learned Vocabulary ({len(vocab)} items):")
    print(sorted(list(vocab)))


# --- Run Simulation with Example Data ---
initial_corpus = {"low": 5, "lower": 2, "newest": 6, "widest": 3}

run_bpe_simulation(initial_corpus, num_merges=4)