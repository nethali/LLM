"""
Claude Tokenizer Hands-on Demonstration
--------------------------------------------------
This script demonstrates how to inspect, encode, decode, and analyze text using 
Claude's BPE Tokenizer via Hugging Face's transformers library.
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from transformers import GPT2TokenizerFast


def main():
    # ==========================================
    # 1. Environment Setup & Loading Tokenizer
    # ==========================================
    print("--- Step 1: Loading Claude Tokenizer ---")

    # Load Claude's BPE tokenizer from Hugging Face checkpoint
    tokenizer = GPT2TokenizerFast.from_pretrained("Xenova/claude-tokenizer")

    # Check vocabulary size (~65,000 tokens)
    vocab_size = tokenizer.vocab_size
    print("Loaded Tokenizer: Claude BPE Tokenizer (Xenova/claude-tokenizer)")
    print(f"Total Vocabulary Size: {vocab_size} tokens\n")

    # ==========================================
    # 2. Invertibility & Decoding Basics
    # ==========================================
    print("--- Step 2: Decoding Token IDs ---")

    # Important: .decode() expects a list of token IDs
    sample_token_id = 1234
    decoded_string = tokenizer.decode([sample_token_id])

    print(f"Token ID {sample_token_id} decodes to: '{decoded_string}'\n")

    # ==========================================
    # 3. Leading Spaces and Casing Sensitivity
    # ==========================================
    print("--- Step 3: Sensitivity to Spaces and Casing ---")

    # Demonstrating how leading spaces alter Token IDs
    text_no_space = "John"
    text_with_space = " John"

    ids_no_space = tokenizer.encode(text_no_space)
    ids_with_space = tokenizer.encode(text_with_space)

    print(f"'{text_no_space}' -> Token IDs: {ids_no_space}")
    print(f"'{text_with_space}' -> Token IDs: {ids_with_space}")

    # Demonstrating how capitalization fragments text into subwords
    words_to_test = [" lanka", "Lanka", "LANKA"]
    print("\nCapitalization and Subword Fragmentation:")
    for word in words_to_test:
        encoded_ids = tokenizer.encode(word)
        decoded_tokens = [tokenizer.decode([tid]) for tid in encoded_ids]
        print(f"  '{word}' -> IDs: {encoded_ids} | Subwords: {decoded_tokens}")
    print()

    # ==========================================
    # 4. End-to-End Raw Text Tokenization
    # ==========================================
    print("--- Step 4: Tokenizing Raw Uncleaned Text ---")

    sample_raw_text = """
    "My MFA seems to have gone on an unexpected vacation." 😅

    Could you help bring it back when you get a chance?
    """

    # Direct encoding of raw text without manual splitting
    raw_text_tokens = tokenizer.encode(sample_raw_text)
    print(f"Raw Text Sample:\n{sample_raw_text.strip()}")
    print(f"Encoded Token Count: {len(raw_text_tokens)}")

    # Print first few token IDs and their exact string representations
    print("\nToken-by-Token Breakdown (First 20 tokens):")
    for tid in raw_text_tokens[:20]:
        token_str = repr(tokenizer.decode([tid]))
        print(f"  ID {tid:>6} -> {token_str}")
    print()

    # ==========================================
    # 5. Analyzing Vocabulary Token Lengths
    # ==========================================
    print("--- Step 5: Analyzing Vocabulary Token Lengths ---")
    print("Calculating character lengths across vocabulary...")

    token_lengths = []
    for i in range(vocab_size):
        try:
            decoded = tokenizer.decode([i])
            token_lengths.append(len(decoded))
        except Exception:
            token_lengths.append(np.nan)

    token_lengths_arr = np.array(token_lengths)
    valid_lengths = token_lengths_arr[~np.isnan(token_lengths_arr)]

    min_len = int(np.min(valid_lengths))
    max_len = int(np.max(valid_lengths))

    print(f"Valid Tokens Analyzed: {len(valid_lengths)}")
    print(f"Min Token Length: {min_len} chars")
    print(f"Max Token Length: {max_len} chars")
    print(f"Mean Token Length: {np.mean(valid_lengths):.2f} chars")

    # Discrete integer bin boundaries for histogram
    integer_bins = np.arange(min_len - 0.5, max_len + 1.5, 1)

    # Plotting the Distribution of Token Lengths
    print("\nGenerating Distribution Plots...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Linear Scale Histogram
    axes[0].hist(
        valid_lengths, bins=integer_bins, color="skyblue", edgecolor="black", alpha=0.7
    )
    axes[0].set_title("Token Length Distribution (Linear Scale)")
    axes[0].set_xlabel("Token Length (Characters)")
    axes[0].set_ylabel("Frequency")
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Plot 2: Logarithmic Scale Histogram
    axes[1].hist(
        valid_lengths,
        bins=integer_bins,
        color="salmon",
        edgecolor="black",
        alpha=0.7,
        log=True,
    )
    axes[1].set_title("Token Length Distribution (Log Scale)")
    axes[1].set_xlabel("Token Length (Characters)")
    axes[1].set_ylabel("Frequency (Log Scale)")
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()