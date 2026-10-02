"""
BERT Tokenizer (`WordPiece`) Hands-on Demonstration
--------------------------------------------------
This script demonstrates how to inspect, encode, decode, and analyze text using 
Hugging Face's `BertTokenizer` (`bert-base-uncased`).
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from transformers import BertTokenizer


def main():
    # ==========================================
    # 1. Environment Setup & Loading Tokenizer
    # ==========================================
    print("--- Step 1: Loading BERT Tokenizer ---")

    # Load the standard WordPiece tokenizer for BERT (uncased)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Check the total vocabulary size
    vocab_size = tokenizer.vocab_size
    print(f"Loaded Tokenizer: {tokenizer.__class__.__name__} (bert-base-uncased)")
    print(f"Total Vocabulary Size: {vocab_size} tokens\n")

    # ==========================================
    # 2. Invertibility & Decoding Basics
    # ==========================================
    print("--- Step 2: Decoding Token IDs ---")

    # Decoding a single token ID
    sample_token_id = 1234
    decoded_string = tokenizer.decode([sample_token_id])

    print(f"Token ID {sample_token_id} decodes to: '{decoded_string}'\n")

    # ==========================================
    # 3. Leading Spaces, Casing, and Special Tokens
    # ==========================================
    print("--- Step 3: Whitespace, Casing, and Special Tokens ---")

    # Demonstrating how BERT strips leading spaces (unlike BPE)
    text_no_space = "John"
    text_with_space = " John"

    ids_no_space = tokenizer.encode(text_no_space, add_special_tokens=False)
    ids_with_space = tokenizer.encode(text_with_space, add_special_tokens=False)

    print(f"'{text_no_space}' -> Token IDs: {ids_no_space}")
    print(f"'{text_with_space}' -> Token IDs: {ids_with_space}")
    print("Notice: BERT strips leading spaces during pre-tokenization normalization.")

    # Demonstrating how capitalization and subwords are handled in 'uncased' BERT
    words_to_test = [" lanka", "Lanka", "LANKA"]
    print("\nCasing Normalization & WordPiece Subword Prefixing ('##'):")
    for word in words_to_test:
        encoded_ids = tokenizer.encode(word, add_special_tokens=False)
        decoded_tokens = tokenizer.convert_ids_to_tokens(encoded_ids)
        print(f"  '{word}' -> IDs: {encoded_ids} | Subwords: {decoded_tokens}")
    print()

    # ==========================================
    # 4. End-to-End Raw Text Tokenization
    # ==========================================
    print("--- Step 4: Tokenizing Raw Text & Special Tokens ([CLS], [SEP], [UNK]) ---")

    sample_raw_text = """
    "My MFA seems to have gone on an unexpected vacation." 😅

    Could you help bring it back when you get a chance?
    """

    # Encoding raw text (BERT automatically adds [CLS] at start and [SEP] at end)
    raw_text_tokens = tokenizer.encode(sample_raw_text)
    print(f"Raw Text Sample:\n{sample_raw_text.strip()}")
    print(f"Encoded Token Count (including [CLS] and [SEP]): {len(raw_text_tokens)}")

    # Print token breakdown
    print("\nToken-by-Token Breakdown (First 20 tokens):")
    converted_tokens = tokenizer.convert_ids_to_tokens(raw_text_tokens[:20])
    for tid, tstr in zip(raw_text_tokens[:20], converted_tokens):
        print(f"  ID {tid:>6} -> {tstr!r}")
    print(
        "\nNotice: The emoji 😅 is converted to '[UNK]' (ID 100) because uncased BERT lacks byte fallback!"
    )
    print()

    # ==========================================
    # 5. Analyzing Vocabulary Token Lengths
    # ==========================================
    print("--- Step 5: Analyzing Vocabulary Token Lengths ---")
    print("Calculating token character lengths across BERT vocabulary...")

    token_lengths = []
    for token, tid in tokenizer.vocab.items():
        # Strip BERT's continuation prefix '##' for accurate character length comparison
        clean_token = token.lstrip("#")
        token_lengths.append(len(clean_token))

    valid_lengths = np.array(token_lengths)

    min_len = int(np.min(valid_lengths))
    max_len = int(np.max(valid_lengths))

    print(f"Valid Tokens Analyzed: {len(valid_lengths)}")
    print(f"Min Token Length: {min_len} chars")
    print(f"Max Token Length: {max_len} chars")
    print(f"Mean Token Length: {np.mean(valid_lengths):.2f} chars")

    # Define exact integer bin boundaries centered on integer lengths (e.g., 0.5 to 1.5 for len=1)
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