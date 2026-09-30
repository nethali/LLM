"""
GPT-4 Tokenizer (`tiktoken`) Hands-on Demonstration
--------------------------------------------------
This script demonstrates how to inspect, encode, decode, and analyze text using 
OpenAI's `tiktoken` library (GPT-4 `cl100k_base` encoding).
"""

import matplotlib.pyplot as plt
import numpy as np
import tiktoken


def main():
    # ==========================================
    # 1. Environment Setup & Loading Tokenizer
    # ==========================================
    print("--- Step 1: Loading GPT-4 Tokenizer ---")

    # Load the base BPE tokenizer for GPT-4 (100k vocabulary)
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Check the total vocabulary size
    vocab_size = tokenizer.n_vocab
    print(f"Loaded Tokenizer: {tokenizer.name}")
    print(f"Total Vocabulary Size: {vocab_size} tokens\n")

    # ==========================================
    # 2. Invertibility & Decoding Basics
    # ==========================================
    print("--- Step 2: Decoding Token IDs ---")

    # Important: .decode() expects a list of token IDs, not a single integer.
    sample_token_id = 12345
    decoded_string = tokenizer.decode([sample_token_id])

    print(f"Token ID {sample_token_id} decodes to: '{decoded_string}'")
    print(
        "(Notice the leading space included directly in the subword token!)\n"
    )

    # ==========================================
    # 3. Leading Spaces and Casing Sensitivity
    # ==========================================
    print("--- Step 3: Sensitivity to Spaces and Casing ---")

    # Demonstrating how leading spaces alter Token IDs
    text_no_space = "Michael"
    text_with_space = " Michael"

    ids_no_space = tokenizer.encode(text_no_space)
    ids_with_space = tokenizer.encode(text_with_space)

    print(f"'{text_no_space}' -> Token IDs: {ids_no_space}")
    print(f"'{text_with_space}' -> Token IDs: {ids_with_space}")

    # Demonstrating how capitalization fragments text into subwords
    words_to_test = [" peach", "Peach", "POach"]
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
    "I stayed silent, and then she touched my hand."
    
    Then I felt on my back another soft tentacle...
    """

    # Direct encoding of raw text without pre-cleaning or manual splitting
    raw_text_tokens = tokenizer.encode(sample_raw_text)
    print(f"Raw Text Sample:\n{sample_raw_text.strip()}")
    print(f"Encoded Token Count: {len(raw_text_tokens)}")

    # Print first few token IDs and their exact string representations
    print("\nToken-by-Token Breakdown (First 10 tokens):")
    for tid in raw_text_tokens[:10]:
        token_str = repr(tokenizer.decode([tid]))
        print(f"  ID {tid:>6} -> {token_str}")
    print()

    # ==========================================
    # 5. Analyzing Vocabulary Token Lengths
    # ==========================================
    print("--- Step 5: Analyzing Vocabulary Token Lengths ---")
    print("Calculating byte lengths across vocabulary... (this may take a few seconds)")

    token_lengths = []
    for i in range(vocab_size):
        try:
            # Get raw byte length for each valid token index
            token_bytes = tokenizer.decode_bytes([i])
            token_lengths.append(len(token_bytes))
        except Exception:
            # Handle unassigned or invalid special token slots safely
            token_lengths.append(np.nan)

    token_lengths_arr = np.array(token_lengths)
    valid_lengths = token_lengths_arr[~np.isnan(token_lengths_arr)]

    print(f"Valid Tokens Analyzed: {len(valid_lengths)}")
    print(f"Min Token Length: {int(np.min(valid_lengths))} bytes")
    print(f"Max Token Length: {int(np.max(valid_lengths))} bytes")
    print(f"Mean Token Length: {np.mean(valid_lengths):.2f} bytes")

    # Plotting the Distribution of Token Lengths
    print("\nGenerating Distribution Plots...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Linear Scale Histogram
    axes[0].hist(
        valid_lengths, bins=50, color="skyblue", edgecolor="black", alpha=0.7
    )
    axes[0].set_title("Token Length Distribution (Linear Scale)")
    axes[0].set_xlabel("Token Length (Bytes)")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Plot 2: Logarithmic Scale Histogram
    axes[1].hist(
        valid_lengths,
        bins=50,
        color="salmon",
        edgecolor="black",
        alpha=0.7,
        log=True,
    )
    axes[1].set_title("Token Length Distribution (Log Scale)")
    axes[1].set_xlabel("Token Length (Bytes)")
    axes[1].set_ylabel("Frequency (Log Scale)")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()