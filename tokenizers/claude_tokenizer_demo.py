"""
Claude Tokenizer Hands-on Demonstration (Colorful Plots)
--------------------------------------------------
This script demonstrates how to inspect, encode, decode, and analyze text using 
Claude's BPE Tokenizer with vibrant, gradient-colored distribution visualizations.
"""

import matplotlib.cm as cm
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

    sample_token_id = 1234
    decoded_string = tokenizer.decode([sample_token_id])

    print(f"Token ID {sample_token_id} decodes to: '{decoded_string}'")
    print(
        "(Notice how Claude's BPE handles subword tokens and whitespace!)\n"
    )

    # ==========================================
    # 3. Leading Spaces and Casing Sensitivity
    # ==========================================
    print("--- Step 3: Sensitivity to Spaces and Casing ---")

    text_no_space = "John"
    text_with_space = " John"

    ids_no_space = tokenizer.encode(text_no_space)
    ids_with_space = tokenizer.encode(text_with_space)

    print(f"'{text_no_space}' -> Token IDs: {ids_no_space}")
    print(f"'{text_with_space}' -> Token IDs: {ids_with_space}")

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

    raw_text_tokens = tokenizer.encode(sample_raw_text)
    print(f"Raw Text Sample:\n{sample_raw_text.strip()}")
    print(f"Encoded Token Count: {len(raw_text_tokens)}")

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

    # Integer bin boundaries matching exact token byte lengths
    integer_bins = np.arange(min_len - 0.5, max_len + 1.5, 1)

    # ==========================================
    # 6. Vibrant Gradient Plotting
    # ==========================================
    print("\nGenerating Vibrant Distribution Plots...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Plot 1: Linear Scale with Viridis Colormap Gradient ---
    n, bins, patches = axes[0].hist(
        valid_lengths, bins=integer_bins, edgecolor="black", linewidth=0.5
    )

    # Color each histogram bin based on its token length value
    col_norm = (bins - min_len) / (max_len - min_len)
    for patch, color_val in zip(patches, col_norm):
        patch.set_facecolor(cm.viridis(color_val))

    axes[0].set_title("Claude Token Length Distribution (Linear Scale)")
    axes[0].set_xlabel("Token Length (Characters)")
    axes[0].set_ylabel("Frequency")
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].grid(True, linestyle="--", alpha=0.4)

    # --- Plot 2: Logarithmic Scale with Plasma Colormap Gradient ---
    n_log, bins_log, patches_log = axes[1].hist(
        valid_lengths, bins=integer_bins, edgecolor="black", linewidth=0.5, log=True
    )

    for patch, color_val in zip(patches_log, col_norm):
        patch.set_facecolor(cm.plasma(color_val))

    axes[1].set_title("Claude Token Length Distribution (Log Scale)")
    axes[1].set_xlabel("Token Length (Characters)")
    axes[1].set_ylabel("Frequency (Log Scale)")
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()