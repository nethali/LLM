import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# 1. GLOVE EMBEDDING LOADER CLASS
# =============================================================================
class LocalGloVeLoader:
    """
    Parses GloVe-formatted text files directly into NumPy arrays.
    """
    def __init__(self, filename="dolma_300_2024_1.2M.100_combined.txt"):
        self.filename = filename
        self.stoi = {}       # String-to-Index lookup dictionary
        self.itos = []       # Index-to-String list
        self.vectors = None  # NumPy array holding shape (Vocab_Size, Dimensions)
        self.dim = None      # Embedding vector dimensionality
        self._load_file()

    def _load_file(self):
        if not os.path.exists(self.filename):
            raise FileNotFoundError(
                f"Could not find '{self.filename}'. Ensure the file is in: {os.getcwd()}"
            )

        print(f"Loading embeddings from '{self.filename}'...")
        vectors_list = []

        # Read line-by-line: [word, dim_1, dim_2, ..., dim_N]
        with open(self.filename, "r", encoding="utf-8", errors="ignore") as f:
            for idx, line in enumerate(f):
                parts = line.strip().split()
                if not parts:
                    continue
                
                word = parts[0]
                try:
                    vec = np.array(parts[1:], dtype=np.float32)
                except ValueError:
                    # Skip malformed metadata lines if present
                    continue

                if self.dim is None:
                    self.dim = len(vec)

                # Skip vector size mismatches
                if len(vec) != self.dim:
                    continue

                self.stoi[word] = len(self.itos)
                self.itos.append(word)
                vectors_list.append(vec)

        self.vectors = np.array(vectors_list)
        print(f"Successfully loaded {len(self.itos):,} words with {self.dim}D dimensions!\n")

    def __getitem__(self, word):
        """Allows direct indexing: glove['king']. Returns zero-vector for OOV terms."""
        word = str(word).lower()
        if word in self.stoi:
            return self.vectors[self.stoi[word]]
        return np.zeros(self.dim, dtype=np.float32)


# =============================================================================
# 2. VECTOR MATHEMATICS & SEARCH FUNCTIONS
# =============================================================================
def get_cosine_similarity(v1, v2):
    """Calculates cosine similarity scalar between two 1D vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def get_nearest_neighbors(glove, word, topn=10):
    """Finds top-N nearest neighbors in the vocabulary via dot product on normalized vectors."""
    word = str(word).lower()
    if word not in glove.stoi:
        print(f"Word '{word}' not found in vocabulary.")
        return []

    word_idx = glove.stoi[word]
    target_vector = glove.vectors[word_idx]

    # Batch normalize all vectors in the matrix
    norms = np.linalg.norm(glove.vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    norm_vectors = glove.vectors / norms

    target_norm = target_vector / (np.linalg.norm(target_vector) + 1e-10)
    similarities = np.dot(norm_vectors, target_norm)

    # Exclude the query word itself
    similarities[word_idx] = -1.0

    top_indices = np.argsort(similarities)[::-1][:topn]
    return [(glove.itos[idx], float(similarities[idx])) for idx in top_indices]


def get_doesnt_match(glove, words):
    """Identifies the semantic outlier in a list of words by comparing average pairwise cosine similarity."""
    valid_words = [w.lower() for w in words if w.lower() in glove.stoi]
    if len(valid_words) < 2:
        return words[0]

    vectors = np.array([glove[w] for w in valid_words])
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    norm_vectors = vectors / norms

    sim_matrix = np.dot(norm_vectors, norm_vectors.T)
    mean_sims = (np.sum(sim_matrix, axis=1) - 1.0) / (len(valid_words) - 1)

    outlier_idx = np.argmin(mean_sims)
    return valid_words[outlier_idx]


def solve_analogy(glove, w1, w2, w3, topn=3):
    """Solves vector analogy equations: w1 - w2 + w3 = ? (e.g., King - Man + Woman = Queen)"""
    w1, w2, w3 = w1.lower(), w2.lower(), w3.lower()
    if not all(w in glove.stoi for w in [w1, w2, w3]):
        return []

    target_vec = glove[w1] - glove[w2] + glove[w3]

    norms = np.linalg.norm(glove.vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    norm_vectors = glove.vectors / norms

    target_norm = target_vec / (np.linalg.norm(target_vec) + 1e-10)
    similarities = np.dot(norm_vectors, target_norm)

    # Mask input words to prevent returning query words as answers
    for w in [w1, w2, w3]:
        similarities[glove.stoi[w]] = -1.0

    top_indices = np.argsort(similarities)[::-1][:topn]
    return [(glove.itos[idx], float(similarities[idx])) for idx in top_indices]


# =============================================================================
# 3. DEMONSTRATION WORKFLOW PIPELINE
# =============================================================================
def main():
    # Set target file name
    GLOVE_FILENAME = "dolma_300_2024_1.2M.100_combined.txt"

    # --- STEP 1: LOAD EMBEDDINGS ---
    print("=================================================================")
    print(" SECTION 1: Loading Vocabulary and Embedding Matrix")
    print("=================================================================")
    glove = LocalGloVeLoader(filename=GLOVE_FILENAME)

    vocab_size = len(glove.itos)
    print(f"Total vocabulary size : {vocab_size:,} tokens")
    print(f"Embedding dimensions  : {glove.dim}D")

    sample_indices = [10, 500, 12000, 45000, 180000]
    print("\nSample words across vocabulary index frequencies:")
    for idx in sample_indices:
        if idx < vocab_size:
            print(f"  Index {idx:6d}: {glove.itos[idx]}")
    print()

    # --- STEP 2: WORD LENGTH DISTRIBUTION ---
    print("=================================================================")
    print(" SECTION 2: Analyzing Word Length Distribution")
    print("=================================================================")
    word_lengths = [len(word) for word in glove.itos]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(word_lengths, bins=range(1, 35), color="skyblue", edgecolor="black")
    axes[0].set_title("Word Length Distribution (Linear Scale)")
    axes[0].set_xlabel("Word Length (Characters)")
    axes[0].set_ylabel("Token Count")

    axes[1].hist(word_lengths, bins=range(1, 35), color="skyblue", edgecolor="black", log=True)
    axes[1].set_title("Word Length Distribution (Log Scale)")
    axes[1].set_xlabel("Word Length (Characters)")
    axes[1].set_ylabel("Log(Token Count)")

    plt.tight_layout()
    plt.show()

    # --- STEP 3: MATRIX GLOBAL STATISTICS ---
    print("=================================================================")
    print(" SECTION 3: Global Embedding Matrix Statistics")
    print("=================================================================")
    means = np.mean(glove.vectors, axis=1)
    stds = np.std(glove.vectors, axis=1)

    df = pd.DataFrame({"Mean": means, "Standard Deviation": stds})
    grid = sns.jointplot(
        data=df, x="Mean", y="Standard Deviation", alpha=0.1, color="teal", height=6
    )
    grid.fig.suptitle("Mean vs Std Dev Across Vocabulary Vectors", y=1.02)
    plt.show()

    # --- STEP 4: VECTOR COMPARISONS & PROFILE PLOTS ---
    print("=================================================================")
    print(" SECTION 4: Vector Profile Comparisons & Cosine Scatter Plots")
    print("=================================================================")
    v_banana = glove["banana"]
    v_apple = glove["apple"]
    v_cosmic = glove["cosmic"]

    plt.figure(figsize=(10, 4))
    plt.plot(v_banana, label="banana", color="blue", linewidth=1.5)
    plt.plot(v_apple, label="apple", color="orange", linewidth=1.5)
    plt.plot(v_cosmic, label="cosmic", color="green", linewidth=1.5)
    plt.title("Dimension Profiles for 'banana', 'apple', and 'cosmic'")
    plt.xlabel("Dimension Index")
    plt.ylabel("Dimension Weight Value")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

    sim_apple_banana = get_cosine_similarity(v_banana, v_apple)
    sim_banana_cosmic = get_cosine_similarity(v_banana, v_cosmic)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.scatter(v_banana, v_apple, color="purple", alpha=0.5)
    ax1.set_title(f"Banana vs Apple\nCosine Similarity: {sim_apple_banana:.4f}")
    ax1.set_xlabel("Banana Dimensions")
    ax1.set_ylabel("Apple Dimensions")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.scatter(v_banana, v_cosmic, color="darkgreen", alpha=0.5)
    ax2.set_title(f"Banana vs Cosmic\nCosine Similarity: {sim_banana_cosmic:.4f}")
    ax2.set_xlabel("Banana Dimensions")
    ax2.set_ylabel("Cosmic Dimensions")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()

    # --- STEP 5: HIGHLIGHT 1 - NEAREST NEIGHBORS ---
    print("=================================================================")
    print(" SECTION 5: Highlight 1 - Nearest Neighbors (Taxonomy & Similarity)")
    print("=================================================================")
    target_words = ["frog", "fashion"]
    for target in target_words:
        if target in glove.stoi:
            neighbors = get_nearest_neighbors(glove, target, topn=8)
            print(f"Closest words to '{target}':")
            for rank, (word, sim) in enumerate(neighbors, 1):
                print(f"  {rank}. {word:18s} (Cosine Sim: {sim:.4f})")
            print()

    # Outlier detection demo
    list_1 = ["apple", "banana", "mango", "pirate"]
    print(f"Outlier check for list {list_1}:")
    print(f"  Outlier detected -> '{get_doesnt_match(glove, list_1)}'\n")

    # --- STEP 6: HIGHLIGHT 2 - LINEAR SUBSTRUCTURES & ANALOGIES ---
    print("=================================================================")
    print(" SECTION 6: Highlight 2 - Linear Substructures & Analogies")
    print("=================================================================")
    analogies = [
        ("king", "man", "woman", "King - Man + Woman -> Queen"),
        ("paris", "france", "germany", "Paris - France + Germany -> Berlin"),
        ("slower", "slow", "fast", "Slower - Slow + Fast -> Faster"),
    ]

    for w1, w2, w3, desc in analogies:
        results = solve_analogy(glove, w1, w2, w3, topn=1)
        top_word, score = results[0] if results else ("N/A", 0.0)
        print(f"Analogy: {desc:38s} => Result: {top_word} ({score:.4f})")

    v_man_woman = glove["man"] - glove["woman"]
    v_king_queen = glove["king"] - glove["queen"]
    v_brother_sister = glove["brother"] - glove["sister"]

    sim_1 = get_cosine_similarity(v_man_woman, v_king_queen)
    sim_2 = get_cosine_similarity(v_man_woman, v_brother_sister)

    print("\nVector Difference Substructure Alignment:")
    print(f"  Cosine Sim (Man - Woman, King - Queen)     : {sim_1:.4f}")
    print(f"  Cosine Sim (Man - Woman, Brother - Sister) : {sim_2:.4f}\n")

    # --- STEP 7: HIGHLIGHT 3 - MODEL OVERVIEW & PROBABILITY RATIOS ---
    print("=================================================================")
    print(" SECTION 7: Highlight 3 - Co-occurrence Probability Ratios")
    print("=================================================================")
    print("GloVe associates probability ratios with vector dot-product differences:")
    print("  dot(v_target, v_probe) ~ log P(probe | target)\n")

    target_ice = glove["ice"]
    target_steam = glove["steam"]
    probe_words = ["solid", "gas", "water", "fashion"]

    print(f"{'Probe Word (x)':15s} | {'dot(v_ice, x)':15s} | {'dot(v_steam, x)':15s} | {'Implied Ratio [P(x|ice)/P(x|steam)]':35s}")
    print("-" * 85)

    for word in probe_words:
        if word in glove.stoi:
            dot_ice = np.dot(target_ice, glove[word])
            dot_steam = np.dot(target_steam, glove[word])
            implied_ratio = np.exp(dot_ice - dot_steam)
            print(f"{word:15s} | {dot_ice:15.4f} | {dot_steam:15.4f} | {implied_ratio:35.4f}")
    print()

    # --- STEP 8: HIGHLIGHT 4 - VISUALIZATIONS (HEATMAPS & BANDING) ---
    print("=================================================================")
    print(" SECTION 8: Highlight 4 - Vector Heatmaps & Banding Visualizations")
    print("=================================================================")
    
    # 1. Horizontal Banded Structure Across Common Concepts
    words_to_plot = [
        "the", "of", "and", "in", "a", "to", "is", "was", "for", "on",
        "water", "ice", "steam", "solid", "gas", "frog", "lizard", "toad",
        "king", "queen", "man", "woman", "boy", "girl", "city", "country"
    ]
    valid_words = [w for w in words_to_plot if w in glove.stoi]
    matrix = np.array([glove[w] for w in valid_words])

    plt.figure(figsize=(14, 6))
    sns.heatmap(matrix, yticklabels=valid_words, cmap="viridis", cbar=True)
    plt.title("GloVe Heatmap: Horizontal Banded Structure Across Vocabulary Dimensions")
    plt.xlabel("Vector Dimension Index")
    plt.ylabel("Words")
    plt.tight_layout()
    plt.show()

    # 2. Vertical Density Bands across Frequency Ranges
    indices_to_sample = np.linspace(0, min(len(glove.itos) - 1, 250000), 200, dtype=int)
    sampled_matrix = glove.vectors[indices_to_sample]

    plt.figure(figsize=(12, 5))
    plt.imshow(sampled_matrix, aspect="auto", cmap="coolwarm", interpolation="nearest")
    plt.colorbar(label="Dimension Weight Value")
    plt.title("Global Space Inspection: Vertical Density Bands Across Frequency Indices")
    plt.xlabel("Dimension Index")
    plt.ylabel("Sampled Vocabulary Index (Frequent -> Rare)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()