import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from transformers import AutoTokenizer, BERTModel, GPT2Model

# =============================================================================
# 1. HELPER FUNCTIONS
# =============================================================================


def get_cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculates cosine similarity between two 1D vectors."""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(dot_product / (norm_v1 * norm_v2))


def style_axis(ax, title: str, xlabel: str, ylabel: str, grid_color="#334155"):
    """Utility function to apply dark-themed axis styling."""
    text_light = "#f8fafc"
    text_muted = "#94a3b8"

    ax.set_facecolor("#1e293b")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10, color=text_light)
    ax.set_xlabel(xlabel, fontsize=9, color=text_muted)
    ax.set_ylabel(ylabel, fontsize=9, color=text_muted)
    ax.tick_params(colors=text_muted, labelsize=8)
    ax.grid(True, linestyle="--", alpha=0.3, color=grid_color)
    for spine in ax.spines.values():
        spine.set_color(grid_color)


# =============================================================================
# 2. DEMONSTRATION WORKFLOW PIPELINE
# =============================================================================


def main():
    # --- STEP 1: LOAD REAL MODEL EMBEDDING MATRICES ---
    print("=================================================================")
    print(" SECTION 1: Reading Parameters & Vectors from Real Models")
    print("=================================================================")

    # Load GPT-2 weights and tokenizer
    print("Loading GPT-2 model parameters...")
    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    gpt2_model = GPT2Model.from_pretrained("gpt2")

    # Extract GPT-2 word embedding weight matrix: shape (50257, 768)
    gpt2_wte = (
        gpt2_model.get_input_embeddings().weight.detach().cpu().numpy()
    )
    vocab_gpt2, d_model_gpt2 = gpt2_wte.shape

    # Load BERT weights and tokenizer
    print("Loading BERT model parameters...")
    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BERTModel.from_pretrained("bert-base-uncased")

    # Extract BERT word embedding weight matrix: shape (30522, 768)
    bert_wte = (
        bert_model.get_input_embeddings().weight.detach().cpu().numpy()
    )
    vocab_bert, d_model_bert = bert_wte.shape

    print(
        f"\n[Extracted] GPT-2 -> Vocab Size: {vocab_gpt2:,} | Vector Dim: {d_model_gpt2}D"
    )
    print(
        f"[Extracted] BERT  -> Vocab Size: {vocab_bert:,} | Vector Dim: {d_model_bert}D"
    )

    # Extract real "the" vector indices and embeddings
    gpt2_the_idx = gpt2_tokenizer.encode("the")[0]
    bert_the_idx = bert_tokenizer.encode("the", add_special_tokens=False)[0]

    the_gpt2 = gpt2_wte[gpt2_the_idx]
    the_bert = bert_wte[bert_the_idx]

    print(f"Token 'the' index in GPT-2: {gpt2_the_idx}")
    print(f"Token 'the' index in BERT : {bert_the_idx}")

    # --- STEP 2: ARCHITECTURAL COMPARISON DASHBOARD ---
    print("\n=================================================================")
    print(" SECTION 2: Plotting Architectural & Structural Dashboard")
    print("=================================================================")
    fig = plt.figure(figsize=(14, 5), facecolor="#0f172a")
    ax1 = fig.add_subplot(111)
    ax1.set_facecolor("#1e293b")
    ax1.axis("off")
    ax1.set_title(
        "1. Architectural & Structural Overview (Extracted Weights)",
        fontsize=14,
        fontweight="bold",
        color="#f8fafc",
        pad=15,
    )

    gpt2_text = (
        "GPT-2 (Autoregressive Generator)\n"
        "──────────────────────────────────\n"
        f"• Vocabulary:    {vocab_gpt2:,} Tokens (BPE)\n"
        f"• Model Dim:     d_model = {d_model_gpt2}\n"
        "• Focus:         Subword / Causal Generation\n"
        "• Initial State: Extracted from 'gpt2'"
    )

    bert_text = (
        "BERT (Bidirectional Encoder)\n"
        "──────────────────────────────────\n"
        f"• Vocabulary:    {vocab_bert:,} Tokens (WordPiece)\n"
        f"• Model Dim:     d_model = {d_model_bert}\n"
        "• Focus:         Word-piece / Masked Language\n"
        "• Initial State: Extracted from 'bert-base-uncased'"
    )

    ax1.text(
        0.05,
        0.5,
        gpt2_text,
        transform=ax1.transAxes,
        fontsize=11,
        fontfamily="monospace",
        color="#38bdf8",
        va="center",
        bbox=dict(
            boxstyle="round,pad=1",
            facecolor="#0f172a",
            edgecolor="#38bdf8",
            alpha=0.8,
        ),
    )

    ax1.text(
        0.55,
        0.5,
        bert_text,
        transform=ax1.transAxes,
        fontsize=11,
        fontfamily="monospace",
        color="#f43f5e",
        va="center",
        bbox=dict(
            boxstyle="round,pad=1",
            facecolor="#0f172a",
            edgecolor="#f43f5e",
            alpha=0.8,
        ),
    )

    plt.tight_layout()
    plt.savefig("01_architectural_overview.png", dpi=300, bbox_inches="tight")
    plt.show()

    # --- STEP 3: STATISTICAL & DISTRIBUTIONAL ANALYSIS ---
    print("\n=================================================================")
    print(" SECTION 3: Distributional Discrepancies Across Real Dimensions")
    print("=================================================================")
    gpt2_means = np.mean(gpt2_wte, axis=1)
    bert_means = np.mean(bert_wte, axis=1)

    print(
        f"Real GPT-2 Embedding Mean Offset: {np.mean(gpt2_means):.6f} | Std: {np.std(gpt2_means):.6f}"
    )
    print(
        f"Real BERT  Embedding Mean Offset: {np.mean(bert_means):.6f} | Std: {np.std(bert_means):.6f}"
    )

    fig = plt.figure(figsize=(10, 5), facecolor="#0f172a")
    ax2 = fig.add_subplot(111)
    style_axis(
        ax2,
        "2. Empirical Embedding Mean Distribution Across Dimensions",
        "Embedding Value",
        "Density",
    )

    ax2.hist(
        gpt2_means,
        bins=50,
        density=True,
        alpha=0.6,
        color="#38bdf8",
        label=f"GPT-2 (Mean: {np.mean(gpt2_means):.4f})",
    )
    ax2.hist(
        bert_means,
        bins=50,
        density=True,
        alpha=0.6,
        color="#f43f5e",
        label=f"BERT (Mean: {np.mean(bert_means):.4f})",
    )
    ax2.axvline(0, color="#ffffff", linestyle=":", alpha=0.5)
    ax2.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc")

    plt.tight_layout()
    plt.savefig("02_distributional_discrepancies.png", dpi=300, bbox_inches="tight")
    plt.show()

    # --- STEP 4: CROSS-MODEL VECTOR ORTHOGONALITY ---
    print("\n=================================================================")
    print(" SECTION 4: Empirical Cross-Model Vector Non-Comparability")
    print("=================================================================")
    corr_coef = np.corrcoef(the_gpt2, the_bert)[0, 1]
    cos_sim = get_cosine_similarity(the_gpt2, the_bert)

    print(f"Empirical Pearson Correlation between 'the' vectors: {corr_coef:.4f}")
    print(f"Empirical Cosine Similarity   between 'the' vectors: {cos_sim:.4f}")

    fig = plt.figure(figsize=(10, 5), facecolor="#0f172a")
    ax3 = fig.add_subplot(111)
    style_axis(
        ax3,
        f"3. Cross-Model Vector Non-Comparability ('the')\nPearson Correlation r = {corr_coef:.4f} (Orthogonal)",
        "GPT-2 Vector Dimensions",
        "BERT Vector Dimensions",
    )

    ax3.scatter(
        the_gpt2, the_bert, alpha=0.4, color="#a855f7", s=15, edgecolors="none"
    )

    m, b = np.polyfit(the_gpt2, the_bert, 1)
    ax3.plot(
        the_gpt2,
        m * the_gpt2 + b,
        color="#f59e0b",
        linestyle="--",
        linewidth=1.5,
        label="Linear Fit",
    )
    ax3.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc")

    plt.tight_layout()
    plt.savefig("03_cross_model_orthogonality.png", dpi=300, bbox_inches="tight")
    plt.show()

    # --- STEP 5: REAL CONTEXTUAL TRAJECTORY THROUGH LAYERS ---
    print("\n=================================================================")
    print(" SECTION 5: Extracting Vector Trajectory Across Transformer Depth")
    print("=================================================================")
    sample_text = "The quick brown fox jumps over the lazy dog"

    # Function to extract hidden states across all 12 layers
    def get_layer_hidden_states(model, tokenizer, text):
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        # outputs.hidden_states contains (Layer 0, Layer 1, ..., Layer 12)
        # We compute L2 norm shift relative to the input embedding layer (Layer 0)
        layer_0 = outputs.hidden_states[0][0, 0, :].numpy()
        shifts = [
            np.linalg.norm(outputs.hidden_states[i][0, 0, :].numpy() - layer_0)
            for i in range(len(outputs.hidden_states))
        ]
        return shifts

    gpt2_shifts = get_layer_hidden_states(
        gpt2_model, gpt2_tokenizer, sample_text
    )
    bert_shifts = get_layer_hidden_states(
        bert_model, bert_tokenizer, sample_text
    )
    layers = np.arange(len(gpt2_shifts))

    fig = plt.figure(figsize=(10, 5), facecolor="#0f172a")
    ax4 = fig.add_subplot(111)
    style_axis(
        ax4,
        "4. Dynamic Evolution: Empirical Vector Shift Across Transformer Depth",
        "Transformer Layer Depth (0 = Input Embedding)",
        "Cumulative Representation Shift relative to Layer 0 (L2 Norm)",
    )

    ax4.plot(
        layers,
        gpt2_shifts,
        marker="o",
        color="#38bdf8",
        linewidth=2,
        label="GPT-2 Hidden Trajectory",
    )
    ax4.plot(
        layers,
        bert_shifts,
        marker="s",
        color="#f43f5e",
        linewidth=2,
        label="BERT Hidden Trajectory",
    )
    ax4.set_xticks(layers)
    ax4.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc")

    plt.tight_layout()
    plt.savefig("04_contextual_evolution.png", dpi=300, bbox_inches="tight")
    plt.show()

    # --- STEP 6: HEATMAP MICRO-STRUCTURE COMPARISON ---
    print("\n=================================================================")
    print(" SECTION 6: Static Embedding Micro-structure Visualizations")
    print("=================================================================")
    fig = plt.figure(figsize=(12, 4), facecolor="#0f172a")
    ax5 = fig.add_subplot(111)
    style_axis(
        ax5,
        "5. Static Embedding Micro-structure (First 50 Dimensions)",
        "Vector Dimensions",
        "Token Sample",
    )

    sample_matrix = np.vstack([the_gpt2[:50], the_bert[:50]])
    im = ax5.imshow(sample_matrix, cmap="mako", aspect="auto")
    ax5.set_yticks([0, 1])
    ax5.set_yticklabels(["GPT-2 ('the')", "BERT ('the')"])

    cbar = fig.colorbar(im, ax=ax5, orientation="horizontal", pad=0.25)
    cbar.ax.tick_params(labelsize=8, colors="#94a3b8")
    cbar.set_label("Raw Vector Weight Value", color="#94a3b8", fontsize=9)

    plt.tight_layout()
    plt.savefig("05_embedding_heatmap.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\nPipeline execution finished successfully.")


if __name__ == "__main__":
    main()