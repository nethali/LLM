"""
Tokenization Compression Ratio Analysis using GPT-4 Tokenizer (tiktoken)
--------------------------------------------------------------------------
This script calculates tokenization compression metrics across 10 Project Gutenberg
books and 10 real-world web pages.

Metrics computed:
1. Bytes per Token  (len(utf8_bytes) / num_tokens)
2. Chars per Token  (len(raw_chars) / num_tokens)
"""

import re
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import tiktoken

# Initialize GPT-4 Tokenizer ('cl100k_base' is used by gpt-4 and gpt-3.5-turbo)
tokenizer = tiktoken.get_encoding("cl100k_base")

# ==============================================================================
# 1. Dataset Definitions
# ==============================================================================

# 10 Public Domain Books directly downloadable from Project Gutenberg (.txt)
BOOKS = {
    "Frankenstein": "https://www.gutenberg.org/files/84/84-0.txt",
    "Pride and Prejudice": "https://www.gutenberg.org/files/1342/1342-0.txt",
    "Moby Dick": "https://www.gutenberg.org/files/2701/2701-0.txt",
    "Alice in Wonderland": "https://www.gutenberg.org/files/11/11-0.txt",
    "The Great Gatsby": "https://www.gutenberg.org/cache/epub/64317/pg64317.txt",
    "Dracula": "https://www.gutenberg.org/files/345/345-0.txt",
    "The Adventures of Sherlock Holmes": "https://www.gutenberg.org/files/1661/1661-0.txt",
    "A Tale of Two Cities": "https://www.gutenberg.org/files/98/98-0.txt",
    "War and Peace": "https://www.gutenberg.org/files/2600/2600-0.txt",
    "Metamorphosis": "https://www.gutenberg.org/files/5200/5200-0.txt",
}

# 10 Popular Public Websites
WEBSITES = {
    "Wikipedia (Main)": "https://en.wikipedia.org/wiki/Main_Page",
    "Python Documentation": "https://docs.python.org/3/",
    "BBC News": "https://www.bbc.com/news",
    "GitHub Trending": "https://github.com/trending",
    "Hacker News": "https://news.ycombinator.com/",
    "ArXiv (AI)": "https://arxiv.org/list/cs.AI/recent",
    "Stack Overflow": "https://stackoverflow.com/questions",
    "NPR": "https://www.npr.org/",
    "Internet Archive": "https://archive.org/",
    "W3Schools": "https://www.w3schools.com/",
}


# ==============================================================================
# 2. Helper Functions
# ==============================================================================


def fetch_text_from_url(url: str, is_html: bool = False) -> str:
    """Fetch raw text content from a URL."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=15)
    
    # Use raise_for_status()
    response.raise_for_status()
    
    # Standard encoding handling in requests
    response.encoding = response.apparent_encoding or "utf-8"

    text = response.text

    if is_html:
        # Clean HTML tags and extract readable visible text
        soup = BeautifulSoup(text, "html.parser")
        for script_or_style in soup(["script", "style", "noscript"]):
            script_or_style.decompose()
        text = soup.get_text(separator=" ")
        # Clean up excessive blank spaces/newlines
        text = re.sub(r"\s+", " ", text).strip()

    return text


def compute_compression_metrics(name: str, category: str, text: str) -> dict:
    """Calculate token count and compression ratios."""
    num_chars = len(text)
    num_bytes = len(text.encode("utf-8"))

    # Tokenize using GPT-4 encoding
    tokens = tokenizer.encode(text)
    num_tokens = len(tokens)

    chars_per_token = num_chars / num_tokens if num_tokens > 0 else 0
    bytes_per_token = num_bytes / num_tokens if num_tokens > 0 else 0

    return {
        "Name": name,
        "Category": category,
        "Characters": num_chars,
        "Bytes": num_bytes,
        "Tokens": num_tokens,
        "Chars / Token": round(chars_per_token, 3),
        "Bytes / Token": round(bytes_per_token, 3),
    }


# ==============================================================================
# 3. Execution & Processing Loop
# ==============================================================================


def main():
    results = []

    print("=" * 70)
    print("1. Downloading and Tokenizing 10 Books (Gutenberg TXT)...")
    print("=" * 70)
    for name, url in BOOKS.items():
        try:
            print(f"  Fetching: {name}...")
            text = fetch_text_from_url(url, is_html=False)
            metrics = compute_compression_metrics(name, "Book", text)
            results.append(metrics)
        except Exception as e:
            print(f"  Failed to fetch {name}: {e}")

    print("\n" + "=" * 70)
    print("2. Downloading and Tokenizing 10 Websites (Cleaned Text)...")
    print("=" * 70)
    for name, url in WEBSITES.items():
        try:
            print(f"  Fetching: {name}...")
            text = fetch_text_from_url(url, is_html=True)
            metrics = compute_compression_metrics(name, "Website", text)
            results.append(metrics)
        except Exception as e:
            print(f"  Failed to fetch {name}: {e}")

    # Convert results into a DataFrame
    df = pd.DataFrame(results)

    # Display Results Table
    print("\n" + "=" * 70)
    print("COMPRESSION RATIO RESULTS (GPT-4 cl100k_base Tokenizer)")
    print("=" * 70)
    print(df.to_string(index=False))

    # Calculate Group Averages
    print("\n" + "=" * 70)
    print("CATEGORY AVERAGES")
    print("=" * 70)
    avg_df = (
        df.groupby("Category")[["Chars / Token", "Bytes / Token"]]
        .mean()
        .reset_index()
    )
    print(avg_df.to_string(index=False))

    # ==============================================================================
    # 4. Visualization
    # ==============================================================================
    plt.figure(figsize=(14, 6))

    # Plot 1: Bytes / Token Comparison
    plt.subplot(1, 2, 1)
    colors = ["#2b5c8f" if c == "Book" else "#d95f02" for c in df["Category"]]
    plt.barh(df["Name"], df["Bytes / Token"], color=colors, alpha=0.85)
    plt.axvline(
        df[df["Category"] == "Book"]["Bytes / Token"].mean(),
        color="#2b5c8f",
        linestyle="--",
        label="Book Avg",
    )
    plt.axvline(
        df[df["Category"] == "Website"]["Bytes / Token"].mean(),
        color="#d95f02",
        linestyle="--",
        label="Website Avg",
    )
    plt.title("Bytes per Token (GPT-4 / cl100k_base)")
    plt.xlabel("Bytes / Token (Higher = Better Compression)")
    plt.legend()
    plt.gca().invert_yaxis()
    plt.grid(True, linestyle=":", alpha=0.6)

    # Plot 2: Chars / Token Comparison
    plt.subplot(1, 2, 2)
    plt.barh(df["Name"], df["Chars / Token"], color=colors, alpha=0.85)
    plt.axvline(
        df[df["Category"] == "Book"]["Chars / Token"].mean(),
        color="#2b5c8f",
        linestyle="--",
        label="Book Avg",
    )
    plt.axvline(
        df[df["Category"] == "Website"]["Chars / Token"].mean(),
        color="#d95f02",
        linestyle="--",
        label="Website Avg",
    )
    plt.title("Characters per Token (GPT-4 / cl100k_base)")
    plt.xlabel("Characters / Token")
    plt.legend()
    plt.gca().invert_yaxis()
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()