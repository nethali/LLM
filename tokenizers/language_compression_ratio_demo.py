"""
Tokenization Compression Ratio Comparison: GPT-4 vs. Multilingual BERT
------------------------------------------------------------------------
This script compares how GPT-4 (BPE) and Multilingual BERT (WordPiece) 
tokenize identical semantic sentences across 7 different languages.
"""

import pandas as pd
from tabulate import tabulate
import tiktoken
from transformers import BertTokenizer

# 1. Initialize Tokenizers
gpt4_tokenizer = tiktoken.get_encoding("cl100k_base")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

# 2. Sentences Dataset in 7 languages
TRANSLATIONS = {
    "English": (
        "I translated one sentence into six languages. The meaning stayed the"
        " same, but the token count exploded so dramatically that my AI budget"
        " immediately requested parental supervision."
    ),
    "Sinhala": (
        "මම එක් වාක්‍යක් භාෂා හයකට පරිවර්තනය කළා. එහි අර්ථය වෙනස් වුණේ නැතත්,"
        " ටෝකන ගණන කෙතරම් සීඝ්‍රයෙන් වැඩි වුණාද කියනවා නම්, මගේ AI අයවැයට"
        " වහාම දෙමාපිය අධීක්ෂණය අවශ්‍ය වුණා."
    ),
    "Japanese": (
        "1つの文章を6つの言語に翻訳しました。意味は同じままでしたが、トークン数が劇的に爆発したため、私のAI予算はすぐに親の監督を要求しました。"
    ),
    "Mandarin Chinese": (
        "我把一句话翻译成了六种语言。意思虽然没变，但 Token 数量剧烈暴涨，以至于我的 AI"
        " 预算立刻要求家长监管。"
    ),
    "Spanish": (
        "Traduje una oración a seis idiomas. El significado se mantuvo igual,"
        " pero el recuento de tokens explotó de forma tan dramática que mi"
        " presupuesto de IA solicitó inmediatamente supervisión paterna."
    ),
    "Arabic": (
        "ترجمتُ جملة واحدة إلى ست لغات. بقي المعنى كما هو، لكن عدد الرموز"
        " (Tokens) انفجر بشكل هائل لدرجة أن ميزانية الذكاء الاصطناعي الخاصة بي"
        " طلبت فورًا إشرافًا أبويًا."
    ),
    "Thai": (
        "ฉันแปลประโยคเดียวเป็นหกภาษา ความหมายยังคงเหมือนเดิม"
        " แต่จำนวนโทเค็นกลับพุ่งสูงขึ้นอย่างรุนแรงจนงบประมาณ AI"
        " ของฉันต้องร้องขอการดูแลจากผู้ปกครองทันที"
    ),
}


def analyze_tokenization():
    records = []

    for lang, text in TRANSLATIONS.items():
        num_chars = len(text)
        num_bytes = len(text.encode("utf-8"))

        # --- GPT-4 Tokenization ---
        gpt4_tokens = gpt4_tokenizer.encode(text)
        gpt4_count = len(gpt4_tokens)
        gpt4_chars_per_tok = num_chars / gpt4_count
        gpt4_bytes_per_tok = num_bytes / gpt4_count

        # --- BERT Tokenization ---
        # add_special_tokens=False to measure pure text tokens without [CLS]/[SEP]
        bert_tokens = bert_tokenizer.encode(text, add_special_tokens=False)
        bert_count = len(bert_tokens)
        bert_chars_per_tok = num_chars / bert_count
        bert_bytes_per_tok = num_bytes / bert_count

        records.append({
            "Language": lang,
            "Chars": num_chars,
            "Bytes": num_bytes,
            "GPT-4 Tokens": gpt4_count,
            "GPT-4 Chars/Tok": round(gpt4_chars_per_tok, 2),
            "GPT-4 Bytes/Tok": round(gpt4_bytes_per_tok, 2),
            "BERT Tokens": bert_count,
            "BERT Chars/Tok": round(bert_chars_per_tok, 2),
            "BERT Bytes/Tok": round(bert_bytes_per_tok, 2),
        })

    df = pd.DataFrame(records)

    # Reordering columns strictly in requested structure
    ordered_columns = [
        "Language",
        "Chars",
        "Bytes",
        "GPT-4 Tokens",
        "GPT-4 Chars/Tok",
        "GPT-4 Bytes/Tok",
        "BERT Tokens",
        "BERT Chars/Tok",
        "BERT Bytes/Tok",
    ]
    df = df[ordered_columns]

    # Display clean table
    print("=" * 110)
    print("TOKENIZATION COMPRESSION RATIO ANALYSIS (GPT-4 vs Multilingual BERT)")
    print("Note: Higher Chars/Tok or Bytes/Tok = Higher Compression Efficiency")
    print("=" * 110)
    print(tabulate(df, headers="keys", tablefmt="presto", showindex=False))


if __name__ == "__main__":
    analyze_tokenization()