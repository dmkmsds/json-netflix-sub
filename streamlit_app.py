#######################
# streamlit_app.py
#######################
import streamlit as st
import pandas as pd
import json
import os
import random
import jieba
from pypinyin import lazy_pinyin, Style
import torch
import itertools
from transformers import AutoTokenizer, AutoModel



########################
# 0)google translate
########################

from google.cloud import translate_v2 as translate


# ------------------ LOAD GOOGLE CLOUD CREDENTIALS FROM SECRETS ------------------

def configure_google_credentials():
    """
    Load Google Cloud service account credentials either from an environment
    variable (GOOGLE_CREDS_JSON) or from st.secrets['google_cloud'].
    Then write them to a temp file and set GOOGLE_APPLICATION_CREDENTIALS.
    """
    creds_dict = None

    # 1) Check if credentials are supplied via environment variable
    if "GOOGLE_CREDS_JSON" in os.environ:
        # Parse JSON from the env variable
        creds_dict = json.loads(os.environ["GOOGLE_CREDS_JSON"])

    # 2) Else check if credentials are in st.secrets
    elif "google_cloud" in st.secrets:
        creds_dict = dict(st.secrets["google_cloud"])

    # 3) If neither is found, raise an error
    else:
        st.error("No Google Cloud credentials found. Please set GOOGLE_CREDS_JSON or st.secrets['google_cloud'].")
        return

    # 4) Write the loaded credentials dict to a temp file
    creds_path = "temp_google_credentials.json"
    with open(creds_path, "w") as f:
        json.dump(creds_dict, f)

    # 5) Set the GOOGLE_APPLICATION_CREDENTIALS env var
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path

# Example usage:
configure_google_credentials()

# create a client using your credentials from st.secrets, etc.
translation_client = translate.Client()



def google_translate_ch_to_en(ch_text: str) -> str:
    """
    Given a Chinese string, returns its English translation via Google Translate.
    If there's an error, return the original text.
    """
    ch_text = ch_text.strip()
    if not ch_text:
        return ""
    try:
        result = translation_client.translate(
            ch_text,
            target_language="en",  # we want English
            source_language="zh-CN" # or 'auto' if your input is definitely Chinese
        )
        return result["translatedText"]
    except Exception as e:
        st.write(f"Google translation error: {e}")
        return ch_text



########################
# 1) LOAD AWESOME-ALIGN MODEL
########################
@st.cache_resource
def load_awesome_align_model():
    """
    Loads a pre-trained awesome-align model.
    See https://github.com/neulab/awesome-align
    For example, "aneuraz/awesome-align-with-co".
    """
    model = AutoModel.from_pretrained("aneuraz/awesome-align-with-co",
                                     use_auth_token = "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")
    tokenizer = AutoTokenizer.from_pretrained("aneuraz/awesome-align-with-co",
                                             use_auth_token = "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")
                                             
    model.eval()
    return model, tokenizer

model, tokenizer = load_awesome_align_model()


########################
# 2) AWESOME-ALIGN FUNCTION
########################
def awesome_align(english_text, chinese_text):
    """
    Replicates the logic from your snippet. We:

    - Tokenize the English text with .split()
    - Tokenize the Chinese text with jieba
    - Embed them with the model
    - Compute alignment at layer=8
    - Return a set of (src_idx, tgt_idx, prob).
    """
    align_layer = 8
    threshold = 1e-3

    sent_src = english_text.strip().split()
    sent_tgt = list(jieba.cut(chinese_text))

    # Tokenize
    token_src = [tokenizer.tokenize(word) for word in sent_src]  # list of lists
    token_tgt = [tokenizer.tokenize(word) for word in sent_tgt]
    wid_src = [tokenizer.convert_tokens_to_ids(x) for x in token_src]
    wid_tgt = [tokenizer.convert_tokens_to_ids(x) for x in token_tgt]

    ids_src = tokenizer.prepare_for_model(
        list(itertools.chain(*wid_src)),
        return_tensors='pt',
        truncation=True
    )['input_ids']
    ids_tgt = tokenizer.prepare_for_model(
        list(itertools.chain(*wid_tgt)),
        return_tensors='pt',
        truncation=True
    )['input_ids']

    # sub2word_map lets us know which "word" each subword belongs to
    sub2word_map_src = []
    for i, word_list in enumerate(token_src):
        sub2word_map_src += [i] * len(word_list)
    sub2word_map_tgt = []
    for i, word_list in enumerate(token_tgt):
        sub2word_map_tgt += [i] * len(word_list)

    with torch.no_grad():
        outputs_src = model(ids_src.unsqueeze(0), output_hidden_states=True)
        out_src = outputs_src.hidden_states[align_layer][0, 1:-1]  # remove [CLS], [SEP]

        outputs_tgt = model(ids_tgt.unsqueeze(0), output_hidden_states=True)
        out_tgt = outputs_tgt.hidden_states[align_layer][0, 1:-1]

        # Dot product
        dot_prod = torch.matmul(out_src, out_tgt.transpose(-1, -2))
        softmax_srctgt = torch.nn.Softmax(dim=-1)(dot_prod)
        softmax_tgtsrc = torch.nn.Softmax(dim=-2)(dot_prod)
        softmax_inter = (softmax_srctgt > threshold) * (softmax_tgtsrc > threshold)

    align_subwords = torch.nonzero(softmax_inter, as_tuple=False)  # (N,2)
    align_dict = {}

    for i_idx, j_idx in align_subwords:
        prob = (softmax_srctgt[i_idx, j_idx].item() + softmax_tgtsrc[i_idx, j_idx].item()) / 2
        src_word_id = sub2word_map_src[i_idx.item()]
        tgt_word_id = sub2word_map_tgt[j_idx.item()]
        key = (src_word_id, tgt_word_id)
        align_dict.setdefault(key, []).append(prob)

    # Aggregated per (word_src_idx, word_tgt_idx)
    aggregated_alignments = []
    for (k0, k1), probs in align_dict.items():
        aggregated_alignments.append((k0, k1, sum(probs)/len(probs)))

    return aggregated_alignments


########################
# 3) COLOR-CODING LOGIC
########################
def color_code_by_alignment(english_text, chinese_text):
    """
    Uses a bright palette suitable for a black background.
    Unaligned tokens default to 'white'.
    Each Chinese word is expanded into multiple pinyin syllables
    that share the same color.
    """
    import random
    import jieba
    from pypinyin import lazy_pinyin, Style

    # Bright colors that pop on black
    normal_palette = [
        "red",       # bright red
        "lime",      # bright lime green
        "cyan",      # bright cyan
        "yellow",    # bright yellow
        "magenta",   # bright magenta
        "orange",    # bright orange
        "pink",      # bright pink
        "gold",      # gold
        "dodgerblue",
        "lawngreen"
    ]
    color_idx = 0

    # English tokens (source)
    src_tokens = english_text.strip().split()
    # Chinese tokens (target) from jieba
    tgt_tokens = list(jieba.cut(chinese_text))

    # Get alignments
    alignments = awesome_align(english_text, chinese_text)

    # Forward & reverse mappings
    mapping_dict = {}
    reverse_mapping = {}
    for (i, j, _) in alignments:
        mapping_dict.setdefault(i, set()).add(j)
        reverse_mapping.setdefault(j, set()).add(i)

    # Assign colors to each English token
    color_mapping_src = {}
    for i, word in enumerate(src_tokens):
        if i in mapping_dict and color_idx < len(normal_palette):
            color_mapping_src[i] = normal_palette[color_idx]
            color_idx = (color_idx + 1) % len(normal_palette)
        else:
            # Default color for unaligned tokens
            color_mapping_src[i] = "white"

    # Assign colors to each Chinese token
    color_mapping_tgt = {}
    for j, word in enumerate(tgt_tokens):
        if j in reverse_mapping:
            # pick any aligned source index
            src_id = list(reverse_mapping[j])[0]
            color_mapping_tgt[j] = color_mapping_src.get(src_id, "white")
        else:
            # Default color
            color_mapping_tgt[j] = "white"

    # Build final English list
    eng_colored = [(word, color_mapping_src.get(i, "white")) for i, word in enumerate(src_tokens)]

    # Chinese list for reference (not required to change)
    chn_colored = [(word, color_mapping_tgt.get(j, "white")) for j, word in enumerate(tgt_tokens)]

    # Expand each Chinese token into multiple pinyin syllables with the same color
    pin_colored_expanded = []
    for j, word in enumerate(tgt_tokens):
        color = color_mapping_tgt.get(j, "white")
        pinyin_syllables = lazy_pinyin(word, style=Style.TONE)
        for syll in pinyin_syllables:
            pin_colored_expanded.append((syll, color))

    return eng_colored, chn_colored, pin_colored_expanded




########################
# 4) STREAMLIT APP
########################
def main():
    st.title("Subtitle Alignment + Color Coding (Awesome-Align)")

    # User uploads a subtitle text file with columns: Time, Subtitle, Translation
    uploaded_file = st.file_uploader("Upload your subtitle file", type=["txt", "tsv", "csv"])
    if uploaded_file is not None:
        # Basic DataFrame reading (tab-separated, ignoring header row if you have it)
        df = pd.read_csv(
            uploaded_file,
            engine="python",
            header=0
        )
        df = df.fillna("")
        st.write("Parsed Data")
        st.dataframe(df)

        final_output = []
        for i in range(len(df)):
            row = df.iloc[i]
            time_str = row["Time"].strip()
            subtitle_ch = row["Subtitle"].strip()       # The Chinese line
            
            # Translate to English via Google
            translation_en = google_translate_ch_to_en(subtitle_ch)

            # Next line's time (for end_time), or blank if last
            if i < len(df) - 1:
                end_time_str = df.iloc[i+1]["Time"].strip()
            else:
                end_time_str = ""

            # Use the snippet logic: English is "source", Chinese is "target"
            eng_colored, chn_colored, pin_colored_expanded = color_code_by_alignment(translation_en, subtitle_ch)
            
            # Build strings for top (pinyin) and bottom (English)
            top_tokens_str   = " ".join(t for (t, c) in pin_colored_expanded)
            top_colors_array = [c for (t, c) in pin_colored_expanded]

            bottom_tokens_str   = " ".join(t for (t, c) in eng_colored)
            bottom_colors_array = [c for (t, c) in eng_colored]

            # IMPORTANT CHANGE: Convert color arrays to single bracketed strings
            # Example: "[red, red, lime]" instead of ["red","red","lime"]
            color_top_string = "[" + ", ".join(top_colors_array) + "]"
            color_bottom_string = "[" + ", ".join(bottom_colors_array) + "]"

            # Build the final dict
            line_data = {
                "start_time": time_str,
                "end_time": end_time_str,
                "top_subtitle": top_tokens_str,
                "bottom_subtitle": bottom_tokens_str,
                "color_top_sub": color_top_string,      # <--- changed
                "color_bottom_sub": color_bottom_string # <--- changed
            }
            final_output.append(line_data)

        # Show final JSON in the UI
        st.subheader("Final Output JSON")
        st.json(final_output)

        # Also provide a download button
        json_str = json.dumps(final_output, ensure_ascii=False, indent=2)
        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name="subtitle_alignment.json",
            mime="application/json",
        )

    else:
        st.info("Please upload a .txt, .tsv, or .csv file with columns: Time, Subtitle, Translation.")


if __name__ == "__main__":
    main()
