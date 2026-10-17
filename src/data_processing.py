import json
import re

from huggingface_hub import snapshot_download
from sklearn.model_selection import train_test_split

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
INPUT_PATH = f"{RAW_DIR}/TeleQnA.txt"
OUTPUT_JSON = f"{RAW_DIR}/TeleQnA.json"

def load_json(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        data = {}
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {file_path}")
        data = {}
    return data

def convert_txt_to_json():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content.startswith("{"):
            content = "{\n" + content + "\n}"
        data = json.loads(content)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def filter_by_release(data):
    filtered_questions = {}
    pattern = re.compile(
        rf"\b(rel(ease)?[-_\s]*17|r17)\b",
        re.IGNORECASE,
    )
    data = load_json(OUTPUT_JSON)

    for q_id, q_info in data.items():
        full_text = " ".join(
            [
                q_info.get("question", ""),
                q_info.get("explanation", ""),
                q_info.get("option 1", ""),
                q_info.get("option 2", ""),
                q_info.get("option 3", ""),
                q_info.get("option 4", ""),
            ]
        )

        if pattern.search(full_text):
            filtered_questions[q_id] = q_info

    return filtered_questions

def save_filtered_data(filtered_data):
    with open(
        f"{PROCESSED_DIR}/TeleQnA_Rel17.json", "w", encoding="utf-8"
    ) as f_out_json:
        json.dump(filtered_data, f_out_json, indent=4, ensure_ascii=False)

def process_data(release_number:int=16):
    convert_txt_to_json()
    filtered_data = filter_by_release(load_json(OUTPUT_JSON))
    save_filtered_data(filtered_data)

def make_sets():
    loaded_data = load_json(f"{PROCESSED_DIR}/TeleQnA_Rel17.json")
    data_list = list(loaded_data.values())

    train_data, dev_data = train_test_split(data_list, test_size=0.3, random_state=42)
    
    json.dump(train_data, open(f"{PROCESSED_DIR}/TeleQnA_Rel17_train.json", "w", encoding="utf-8"), indent=4, ensure_ascii=False)
    json.dump(dev_data, open(f"{PROCESSED_DIR}/TeleQnA_Rel17_dev.json", "w", encoding="utf-8"), indent=4, ensure_ascii=False)


def download_tspec_dataset():
    """
    Download the TSpec dataset from Hugging Face and save it to the specified local directory.
    """
    snapshot_download(
        repo_id="rasoul-nikbakht/TSpec-LLM",
        repo_type="dataset",
        local_dir="data/raw/tspec",
        allow_patterns=["3GPP-clean/Rel-17/38_series/*.md"],
    )

if __name__ == "__main__":
    process_data()
    make_sets()
    download_tspec_dataset()