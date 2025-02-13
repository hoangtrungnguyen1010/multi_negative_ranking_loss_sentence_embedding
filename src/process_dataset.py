from datasets import load_dataset
from datasets import concatenate_datasets
from grouped_topics import *
titles_dict = {
    "country": country_titles,
    "economy": economy_titles,
    "figure": figure_titles,
    "history": history_titles,
    "geography": geography_titles,
    "chemistry": chemistry_titles,
    "astronomy": astronomy_titles,
    "information_technology": it_titles,
    "religion": religion_titles,
    "language_and_linguistics": language_and_linguistics_titles,
    "social_sciences": social_sciences_titles,
    "biology": biology_titles,
    "law_government": law_government_titles,
    "university": university_titles,
    "miscellaneous": miscellaneous_titles
}

# ==========================
# Define Function to Filter Long Sequences
# ==========================
def filter_long_samples(tokenizer, example, max_length=256):
    """Remove rows where any text field exceeds max token length."""
    query_tokens = tokenizer(example["question"], truncation=False, add_special_tokens=True)["input_ids"]
    pos_tokens = tokenizer(example["context"], truncation=False, add_special_tokens=True)["input_ids"]
    

    # Keep only samples where all texts are within limit
    return len(query_tokens) <= max_length and len(pos_tokens) <= max_length

def get_dataset( tokenizer, hf_path = '', group = None, seed = 42, max_length= 256):
    title_list = None
    if group:
        title_list = titles_dict[group]
    dataset = load_dataset(hf_path)
    merged_dataset = concatenate_datasets([dataset["train"], dataset["validation"], dataset["test"]])
    filtered_dataset = merged_dataset.filter(lambda example: filter_long_samples(tokenizer, example, max_length) and not example["is_impossible"] and (example["title"] in title_list or not title_list))

        # Shuffle dataset before splittisng
    filtered_dataset = filtered_dataset.shuffle(seed=seed)

    # Compute split sizes
    train_size = int(0.8 * len(filtered_dataset))
    val_size = int(0.1 * len(filtered_dataset))
    test_size = len(filtered_dataset) - train_size - val_size  # Ensure all data is used

    # Perform the split
    train_dataset = filtered_dataset.select(range(train_size))
    val_dataset = filtered_dataset.select(range(train_size, train_size + val_size))
    test_dataset = filtered_dataset.select(range(train_size + val_size, len(filtered_dataset)))

    return train_dataset, val_dataset, test_dataset

