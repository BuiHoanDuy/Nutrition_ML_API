"""
Train a NER model using PhoBERT from HuggingFace transformers.
"""

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import torch

LABELS = ["O", "B-FOOD", "I-FOOD", "B-QUANTITY", "I-QUANTITY"]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}

def encode_examples(example, tokenizer):
    # Try to get offset mappings from the tokenizer (fast tokenizer)
    try:
        enc = tokenizer(example["text"], return_offsets_mapping=True, truncation=True)
        offset_mapping = enc["offset_mapping"]
    except NotImplementedError:
        # Fallback for slow (Python) tokenizers that don't provide offset mappings.
        # We approximate offsets by splitting the text into words and assigning each
        # token produced from a word the whole word span. This is not perfect for
        # subword tokenizers but works reasonably for NER on word-level entities.
        text = example["text"]
        words = text.split()
        word_spans = []
        cursor = 0
        for w in words:
            idx = text.find(w, cursor)
            if idx == -1:
                idx = cursor
            start = idx
            end = start + len(w)
            word_spans.append((w, start, end))
            cursor = end

        tokens = []
        offset_mapping = []
        for w, s, e in word_spans:
            toks = tokenizer.tokenize(w)
            for t in toks:
                tokens.append(t)
                offset_mapping.append((s, e))

        # Convert tokens to ids and add special tokens so inputs match model expectations
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = tokenizer.build_inputs_with_special_tokens(token_ids)
        attention_mask = [1] * len(input_ids)

        # Add placeholder None offsets for special tokens (common case: 1 at start and 1 at end)
        try:
            n_special = tokenizer.num_special_tokens_to_add(pair=False)
        except Exception:
            n_special = 0
        if n_special == 2:
            offset_mapping = [(None, None)] + offset_mapping + [(None, None)]
        elif n_special == 1:
            offset_mapping = [(None, None)] + offset_mapping

        enc = {"input_ids": input_ids, "attention_mask": attention_mask, "offset_mapping": offset_mapping}

    labels = ["O"] * len(enc["offset_mapping"])

    for ent in example["entities"]:
        s, e, lab = ent["start"], ent["end"], ent["label"]
        for i, (off_s, off_e) in enumerate(enc["offset_mapping"]):
            if off_s is None:
                continue
            if not (off_e <= s or off_s >= e):
                labels[i] = "B-" + lab if off_s <= s else "I-" + lab

    label_ids = [label2id.get(l, 0) for l in labels]
    enc["labels"] = label_ids
    enc.pop("offset_mapping")
    return enc

def main():
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=True)
    ds = load_dataset("json", data_files={"train": "data/ner_generated.jsonl"}, split="train")
    ds = ds.map(lambda x: encode_examples(x, tokenizer), remove_columns=ds.column_names)

    # Don't convert to torch tensors here — let the data collator pad batches to equal length
    # Use DataCollatorForTokenClassification which pads inputs and labels properly
    data_collator = DataCollatorForTokenClassification(tokenizer)

    model = AutoModelForTokenClassification.from_pretrained(
        "vinai/phobert-base",
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id
    )

    args = TrainingArguments(
        output_dir="services/models/ner_phobert",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        save_strategy="epoch",
        logging_steps=10
    )

    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=data_collator)
    trainer.train()
    trainer.save_model("services/models/ner_phobert")

    print("✅ NER model saved to services/models/ner_phobert")

if __name__ == "__main__":
    main()
