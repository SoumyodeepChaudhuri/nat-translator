from typing import List, Dict
import torch
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, src_lang_key, tgt_lang_key, max_seq_len=128):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.src_lang_key = src_lang_key
        self.tgt_lang_key = tgt_lang_key
        self.pad_token_id = tokenizer.pad_token_id
        
        try:
             self.bos_token_id = tokenizer.lang_code_to_id[tokenizer.tgt_lang]
        except AttributeError:
             self.bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id else 0
             print(f"Warning: Using fallback BOS token ID: {self.bos_token_id}")

        self.eos_token_id = tokenizer.eos_token_id
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]['translation']
        src_text = item[self.src_lang_key]
        tgt_text = item[self.tgt_lang_key]

        src_encoded = self.tokenizer(src_text, truncation=True, max_length=self.max_seq_len, padding=False, return_tensors=None)
        src_input_ids = src_encoded['input_ids']
        src_attention_mask = src_encoded['attention_mask']

        tgt_encoded = self.tokenizer(text_target=tgt_text, truncation=True, max_length=self.max_seq_len, padding=False, return_tensors=None)
        tgt_ids = tgt_encoded['input_ids']

        src_len = len(src_input_ids)
        tgt_len = len(tgt_ids)

        tgt_input_ids = [self.bos_token_id] + [self.pad_token_id] * (tgt_len - 1) if tgt_len > 0 else []

        return {
            "src_input_ids": torch.tensor(src_input_ids, dtype=torch.long),
            "src_attention_mask": torch.tensor(src_attention_mask, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long),
            "tgt_input_ids": torch.tensor(tgt_input_ids[:self.max_seq_len], dtype=torch.long),
            "tgt_len": torch.tensor(tgt_len, dtype=torch.long),
            "src_len": torch.tensor(src_len, dtype=torch.long)
        }

def collate_fn_hf(batch: List[Dict[str, torch.Tensor]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    max_src_len = max(item["src_input_ids"].shape[0] for item in batch)
    max_tgt_len = max(item["tgt_ids"].shape[0] for item in batch)
    max_tgt_input_len = max(item["tgt_input_ids"].shape[0] for item in batch)

    src_ids_padded, src_masks_padded, tgt_ids_padded = [], [], []
    tgt_input_ids_padded, tgt_masks_padded = [], []
    src_lens, tgt_lens = [], []

    for item in batch:
        src_len = item["src_input_ids"].shape[0]
        src_pad_len = max_src_len - src_len
        src_ids_padded.append(torch.cat([item["src_input_ids"], torch.full((src_pad_len,), pad_token_id, dtype=torch.long)], dim=0))
        src_masks_padded.append(torch.cat([item["src_attention_mask"], torch.zeros(src_pad_len, dtype=torch.long)], dim=0))
        src_lens.append(item["src_len"])

        tgt_len = item["tgt_ids"].shape[0]
        tgt_pad_len = max_tgt_len - tgt_len
        tgt_ids_padded.append(torch.cat([item["tgt_ids"], torch.full((tgt_pad_len,), pad_token_id, dtype=torch.long)], dim=0))
        tgt_lens.append(item["tgt_len"])

        tgt_input_len = item["tgt_input_ids"].shape[0]
        tgt_input_pad_len = max_tgt_input_len - tgt_input_len
        tgt_input_ids_padded.append(torch.cat([item["tgt_input_ids"], torch.full((tgt_input_pad_len,), pad_token_id, dtype=torch.long)], dim=0))
        tgt_masks_padded.append(torch.cat([torch.zeros(tgt_input_len, dtype=torch.bool), torch.ones(tgt_input_pad_len, dtype=torch.bool)], dim=0))

    return {
        "src_input_ids": torch.stack(src_ids_padded, dim=0),
        "src_attention_mask": torch.stack(src_masks_padded, dim=0),
        "tgt_ids": torch.stack(tgt_ids_padded, dim=1),
        "tgt_input_ids": torch.stack(tgt_input_ids_padded, dim=1),
        "tgt_padding_mask": torch.stack(tgt_masks_padded, dim=0),
        "src_len": torch.stack(src_lens, dim=0),
        "tgt_len": torch.stack(tgt_lens, dim=0)
    }