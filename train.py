import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer

from models.nat_model import NATransformer
from utils.dataset import TranslationDataset, collate_fn_hf
from utils.engine import train_epoch, evaluate

def main():
    DATASET_NAME = "wmt14"
    DATASET_CONFIG = "fr-en"
    SRC_LANG = "fr"
    TGT_LANG = "en"
    PRETRAINED_MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
    SRC_LANG_CODE = "fr_XX"
    TGT_LANG_CODE = "en_XX"

    # Hyperparameters
    N_HEADS = 8
    NUM_DECODER_LAYERS = 6
    DIM_FEEDFORWARD = 2048
    DROPOUT = 0.1
    MAX_POS_ENCODING = 1024
    NUM_LENGTH_BINS = 21
    LENGTH_BIN_OFFSET = 10
    BATCH_SIZE = 8 
    NUM_EPOCHS = 50
    LEARNING_RATE = 3e-5 
    GRAD_CLIP = 1.0
    LENGTH_LOSS_WEIGHT = 0.1
    VALIDATION_SPLIT = 0.01 
    MAX_SEQ_LEN = 128 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME, src_lang=SRC_LANG_CODE, tgt_lang=TGT_LANG_CODE)
        config = AutoConfig.from_pretrained(PRETRAINED_MODEL_NAME)
    except Exception as e:
        print(f"Error loading pretrained model/tokenizer: {e}")
        return

    D_MODEL = config.hidden_size
    vocab_size = tokenizer.vocab_size
    pad_token_id = tokenizer.pad_token_id

    print(f"Loading {DATASET_NAME} dataset...")
    try:
        dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, trust_remote_code=True)
        dataset['train'] = dataset['train'].select(range(45000))

        if 'validation' not in dataset or len(dataset['validation']) == 0:
             if len(dataset['train']) > 10000:
                 train_test_split = dataset['train'].train_test_split(test_size=VALIDATION_SPLIT, seed=42)
                 dataset['train'] = train_test_split['train']
                 dataset['validation'] = train_test_split['test']
             else:
                 dataset['validation'] = dataset['train']
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    train_dataset = TranslationDataset(dataset['train'], tokenizer, SRC_LANG, TGT_LANG, max_seq_len=MAX_SEQ_LEN)
    val_dataset = TranslationDataset(dataset['validation'], tokenizer, SRC_LANG, TGT_LANG, max_seq_len=MAX_SEQ_LEN)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=lambda b: collate_fn_hf(b, pad_token_id), num_workers=2, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                collate_fn=lambda b: collate_fn_hf(b, pad_token_id), num_workers=2, pin_memory=True)

    model = NATransformer(
        tgt_vocab_size=vocab_size, d_model=D_MODEL, nhead=N_HEADS,
        num_decoder_layers=NUM_DECODER_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT, max_len=MAX_POS_ENCODING, pretrained_encoder_name=PRETRAINED_MODEL_NAME,
        num_length_bins=NUM_LENGTH_BINS
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion_token = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    criterion_length = nn.CrossEntropyLoss()

    best_bleu = -1.0
    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()
        avg_loss, avg_token_loss, avg_length_loss = train_epoch(
            model, train_dataloader, optimizer, criterion_token, criterion_length, device, pad_token_id,
            LENGTH_LOSS_WEIGHT, GRAD_CLIP, NUM_LENGTH_BINS, LENGTH_BIN_OFFSET
        )
        epoch_mins, epoch_secs = divmod(time.time() - start_time, 60)

        print(f"\nEpoch {epoch}/{NUM_EPOCHS} | Time: {int(epoch_mins)}m {int(epoch_secs)}s")
        print(f"\tTrain Loss: {avg_loss:.4f} | Token Loss: {avg_token_loss:.4f} | Length Loss: {avg_length_loss:.4f}")

        if val_dataloader:
            epoch_bleu = evaluate(model, val_dataloader, tokenizer, device, NUM_LENGTH_BINS, LENGTH_BIN_OFFSET)
            print(f"\tValidation BLEU: {epoch_bleu:.2f}")
            if epoch_bleu > best_bleu:
                best_bleu = epoch_bleu
                torch.save(model.state_dict(), 'best_model.pt')

    torch.save(model.state_dict(), 'NAT_mBART_final.pt')
    print("Training Complete. Final Model Saved.")

if __name__ == "__main__":
    main()