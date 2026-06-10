import traceback
import sacrebleu
import torch

def train_epoch(model, dataloader, optimizer, criterion_token, criterion_length, device, pad_token_id,
                length_loss_weight=0.1, grad_clip=1.0, num_length_bins=21, length_bin_offset=10):
    model.train()
    total_loss, total_token_loss, total_length_loss = 0, 0, 0
    num_batches = len(dataloader)

    for batch in dataloader:
        src_input_ids = batch["src_input_ids"].to(device)
        src_attention_mask = batch["src_attention_mask"].to(device)
        tgt_ids = batch["tgt_ids"].to(device)
        tgt_input_ids = batch["tgt_input_ids"].to(device)
        tgt_padding_mask = batch["tgt_padding_mask"].to(device)
        tgt_len = batch["tgt_len"].to(device)
        src_len = batch["src_len"].to(device)

        optimizer.zero_grad()
        logits, length_logits = model(
            src_input_ids=src_input_ids, src_attention_mask=src_attention_mask,
            tgt_input_ids=tgt_input_ids, tgt_padding_mask=tgt_padding_mask
        )

        loss_token = criterion_token(logits.view(-1, logits.shape[-1]), tgt_ids.view(-1))
        
        length_diff = tgt_len - src_len
        target_length_bin = (length_diff + length_bin_offset).long()
        target_length_bin = torch.clamp(target_length_bin, 0, num_length_bins - 1)
        loss_length = criterion_length(length_logits, target_length_bin)

        combined_loss = loss_token + length_loss_weight * loss_length
        combined_loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += combined_loss.item()
        total_token_loss += loss_token.item()
        total_length_loss += loss_length.item()

    return total_loss / num_batches, total_token_loss / num_batches, total_length_loss / num_batches


def evaluate(model, dataloader, tokenizer, device, num_length_bins=21, length_bin_offset=10):
    model.eval()
    hypotheses, references_for_bleu = [], []
    pad_token_id = tokenizer.pad_token_id
    
    try:
        bos_token_id = tokenizer.lang_code_to_id[tokenizer.tgt_lang]
    except AttributeError:
        bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id else 0

    print("\n--- Running Evaluation ---")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            src_input_ids = batch["src_input_ids"].to(device)
            src_attention_mask = batch["src_attention_mask"].to(device)
            tgt_ids_gt = batch["tgt_ids"].to(device)
            tgt_len_gt = batch["tgt_len"].to(device)
            src_len = batch["src_len"].to(device)

            try:
                encoder_outputs = model.encoder(input_ids=src_input_ids, attention_mask=src_attention_mask, return_dict=True)
                memory = encoder_outputs.last_hidden_state.permute(1, 0, 2)
                memory_key_padding_mask = (src_attention_mask == 0)

                length_logits = model.length_predictor(memory, memory_key_padding_mask)
                predicted_bin_index = length_logits.argmax(dim=-1)
                predicted_diff = predicted_bin_index - length_bin_offset
                predicted_lengths = src_len + predicted_diff
                
                max_src_len_batch = src_input_ids.size(1)
                predicted_lengths = torch.clamp(predicted_lengths, min=2, max=max_src_len_batch + 50)

                max_pred_len = predicted_lengths.max().item()
                batch_size = src_input_ids.size(0)
                decoder_input_ids = torch.full((max_pred_len, batch_size), pad_token_id, dtype=torch.long, device=device)
                if max_pred_len > 0: 
                    decoder_input_ids[0, :] = bos_token_id
                tgt_padding_mask_gen = torch.arange(max_pred_len, device=device).unsqueeze(0).expand(batch_size, -1) >= predicted_lengths.unsqueeze(1)

                tgt_emb = model.pos_encoder(model.tgt_embedding(decoder_input_ids) * math.sqrt(model.d_model))
                output = model.decoder(
                    tgt=tgt_emb, memory=memory, tgt_mask=None, memory_mask=None,
                    tgt_key_padding_mask=tgt_padding_mask_gen, memory_key_padding_mask=memory_key_padding_mask,
                )

                logits = model.output_projection(output)
                preds = logits.argmax(dim=-1)

            except Exception as e:
                print(f"\nERROR during generation in batch {batch_idx}: {e}")
                traceback.print_exc()
                continue

            try:
                preds_np = preds.cpu().numpy().T
                tgt_ids_gt_np = tgt_ids_gt.cpu().numpy().T if tgt_ids_gt.dim() > 1 else tgt_ids_gt.cpu().numpy()
                if tgt_ids_gt_np.ndim == 1 and batch_size == 1:
                    tgt_ids_gt_np = tgt_ids_gt_np.reshape(1, -1)

                for i in range(batch_size):
                    actual_pred_len = predicted_lengths[i].item()
                    pred_token_ids_for_item = preds_np[i, :actual_pred_len]
                    hypothesis = tokenizer.decode(pred_token_ids_for_item, skip_special_tokens=True)
                    hypotheses.append(hypothesis)

                    actual_gt_len = int(tgt_len_gt[i].item())
                    gt_token_ids_for_item = tgt_ids_gt_np[i, :actual_gt_len]
                    reference = tokenizer.decode(gt_token_ids_for_item, skip_special_tokens=True)
                    references_for_bleu.append([reference])

            except Exception as e:
                print(f"\nERROR during detokenization in batch {batch_idx}: {e}")
                traceback.print_exc()
                continue

    bleu_score = 0.0
    if hypotheses and references_for_bleu and len(hypotheses) == len(references_for_bleu):
        print(f"Calculating BLEU score for {len(hypotheses)} pairs...")
        try:
            bleu = sacrebleu.corpus_bleu(hypotheses, references_for_bleu, lowercase=True)
            bleu_score = bleu.score
            print(f"\n--- BLEU Calculation Result ---\n{bleu}\n-----------------------------")
        except Exception as e:
             print(f"\nERROR during SacreBLEU calculation: {e}")
    
    print("------ Sample Sentences ------")
    print("Hypothesis: ", hypotheses[:5])
    print("\nReference: ", references_for_bleu[:5])
    print("-----------------------------")

    model.train()
    return bleu_score