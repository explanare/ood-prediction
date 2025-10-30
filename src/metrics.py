import torch
from tqdm.auto import tqdm


def compute_per_token_loss(model, encoded_inputs, labels, temperature=None):
  loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
  with torch.no_grad():
    # position_ids is required if there are paddings.
    position_ids = encoded_inputs.attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(encoded_inputs.attention_mask == 0, 1)
    outputs = model(**encoded_inputs, position_ids=position_ids, labels=labels)
    shift_logits = outputs.logits[:, :-1, :].contiguous()
    if temperature is not None:
      shift_logits = shift_logits / temperature
    labels = labels[:, 1:].unsqueeze(-1).contiguous().to(torch.int64)
    loss = loss_fn(
      shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1)
    )
    loss = loss.view(labels.size(0), -1)
    return loss


def compute_per_token_loss_batched(
  model, encoded_inputs, labels, temperature=None, batch_size=64
):
  loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
  loss_agg = []
  for b_i in tqdm(range(0, len(labels), batch_size)):
    with torch.no_grad():
      # position_ids is required if there are paddings.
      input_ids = encoded_inputs.input_ids[b_i : b_i + batch_size]
      attention_mask = encoded_inputs.attention_mask[b_i : b_i + batch_size]
      batch_labels = labels[b_i : b_i + batch_size]
      position_ids = attention_mask.long().cumsum(-1) - 1
      position_ids.masked_fill_(attention_mask == 0, 1)
      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        labels=batch_labels,
      )
      shift_logits = outputs.logits[:, :-1, :].contiguous()
      if temperature is not None:
        shift_logits = shift_logits / temperature
      batch_labels = (
        batch_labels[:, 1:].unsqueeze(-1).contiguous().to(torch.int64)
      )
      loss = loss_fn(
        shift_logits.view(-1, shift_logits.size(-1)), batch_labels.view(-1)
      )
      loss = loss.view(batch_labels.size(0), -1)
      loss_agg.extend(loss.tolist())
  return loss_agg


def compute_per_token_probability(
  model, encoded_inputs, labels, temperature=None
):
  with torch.no_grad():
    # position_ids is required if there are paddings.
    position_ids = encoded_inputs.attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(encoded_inputs.attention_mask == 0, 1)
    attention_mask = encoded_inputs.attention_mask
    outputs = model(
      input_ids=encoded_inputs.input_ids,
      attention_mask=attention_mask,
      position_ids=position_ids,
      labels=labels,
    )

    shift_logits = outputs.logits[:, :-1, :].contiguous()
    if temperature is not None:
      shift_logits = shift_logits / temperature
    full_dist = torch.nn.functional.softmax(shift_logits, dim=-1)
    labels = labels[:, 1:].unsqueeze(-1).contiguous().to(torch.int64)
    # Index into prob with labels
    safe_labels = torch.maximum(labels, torch.zeros_like(labels))
    assert torch.max(safe_labels).tolist() < full_dist.shape[-1]
    assert torch.min(safe_labels).tolist() >= 0
    label_prob = torch.gather(full_dist, -1, safe_labels)
    label_prob = torch.where(
      labels >= 0, label_prob, torch.ones_like(label_prob) * -1
    )
    return label_prob, full_dist


def compute_confidence_score(
  model,
  tokenizer,
  prompt_batch,
  output_batch,
  score_fn=None,
  max_length=128,
  batch_size=32,
  temperature=None,
):
  assert len(prompt_batch) == len(output_batch)
  input_batch = [
    prompt_batch[i] + output_batch[i] for i in range(len(prompt_batch))
  ]

  all_scores = []
  for i in tqdm(range(0, len(input_batch), batch_size)):
    encoded_inputs = tokenizer(
      input_batch[i : i + batch_size],
      return_tensors="pt",
      max_length=max_length,
      truncation=True,
      padding="max_length",
    ).to(model.device)
    encoded_outputs = tokenizer(
      output_batch[i : i + batch_size],
      return_tensors="pt",
      max_length=max_length,
      truncation=True,
      padding="max_length",
    ).to(model.device)
    labels = encoded_inputs["input_ids"].clone()
    if (
      tokenizer.bos_token_id is not None
      and tokenizer.bos_token_id != tokenizer.pad_token_id
    ):
      # Tokenizer with a BOS token.
      prompt_mask = ~torch.logical_and(
        encoded_outputs.attention_mask.to(torch.bool),
        encoded_outputs.input_ids != tokenizer.bos_token_id,
      )
    else:
      prompt_mask = ~encoded_outputs.attention_mask.to(torch.bool)
    labels = torch.where(prompt_mask, torch.ones_like(labels) * -100, labels)
    label_prob, full_dist = compute_per_token_probability(
      model, encoded_inputs, labels, temperature
    )
    # Mask out prompt and padding tokens.
    full_dist = torch.where(
      prompt_mask[:, 1:].unsqueeze(-1), torch.zeros_like(full_dist), full_dist
    )
    if score_fn is None:
      # Take the probability of the label token of the last step.
      scores = label_prob[:, -1, 0].tolist()
    else:
      scores = score_fn(full_dist)
    all_scores.extend(scores)

  return all_scores


def pool_confidence_score(
  tokenizer, prob, next_n_tokens=16, kept_token_ids=None, mode="mean"
):
  epsilon = 1e-4
  pred_tokens = torch.argmax(prob, dim=-1)
  top_token_conf = torch.max(prob, dim=-1)[0]
  confidence_score = []
  for b_i in range(len(prob)):
    # 1) Find how many tokens are effectively "non-zero"
    num_pad = torch.sum(torch.max(prob[b_i], dim=-1)[0] < epsilon)
    kept_prob = top_token_conf[b_i]
    if kept_token_ids is not None:
      # Possibly restrict to certain token IDs
      mask_any = torch.stack(
        [(pred_tokens[b_i] == tid) for tid in kept_token_ids]
      )
      mask_any = torch.any(mask_any, dim=0)
      kept_prob = torch.where(mask_any, kept_prob, torch.zeros_like(kept_prob))
    # 2) Slice the first N tokens after skipping padding
    kept_prob = kept_prob[num_pad : num_pad + next_n_tokens]
    # 3) Avoid dividing by zero:
    nonzero = kept_prob[kept_prob > epsilon]
    if len(nonzero) == 0:
      # no valid tokens => fallback to 0.0 or some small number
      confidence_score.append(0.0)
      continue
    if mode == "mean":
      nonzero = torch.log(nonzero)
      confidence_score.append((nonzero.mean().exp().item()))
      # confidence_score.append((nonzero.mean().item()))
    elif mode == "max":
      confidence_score.append((nonzero.max().item()))
    elif mode == "min":
      confidence_score.append((nonzero.min().item()))
    else:
      raise ValueError(f"Unknown mode: {mode}")
  return confidence_score
