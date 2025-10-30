import torch
from tqdm import tqdm


def generate(
  model,
  tokenizer,
  prompt,
  max_new_tokens=4,
  num_beams=1,
  return_output_only=False,
  show_tokenization=False,
):
  input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(
    model.device
  )
  if show_tokenization:
    print(tokenizer.batch_decode(input_ids[0]))
  with torch.no_grad():
    outputs = tokenizer.batch_decode(
      model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        num_beams=num_beams,
        use_cache=True,
      ),
      skip_special_tokens=True,
    )
  if return_output_only:
    outputs = outputs[0][len(prompt) :]
  return outputs


def generate_distribution(model, tokenizer, prefix, top_k=10):
  input_ids = tokenizer(prefix, return_tensors="pt")["input_ids"].to(
    model.device
  )
  with torch.no_grad():
    logits = torch.nn.functional.softmax(
      model(input_ids).logits[:, -1, :], dim=-1
    )
    topk_index = torch.argsort(logits, descending=True)[:, :top_k]
  topk_tokens = [
    list(
      zip(
        tokenizer.batch_decode(topk_index[i]), logits[i][topk_index[i]].tolist()
      )
    )
    for i in range(len(topk_index))
  ]
  return topk_tokens


def generate_distribution_batched(
  model, tokenizer, prefixes, top_k=10, prompt_max_length=None, batch_size=32
):
  all_topk_tokens = []
  for b_i in tqdm(range(0, len(prefixes), batch_size)):
    encoded_inputs = tokenizer(
      prefixes[b_i : b_i + batch_size],
      return_tensors="pt",
      padding="max_length" if prompt_max_length else "longest",
      max_length=prompt_max_length,
    ).to(model.device)
    with torch.no_grad():
      position_ids = encoded_inputs.attention_mask.long().cumsum(-1) - 1
      position_ids.masked_fill_(encoded_inputs.attention_mask == 0, 1)
      logits = model(**encoded_inputs, position_ids=position_ids).logits[
        :, -1, :
      ]
      probs = torch.nn.functional.softmax(logits, dim=-1)
      topk_index = torch.argsort(probs, descending=True, dim=-1)[:, :top_k]
      topk_tokens = [
        list(
          zip(
            tokenizer.batch_decode(topk_index[i]),
            # logits[i][topk_index[i]].tolist()))
            probs[i][topk_index[i]].tolist(),
          )
        )
        for i in range(len(topk_index))
      ]
    all_topk_tokens.extend(topk_tokens)
  return all_topk_tokens


def _generate_single_batch(
  pretrained_model,
  tokenizer,
  prompt_batch,
  max_length=None,
  prompt_max_length=32,
  max_new_tokens=None,
  sample_n=None,
  **kwargs,
):
  if not sample_n:
    sample_n = 1
  if not max_new_tokens:
    assert max_length and prompt_max_length
    max_new_tokens = max_length - prompt_max_length
  input_batch = tokenizer(
    prompt_batch,
    return_tensors="pt",
    padding="max_length",
    max_length=prompt_max_length,
    truncation=True,
  )
  input_ids = input_batch["input_ids"].to(pretrained_model.device)
  attention_mask = input_batch["attention_mask"].to(pretrained_model.device)
  with torch.no_grad():
    outputs = pretrained_model.generate(
      input_ids,
      attention_mask=attention_mask,
      max_new_tokens=max_new_tokens,
      do_sample=True if sample_n > 1 else False,
      num_return_sequences=sample_n,
      return_dict_in_generate=False,
      pad_token_id=tokenizer.pad_token_id,
      **kwargs,
    )
  preds = [
    (prompt_batch[i // sample_n], p)
    for i, p in enumerate(
      tokenizer.batch_decode(outputs, skip_special_tokens=True)
    )
  ]
  return preds


def generate_batched(
  pretrained_model,
  tokenizer,
  all_prompts,
  max_length=None,
  prompt_max_length=None,
  max_new_tokens=None,
  sample_n=None,
  batch_size=32,
  **kwargs,
):
  print("Total #prompts=%d" % len(all_prompts))
  pretrained_model = pretrained_model.eval()
  if prompt_max_length is None:
    # Estimate the max prompt length from the longest sequence in the batch.
    # This estimation assumes all sequences have similar token to character
    # ratio, which would not hold if some sequences are mostly English words,
    # while others are mostly digits, punctuations, etc.
    max_length_prompt = max(all_prompts, key=len)
    prompt_max_length = 8 * (
      len(tokenizer(max_length_prompt).input_ids) // 8 + 1
    )
    print("Set prompt_max_length=%d" % prompt_max_length)
  prompt_to_raw_outputs = []
  for batch_begin in tqdm(range(0, len(all_prompts), batch_size)):
    batch_prompts = all_prompts[batch_begin : batch_begin + batch_size]
    output_texts = _generate_single_batch(
      pretrained_model,
      tokenizer,
      batch_prompts,
      prompt_max_length=prompt_max_length,
      max_new_tokens=max_new_tokens,
      max_length=max_length,
      sample_n=sample_n,
      **kwargs,
    )
    prompt_to_raw_outputs.extend(output_texts)
  prompt_to_raw_outputs = {
    p: out[
      len(tokenizer.decode(tokenizer(p).input_ids, skip_special_tokens=True)) :
    ]
    for p, out in prompt_to_raw_outputs
  }
  return prompt_to_raw_outputs


def add_user_prompt(p):
  return [{"role": "user", "content": p}]


def identity_fn(p):
  return p


def apply_chat_template(
  tokenizer, prompts, per_template_fn=None, post_template_fn=None
):
  if per_template_fn is None:
    per_template_fn = add_user_prompt
  if post_template_fn is None:
    post_template_fn = identity_fn
  all_messages = []
  for i, p in enumerate(prompts):
    messages = per_template_fn(p)
    tokens = tokenizer.apply_chat_template(
      [{k: v for k, v in t.items()} for t in messages],
      add_generation_prompt=True,
      return_tensors="pt",
    )
    all_messages.append(post_template_fn(tokenizer.decode(tokens[0][1:])))
  return all_messages
