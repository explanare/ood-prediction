import collections
import copy
import random

import datasets
import torch
import transformers
from datasets import Dataset
from torch.utils.data import DataLoader

_BASE_TEMPLATE = "BASE_TEMPLATE"
_SOURCE_TEMPLATE = "SOURCE_TEMPLATE"
FEATURE_TYPES = datasets.Features(
  {
    "input": datasets.Value("string"),
    "label": datasets.Value("string"),
    "source_input": datasets.Value("string"),
    "source_label": datasets.Value("string"),
    "inv_label": datasets.Value("string"),
    "split": datasets.Value("string"),
    "source_split": datasets.Value("string"),
  }
)


def preproc_tokenize(
  tokenizer,
  max_input_length,
  max_output_length,
  examples,
  input_feature=None,
  label_feature=None,
  extra_input_to_tokenize=None,
  extra_label_to_tokenize=None,
):
  assert tokenizer.padding_side == "left"
  input_batch = copy.deepcopy(examples)
  input_feature = input_feature or "input"
  label_feature = label_feature or "label"
  input_batch.update(
    tokenizer(
      examples[input_feature],
      padding="max_length",
      max_length=max_input_length,
      return_tensors="pt",
      truncation=True,
    )
  )
  # Right padding labels.
  tokenizer.padding_side = "right"
  labels = tokenizer(
    examples[label_feature],
    padding="max_length",
    max_length=max_output_length,
    return_tensors="pt",
    truncation=True,
  )["input_ids"]
  tokenizer.padding_side = "left"
  labels[labels == tokenizer.pad_token_id] = -100
  input_batch["labels"] = labels
  if extra_input_to_tokenize:
    for feat in extra_input_to_tokenize:
      tokenized_feat = tokenizer(
        examples[feat],
        padding="max_length",
        max_length=max_input_length,
        return_tensors="pt",
        truncation=True,
      )
      input_batch[f"{feat.replace('_input', '')}_input_ids"] = tokenized_feat[
        "input_ids"
      ]
      input_batch[f"{feat.replace('_input', '')}_attention_mask"] = (
        tokenized_feat["attention_mask"]
      )
  if extra_label_to_tokenize:
    for label in extra_label_to_tokenize:
      # Right padding labels.
      tokenizer.padding_side = "right"
      tokenized_feat = tokenizer(
        examples[label],
        padding="max_length",
        max_length=max_output_length,
        return_tensors="pt",
        truncation=True,
      )
      tokenizer.padding_side = "left"
      input_batch[f"{label.split('_')[0] + '_labels'}"] = tokenized_feat[
        "input_ids"
      ]
  # Remove extra nesting.
  for k in input_batch:
    if isinstance(input_batch[k], list) or isinstance(
      input_batch[k], torch.Tensor
    ):
      input_batch[k] = input_batch[k][0]
  return input_batch


def kept_first_n_label_token(x, first_n, padding_offset=3):
  x["base_labels"] = x["labels"][padding_offset : padding_offset + first_n]
  # Remove the <s> and SOS Pad token.
  x["labels"] = x["inv_labels"][padding_offset : padding_offset + first_n]
  return x


def get_label_offset(tokenizer):
  if isinstance(tokenizer, transformers.LlamaTokenizerFast):
    return 3
  elif "Llama-3" in tokenizer.name_or_path:
    # Tokenizer with the BOS token.
    return 1
  elif "gemma" in tokenizer.name_or_path:
    # Tokenizer with the BOS token.
    return 1
  return 0


def get_dataloader(
  eval_dataset,
  tokenizer,
  batch_size,
  prompt_max_length,
  output_max_length,
  first_n=1,
  drop_last=False,
  shuffle=True,
):
  eval_dataset = eval_dataset.map(
    lambda x: preproc_tokenize(
      tokenizer,
      prompt_max_length,
      output_max_length,
      x,
      extra_input_to_tokenize=["source_input"],
      extra_label_to_tokenize=["inv_label"],
    )
  )
  eval_dataset = eval_dataset.map(
    lambda x: kept_first_n_label_token(
      x, first_n, padding_offset=get_label_offset(tokenizer)
    )
  )
  eval_dataset = eval_dataset.with_format("torch")
  eval_dataloader = DataLoader(
    eval_dataset, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle
  )
  return eval_dataloader


def load_intervention_data(
  mode,
  verified_examples,
  data_split,
  prompt_to_vars,
  inv_label_fn,
  filter_fn=None,
  bos_pad=None,
  max_example_per_split=20480,
  max_example_per_eval_split=10,
):
  # inv_label_fn: A callable that takes in the variables parsed from the
  # base and source input, i.e., two dictionaries and returns a boolean.
  # verified_examples: data['train']['correct']
  if mode == "val":
    base_examples = verified_examples
    source_examples = data_split["val"]["correct"] + data_split["val"]["wrong"]
  elif mode == "test":
    base_examples = verified_examples
    source_examples = (
      data_split["test"]["correct"] + data_split["test"]["wrong"]
    )
  elif mode == "das":
    # base_examples are ...['train']['correct']
    # source examples are from ...['train']['correct'] + ...['test']
    base_examples = verified_examples
    source_examples = (
      base_examples + data_split["val"]["correct"] + data_split["val"]["wrong"]
    )
  else:
    raise ValueError("Invalid mode.")

  source_example_calculation = ""
  if mode == "das":
    source_example_calculation = (
      f"{len(base_examples)}+{len(data_split['test']['correct'])}+"
      f"{len(data_split['test']['wrong'])}={len(source_examples)}"
    )
  elif mode == "test" or mode == "val":
    source_example_calculation = (
      f"{len(data_split[mode]['correct'])}+{len(data_split[mode]['wrong'])}="
      f"{len(source_examples)}"
    )
  print(
    f"mode={mode}, "
    f"#base_examples={len(base_examples)}, "
    f"#source_examples={source_example_calculation}"
  )

  train_num_calculation, val_num_calculation, test_num_calculation = "", "", ""
  # gathers all pairs of (base, source) examples into a dictionary
  # keyed by their “split key.”
  split_to_raw_example = collections.defaultdict(list)
  for j in range(len(source_examples)):
    for i in range(len(base_examples)):
      base_vars = prompt_to_vars[base_examples[i]]
      source_vars = prompt_to_vars[source_examples[j]]
      if filter_fn and not filter_fn(base_vars, source_vars):
        continue
      # Set split.
      # split_key = "...-train" or "...-val" or "...-test"
      # each key is a “split identifier,” and
      # the values are lists of examples (dictionaries)
      # {
      # "das-train": [ { ... }, { ... }, ... ],
      # "source-foo-correct-test": [ { ... }, ... ],
      # ...
      # }
      src_is_correct = any(
        source_examples[j] in data_split[s]["correct"] for s in data_split
      )
      split_key = (
        f"source-{source_examples[j]}-"
        f"{'correct' if src_is_correct else 'wrong'}-test"
      )
      # Before-split formulas
      test_num_calculation = (
        f"{len(base_examples)}*({len(data_split['test']['correct'])}+"
        f"{len(data_split['test']['wrong'])})"
      )
      if mode == "das":
        # base_examples[i] is always in ...['train']['correct']
        # source_examples[j] can be in ...['train']['correct'] or
        #   ...['test']['correct'] or ...['test']['wrong']
        if (
          base_examples[i] in data_split["train"]["correct"]
          and source_examples[j] in data_split["train"]["correct"]
        ):
          split_key = "das-train"
        elif (
          base_examples[i] in data_split["train"]["correct"]
          and source_examples[j] in data_split["val"]["correct"]
        ):
          split_key = f"source-{source_examples[j]}-correct-test"
        elif (
          base_examples[i] in data_split["train"]["correct"]
          and source_examples[j] in data_split["val"]["wrong"]
        ):
          split_key = f"source-{source_examples[j]}-wrong-test"
        else:
          continue
        train_num_calculation = f"{len(base_examples)}*{len(base_examples)}"
        val_num_calculation = ""
        test_num_calculation = (
          f"{len(base_examples)}*({len(data_split['test']['correct'])}+"
          f"{len(data_split['test']['wrong'])})"
        )
      if i < 3 and j < 3:
        print(f"base_vars={base_vars}, source_vars={source_vars}")
      split_to_raw_example[split_key].append(
        {
          "input": base_examples[i],
          "label": base_vars["label"],
          "source_input": source_examples[j],
          "source_label": source_vars["label"],
          "inv_label": inv_label_fn(base_vars, source_vars),
          # Determine the intervention locations.
          "split": base_vars["split"],
          "source_split": source_vars["split"],
        }
      )
  split_to_raw_example = dict(split_to_raw_example)
  bos_pad = bos_pad or ""
  for split in split_to_raw_example:
    for i in range(len(split_to_raw_example[split])):
      split_to_raw_example[split][i]["inv_label"] = (
        bos_pad + split_to_raw_example[split][i]["inv_label"]
      )
      split_to_raw_example[split][i]["label"] = (
        bos_pad + split_to_raw_example[split][i]["label"]
      )

  # Preprocess the dataset.
  for split in split_to_raw_example:
    # Shuffle examples
    random.shuffle(split_to_raw_example[split])
  # Remove empty splits.
  split_to_raw_example = {
    k: v for k, v in split_to_raw_example.items() if len(v) > 0
  }
  # These counts reflect the raw examples found in split_to_raw_example—before
  # any further subsampling.
  print(
    f"BEFORE FILTERING: "
    f"#Training examples={train_num_calculation}="
    f"{sum(map(len, [v for k, v in split_to_raw_example.items() if k.endswith('-train')]))}, "
    f"#Validation examples={val_num_calculation}="
    f"{sum(map(len, [v for k, v in split_to_raw_example.items() if k.endswith('-val')]))}, "
    f"#Test examples={test_num_calculation}="
    f"{sum(map(len, [v for k, v in split_to_raw_example.items() if k.endswith('-test')]))}"
  )

  split_to_dataset = {
    split: Dataset.from_list(
      [
        x
        for x in split_to_raw_example[split][
          : max_example_per_eval_split
          if mode == "localization"
          or (mode == "das" and split.endswith("-test"))
          else max_example_per_split
        ]
      ],
      features=FEATURE_TYPES,
    )
    for split in split_to_raw_example
  }

  if mode == "das":
    train_num_calculation_after = (
      f"min({len(base_examples)}*{len(base_examples)}, 20480)"
    )
    test_num_calculation_after = (
      f"{len(data_split['test']['correct'])}*{max_example_per_eval_split}+"
      f"{len(data_split['test']['wrong'])}*{max_example_per_eval_split}"
    )
    val_num_calculation_after = ""
  else:
    train_num_calculation_after = ""
    test_num_calculation_after = (
      f"({len(data_split['test']['correct'])}+"
      f"{len(data_split['test']['wrong'])})*{max_example_per_eval_split}"
    )
    val_num_calculation_after = ""

  print(
    f"AFTER FILTERING KEPT: "
    f"#Training examples={train_num_calculation_after}="
    f"{sum(map(len, [v for k, v in split_to_dataset.items() if k.endswith('-train')]))}, "
    f"#Validation examples={val_num_calculation_after}="
    f"{sum(map(len, [v for k, v in split_to_dataset.items() if k.endswith('-val')]))}, "
    f"#Test examples={test_num_calculation_after}="
    f"{sum(map(len, [v for k, v in split_to_dataset.items() if k.endswith('-test')]))}"
  )
  return split_to_raw_example, split_to_dataset
