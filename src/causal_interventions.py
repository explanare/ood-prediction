import collections
import copy
import gc
import numpy as np
import re

import datasets
import pyvene as pv
import torch
from tqdm.auto import tqdm

from causal_data_utils import get_dataloader, get_label_offset


def get_intervention_config(
  model_type,
  intervention_representations,
  layers,
  intervention_type,
  intervention_units=None,
  intervention_dimension=None,
  num_unit=1,
):
  if isinstance(layers, int):
    layers = [layers]
  if isinstance(intervention_representations, str):
    intervention_representations = [intervention_representations] * len(layers)
  if isinstance(intervention_units, str) or intervention_units is None:
    intervention_units = [
      intervention_units if intervention_units is not None else "pos"
    ]
  assert len(intervention_representations) == len(layers)
  assert len(intervention_representations) == len(intervention_units)
  inv_config = pv.IntervenableConfig(
    model_type=model_type,
    representations=[
      pv.RepresentationConfig(
        layer,  # layer
        intervention_representations[i],  # intervention repr
        intervention_units[i],  # intervention unit
        num_unit,  # max number of unit
        intervention_dimension,
      )
      for i, layer in enumerate(layers)
    ],
    intervention_types=intervention_type,
  )
  return inv_config


def remove_invalid_token_id(token_ids, pad_id=2):
  token_ids = token_ids.clone()
  token_ids[token_ids < 0] = pad_id
  return token_ids


def remove_all_forward_hooks(model):
  for name, child in model._modules.items():
    if child is not None:
      if hasattr(child, "_forward_hooks") and len(child._forward_hooks) > 0:
        print(child._forward_hooks)
        print(name, child)
        child._forward_hooks = collections.OrderedDict()
      remove_all_forward_hooks(child)


class PretrainedFeaturizer(torch.nn.Module):
  """A pretrained featurizer, which is typically a linear layer."""

  def __init__(self, pretrained_weight_or_path):
    super().__init__()
    if isinstance(pretrained_weight_or_path, str):
      if pretrained_weight_or_path.endswith(".pt"):
        self.weight = torch.load(pretrained_weight_or_path)
      elif pretrained_weight_or_path.endswith(".npy"):
        self.weight = torch.tensor(np.load(pretrained_weight_or_path))
    else:
      # Convert input weight to torch.Tensor.
      self.weight = torch.tensor(pretrained_weight_or_path)
    if self.weight.shape[0] > self.weight.shape[1]:
      self.weight = self.weight.T

  def forward(self, x):
    return torch.matmul(x.to(self.weight.dtype), self.weight.T)


def load_intervenable(
  base_model, pretrained_weight_or_path, intervention_representations=None
):
  """Load interventions that involve a linear transformation."""

  run_name = pretrained_weight_or_path.rsplit(".", 1)[0].rsplit("/", 1)[-1]
  # Support formats: {inv_key: torch.Tensor}, torch.Tensor, numpy.array
  rotate_layers = {}
  if pretrained_weight_or_path.endswith(".pt"):
    inv_key_to_weights = torch.load(pretrained_weight_or_path)
    if isinstance(inv_key_to_weights, dict):
      for k, v in inv_key_to_weights.items():
        rotate_layer = PretrainedFeaturizer(v).eval()
        print(k)
        print(
          "Loaded feature projection matrix shape:", rotate_layer.weight.shape
        )
        rotate_layers[k] = rotate_layer
    else:
      # Weights saved without intervention key.
      layer_match = re.search(r"layer(\d+)[\-_.]", run_name)
      layer = int(layer_match.group(1))
      rotate_layer = PretrainedFeaturizer(inv_key_to_weights).eval()
      print(
        "Loaded feature projection matrix shape:", rotate_layer.weight.shape
      )
      rotate_layers[f"layer.{layer}.comp.block_output.unit.pos.nunit.1#0"] = (
        rotate_layer
      )
  layers = [int(k.split(".")[1]) for k in rotate_layers]
  intervention_representations = intervention_representations or "block_output"
  inv_config = get_intervention_config(
    type(base_model),
    intervention_representations,
    layers,
    LowRankRotatedSpaceIntervention,
    intervention_dimension=0,
  )
  intervenable = pv.IntervenableModel(inv_config, base_model)
  intervenable.set_device("cuda")
  intervenable.disable_model_gradients()
  for k, v in rotate_layers.items():
    intervenable.interventions[k][0].rotate_layer = v
    intervenable.interventions[k][0].set_interchange_dim(
      interchange_dim=v.weight.shape[0]
    )
  intervenable.model.eval()
  return intervenable


class LowRankRotatedSpaceIntervention(pv.TrainableIntervention):
  """Intervene in the rotated subspace defined by (low-rank) DAS."""

  def __init__(self, embed_dim, **kwargs):
    super().__init__()
    self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(
      torch.nn.Linear(embed_dim, kwargs["low_rank_dimension"], bias=False)
    )
    self.embed_dim = embed_dim

  def forward(self, base, source, subspaces=None):
    input_dtype, model_dtype = base.dtype, self.rotate_layer.weight.dtype
    base, source = base.to(model_dtype), source.to(model_dtype)
    rotated_base = self.rotate_layer(base)
    rotated_source = self.rotate_layer(source)
    # Apply interchange interventions.
    output = base + torch.matmul(
      (rotated_source - rotated_base), self.rotate_layer.weight
    )
    return output.to(input_dtype)


def train_intervention_step(
  intervenable, inputs, split_to_inv_locations, pad_token_id
):
  inputs = copy.deepcopy(inputs)
  b_s = inputs["input_ids"].shape[0]
  # Set intervention locations.
  # These locations are invariant to the label appended later.
  num_inv = len(intervenable.interventions)
  intervention_locations = {
    "sources->base": (
      [
        [
          split_to_inv_locations[inputs["source_split"][i]]["inv_position"]
          for i in range(b_s)
        ]
      ]
      * num_inv,
      [
        [
          split_to_inv_locations[inputs["split"][i]]["inv_position"]
          for i in range(b_s)
        ]
      ]
      * num_inv,
    )
  }
  # Append label to input.
  inputs["labels"][inputs["labels"] < 0] = pad_token_id
  inputs["input_ids"] = torch.cat(
    [inputs["input_ids"], inputs["labels"]], dim=-1
  )
  inputs["attention_mask"] = torch.zeros(
    inputs["input_ids"].shape,
    dtype=inputs["attention_mask"].dtype,
    device=inputs["attention_mask"].device,
  )
  inputs["attention_mask"][inputs["input_ids"] != pad_token_id] = 1
  position_ids = {
    f"{prefix}position_ids": intervenable.model.prepare_inputs_for_generation(
      input_ids=inputs[f"{prefix}input_ids"],
      attention_mask=inputs[f"{prefix}attention_mask"],
    )["position_ids"]
    for prefix in ("", "source_")
  }
  inputs.update(position_ids)
  _, counterfactual_outputs = intervenable(
    {
      "input_ids": inputs["input_ids"],
      "attention_mask": inputs["attention_mask"],
      "position_ids": inputs["position_ids"],
    },
    [
      {
        "input_ids": inputs["source_input_ids"],
        "attention_mask": inputs["source_attention_mask"],
        "position_ids": inputs["source_position_ids"],
      }
    ]
    * num_inv,
    intervention_locations,
  )
  return counterfactual_outputs


def compute_metrics(
  keyed_eval_preds,
  eval_labels,
  pad_token_id,
  last_n_tokens=1,
  inference_mode=None,
  **kwargs,
):
  """Computes squence-level and token-level accuracy."""
  metrics = {}
  for key, eval_preds in keyed_eval_preds.items():
    total_count, total_token_count = 0, 0
    correct_count, correct_token_count = 0, 0
    class_0_correct_count = 0
    class_0_val = eval_labels[0][0, -1]
    for eval_pred, eval_label in zip(eval_preds, eval_labels):
      if inference_mode == "force_decode":
        eval_pred = eval_pred[:, :-1]

      actual_test_labels = eval_label[:, -last_n_tokens:]

      if actual_test_labels.shape[0] == 0:
        continue
      if len(eval_pred.shape) == 3:
        # eval_preds is in the form of logits.
        pred_test_labels = torch.argmax(eval_pred[:, -last_n_tokens:], dim=-1)
      else:
        # eval_preds is in the form of token ids.
        pred_test_labels = eval_pred[:, -last_n_tokens:]

      padding_tokens = torch.logical_or(
        actual_test_labels == pad_token_id, actual_test_labels < 0
      )
      match_tokens = actual_test_labels == pred_test_labels
      correct_labels = torch.logical_or(match_tokens, padding_tokens)
      total_count += len(correct_labels)
      correct_count += torch.all(correct_labels, axis=-1).float().sum().tolist()
      total_token_count += (~padding_tokens).float().sum().tolist()
      correct_token_count += (
        (~padding_tokens & match_tokens).float().sum().tolist()
      )
      # For binary classification, log the actual prediction by comparing to
      # a single-side label
      class_0_labels = (torch.ones_like(actual_test_labels) * class_0_val).to(
        actual_test_labels.dtype
      )
      class_0_match_tokens = class_0_labels == pred_test_labels
      class_0_correct_labels = torch.logical_or(
        class_0_match_tokens, padding_tokens
      )
      class_0_correct_count += (
        torch.all(class_0_correct_labels, axis=-1).float().sum().tolist()
      )

    accuracy = round(correct_count / total_count, 2)
    token_accuracy = round(correct_token_count / max(1, total_token_count), 2)
    class_0_accuracy = round(class_0_correct_count / total_count, 2)
    metrics[key] = {
      "accuracy": accuracy,
      "token_accuracy": token_accuracy,
      "class_0_accuracy": class_0_accuracy,
    }
  # For compatablity with other metrics.
  metrics["accuracy"] = metrics["inv_outputs"]["accuracy"]
  return metrics


def compute_string_based_metrics(
  keyed_eval_preds,
  eval_labels,
  pad_token_id,
  last_n_tokens=1,
  tokenizer=None,
  extract_prediction_fn=None,
  empty_token="<EMPTY>",
  **kwargs,
):
  """Computes squence-level and token-level accuracy."""
  metrics = {}
  # Add another key 'inv_parsed_outputs' as label_match and pred_match
  for key, eval_preds in keyed_eval_preds.items():
    total_count = 0
    correct_count = 0
    for eval_pred, eval_label in zip(eval_preds, eval_labels):
      actual_test_labels = eval_label[:, -last_n_tokens:]
      if len(eval_pred.shape) == 3:
        # Eval_preds is in the form of logits.
        # (batch_size, seq_length, vocab_size) => logits => take argmax
        pred_test_labels = torch.argmax(eval_pred[:, -last_n_tokens:], dim=-1)
      else:
        # Eval_preds is in the form of token ids.
        # (batch_size, seq_length) => token IDs => just slice
        pred_test_labels = eval_pred[:, -last_n_tokens:]
      # Replaces negative or out-of-range IDs with pad or 0
      label_text = tokenizer.batch_decode(
        remove_invalid_token_id(
          token_ids=actual_test_labels, pad_id=tokenizer.pad_token_id
        ),
        skip_special_tokens=True,
      )
      pred_text = tokenizer.batch_decode(
        remove_invalid_token_id(
          token_ids=pred_test_labels, pad_id=tokenizer.pad_token_id
        ),
        skip_special_tokens=True,
      )
      # For each item in the batch, parse out the final label
      for i in range(len(label_text)):
        label_match = extract_prediction_fn(label_text[i])
        pred_match = extract_prediction_fn(pred_text[i])
        if (
          label_match is not None
          and pred_match is not None
          and label_match != empty_token
          and pred_match != empty_token
          and label_match == pred_match
        ):
          correct_count += 1
        total_count += 1
    accuracy = round(correct_count / total_count, 2)
    token_accuracy = 0  # Do not do per-token accuracy in string-based methods.
    metrics[key] = {"accuracy": accuracy, "token_accuracy": token_accuracy}

  # For compatablity with other metrics.
  metrics["accuracy"] = metrics["inv_outputs"]["accuracy"]
  return metrics


def compute_metrics_case_normalized(
  keyed_eval_preds, eval_labels, pad_token_id, last_n_tokens=1, **kwargs
):
  """Computes squence-level and token-level accuracy."""
  metrics = {}
  for key, eval_preds in keyed_eval_preds.items():
    total_count, total_token_count = 0, 0
    correct_count, correct_token_count = 0, 0
    for eval_pred, eval_label in zip(eval_preds, eval_labels):
      actual_test_labels = eval_label[:, -last_n_tokens:]
      if len(eval_pred.shape) == 3:
        # eval_preds is in the form of logits.
        pred_test_labels = torch.argmax(eval_pred[:, -last_n_tokens:], dim=-1)
      else:
        # eval_preds is in the form of token ids.
        pred_test_labels = eval_pred[:, -last_n_tokens:]
      padding_tokens = torch.logical_or(
        actual_test_labels == pad_token_id, actual_test_labels < 0
      )
      match_tokens = actual_test_labels == pred_test_labels
      correct_labels = torch.logical_or(match_tokens, padding_tokens)
      total_count += len(correct_labels)
      correct_count += torch.all(correct_labels, axis=-1).float().sum().tolist()
      total_token_count += (~padding_tokens).float().sum().tolist()
      correct_token_count += (
        (~padding_tokens & match_tokens).float().sum().tolist()
      )
    accuracy = round(correct_count / total_count, 2)
    token_accuracy = round(correct_token_count / total_token_count, 2)
    metrics[key] = {"accuracy": accuracy, "token_accuracy": token_accuracy}
  # For compatablity with other metrics.
  metrics["accuracy"] = metrics["inv_outputs"]["accuracy"]
  return metrics


def extract_tokens_from_output(
  mode, model_outputs, max_new_tokens, prompt_tokens=None
):
  # Model output could either be logits or token_ids depending on the mode.
  if mode == "generate":
    # Output the greedy decoded sequence.
    if prompt_tokens is not None:
      tokens = model_outputs[
        :, prompt_tokens.shape[1] : prompt_tokens.shape[1] + max_new_tokens
      ]
    else:
      tokens = model_outputs[:, -max_new_tokens:]

  elif mode == "forward":
    # Output the topK token.
    tokens = torch.argsort(model_outputs[:, -1, :], dim=-1, descending=True)[
      :, :max_new_tokens
    ]
  elif mode == "force_decode":
    # Output the force decoded sequence.
    tokens = torch.argsort(
      model_outputs[:, -max_new_tokens - 1 : -1, :], dim=-1, descending=True
    )[:, :, 0]
  else:
    raise ValueError(f"Unknown mode: {mode}")
  return tokens


def eval_with_interventions_batched(
  intervenable,
  split_to_dataset,
  split_to_inv_locations,
  tokenizer,
  compute_metrics_fn,
  max_input_length=None,
  max_new_tokens=1,
  eval_batch_size=16,
  debug_print=False,
  inference_mode=None,
  intervention_location_fn=None,
):
  """Fully batched interchange intervention evaluation."""
  if inference_mode is None:
    # Default to generate.
    inference_mode = "generate"
  assert inference_mode in ("generate", "forward", "force_decode")
  if compute_metrics_fn is None:
    compute_metrics_fn = compute_metrics
  split_to_eval_metrics = {}
  padding_offset = get_label_offset(tokenizer)
  num_inv = len(intervenable.interventions)
  # Merge all splits to allow more efficient batching.
  merged_dataset = datasets.concatenate_datasets(
    [split_to_dataset[split] for split in split_to_dataset]
  )
  split_to_index = {}
  offset = 0
  for split in split_to_dataset:
    split_to_index[split] = [offset, len(split_to_dataset[split]) + offset]
    offset += len(split_to_dataset[split])
  if max_input_length is None:
    # Asssume all inputs have the same max length.
    max_input_length = split_to_inv_locations[merged_dataset[0]["split"]][
      "max_input_length"
    ]
  eval_dataloader = get_dataloader(
    merged_dataset,
    tokenizer=tokenizer,
    batch_size=eval_batch_size,
    prompt_max_length=max_input_length,
    output_max_length=padding_offset + max_new_tokens,
    first_n=max_new_tokens,
    shuffle=False,
  )
  eval_labels = collections.defaultdict(list)
  eval_preds = collections.defaultdict(list)
  var_code = []
  source_label = []
  current_split = list(split_to_dataset)[0]
  with torch.no_grad():
    epoch_iterator = tqdm(eval_dataloader, desc="Test")
    for step, inputs in enumerate(epoch_iterator):
      torch.cuda.empty_cache()
      b_s = inputs["input_ids"].shape[0]
      position_ids = {
        f"{prefix}position_ids": intervenable.model.prepare_inputs_for_generation(
          input_ids=inputs[f"{prefix}input_ids"],
          attention_mask=inputs[f"{prefix}attention_mask"],
        )["position_ids"]
        for prefix in ("", "source_")
      }
      inputs.update(position_ids)
      for key in inputs:
        if key in (
          "input_ids",
          "source_input_ids",
          "attention_mask",
          "source_attention_mask",
          "position_ids",
          "source_position_ids",
          "labels",
          "base_labels",
        ):
          inputs[key] = inputs[key].to(intervenable.model.device)
      if intervention_location_fn is not None:
        intervention_locations = intervention_location_fn(inputs, num_inv)
      else:
        intervention_locations = {
          "sources->base": (
            [
              [
                split_to_inv_locations[inputs["source_split"][i]][
                  "inv_position"
                ]
                for i in range(b_s)
              ]
            ]
            * num_inv,
            [
              [
                split_to_inv_locations[inputs["split"][i]]["inv_position"]
                for i in range(b_s)
              ]
            ]
            * num_inv,
          )
        }
      if inference_mode == "generate":
        base_outputs, counterfactual_outputs = intervenable.generate(
          {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
          },
          [
            {
              "input_ids": inputs["source_input_ids"],
              "attention_mask": inputs["source_attention_mask"],
              "position_ids": inputs["source_position_ids"],
            }
          ],
          intervention_locations,
          max_new_tokens=max_new_tokens,
          do_sample=False,
          intervene_on_prompt=True,
          pad_token_id=tokenizer.pad_token_id,
          output_original_output=True,
        )

        base_outputs = base_outputs[:, inputs["input_ids"].shape[1] :]
        counterfactual_outputs = counterfactual_outputs[
          :, inputs["input_ids"].shape[1] :
        ]
      elif inference_mode == "forward":
        base_outputs, counterfactual_outputs = intervenable(
          {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "position_ids": inputs["position_ids"],
          },
          [
            {
              "input_ids": inputs["source_input_ids"],
              "attention_mask": inputs["source_attention_mask"],
              "position_ids": inputs["source_position_ids"],
            }
          ],
          intervention_locations,
          output_original_output=True,
        )
        counterfactual_outputs = counterfactual_outputs.logits
        base_outputs = base_outputs.logits
      elif inference_mode == "force_decode":
        # Append the counterfactual label to the base input.
        # There are paddings both before the input and after the label.
        # P(y|x) = sum_i P(y_i|x_i, y_{j<i})
        force_decode_input_ids = torch.concat(
          [inputs["input_ids"], inputs["labels"]], dim=-1
        )
        force_decode_label = torch.concat(
          [torch.ones_like(inputs["input_ids"]) * -100, inputs["labels"]],
          dim=-1,
        )
        force_decode_attn_mask = torch.concat(
          [inputs["attention_mask"], torch.ones_like(inputs["labels"])], dim=-1
        )
        force_decode_position_ids = (
          intervenable.model.prepare_inputs_for_generation(
            input_ids=force_decode_input_ids,
            attention_mask=force_decode_attn_mask,
          )["position_ids"]
        )
        base_outputs, counterfactual_outputs = intervenable(
          {
            "input_ids": force_decode_input_ids,
            "attention_mask": force_decode_attn_mask,
            "position_ids": force_decode_position_ids,
            "labels": force_decode_label,
          },
          [
            {
              "input_ids": inputs["source_input_ids"],
              "attention_mask": inputs["source_attention_mask"],
              "position_ids": inputs["source_position_ids"],
            }
          ],
          # The appended labels do not change the intervention location, as
          # the intervention position is counted from the left.
          intervention_locations,
          output_original_output=True,
        )
        counterfactual_outputs = counterfactual_outputs.logits
        base_outputs = base_outputs.logits

      split_offset = [
        0,
        split_to_index[current_split][1] - step * eval_batch_size,
      ]
      while True:
        # Check if a split is finished or we are at the last batch.
        if (
          split_offset[1] >= 0 and split_offset[1] < eval_batch_size
        ) or step == len(epoch_iterator) - 1:
          if split_offset[1] > 0:
            eval_preds["base_outputs"].append(
              base_outputs[split_offset[0] : split_offset[1]]
            )
            eval_preds["inv_outputs"].append(
              counterfactual_outputs[split_offset[0] : split_offset[1]]
            )
            for label_type in ["base_labels", "labels"]:
              eval_labels[label_type].append(
                inputs[label_type][split_offset[0] : split_offset[1]]
              )
              eval_labels["base_outputs"].append(
                base_outputs[
                  split_offset[0] : split_offset[1], -max_new_tokens:
                ]
              )
            var_code.extend(
              inputs["var_code"][split_offset[0] : split_offset[1]]
              if "var_code" in inputs
              else inputs["label"][split_offset[0] : split_offset[1]]
            )
            source_label.extend(
              inputs["source_label"][split_offset[0] : split_offset[1]]
            )

          # Aggregate metrics
          eval_metrics = {
            label_type: compute_metrics_fn(
              keyed_eval_preds=eval_preds,
              eval_labels=eval_labels[label_type],
              last_n_tokens=max_new_tokens,
              pad_token_id=tokenizer.pad_token_id,
              extra_labels=eval_labels,
              eval_label_type=label_type,
            )
            for label_type in eval_labels
            if label_type.endswith("labels")
          }

          inv_output_tokens = [
            extract_tokens_from_output(inference_mode, i, max_new_tokens, None)
            for i in eval_preds["inv_outputs"]
          ]

          inv_outputs = [
            tokenizer.batch_decode(
              remove_invalid_token_id(
                token_ids=i, pad_id=tokenizer.pad_token_id
              ),
              skip_special_tokens=True,
            )
            for i in inv_output_tokens
          ]
          # Merge inv_outputs into a single list
          inv_outputs = sum(inv_outputs, [])
          assert len(var_code) == len(inv_outputs), (
            f"len(var_code)={len(var_code)}, len(inv_outputs)={len(inv_outputs)}"
          )
          split_to_eval_metrics[current_split] = {
            "metrics": eval_metrics,
            "inv_outputs": inv_outputs,
            "inv_labels": tokenizer.batch_decode(
              remove_invalid_token_id(
                token_ids=inputs["labels"][:, :max_new_tokens],
                pad_id=tokenizer.pad_token_id,
              ),
              skip_special_tokens=True,
            ),
            "base_labels": tokenizer.batch_decode(
              remove_invalid_token_id(
                token_ids=inputs["base_labels"][:, :max_new_tokens],
                pad_id=tokenizer.pad_token_id,
              ),
              skip_special_tokens=True,
            ),
            "source_labels": source_label,
            "var_code": var_code,
          }
          if debug_print:
            print("\n", repr(current_split) + ":", eval_metrics)
          eval_preds = collections.defaultdict(list)
          eval_labels = collections.defaultdict(list)
          var_code = []
          source_label = []
          # Need to empty eval_preds to prevent OOM
          gc.collect()
          torch.cuda.empty_cache()
          # Check for termination condition.
          if len(split_to_eval_metrics) == len(split_to_dataset):
            break
          # Run the next split.
          current_split = list(split_to_dataset)[len(split_to_eval_metrics)]
          split_offset = [
            split_offset[1],
            split_to_index[current_split][1] - step * eval_batch_size,
          ]
        else:
          # Add the rest part of the split.
          # We could not compute the aggregated metrics now as we don't know if
          # there will be more examples from the same split in the next batch.
          eval_preds["base_outputs"].append(
            base_outputs[split_offset[0] : split_offset[1]]
          )
          eval_preds["inv_outputs"].append(
            counterfactual_outputs[split_offset[0] : split_offset[1]]
          )
          for label_type in ["base_labels", "labels"]:
            eval_labels[label_type].append(
              inputs[label_type][split_offset[0] : split_offset[1]]
            )
            eval_labels["base_outputs"].append(
              base_outputs[split_offset[0] : split_offset[1], -max_new_tokens:]
            )
          var_code.extend(
            inputs["var_code"][split_offset[0] : split_offset[1]]
            if "var_code" in inputs
            else inputs["label"][split_offset[0] : split_offset[1]]
          )
          source_label.extend(
            inputs["source_label"][split_offset[0] : split_offset[1]]
          )
          break

      # Debug logging.
      if debug_print and step < 3:
        # Check if the first entry is a 'pos' or 'h.pos'
        base_locs = (
          intervention_locations["sources->base"][1][0]
          if isinstance(
            intervention_locations["sources->base"][1][0][0][0], int
          )
          else intervention_locations["sources->base"][1][0][1]
        )
        source_locs = (
          intervention_locations["sources->base"][0][0]
          if isinstance(
            intervention_locations["sources->base"][0][0][0][0], int
          )
          else intervention_locations["sources->base"][0][0][1]
        )
        print("\nInputs:")
        print("Base:", inputs["input"][:3])
        print("Source:", inputs["source_input"][:3])
        print("Tokens to intervene:")
        print(
          "    Base:",
          tokenizer.batch_decode(
            [
              inputs["input_ids"][i][base_locs[i]]
              for i in range(len(inputs["split"]))
            ]
          ),
        )
        print(
          "    Source:",
          tokenizer.batch_decode(
            [
              inputs["source_input_ids"][i][source_locs[i]]
              for i in range(len(inputs["split"]))
            ]
          ),
        )

        print("Outputs:")
        for output_type, outputs in [
          ("base", base_outputs),
          ("counterfactual", counterfactual_outputs),
        ]:
          output_tokens = extract_tokens_from_output(
            inference_mode, outputs, max_new_tokens, prompt_tokens=None
          )
          if inference_mode == "generate" or inference_mode == "force_decode":
            output_text = tokenizer.batch_decode(output_tokens)
            if output_type == "base":
              base_output_text = output_text
          else:
            output_text = [
              "Top K:"
              + "|".join([tokenizer.decode(t) for t in output_tokens[i]])
              for i in range(len(output_tokens))
            ]
          print(f"{output_type.title()} Output:".rjust(22), output_text)

        base_label_text = []
        if inference_mode == "generate" or inference_mode == "force_decode":
          base_label_text = tokenizer.batch_decode(
            remove_invalid_token_id(
              token_ids=inputs["base_labels"][:, :max_new_tokens],
              pad_id=tokenizer.pad_token_id,
            ),
            skip_special_tokens=True,
          )
        else:
          base_label_text = [
            "|".join(tokenizer.batch_decode(label[:max_new_tokens]))
            for label in inputs["base_labels"]
          ]
        print("Labels:")
        print("Base Label:".rjust(22), base_label_text)
        print(
          "Counterfactual Label:".rjust(22),
          tokenizer.batch_decode(
            remove_invalid_token_id(
              token_ids=inputs["labels"][:, :max_new_tokens],
              pad_id=tokenizer.pad_token_id,
            ),
            skip_special_tokens=True,
          ),
        )
        if base_label_text != base_output_text and inference_mode == "generate":
          print("WARNING: Base outputs does not match base labels!")
  return split_to_eval_metrics


def compute_logits_metrics(
  keyed_eval_preds, eval_labels, pad_token_id, last_n_tokens=1, **kwargs
):
  """Computes logprob/loss/loss of mean logits of the eval_labels."""
  metrics = {}
  loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
  for key, eval_preds in keyed_eval_preds.items():
    logprob_agg = []
    loss_agg, loss_exp_agg = [], []
    pred_agg = []
    label_agg = []
    for eval_pred, eval_label in zip(eval_preds, eval_labels):
      eval_pred = eval_pred[:, -last_n_tokens - 1 : -1].contiguous()
      eval_label = eval_label[:, -last_n_tokens:].unsqueeze(-1).contiguous()
      assert len(eval_pred.shape) == 3
      assert eval_pred.shape[:2] == eval_label.shape[:2]
      # Index into prob with labels
      safe_labels = torch.maximum(eval_label, torch.zeros_like(eval_label))
      assert torch.max(safe_labels).tolist() < eval_pred.shape[-1]
      assert torch.min(safe_labels).tolist() >= 0
      probs = torch.nn.functional.softmax(eval_pred, dim=-1)
      logprob = torch.log(torch.gather(probs, -1, safe_labels))
      # Remove paddings
      is_padding = torch.logical_or(
        (safe_labels == pad_token_id), (safe_labels == 0)
      )
      logprob = torch.where(is_padding, torch.zeros_like(logprob), logprob)
      # Average the logprob of the whole sequence
      logprob = torch.sum(logprob, dim=-1) / torch.sum(~is_padding, dim=-1)
      logprob_agg.append(logprob.tolist())
      # Loss
      loss = loss_fn(
        eval_pred.view(-1, eval_pred.size(-1)), eval_label.view(-1)
      )
      loss = loss.view(eval_label.size(0), -1)
      loss_agg.extend(loss.mean(dim=-1).tolist())
      loss_exp_agg.extend((-loss.mean(dim=-1)).exp().tolist())
      pred_agg.append(eval_pred)
      label_agg.append(eval_label)
    # Compute the mean logits representation instead of the mean loss.
    # For a given key, all eval labels should be the same.
    loss_mean_repr = loss_fn(
      torch.mean(torch.cat(pred_agg, dim=0), dim=0).view(
        -1, eval_pred.size(-1)
      ),
      eval_label[0].view(-1),
    )
    metrics[key] = {
      "accuracy": -1,
      "token_accuracy": -1,
      "loss": np.mean(loss_agg).tolist(),
      "loss_exp": np.mean(loss_exp_agg).tolist(),
      "loss_max": np.max(loss_agg).tolist(),
      "loss_exp_max": np.max(loss_exp_agg).tolist(),
      "loss_min": np.min(loss_agg).tolist(),
      "loss_exp_min": np.min(loss_exp_agg).tolist(),
      "loss_mean_repr": loss_mean_repr[0].tolist(),
    }
  metrics["accuracy"] = metrics["inv_outputs"]["loss_mean_repr"]
  return metrics
