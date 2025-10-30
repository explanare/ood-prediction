"""Extrat hidden states and train linear probes."""

import collections
import numpy as np

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC


def extract_all_hidden_states_batched(
  model,
  tokenizer,
  prompts,
  input_max_length,
  token_positions=None,
  batch_size=64,
):
  # Limit RAM usage to 4 GB
  assert len(prompts) < 2048 or token_positions is not None
  all_hidden_states = []
  with torch.no_grad():
    for b_i in range(0, len(prompts), batch_size):
      batch_prompts = prompts[b_i : b_i + batch_size]
      model.config.output_hidden_states = True
      encoded_inputs = tokenizer(
        batch_prompts,
        return_tensors="pt",
        padding="max_length",
        max_length=input_max_length,
      ).to(model.device)
      position_ids = encoded_inputs.attention_mask.long().cumsum(-1) - 1
      position_ids.masked_fill_(encoded_inputs.attention_mask == 0, 1)
      output = model(**encoded_inputs, position_ids=position_ids)
      hidden_states = (
        torch.stack(output.hidden_states, dim=1).detach().cpu()
      )  # B * L * S * DIM
      if token_positions is not None:
        ordered_pos = list(token_positions.values())
        hidden_states = hidden_states[:, :, ordered_pos]
      all_hidden_states.append(hidden_states)
  all_hidden_states = (
    torch.cat(all_hidden_states, dim=0).to(torch.float16).numpy()
  )
  pos_to_all_hidden_states = {
    k: all_hidden_states[:, :, i]
    for i, (k, _) in enumerate(token_positions.items())
  }
  return pos_to_all_hidden_states


def train_linear_binary_classifier(
  train_feats,
  val_feats,
  cls_type="SVC",
  class_labels=None,
  return_accuracy=False,
):
  if class_labels is None:
    class_labels = list(train_feats)
  train_feats_np, val_feats_np = {}, {}
  if isinstance(train_feats[class_labels[0]], torch.Tensor):
    for k in train_feats:
      train_feats_np[k] = train_feats[k].cpu().float().numpy()
    for k in val_feats:
      val_feats_np[k] = val_feats[k].cpu().float().numpy()
  else:
    train_feats_np = train_feats
    val_feats_np = val_feats
  X = np.concatenate([train_feats_np[k] for k in class_labels], axis=0)
  Y = np.array(
    [
      i
      for i, k in enumerate(class_labels)
      for _ in range(train_feats_np[k].shape[0])
    ]
  )
  if cls_type == "NB":
    clf = GaussianNB()
  elif cls_type == "SVC":
    clf = LinearSVC(C=0.05, penalty="l1", dual=False, max_iter=1000, tol=0.01)
  elif cls_type == "LR":
    clf = LogisticRegression(max_iter=1000, tol=0.01, C=0.05)
  else:
    raise ValueError("Unknown classifier type")
  clf = clf.fit(X, Y)
  X_val = np.concatenate([val_feats_np[k] for k in class_labels], axis=0)
  Y_val = np.array(
    [
      i
      for i, k in enumerate(class_labels)
      for _ in range(val_feats_np[k].shape[0])
    ]
  )
  accuracy = {"train": clf.score(X, Y), "val": clf.score(X_val, Y_val)}
  print("Training set accuracy:", accuracy["train"])
  print("Validation set accuracy:", accuracy["val"])
  if return_accuracy:
    return clf, accuracy
  return clf


def compute_auc_roc_for_all_locations(hidden_states, example_labels, method):
  def softmax(x):
    return torch.softmax(x, dim=-1)

  device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  )
  class_labels = list(set(example_labels["train"] + example_labels["val"]))
  print(len(class_labels), class_labels)
  split_type_to_metrics = collections.defaultdict(dict)
  for loc_key in hidden_states["train"]:
    for layer in range(hidden_states["train"][loc_key].shape[1]):
      print(layer, loc_key)
      probe_reprs_train = {
        k: hidden_states["train"][loc_key][
          [i for i, y in enumerate(example_labels["train"]) if y == k], layer, :
        ]
        for k in class_labels
      }
      probe_reprs_val = {
        k: hidden_states["val"][loc_key][
          [i for i, y in enumerate(example_labels["val"]) if y == k], layer, :
        ]
        for k in class_labels
      }
      if method == "svm":
        clf, accuracy = train_linear_binary_classifier(
          probe_reprs_train,
          probe_reprs_val,
          class_labels=class_labels,
          return_accuracy=True,
        )
      elif method == "lr":
        clf, accuracy = train_linear_binary_classifier(
          probe_reprs_train,
          probe_reprs_val,
          cls_type="LR",
          class_labels=class_labels,
          return_accuracy=True,
        )
      elif method == "knn" or method == "nb":
        clf = (
          KNeighborsClassifier(n_neighbors=64, weights="distance")
          if method == "knn"
          else GaussianNB()
        )
        X_train = np.concatenate(list(probe_reprs_train.values()), axis=0)
        Y_train = [
          k for k in class_labels for _ in range(len(probe_reprs_train[k]))
        ]
        clf = clf.fit(X_train, Y_train)
      # Test
      probe_reprs_test = {
        k: hidden_states["test"][loc_key][
          [i for i, y in enumerate(example_labels["test"]) if y == k], layer, :
        ]
        for k in ("correct", "wrong")
      }
      if method == "svm" or method == "lr":
        logits = np.concatenate(
          list(probe_reprs_test.values()), axis=0
        ) @ clf.coef_.T + clf.intercept_.reshape([1, -1])
        if logits.shape[-1] == 1:
          act_fn = torch.sigmoid
        else:
          act_fn = softmax
        prob = (
          act_fn(torch.tensor(logits, dtype=torch.float32, device=device))
          .cpu()
          .numpy()
        )
      elif method == "knn":
        prob = clf.predict_proba(
          np.concatenate(list(probe_reprs_test.values()), axis=0)
        )
      if "correct" not in class_labels and len(class_labels) == 2:
        prob = np.abs(prob - 0.5)
      if "correct" not in class_labels and len(class_labels) > 1:
        prob = np.max(prob, axis=-1)
      if "correct" in class_labels:
        label = np.array(
          [class_labels.index("correct")] * len(probe_reprs_test["correct"])
          + [class_labels.index("wrong")] * len(probe_reprs_test["wrong"])
        )
      else:
        label = np.array(
          [1] * len(probe_reprs_test["correct"])
          + [0] * len(probe_reprs_test["wrong"])
        )
      auc_roc = roc_auc_score(label, prob)
      split_type_to_metrics[layer][loc_key] = {
        "auc_roc": auc_roc,
        "val_accuracy": accuracy["val"],
      }
  split_type_to_metrics = {k: dict(v) for k, v in split_type_to_metrics.items()}
  return split_type_to_metrics
