# Internal Causal Mechanisms Robustly Predict Language Model Out-of-Distribution Behaviors

Interpretability research now offers a variety of techniques for identifying abstract internal mechanisms in neural networks. Can such techniques be used to predict how models will behave on out-of-distribution examples?

In this work, we evaluate methods using output probabilities, internal causal-agnostic features, internal causal features to predict correctness of LLM outputs. We show that the most robust features for correctness prediction are those that play a distinctive causal role in the model's behavior.

![](/figures/abstraction-prediction.svg)
*The hypothesized correspondence between internal mechanisms and generalization behaviors. In this work, we focus on the prediction direction.*

## Dataset

We release a dataset of five correctness prediction tasks. Given a task input and an LLM, the goal is to predict whether the LLM output is correct.

### Tasks

The five tasks cover symbol manipulation, knowledge retrieval, and instruction following, as shown below.

| Task Type | Have Known Internal Mechanisms | Task Names | 
|:--:|:--:|:--:|
| Symbol manipulation | Fully known | [Indirect Object Identification (IOI)](https://openreview.net/forum?id=NpsVSN6o4ul); [PriceTag](https://openreview.net/forum?id=nRfClnMhVX) |
| Knowledge retrieval | Partially known | [RAVEL](https://aclanthology.org/2024.acl-long.470/); [MMLU](https://openreview.net/forum?id=d7KBjmI3GmQ) |
| Instruction following | Partially known | [Unlearn Harry Potter](https://arxiv.org/abs/2403.03329) |


### Data format

Each JSON file represents one fold of a task, structured as follows:

```python
{
  "train" : {
    "correct": [
      "prompt_0",
      "prompt_1",
      ...
    ],
    "wrong": [
      "prompt_0",
      "prompt_1",
      ...
    ]
  },
  "val": {
    ...
  },
  "test": {
    ...
  }
}
```

We release the prompts used in our experiment, where the "correct" and "wrong" labels are determined using `Llama-3-8B-Instruct` as the target model.

If you are using these tasks to predict behaviors of a different target model, you need to regenerate the correctness label of these prompts.

## Methods

We evaluate four correctness prediction methods, categorized by the type of features they use.

|Method| Feature Type | Requires Training | Requires Wrong Samples | Requires Counterfactuals | Requires Decoding |
|:-:|:-:|:-:|:-:|:-:|:-:|
| Confidence Score | Output probabilities | &#x2717; | &#x2717; | &#x2717; | &#x2713; |
| Correctness Probing | Internal causal-agnostic features | &#x2713; | &#x2713; | &#x2717; | Maybe |
| Counterfactual Simulation | Internal causal features | Localization only | &#x2717; | &#x2713; | &#x2713; |
| Value Probing | Internal causal features | &#x2713; | &#x2717; | Localization only | Maybe |

#### Demo

We provide a demo evaluating each method on the MMLU correctness prediction task.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19DlL208ggxzuFBr_g57B46m2nUCTtQrN?usp=sharing)


## Citation

If you use the content of this repo, please kindly consider citing the following work

```
@inproceedings{
huang2025internal,
title={Internal Causal Mechanisms Robustly Predict Language Model Out-of-Distribution Behaviors},
author={Jing Huang and Junyi Tao and Thomas Icard and Diyi Yang and Christopher Potts},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=Ofa1cspTrv}
}
```
