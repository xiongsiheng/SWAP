# SWAP: Deliberate Reasoning in Language Models as Structure-Aware Planning with an Accurate World Model

This repository contains the code for the paper [ACL 25 (main)] [Deliberate Reasoning in Language Models as Structure-Aware Planning with an Accurate World Model](https://aclanthology.org/2025.acl-long.1540.pdf).



## **Overview**

**SWAP** introduces a structure-aware planning framework for deliberate multi-step reasoning in language models.
The framework consists of two core components:

* **Generator**
* **Discriminator**

Within this framework, the generator is repurposed to serve three roles:

* **Policy model** ($M_{\pi}$)
* **World model** ($M_{\text{wm}}$)
* **Controller** ($M_{\text{c}}$)

Given a goal $G$ and an initial state $(s_0, g_0)$, SWAP operates as follows:

1. **Planning:** The policy model $M_{\pi}$ generates an optimized plan $H$.
2. **Action generation:** Using $G$, $H$, and the current state $(s_t, g_t)$, the policy model proposes the next action $a_t$ through deliberate planning.
3. **State prediction:** The world model $M_{\text{wm}}$ predicts the next state $s_{t+1}$ and updates the entailment graph $g_{t+1}$.
4. **Control:** Based on $G$ and the updated state $(s_{t+1}, g_{t+1})$, the controller $M_{\text{c}}$ decides whether to continue the reasoning process or output the final answer.

During this process, the generator explores multiple candidate actions, and the discriminator evaluates the resulting partial trajectories to determine which trajectory should be continued.

<p align="center">
  <img src="https://raw.githubusercontent.com/xiongsiheng/SWAP/main/misc/Framework.png" width="650">
</p>

SWAP performs **multi-step reasoning** through structure-aware planning in tasks such as **FOLIO** (left) and **MATH** (right).
At each step, given the current state (represented as a graph) and an action, the world model predicts the next state as an updated graph.
The policy model is guided by this graph to propose the next action.

<p align="center">
  <img src="https://raw.githubusercontent.com/xiongsiheng/SWAP/main/misc/Example_tree_search.png" width="650">
</p>



## **Quick Start**

### **Directory structure**

```
SWAP/
├── checkpoints/
├── model_weights/
├── output/
├── scripts/
└── src/
```

### **Setup**

```bash
git clone https://github.com/xiongsiheng/SWAP.git
cd SWAP
```

For training:
```bash
# Create and activate the training environment
conda create -n swap_train python=3.10 -y
conda activate swap_train

# Install training dependencies
pip install -r requirements_train.txt
```

For evaluation:
```bash
# Create and activate the evaluation environment
# vLLM is used to substantially accelerate evaluation
conda create -n swap_eval python=3.10 -y
conda activate swap_eval

# Install evaluation dependencies
pip install -r requirements_eval.txt
```


## **Training**

```bash
# Train the generator
bash script/train_sft_generator_gsm8k.sh

# Train the discriminator
bash script/train_sft_discriminator_gsm8k.sh

# Optional: distributed training
bash script/train_sft_discriminator_gsm8k_dist.sh

# Optional: DPO training
bash script/train_dpo_discrimintor_gsm8k.sh
```


## **Evaluation**

(Optional) Download our checkpoints:
```bash
GENERATOR_CHECKPOINT=checkpoints/SWAP_v1_GSM8K_Gen_Llama3-8B
DISCRIMINATOR_CHECKPOINT=checkpoints/SWAP_v1_GSM8K_Disc_Llama3-8B

hf download sxiong/SWAP_v1_GSM8K_Gen_Llama3-8B \
  --local-dir "${GENERATOR_CHECKPOINT}"
hf download sxiong/SWAP_v1_GSM8K_Disc_Llama3-8B \
  --local-dir "${DISCRIMINATOR_CHECKPOINT}"
```

Run the evaluation:
```bash
# Evaluate the generator (without planning)
bash scripts/eval_generator_gsm8k.sh "${GENERATOR_CHECKPOINT}"

# Evaluate the full system
bash scripts/eval_system_gsm8k.sh "${GENERATOR_CHECKPOINT}" "${DISCRIMINATOR_CHECKPOINT}"

# Optional: distributed evaluation
CUDA_VISIBLE_DEVICES=0 NUM_SHARDS=4 SHARD_INDEX=0 bash scripts/eval_system_gsm8k.sh "${GENERATOR_CHECKPOINT}" "${DISCRIMINATOR_CHECKPOINT}"
CUDA_VISIBLE_DEVICES=1 NUM_SHARDS=4 SHARD_INDEX=1 bash scripts/eval_system_gsm8k.sh "${GENERATOR_CHECKPOINT}" "${DISCRIMINATOR_CHECKPOINT}"
CUDA_VISIBLE_DEVICES=2 NUM_SHARDS=4 SHARD_INDEX=2 bash scripts/eval_system_gsm8k.sh "${GENERATOR_CHECKPOINT}" "${DISCRIMINATOR_CHECKPOINT}"
CUDA_VISIBLE_DEVICES=3 NUM_SHARDS=4 SHARD_INDEX=3 bash scripts/eval_system_gsm8k.sh "${GENERATOR_CHECKPOINT}" "${DISCRIMINATOR_CHECKPOINT}"
```

### SWAP_v2

#### GSM8K

Download our checkpoints:
```bash
GSM8K_V2_GENERATOR_CHECKPOINT=checkpoints/SWAP_v2_GSM8K_Gen_Llama3-8B-LoRA
GSM8K_V2_DISCRIMINATOR_CHECKPOINT=checkpoints/SWAP_v2_GSM8K_Disc_Llama3-8B-LoRA

hf download sxiong/SWAP_v2_GSM8K_Gen_Llama3-8B-LoRA \
  --local-dir "${GSM8K_V2_GENERATOR_CHECKPOINT}"
hf download sxiong/SWAP_v2_GSM8K_Disc_Llama3-8B-LoRA \
  --local-dir "${GSM8K_V2_DISCRIMINATOR_CHECKPOINT}"
```

Run the evaluation:
```bash
# Evaluate the generator (without planning)
bash scripts/eval_generator_v2_gsm8k.sh "${GSM8K_V2_GENERATOR_CHECKPOINT}"

# Evaluate the full system
bash scripts/eval_system_v2_gsm8k.sh "${GSM8K_V2_GENERATOR_CHECKPOINT}" "${GSM8K_V2_DISCRIMINATOR_CHECKPOINT}"
```

#### MATH500

Download our checkpoints:
```bash
MATH500_V2_GENERATOR_CHECKPOINT=checkpoints/SWAP_v2_MATH500_Gen_Llama3-8B-LoRA
MATH500_V2_DISCRIMINATOR_CHECKPOINT=checkpoints/SWAP_v2_MATH500_Disc_Llama3-8B-LoRA

hf download sxiong/SWAP_v2_MATH500_Gen_Llama3-8B-LoRA \
  --local-dir "${MATH500_V2_GENERATOR_CHECKPOINT}"
hf download sxiong/SWAP_v2_MATH500_Disc_Llama3-8B-LoRA \
  --local-dir "${MATH500_V2_DISCRIMINATOR_CHECKPOINT}"
```

Run the evaluation:
```bash
# Evaluate the generator (without planning)
bash scripts/eval_generator_v2_math500.sh "${MATH500_V2_GENERATOR_CHECKPOINT}"

# Evaluate the full system
bash scripts/eval_system_v2_math500.sh "${MATH500_V2_GENERATOR_CHECKPOINT}" "${MATH500_V2_DISCRIMINATOR_CHECKPOINT}"
```

### MultiBench

Supported datasests: GSM8K, MATH500, FOLIO, ReClor, HumanEval, MBPP

```bash
# Download our checkpoints:
# - sxiong/SWAP_{dataset}_Gen_Llama3-8B-LoRA
# - sxiong/SWAP_{dataset}_Disc_Llama3-8B-LoRA

hf download sxiong/SWAP_FOLIO_Gen_Llama3-8B-LoRA \
  --local-dir checkpoints/SWAP_FOLIO_Gen_Llama3-8B-LoRA
hf download sxiong/SWAP_FOLIO_Disc_Llama3-8B-LoRA \
  --local-dir checkpoints/SWAP_FOLIO_Disc_Llama3-8B-LoRA

# Evaluate the generator and the full system:
# - multibench/scripts/eval_system_{dataset}.sh

bash multibench/scripts/eval_system_folio.sh
```

For detailed descriptions of the available arguments and configuration options, please refer to the source code.


## **Data & Checkpoints & Output**

All data used in SWAP (GSM8K, MATH, FOLIO, ReClor, HumanEval, MBPP) are available [here](https://huggingface.co/datasets/sxiong/SWAP):

```python
from datasets import load_dataset
dataset = load_dataset("sxiong/SWAP", "gsm8k")
train_data = dataset["train"]

print(train_data)
```

We also provide the corresponding [checkpoints](https://huggingface.co/collections/sxiong/swap) and [output](https://huggingface.co/datasets/sxiong/SWAP_output).

In addition, we release an updated version (SWAP_v2) of [data](https://huggingface.co/datasets/sxiong/SWAP_v2) and [checkpoints](https://huggingface.co/collections/sxiong/swap).


## **Citation**

```
@inproceedings{xiong2025deliberate,
  title={Deliberate reasoning in language models as structure-aware planning with an accurate world model},
  author={Xiong, Siheng and Payani, Ali and Yang, Yuan and Fekri, Faramarz},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={31900--31931},
  year={2025}
}
```
