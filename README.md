# SWAP: Deliberate Reasoning in Language Models as Structure-Aware Planning with an Accurate World Model

This repository contains the code for the paper [ACL 25 (main)] [Deliberate Reasoning in Language Models as Structure-Aware Planning with an Accurate World Model](https://aclanthology.org/2025.acl-long.1540.pdf).



## **Overview**

**SWAP** introduces a structure-aware planning framework that enables deliberate multi-step reasoning in language models.
It comprises three main components:

* **Policy model** ($M_{\pi}$)
* **World model** ($M_{\text{wm}}$)
* **Controller** ($M_{\text{c}}$)

Starting from the goal $G$ and the initial state $(s_0, g_0)$:

1. **Planning:** The policy model $M_{\pi}$ generates an optimized plan $H$.
2. **Action generation:** Using $G$, $H$, and the current state $(s_t, g_t)$, the policy model proposes the next action $a_t$ through deliberate planning.
3. **State prediction:** The world model $M_{\text{wm}}$ predicts the next state $s_{t+1}$ and updates the entailment graph $g_{t+1}$.
4. **Control:** Based on $G$ and the updated state $(s_{t+1}, g_{t+1})$, the controller $M_{\text{c}}$ decides whether to continue the reasoning process or output the final answer.

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

We use the [Hugging Face](https://huggingface.co/) platform to load base models such as **Llama 3** and **Mistral**.
Ensure you have a Hugging Face account ([guidelines](https://huggingface.co/blog/llama3)) before starting.


### **Directory structure**

```
SWAP/
├── materials/
├── model_weights/
├── results/
└── src/
```

### **Setup**

```bash
git clone https://github.com/xiongsiheng/SWAP.git
cd SWAP

# Create and activate environment
conda create -n swap python=3.10 -y
conda activate swap

# Install dependencies
pip install -r requirements.txt
```


## **Training**

### **Base Model Fine-Tuning**

```bash
# Train the generator
accelerate launch SFT_Generator.py --dataset MATH --subset algebra --prob_type math --train --print_example

# Train the semantic-equivalence LoRA
accelerate launch SFT_sem_equ_LoRA.py --dataset MATH --subset algebra --train --print_example

# Train the discriminator
accelerate launch SFT_Discriminator.py --dataset MATH --subset algebra --prob_type math --group_size 2 --train --print_example
```


## **Inference**

```bash
accelerate launch main.py \
  --dataset MATH \
  --subset algebra \
  --prob_type math \
  --enable_DM \
  --visualize \
  --max_steps 20 \
  --num_rollouts 3 \
  --num_generations 3 \
  --group_size 2
```

(Refer to the source code for detailed parameter descriptions.)


## **Datasets**

All datasets used in SWAP (GSM8K, MATH, FOLIO, ReClor, HumanEval, MBPP) with trajectory and process supervision are available on [Hugging Face Datasets](https://huggingface.co/datasets/sxiong/SWAP):

```python
from datasets import load_dataset

dataset = load_dataset("sxiong/SWAP", "MATH_trajectory")
print(dataset)
split = dataset["train"]
```

We also provide an updated version (**[SWAP_v2](https://huggingface.co/datasets/sxiong/SWAP_v2)**) featuring **DeepSeek V3.2** and provides the corresponding [model weights](https://huggingface.co/sxiong/SWAP_LLM_v2).



## **Acceleration with Multiple GPUs**

The default configuration targets a single **A100 (80 GB)** GPU.
To accelerate **training**, we recommend distributed execution with **[DeepSpeed](https://huggingface.co/docs/peft/en/accelerate/deepspeed)**.
**Inference** can also be parallelized across multiple GPUs for efficiency.



## **Contact**

If you have any inquiries, please feel free to raise an issue or reach out to sxiong45@gatech.edu.



## **Citation**

```
@inproceedings{xiong-etal-2025-deliberate,
    title = "Deliberate Reasoning in Language Models as Structure-Aware Planning with an Accurate World Model",
    author = "Xiong, Siheng  and
      Payani, Ali  and
      Yang, Yuan  and
      Fekri, Faramarz",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.1540/",
    doi = "10.18653/v1/2025.acl-long.1540",
    pages = "31900--31931",
    ISBN = "979-8-89176-251-0"
}
```

