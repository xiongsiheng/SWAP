# SWAP: Deliberate Reasoning for LLMs as Structure-aware Planning with Accurate World Model

This repository contains the code for the paper [Deliberate Reasoning for LLMs as Structure-aware Planning with Accurate World Model](https://arxiv.org/pdf/2410.03136).

SWAP consists of three main components: the policy model $M_{\pi}$, the world model $M_{\text{wm}}$, and the controller $M_\text{c}$. Starting with the goal $G$ and the initial state $(s_0, g_0)$, the policy model $M_{\pi}$ generates an optimized plan $H$. Using $G$, $H$, and the current state $(s_t, g_t)$, $M_{\pi}$ proposes the next action $a_t$ through deliberate planning. After the action is chosen, the world model $M_{\text{wm}}$ predicts the next state $s_{t+1}$ and updates the entailment graph $g_{t+1}$. Finally, based on $G$ and the updated state $(s_{t+1}, g_{t+1})$, the controller $M_c$ decides whether to continue the process or output the final answer.

<br>

<p align="center">
  <img src='https://raw.githubusercontent.com/xiongsiheng/SWAP/main/misc/Framework.png' width=650>
</p>

<br>

SWAP performs multi-step reasoning through structure-aware planning in FOLIO (left) and MATH (right). At each step, given the current state, represented as a graph, and an action, the world model predicts the next state as an updated graph.

<br>
<p align="center">
  <img src='https://raw.githubusercontent.com/xiongsiheng/SWAP/main/misc/Example_tree_search.png' width=650>
</p>





## Quick Start
We use [Hugging Face](https://huggingface.co/) platform to load the Llama3 model family. Make sure you have an account ([Guidance](https://huggingface.co/blog/llama3)).

The structure of the file folder should be like
```sh
SWAP/
│
├── materials/
│
├── model_weights/
│
├── results/
│
└── src/
```

<h4> Preparation: </h4>

```sh
# git clone this repo

# create a new environment with anaconda and install the necessary Python packages

# install hugging face packages to load the base models and datasets

# create the folders
cd SWAP
mkdir model_weights
mkdir results
mkdir materials
cd src
```

<h4> For our SWAP framework: </h4>

- Base model fine-tuning

```sh
# Train the generator
python SFT_Generator.py --dataset MATH --subset algebra --prob_type math --train --print_example

# Train the semantical equivalence LoRA
python SFT_sem_equ_LoRA.py --dataset MATH --subset algebra --train --print_example

# Train the discriminator
python SFT_Discriminator.py --dataset MATH --subset algebra --prob_type math --group_size 2 --train --print_example 
```

- Inference

```sh
python main.py --dataset MATH --subset algebra --prob_type math --enable_DBM --visualize --max_steps 20 --num_rollouts 3 --num_generations 3 --group_size 2
```

Please check the source code for detailed parameter explanation.

## Datasets

All the datasets (gsm8k, MATH, FOLIO, ReClor, HumanEval, MBPP) with trajectories and process supervision can be found [here](https://huggingface.co/datasets/sxiong/SWAP).

To download the dataset, install [Huggingface Datasets](https://huggingface.co/docs/datasets/quickstart) and then use the following command:

```python
from datasets import load_dataset
dataset = load_dataset("sxiong/SWAP", "MATH_trajectory")
print(dataset)
split = dataset['train']
```

## Accelerate with Multi GPUs
The default training/inference arguments are for a single A100 (GPU memory: 80G). If you have multiple GPUs, the **training** process can be accelerated in a distributed way. Here we recommend the library of **DeepSpeed** [[docs]](https://huggingface.co/docs/peft/en/accelerate/deepspeed).

Also, you can accelerate the **inference** with multiple GPUs.

## Contact
If you have any inquiries, please feel free to raise an issue or reach out to sxiong45@gatech.edu.

## Citation
```
@article{xiong2024deliberate,
  title={Deliberate Reasoning for LLMs as Structure-aware Planning with Accurate World Model},
  author={Xiong, Siheng and Payani, Ali and Yang, Yuan and Fekri, Faramarz},
  journal={arXiv preprint arXiv:2410.03136},
  year={2024}
}
```
