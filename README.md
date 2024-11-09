# SWAP: Deliberate Reasoning for LLMs as Structure-aware Planning with Accurate World Model

This repository contains the code for the paper [Deliberate Reasoning for LLMs as Structure-aware Planning with Accurate World Model](https://arxiv.org/pdf/2410.03136).

SWAP consists of three main components: the policy model $M_{\pi}$, the world model $M_{\text{wm}}$, and the controller $M_\text{c}$. Starting with the goal $G$ and the initial state $s_0(g_0)$, the policy model $M_{\pi}$ generates an optimized plan $PL$. Using $G$, $PL$, and the current state $s_t(g_t)$, $M_{\pi}$ proposes the next action $a_t$ through deliberate planning. After the action is chosen, the world model $M_{\text{wm}}$ predicts the next state $s_{t+1}$ and updates the entailment graph $g_{t+1}$. Finally, based on $G$ and the updated state $s_{t+1}(g_{t+1})$, the controller $M_c$ decides whether to continue the process or output the final answer.

<br>

<p align="center">
  <img src='https://raw.githubusercontent.com/xiongsiheng/SWAP/main/misc/Framework.png' width=650>
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

## Code & Data
Under construction

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
