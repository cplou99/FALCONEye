
<h1 align="center">FALCONEye: Finding Answers and Localizing Content in ONE-hour-long videos with multi-modal LLMs</h1>
<h3 align="center">WACV 2026</h3>
 <div align="center">
    <a href="https://cplou99.github.io/web/" target="_blank">Carlos Plou</a>,
    <a href="https://www.linkedin.com/in/cesar-borja-moreno/" target="_blank">Cesar Borja</a>,
    <a href="https://webdiis.unizar.es/~rmcantin/" target="_blank">Ruben Martinez-Cantin</a>,
    <a href="https://sites.google.com/unizar.es/anac/home?authuser=0" target="_blank">Ana C. Murillo</a>,
</div>


<div align="center">
   <a href="https://cplou99.github.io/FALCONEye/"><strong>🌍 Homepage</strong></a> | <a href="https://huggingface.co/datasets/cplou99/FALCON-Bench"><strong>🤗 Benchmark</strong></a> |  <a href="https://arxiv.org/abs/2503.19850"><strong>📝 ArXiv</strong></a> |  <a href="docs/img/falconeye_poster.png"><strong>🖼️ Poster</strong></a> |  <a href="https://youtu.be/watch?v=HDyc6uejWko"><strong>🎬 Presentation</strong></a>
   </div>   


## 🔔 News:
- 🆕 08/2026: New Gemini 3.5 Flash results on FALCON-Bench
- 🤗 12/2025: Code released!
- 🥳 11/2025: Paper accepted at WACV 2026!
- ⭐ 3/2025: We have released the [FALCON-Bench](https://huggingface.co/datasets/cplou99/FALCON-Bench) and [Paper](https://arxiv.org/abs/2503.19850)! 🔥

## Requirements
1. Follow [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) installation instructions. 

2. FALCON-Bench additionally requires the `soccernet` Python package. You can install it via pip:

```bash
pip install soccernet
```

## Description
This repo contains the code presented in the paper [FALCONEye](https://arxiv.org/abs/2503.19850): a **novel video agent for video question answering**.
FALCONEye code was built under the [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) framework. Specifically, the main contributions of this repo are:
- FALCON-Bench: lmms_eval/tasks/FALCONBench/
- FALCONEye meta-architecture: lmms_eval/models/meta_architecture/falcon_eye.py
- Agent baselines such as socratic, sequential, and sequentialBP are present in lmms_eval/models/meta_architecture. 

## FALCON-Bench Evaluation

### Recommendations
To evaluate FALCON-Bench with the latest LLMs, you can evaluate it from the [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) repository which is actively maintained. Otherwise, you can also use this repository which is a branch of lmms-eval frozen at the time of the FALCONEye paper submission. Instructions for both options are provided below.

### Setup Instructions

Before using FALCONBench, you must complete the following steps.

1. **Download Video Data**
	 - **SoccerNet:**  
		 - Fill out the [SoccerNet NDA form](https://docs.google.com/forms/d/e/1FAIpQLSfYFqjZNm4IgwGnyJXDPk2Ko_lZcbVtYX73w5lf6din5nxfmA/viewform).
		 - Save the password sent to your email as the environment variable `SOCCERNET_PWD`.
	 - **MovieChat-1K:**  
		 - Request access at [MovieChat-1K on HuggingFace](https://huggingface.co/datasets/Enxin/MovieChat-1K_train).
	 - **Walking Tours:**  
		 - These videos are already included in the Huggingface repository.

2. **Set Environment Variables**
	 - `SOCCERNET_PWD`: Password for SoccerNet video download.
	 - `OPENAI_API_KEY`: Required for open-ended question evaluation (OQ tasks).

	 Example (Linux):
	 ```bash
	 export SOCCERNET_PWD=your_soccernet_password
	 export OPENAI_API_KEY=your_openai_api_key
	 ```

3. **Download and Organize Videos**
	 - The first time you run the benchmark, the script will download the videos from the different sources and organize them in dataset_kwargs['cache_dir']/full_videos directory if they are not already present.

### Tasks Overview

FALCONBench includes four main tasks:

| Task Name                | Multiple-Choice | Open-Ended | Temporal Localization | Output Format |
|-------------------------|:--------------:|:----------:|:--------------------:|:-------------:|
| FALCONBench_mcq         |      ✅        |     ❌     |         ❌           |   String      |
| FALCONBench_mcq_temploc |      ✅        |     ❌     |         ✅           |   Dict        |
| FALCONBench_oq          |      ❌        |     ✅     |         ❌           |   String      |
| FALCONBench_oq_temploc  |      ❌        |     ✅     |         ✅           |   Dict        |

#### Example Dict Output Format for Temporal Localization Tasks

The model should return:

```json
{
	"response": "A person running",
	"temporal_window": [105, 140]
}
```

### Example: Running FALCONBench with LLaVA-Video

To launch the FALCONBench_mcq task using the LLaVA-Video model, use the following command:

```bash
bash examples/models/llava_video.sh
```

**Note1:** In the FALCONEye paper, results for small 7B VLMs are reported only for the MCQ and OQ tasks (without temporal localization) because these models struggle to output a json dictionary with both the answer and the temporal window, leading to a significant drop in accuracy when required to do so. 

**Note2:** In the FALCONEye paper, meta architectures were evaluated using FALCONBench_oq_temploc_metaarch and FALCONBench_mcq_temploc_metaarch tasks, which are equal to the temporal localization tasks but do not ask the model to return the temporal window, as this is handled by the meta architecture itself.

## FALCONEye

To easily run FALCONEye, simply execute the script:

```bash
bash examples/meta_architectures/falconeye.sh
```

This script provides ready-to-use commands for different settings, including the standard and "flash" versions, and allows you to vary the LLM (e.g., GPT-4o, Gemini) and VLM (e.g., Qwen2.5-VL, LLaVA-Video).

### Extending FALCONEye to Other Models

If you wish to use FALCONEye with any other VLM or LLM, you only need to implement an `inference` function following the examples provided:
- For VLMs, see the `inference` function in [lmms_eval/models/simple/qwen2_5_vl.py](lmms_eval/models/simple/qwen2_5_vl.py).
- For LLMs, see the `inference` function in [lmms_eval/models/simple/gpt4v.py](lmms_eval/models/simple/gpt4v.py).

With these minimal changes, you can extend FALCONEye to support additional models.

## FALCON-Bench Evaluation Results

Below are the performance comparisons of state-of-the-art VLMs and our agentic framework (FALCONEye) on the FALCON-Bench test split. The results evaluate both answering accuracy and temporal localization capability.

### 1. Multiple Choice Questions (MCQ)
| Type | Model | Temp Loc | SoccerNet (Acc) | MovieChat-1K (Acc) | Walking Tours (Acc) | **Average (Acc / mGToU)** |
|:---:|-------|:---:|:---:|:---:|:---:|:---:|
| **VLM** | Qwen2.5-VL (7B) | ❌ | 24.4% | 48.0% | 50.0% | 40.8% / - |
| **VLM** | LLaVA-Video (7B) | ❌ | 38.9% | 50.0% | 58.0% | 48.9% / - |
| **VLM** | Apollo (7B) | ❌ | 35.8% | 51.9% | 64.0% | 50.5% / - |
| **VLM** | **Gemini 3.5 Flash** | ❌ | 97.74% | 81.37% | 64.00% | **81.04% / -** |
| **VLM** | **Gemini 3.5 Flash** | ✅ | 98.31% | 81.37% | 64.00% | **81.23% / 7.32%** |
| ──────── | ──────────── | ─── | ────────── | ────────── | ────────── | ────────── |
| **Agent** | Socratic | ✅ | 54.0% | 51.9% | 48.0% | 53.9% / 21.3% |
| **Agent** | Lifelong Memory | ✅ | 41.5% | 45.0% | 38.0% | 41.8% / 0.00% |
| **Agent** | VideoAgent+OQ | ✅ | 52.2% | 48.0% | 36.0% | 45.4% / 19.1% |
| **Agent** | *FALCONEye (Flash)* | ✅ | 68.0% | 64.7% | 58.4% | *67.6% / 25.2%* |
| **Agent** | *FALCONEye (Pro)*| ✅ | 74.0% | 61.5% | 74.5% | *70.0% / 27.7%* |

### 2. Open-Ended Questions (OQ)
| Type | Model | Temp Loc | SoccerNet (Acc / Score) | MovieChat-1K (Acc / Score) | Walking Tours (Acc / Score) | **Average (Acc / Score / mGToU)** |
|:---:|-------|:---:|:---:|:---:|:---:|:---:|
| **VLM** | Qwen2.5-VL (7B) | ❌ | 8.19% / 0.25 | 11.7% / 0.72 | 10.0% / 0.53 | 9.96% / 0.62 / - |
| **VLM** | LLaVA-Video (7B) | ❌ | 12.0% / 0.86 | 10.7% / 0.76 | 11.0% / 0.66 | 11.2% / 0.76 / - |
| **VLM** | Apollo (7B) | ❌ | 18.0% / 0.68 | 16.6% / 0.96 | 10.7% / 1.06 | 15.1% / 0.90 / - |
| **VLM** | **Gemini 3.5 Flash** | ❌ | 86.44% / 4.43 | 50.00% / 2.65 | 30.00% / 1.68 | **55.48% / 2.92 / -** |
| **VLM** | **Gemini 3.5 Flash** | ✅ | 82.49% / 4.38 | 50.98% / 2.70 | 28.00% / 1.50 | **53.82% / 2.86 / 6.97%** |
| ──────── | ──────────── | ─── | ────────── | ────────── | ────────── | ────────── |
| **Agent** | Socratic | ✅ | 6.86% / 1.71 | 25.4% / 1.16 | 8.82% / 1.37 | 13.8% / 1.45 / 19.6% |
| **Agent** | Lifelong Memory | ✅ | 20.0% / 0.53 | 29.0% / 1.49 | 22.3% / 0.56 | 24.8% / 0.82 / 0.00% |
| **Agent** | VideoAgent+OQ | ✅ | 24.8% / 1.38 | 8.00% / 0.56 | 8.00% / 0.44 | 13.8% / 0.79 / 13.3% |
| **Agent** | *FALCONEye (Flash)* | ✅ | 38.4% / 2.03 | 43.1% / 2.36 | 42.0% / 2.28 | *41.1% / 2.22 / 22.7%* |
| **Agent** | *FALCONEye (Pro)*| ✅ | 37.2% / 1.96 | 50.9% / 2.67 | 46.0% / 2.50 | *44.7% / 2.38 / 24.9%* |

> *Note: All agent meta-architectures are implemented using the same configuration to ensure fair comparison: Qwen2.5-VL (7B) as the underlying VLM and GPT-4o-mini as the LLM.*

## Licenses

License: This project is released under the CC BY-NC 4.0 license for academic and research purposes. The codebase is built upon lmms-eval (Apache 2.0).

## 📝 Citation
```
@article{plou2025falconeye,
      title={FALCONEye: Finding Answers and Localizing Content in ONE-hour-long videos with multi-modal LLMs}, 
      author={Carlos Plou and Cesar Borja and Ruben Martinez-Cantin and Ana C. Murillo},
      booktitle={Proceedings of Winter Conference on Applications of Computer Vision},
      year={2026},
      eprint={2503.19850},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.19850},
}
```

## Acknowledgements

This work was supported by a DGA scholarship and by DGA project T45_23R, and grants AIA2025-163563-C31, PID2024-159284NB-I00, PID2021-125514NB-I00 and PID2024-158322OB-I00 funded by MCIN/AEI/10.13039/501100011033 and ERDF.
