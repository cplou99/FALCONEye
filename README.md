
<h1 align="center">FALCONEye: Finding Answers and Localizing Content in ONE-hour-long videos with multi-modal LLMs</h1>
<h3 align="center">WACV 2026</h3>
 <div align="center">
    <a href="https://cplou99.github.io/web/" target="_blank">Carlos Plou</a>,
    <a href="https://www.linkedin.com/in/cesar-borja-moreno/" target="_blank">Cesar Borja</a>,
    <a href="https://webdiis.unizar.es/~rmcantin/" target="_blank">Ruben Martinez-Cantin</a>,
    <a href="https://sites.google.com/unizar.es/anac/home?authuser=0" target="_blank">Ana C. Murillo</a>,
</div>


<div align="center">
   <a href="https://cplou99.github.io/FALCONEye/"><strong>🌍 Homepage</strong></a> | <a href="https://huggingface.co/datasets/cplou99/FALCON-Bench"><strong>🤗 Benchmark</strong></a> |  <a href="https://arxiv.org/abs/2503.19850"><strong>📝 ArXiv</strong></a>
   </div>   


## 🔔 News:
- 🔜 Code will be released soon.
- 🥳 11/2025: Paper accepted at WACV 2026!
- ⭐ 3/2025: We have released the [FALCON-Bench](https://huggingface.co/datasets/cplou99/FALCON-Bench) and [Paper](https://arxiv.org/abs/2503.19850)! 🔥

## Description
This repo contains the code presented in the paper "[FALCONEye: Finding Answers and Localizing Content in ONE-hour-long videos with multi-modal LLMs](https://arxiv.org/abs/2503.19850)".
FALCONEye code was built under the [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) framework. Specifically, the main contributions of this repo are:
- The FALCON-Bench benchmark for evaluating multi-modal LLMs in long videos with temporal localization: lmms_eval/tasks/FALCONBench/
- FALCONEye meta-architecture for long video understanding with multi-modal LLMs: lmms_eval/models/falconeye.py
Additionally, some baselines such as socratic, and  are present in lmms_eval/models/. And the VLMs and LLMs used as part of the meta-architecture are addes a function called inference to accept individual inferences.

## FALCON-Bench

### Recommendations
To evaluate FALCON-Bench with the latest models, you can evaluate it from the [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) repository which is actively maintained. Otherwise, you can also use this repository which is a branch of lmms-eval frozen at the time of the FALCONEye paper submission. Instructions for both options are provided below.

### Setup Instructions

Before using FALCONBench, you must complete the following steps:

**Note:** The benchmark requires the `soccernet` Python package. You can install it via pip:

```bash
pip install soccernet
```

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

- **FALCONBench_mcq:**  
	Multiple-choice questions about video content. The model selects the correct option (A, B, C, ...).

- **FALCONBench_mcq_temploc:**  
	Multiple-choice with temporal localization. The model selects the correct option and specifies the time window (in seconds) where the answer is observed.

- **FALCONBench_oq:**  
	Open-ended questions. The model generates a free-form answer, which is evaluated for semantic correctness.

- **FALCONBench_oq_temploc:**  
	Open-ended with temporal localization. The model generates an answer and specifies the relevant time window.

### Example Output Format for Temporal Localization Tasks

The model should return:

```json
{
	"response": "A person running",
	"temporal_window": [105, 140]
}
```

### Example: Running FALCONBench with LLaVA-Video

To launch the benchmark using the LLaVA-Video model, use the following command:

```bash
accelerate launch --num_processes 1 --main_process_port 12345 -m lmms_eval \
		--model llava_vid \
		--model_args pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,video_decode_backend=decord,max_frames_num=32,mm_spatial_pool_mode=average,mm_newline_position=grid,mm_resampler_location=after \
		--tasks FALCONBench_mcq \
		--batch_size 1 \
		--log_samples \
		--log_samples_suffix falconbench-llava_vid_7B \
		--output_path ./logs/
```

**Note:** In the FALCONEye paper, results for small 7B VLMs are reported only for the MCQ and OQ tasks (without temporal localization) because these models struggle to output a json dictionary with both the answer and the temporal window, leading to a significant drop in accuracy when required to do so.

Replace `--tasks FALCONBench_mcq` with any of the other tasks (`FALCONBench_mcq_temploc`, `FALCONBench_oq`, `FALCONBench_oq_temploc`) as needed.


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
