# Run and exactly reproduce qwen2vl results!
# mme as an example
export HF_HOME="~/.cache/huggingface"

# Qwen2.5-vl + GPT-4o mini (FALCONEye-Pro paper setting)
CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
    --model falcon_eye \
    --model_args "vlm_name=qwen2_5_vl,vlm_vqa_fps=0.5,vlm_config={pretrained#Qwen/Qwen2.5-VL-7B-Instruct;max_pixels#768*28*28;total_pixels#15840*28*28},llm_reasoning_name=gpt4v,llm_reasoning_config={model_version#gpt-4o-mini-2024-07-18;modality#blind}" \
    --tasks FALCONBench_mcq_temploc_metaarch \
    --batch_size 1

# Qwen2.5-vl + GPT-4o mini (FALCONEye-Flash paper setting)
# CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
#    --model falcon_eye \
#    --model_args "vlm_name=qwen2_5_vl,flash_mode=True,max_num_vqa_inf=10,vlm_vqa_fps=0.5,vlm_config={pretrained#Qwen/Qwen2.5-VL-7B-Instruct},llm_reasoning_name=gpt4v,llm_reasoning_config={model_version#gpt-4o-mini-2024-07-18;modality#blind}" \
#    --tasks FALCONBench_mcq_temploc_metaarch \
#    --batch_size 1

# Qwen2.5-vl + Gemini 2.5 Pro (Ablation setting)
# CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
#    --model falcon_eye \
#    --model_args=vlm_name=vlm_name=qwen2_5_vl,vlm_vqa_fps=0.5,vlm_config={pretrained#Qwen/Qwen2.5-VL-7B-Instruct;max_pixels#768*28*28;total_pixels#15840*28*28},llm_reasoning_name=gemini_api,llm_reasoning_config={model_version#gemini-2.5-pro} \
#    --tasks FALCONBench_mcq \
#    --batch_size 1