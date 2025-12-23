# Run and exactly reproduce qwen2vl results!
# mme as an example
export HF_HOME="~/.cache/huggingface"

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
    --model socratic_model \
    --model_args=vlm_caption_name=qwen2_5_vl,vlm_caption_config={pretrained#Qwen/Qwen2.5-VL-7B-Instruct},llm_vqa_name=gpt4v,llm_vqa_config={model_version#gpt-4o-mini-2024-07-18;modality#blind} \
    --tasks FALCONBench_mcq \
    --batch_size 1