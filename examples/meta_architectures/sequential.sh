# Run and exactly reproduce qwen2vl results!
# mme as an example
export HF_HOME="~/.cache/huggingface"

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
    --model sequential_model \
    --model_args=vlm_pred_name=qwen2_5_vl,vlm_pred_config={pretrained#Qwen/Qwen2.5-VL-7B-Instruct}\
    --tasks FALCONBench_mcq_temploc_metaarch \
    --batch_size 1