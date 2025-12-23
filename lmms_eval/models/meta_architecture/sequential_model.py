import math
import os
from datetime import timedelta
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import copy
import time
import json
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from decord import VideoReader, cpu
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.model.language_model.llava_llama import LlavaConfig
from llava.model.language_model.llava_qwen import LlavaQwenConfig

# eval_logger = logging.getLogger("lmms-eval")
# import sys;sys.path.append("llava-video")
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav
from lmms_eval.utils import handle_arg_string
from lmms_eval.models import get_model

AutoConfig.register("llava_llama", LlavaConfig)
AutoConfig.register("llava_qwen", LlavaQwenConfig)


@register_model("sequential_model")
class Sequential(lmms):
    """
    Sequential Model
    """

    def __init__(
        self,
        vlm_pred_name: str,
        vlm_pred_config: str,
        frames_sampling_strategy: Optional[str] = "uniform", # ffmpeg_keyframes, resnet_keyframes
        num_frames_sampled: Optional[int] = 32,
        batch_size: Optional[int] = 1, 
        vlm_pred_device: Optional[str] = "cuda",
        device: Optional[str] = "cuda",
        video_decode_backend: Optional[str] = "decord",
        add_time_instruction: Optional[bool] = False,
        window_span: Optional[float] = 60,
        conf_thres_w_options: Optional[float] = 0.9,
        conf_thres_wo_options: Optional[float] = 0.8,
        vqa_dir: Optional[str] = f"{os.path.dirname(os.getcwd())}/features/vqa",
        save_vqa: Optional[bool] = True,
        load_vqa: Optional[bool] = True,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.vlm_pred_name = vlm_pred_name
        self.vlm_pred_config = vlm_pred_config
        self.frames_sampling_strategy = frames_sampling_strategy   
        self.num_frames_sampled = num_frames_sampled
        self.vlm_pred_device = vlm_pred_device
        self.video_decode_backend = video_decode_backend
        self.add_time_instruction = add_time_instruction
        self.window_span = window_span
        self.conf_thres = None
        self.conf_thres_w_options = conf_thres_w_options
        self.conf_thres_wo_options = conf_thres_wo_options
        vlm_pred_ModelClass = get_model(self.vlm_pred_name)

        self.vlm_pred_config = self.vlm_pred_config[1:-1].replace(";", ",").replace("#", "=")
        self.vlm_pred = vlm_pred_ModelClass.create_from_arg_string(
           self.vlm_pred_config,
            {
                "batch_size": batch_size,
                "device": self.vlm_pred_device,
            },
        )
        self.vlm_pred_max_frames_num = self.vlm_pred.max_frames_num
        self.vqa_dir = vqa_dir
        self.save_vqa = save_vqa
        self.load_vqa = load_vqa
        os.makedirs(self.vqa_dir, exist_ok=True)
        if not hasattr(self.vlm_pred, "inference"):
            raise AttributeError(f"Class '{self.vlm_pred.__name__}' does not have a function 'inference'. This function should act as generate_until but for a unique question and a video. Refer to qwen-2_5_vl.py as example.")

    def load_vqa_func(self, filename, question_id, gpt_eval):
        vqa_dir = os.path.join(self.vqa_dir, filename)
        os.makedirs(vqa_dir, exist_ok=True)

        vqa_filename = self.vlm_pred_name + question_id
        if gpt_eval:
            vqa_filename = f"{vqa_filename}_gpteval"
        vqa_filename = os.path.join(vqa_dir, vqa_filename+".json")
        if os.path.isfile(vqa_filename):
            with open(vqa_filename, "r") as f:
                vqa_dict = json.load(f)
        else:
            vqa_dict = {}
        return vqa_dict
    
    def numpy_converter(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert NumPy arrays to lists
        elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
            return obj.item()  # Convert NumPy scalars to Python scalars
        raise TypeError(f"Type {type(obj)} not serializable")

    def save_vqa_func(self, filename, question_id, gpt_eval, key_sampling_frames_info, vqa_dict, output_dict):
        vqa_dir = os.path.join(self.vqa_dir, filename)
        os.makedirs(vqa_dir, exist_ok=True)

        vqa_filename = self.vlm_pred_name + question_id
        if gpt_eval:
            vqa_filename = f"{vqa_filename}_gpteval"
        vqa_filename = os.path.join(vqa_dir, vqa_filename+".json")

        vqa_dict[key_sampling_frames_info] = output_dict

        with open(vqa_filename, "w") as f:
            json.dump(vqa_dict, f, default=self.numpy_converter)

    def add_time_instruction_to_contexts(self, contexts, video_info, sampling_frames_info):
        num_frames = sampling_frames_info["num_frames"]
        window = sampling_frames_info["window"]
        video_time = video_info["total_duration"]
        frames_times = f"{num_frames} frames are uniformly sampled from the clip {window}. These frames are located at {frames_times}."
        time_instruction = f"The video lasts for {video_time:.2f} seconds. {frames_times} Please answer the following questions related to this video."
        return f"{time_instruction}\n{contexts}"

    def get_video_info(self, video_file):
        from decord import VideoReader

        vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
        fps = vr.get_avg_fps()

        total_frames = len(vr)
        video_time = total_frames / fps
        video_info = {"vr": vr, "fps": fps, "total_frames": total_frames, "total_duration": video_time, "path": video_file}
        return video_info

    def load_video(self, video_info, max_num_frames, window_time=None):
        fps = video_info["fps"]
        total_valid_frames = video_info["total_frames"]
        video_time = video_info["total_duration"]
        vr = video_info["vr"]

        if window_time is None:
            num_frames = min(max_num_frames, int(total_valid_frames))
            frame_indices = [int(total_valid_frames / num_frames) * i for i in range(num_frames)]
        else:
            start_time, end_time = window_time[0], window_time[1]
            end_time = min(end_time, video_time)
            start_frame, end_frame = int(start_time * fps), int(end_time * fps)
            total_window_frames = int((end_time - start_time) * fps) 
            num_frames = min(max_num_frames, total_window_frames)
            frame_indices = [int(total_window_frames / num_frames) * i + start_frame for i in range(num_frames)]

        frames = vr.get_batch(frame_indices)
        if isinstance(frames, torch.Tensor):
            frames = frames.numpy()
        else:
            frames = frames.asnumpy()
        frame_timestamps = [frame_index / fps for frame_index in frame_indices]
        frame_timestamps = ",".join([f"{i:.2f}s" for i in frame_timestamps])
        return frames, frame_timestamps, video_time

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        print("TODO: NOT IMPLEMENTED YET")
        return None

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def get_confidence_from_outputs(self, outputs, choices_in_question=False):
        num_tokens = outputs["num_tokens"]
        tokens = outputs["tokens"]

        if choices_in_question:
            options_token = [tok for tok in tokens.values() if tok["token"] in ["A", "B", "C", "D"]]
            if len(options_token) == 0:
                response_probs = [float(tok["top5_probs"][0]) for tok in tokens.values()]
                conf = math.prod(response_probs) ** (1/len(response_probs))
            else:
                options_token = options_token[0]
                conf = options_token["top5_probs"][0]
        else:
            response_probs = [float(tok["top5_probs"][0]) for tok in tokens.values()]
            conf = math.prod(response_probs) ** (1/len(response_probs))
        
        return conf


    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            if "return_tempwindow" in gen_kwargs and gen_kwargs["return_tempwindow"]:
                return_tempwindow = True
            else:
                return_tempwindow = False
            if "question_format" in gen_kwargs and gen_kwargs["question_format"] == "oq":
                choices_in_question = False
                self.conf_thres = self.conf_thres_wo_options
            else:
                choices_in_question = True
                self.conf_thres = self.conf_thres_w_options

            times_and_inferences = {}
            visuals = doc_to_visual(self.task_dict[task][split][doc_id])

            video_path = visuals[0]
            video_name = video_path.split("/")[-1].split(".")[0]  # Extract video name
            if self.load_vqa:
                question_vqa_dict = self.load_vqa_func(video_name, str(self.task_dict[task][split][doc_id]['id']), "gpteval" in task)

            video_info = self.get_video_info(visuals[0])
            spf = 1.0 / video_info["fps"]
            confidence, best_confidence = 0, 0
            window_start, window_end = 0.0, self.window_span
            num_inferences = 0
            finished_video = False
            t0 = time.time()
            times_loaded = 0
            
            while not finished_video:
                window_time = [window_start, window_end]
                print("-------------")
                print(f"Processing window: {window_time}")
                if self.vlm_pred_name != "qwen2_5_vl":
                    sampling_frames_info = {"num_frames": self.vlm_pred_max_frames_num, "num_tiles": None, "window": window_time}
                else:
                    sampling_frames_info = {"window": window_time} 

                if self.add_time_instruction:
                    contexts = self.add_time_instruction_to_contexts(contexts, video_info, sampling_frames_info)

                gen_kwargs["return_dict_in_generate"] = True
                gen_kwargs["output_scores"] = True
                # gen_kwargs["output_logits"] = True

                key_sampling_frames_info = "".join([f"{key}_{value}" for key, value in sampling_frames_info.items()])
                if self.load_vqa and key_sampling_frames_info in question_vqa_dict:
                    outputs = question_vqa_dict.get(key_sampling_frames_info)
                    times_loaded += outputs["vqa_time"]
                else:
                    t_init = time.time()
                    outputs = self.vlm_pred.inference(video_info, sampling_frames_info, contexts, gen_kwargs)
                    if self.save_vqa:
                        outputs["vqa_time"] = time.time() - t_init
                        self.save_vqa_func(video_name, str(self.task_dict[task][split][doc_id]['id']), "gpteval" in task, key_sampling_frames_info, question_vqa_dict, outputs)

                if outputs is None:
                    print(f"The VLM model returned None for the outputs. Skipping this window: {window_time}.")
                    continue

                confidence = self.get_confidence_from_outputs(outputs, choices_in_question)
                window_start = window_end
                window_end = min(window_start + self.window_span, video_info["total_duration"])
                num_inferences += 1
                print("Response: ", outputs["response"], "with confidence: ", confidence)
                print("-------------")
                
                if confidence > best_confidence:
                    best_outputs = outputs
                    best_confidence = confidence
                    window2answer = [window_start-self.window_span, window_start]
                if window_start + spf >= video_info["total_duration"]:
                    finished_video = True
            
            answer = best_outputs
            times_and_inferences["true_total_time"] = (time.time() - t0) + times_loaded
            times_and_inferences["num_vqa_inferences"] = num_inferences
            times_and_inferences["num_vlm_inferences"] = num_inferences
            answer["temporal_window"] = window2answer
            answer["times_and_inferences"] = times_and_inferences
            if return_tempwindow:
                res.append(answer)
            else:
                res.append(answer["response"])
            print("-----------------------------------------------------------------------------------------------------------------------")
            print("Final Answer: ", answer["response"], "with confidence: ", best_confidence, "located at temporal window: ", answer["temporal_window"], "after num_inferences: ", num_inferences)
            print("-----------------------------------------------------------------------------------------------------------------------")
            pbar.update(1)
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for LLaVAVid")
