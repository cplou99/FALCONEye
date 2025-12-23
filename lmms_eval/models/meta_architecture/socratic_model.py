import math
import os
from datetime import timedelta
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import copy
import json
import time
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from decord import VideoReader, cpu

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

from pydantic import BaseModel
class TimeWindow(BaseModel):
    start: float
    end: float

class AnswerVQA(BaseModel):
    answer: str
    window: TimeWindow

@register_model("socratic_model")
class Socratic(lmms):
    """
    Socratic Model
    """

    def __init__(
        self,
        vlm_caption_name: str,
        vlm_caption_config: str,
        llm_vqa_name: str,
        llm_vqa_config: str,
        frames_sampling_strategy: Optional[str] = "uniform", # ffmpeg_keyframes, resnet_keyframes
        frame_rate: Optional[int] = 0.5,
        batch_size: Optional[int] = 1, 
        vlm_caption_device: Optional[str] = "cuda",
        llm_vqa_device: Optional[str] = "cuda",
        device: Optional[str] = "cuda",
        video_decode_backend: Optional[str] = "decord",
        add_time_instruction: Optional[bool] = False,
        window_span: Optional[float] = 60,
        vlm_caption_max_new_tokens: Optional[int] = 64, #Socratic original: 768
        save_captions: Optional[bool] = True,
        load_captions: Optional[bool] = True,
        captions_dir: Optional[str] = f"{os.path.dirname(os.getcwd())}/features/captions",
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.vlm_caption_name = vlm_caption_name
        self.vlm_caption_config = vlm_caption_config
        self.llm_vqa_name = llm_vqa_name
        self.llm_vqa_config = llm_vqa_config
        self.frames_sampling_strategy = frames_sampling_strategy   
        self.frame_rate = frame_rate
        self.vlm_caption_device = vlm_caption_device
        self.llm_vqa_device = llm_vqa_device
        self.video_decode_backend = video_decode_backend
        self.add_time_instruction = add_time_instruction
        self.window_span = window_span
        self.vlm_caption_max_new_tokens = vlm_caption_max_new_tokens
        self.captions_dir = captions_dir
        self.load_captions = load_captions
        self.save_captions = save_captions
        self.curr_filename = None
        self.curr_captions_dict = None
        vlm_caption_ModelClass = get_model(self.vlm_caption_name)

        self.vlm_caption_config = self.vlm_caption_config[1:-1].replace(";", ",").replace("#", "=")
        self.vlm_caption = vlm_caption_ModelClass.create_from_arg_string(
           self.vlm_caption_config,
            {
                "batch_size": batch_size,
                "device": self.vlm_caption_device,
            },
        )
        self.vlm_caption_max_frames_num = self.vlm_caption.max_frames_num

        self.vlm_caption_prompt = "Provide a short description of this clip that belongs to a long video. 15 words maximum. Be confident of your description, DO NOT PROVIDE INCORRECT INFORMATION."
        self.vlm_key_caption_prompt = "".join([w[0] for w in self.vlm_caption_prompt.split(" ")])
        llm_vqa_ModelClass = get_model(self.llm_vqa_name)
        self.llm_vqa_config = self.llm_vqa_config[1:-1].replace(";", ",").replace("#", "=")
        self.llm_vqa = llm_vqa_ModelClass.create_from_arg_string(
           self.llm_vqa_config,
            {
                "batch_size": batch_size,
                "device": self.llm_vqa_device,
            },
        )
        self.llm_vqa_max_frames_num = self.llm_vqa.max_frames_num

        if self.save_captions or self.load_captions:
            os.makedirs(self.captions_dir, exist_ok=True)
            self.vlm_key_caption_prompt = "".join([w[0] for w in self.vlm_caption_prompt.split(" ")])
            

        if not hasattr(self.vlm_caption, "inference"):
            raise AttributeError(f"Class '{self.vlm_caption.__name__}' does not have a function 'inference'. This function should act as generate_until but for a unique question and a video. Refer to qwen-2_5_vl.py as example.")
        if not hasattr(self.llm_vqa, "inference_format"):
            raise AttributeError(f"Class '{self.llm_vqa.__name__}' does not have a function 'inference_format'. This function should act as generate_until but for a unique question and a video. Refer to qwen-2_5_vl.py as example.")
    
   
    def load_captions_func(self, filename):
        if filename == self.curr_filename and self.curr_captions_dict is not None:
            return self.curr_captions_dict
        else:
            captions_dir = os.path.join(self.captions_dir, filename)
            os.makedirs(captions_dir, exist_ok=True)
            captions_filename = f'{self.vlm_caption_name}_uniform_window{self.window_span}s_fmin4_fmax32_toks{self.vlm_caption_max_new_tokens}_{self.vlm_key_caption_prompt}.json'
            captions_filename = os.path.join(captions_dir, captions_filename)
            if os.path.isfile(captions_filename):
                with open(captions_filename, "r") as f:
                    captions_dict = json.load(f)
            else:
                captions_dict = {"captions": {}, "num_inferences": 0, "total_time": 0}
            
            return captions_dict

    def save_captions_func(self, filename, captions):
        captions_dir = os.path.join(self.captions_dir, filename)
        captions_filename = f'{self.vlm_caption_name}__uniform_window{self.window_span}s_fmin4_fmax32_toks{self.vlm_caption_max_new_tokens}_{self.vlm_key_caption_prompt}.json'
        captions_filename = os.path.join(captions_dir, captions_filename)
        with open(captions_filename, "w") as f:
            json.dump(captions, f)
        
        self.curr_captions_dict = captions
        self.curr_filename = filename

    def get_video_info(self, video_file):
        from decord import VideoReader

        vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
        fps = vr.get_avg_fps()

        total_frames = len(vr)
        video_time = total_frames / fps
        video_info = {"vr": vr, "fps": fps, "total_frames": total_frames, "total_duration": video_time, "path": video_file}
        return video_info
    
    def add_time_instruction_to_contexts(self, contexts, video_info, sampling_frames_info):
        num_frames = sampling_frames_info["num_frames"]
        window = sampling_frames_info["window"]
        video_time = video_info["total_duration"]
        frames_times = f"{num_frames} frames are uniformly sampled from the clip {window}. These frames are located at {frames_times}."
        time_instruction = f"The video lasts for {video_time:.2f} seconds. {frames_times} Please answer the following questions related to this video."
        return f"{time_instruction}\n{contexts}"

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


    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            visuals = doc_to_visual(self.task_dict[task][split][doc_id])

            if "return_tempwindow" in gen_kwargs and gen_kwargs["return_tempwindow"]:
                return_tempwindow = True
            else:
                return_tempwindow = False
            if "question_format" in gen_kwargs and gen_kwargs["question_format"] == "oq":
                self.conf_thres = self.conf_thres_wo_options
            else:
                self.conf_thres = self.conf_thres_w_options

            times_and_inferences = {}
            video_info = self.get_video_info(visuals[0])
            video_time = video_info["total_duration"]
            filename = visuals[0].split("/")[-1].split(".")[0]
            t_infs = []

            if self.load_captions:
                captions = self.load_captions_func(filename)
                captions_dict = captions["captions"]
                num_inferences = captions["num_inferences"]
                caption_time = captions["total_time"]
                num_windows = math.ceil(video_time / self.window_span) - 1
                all_windows = [f"[{k*self.window_span}s-{(k+1)*self.window_span}s]" for k in range(num_windows)]
                all_captions_dict = {key: captions_dict[key] for key in all_windows if key in captions_dict}
                print("Number of windows that should have been infered", num_windows, "Number of captions in the file", len(captions_dict), "Number of captions taken from the file", len(all_captions_dict))

            if not self.load_captions or len(captions_dict) < num_windows:
                window_start, window_end = 0.0, self.window_span
                num_inferences = 0
                all_captions_dict = {}
                spf = 0.5
                gen_kwargs["max_new_tokens"] = self.vlm_caption_max_new_tokens
                t0 = time.time()
                while window_start + spf <= video_info["total_duration"]:
                    print("-------------")
                    window_time = [window_start, min(window_end, video_info["total_duration"])]
                    print("Generating caption from window: ", window_time)
                    curr_window_span = window_time[1] - window_time[0]
                    num_frames = min(int(self.frame_rate * curr_window_span), self.vlm_caption_max_frames_num)

                    if num_frames < 1:
                        print("No frames to sample in the window: ", window_time, "in the video: ", video_info["path"], "of total duration: ", video_info["total_duration"])
                        window_start = window_end
                        window_end = window_start + self.window_span
                        break

                    if self.vlm_caption_name != "qwen2_5_vl":
                        sampling_frames_info = {"num_frames": num_frames, "num_tiles": None, "window": window_time}
                    else:
                        sampling_frames_info = {"num_frames": num_frames, "window": window_time} 

                    if self.add_time_instruction:
                        caption_prompt = self.add_time_instruction_to_contexts(self.vlm_caption_prompt, video_info, sampling_frames_info)
                    else:
                        caption_prompt = self.vlm_caption_prompt
                    t_inf_s = time.time()
                    caption = self.vlm_caption.inference(video_info, sampling_frames_info, caption_prompt, gen_kwargs)
                    t_inf = time.time() - t_inf_s
                    print("Time spent at vlm inference: ",  t_inf)
                    t_infs.append(t_inf)
                    print("Caption: ", caption)
                    if caption is None:
                        print("No caption generated for the window: ", window_time, "in the video: ", video_info["path"], "of total duration: ", video_info["total_duration"])
                    else:
                        all_captions_dict[f"[{window_start}s-{window_end}s]"] = caption

                    window_start = window_end
                    if window_end+spf >= video_info["total_duration"]:
                        break
                    else:
                        window_start = window_end
                        window_end = min(window_start + self.window_span, video_info["total_duration"])
                    num_inferences += 1
                    print("-------------")

                t1 = time.time()
                caption_time = t1 - t0
                if self.save_captions:
                    file_dict = {"captions": all_captions_dict, "num_inferences": num_inferences, "total_time": caption_time}
                    self.save_captions_func(filename, file_dict)
            else:
                print("Captions loaded from file")

            t1 = time.time()

            all_captions = [f"{key}: {value}" for key, value in all_captions_dict.items()]
            messages = [
                {"role": "system", "content": "Answer the next question from the following captions of a video and return the temporal window of the video in which the answer is contained."},
                {"role": "user", "content": f"Question: {contexts}"},
                {"role": "user", "content": f"Captions: {all_captions}"}
            ]
            t_llm_s = time.time()
            answer_format = self.llm_vqa.inference_format(AnswerVQA, messages, False)
            t_llm_inf = time.time() - t_llm_s
            # print("Time spent at llm inference: ",  t_llm_inf, "tokens usage: ", answer_format.usage.total_tokens)
            response = answer_format.choices[0].message.parsed.answer
            temporal_window = [answer_format.choices[0].message.parsed.window.start, answer_format.choices[0].message.parsed.window.end]

            t2 = time.time()

            vqa_time = t2 - t1

            times_and_inferences["true_total_time"] = caption_time + vqa_time
            times_and_inferences["num_llm_inferences"] = 1
            times_and_inferences["num_vlm_inferences"] = num_inferences
            times_and_inferences["vqa_time"] = vqa_time
            times_and_inferences["caption_video_time"] = caption_time
            times_and_inferences["caption_time"] = caption_time
            times_and_inferences["llm_tokens_usage"] = answer_format.usage.total_tokens
            
            answer = {
                "response": response,
                "times_and_inferences": times_and_inferences,
                "temporal_window": temporal_window
            }
            print("-----------------------------------------------------------------------------------------------------------------------")
            print("Final Answer: ", answer["response"], "located at temporal window: ", answer["temporal_window"], "after num_inferences: ", num_inferences)
            print("-----------------------------------------------------------------------------------------------------------------------")
            if return_tempwindow:
                res.append(answer)
            else:
                res.append(answer["response"])

            pbar.update(1)
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for LLaVAVid")
