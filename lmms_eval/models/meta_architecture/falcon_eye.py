import math
import os
from datetime import timedelta
import random
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import copy
import json
import time
import subprocess
import gc
import psutil
import pickle

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

class Scene(BaseModel):
    explanation: str
    cliptimeinterval: TimeWindow

class VideoReasoning(BaseModel):
    scenes: list[Scene]
        
@register_model("falcon_eye")
class FALCONEye(lmms):
    """
    FALCONEye Model
    """

    def __init__(
        self,
        vlm_name: str,
        vlm_config: str,
        llm_reasoning_name: str,
        llm_reasoning_config: str,
        flash_mode: Optional[bool] = False,
        trust_in_confidence_mode: Optional[bool] = False,
        frames_sampling_strategy: Optional[str] = "uniform", # ffmpeg_keyframes, resnet_keyframes
        max_num_vqa_inf: Optional[int] = 45,
        max_returned_window_size: Optional[int] = 60,
        conf_thres_w_options: Optional[float] = 0.9,
        conf_thres_wo_options: Optional[float] = 0.8,
        min_window_duration: Optional[int] = 2,
        add_time_instruction: Optional[bool] = False,
        window_span: Optional[float] = 60,
        vlm_caption_prompt: Optional[str] = "Provide a short description of this clip that belongs to a long video. 15 words maximum. Be confident of your description, DO NOT PROVIDE INCORRECT INFORMATION.",
        vlm_caption_max_new_tokens: Optional[int] = 64,
        vlm_caption_max_num_frames: Optional[int] = 32,
        vlm_captions_min_num_frames: Optional[int] = 4,
        vlm_captions_fps: Optional[int] = 0.5,
        vlm_vqa_max_frames_num: Optional[int] = 256,
        vlm_vqa_min_frames_num: Optional[int] = 1,
        vlm_vqa_fps: Optional[int] = 0.5,
        adjust_num_frames2window: Optional[bool] = True,
        ffmpeg_scene_threshold: Optional[float] = 0.4,
        ffmpeg_min_segments: Optional[int] = 16,
        ffmpeg_max_segments: Optional[int] = 64,
        ffmpeg_skip_frames: Optional[int] = 1,
        ffmpeg_width_res: Optional[int] = 128,
        uniform_maxwindow2caption: Optional[int] = 60,
        uniform_minwindow2caption: Optional[int] = 5,
        random_vqa_candidates: Optional[bool] = False,
        llm_return_logprobs: Optional[bool] = False,
        batch_size: Optional[int] = 1, 
        vlm_device: Optional[str] = "cuda",
        llm_reasoning_device: Optional[str] = "cuda",
        device: Optional[str] = "cuda",
        video_decode_backend: Optional[str] = "decord",
        save_captions: Optional[bool] = True,
        load_captions: Optional[bool] = True,
        save_vqa: Optional[bool] = True,
        load_vqa: Optional[bool] = True,
        save_summary: Optional[bool] = True,
        load_summary: Optional[bool] = True,
        save_llm_reasons: Optional[bool] = True,
        load_llm_reasons: Optional[bool] = True,
        captions_dir: Optional[str] = f"{os.path.dirname(os.getcwd())}/features/captions",
        vqa_dir: Optional[str] = f"{os.path.dirname(os.getcwd())}/features/vqa_vfinal_nozoom",
        llm_reasons_dir: Optional[str] = f"{os.path.dirname(os.getcwd())}/features/new_llm_reasonings",
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.vlm_name = vlm_name
        self.vlm_config = vlm_config
        self.llm_reasoning_name = llm_reasoning_name
        self.llm_reasoning_config = llm_reasoning_config
        self.vlm_device = vlm_device
        self.llm_reasoning_device = llm_reasoning_device
        self.flash_mode = flash_mode
        self.trust_in_confidence_mode = trust_in_confidence_mode
        self.video_decode_backend = video_decode_backend
        self.add_time_instruction = add_time_instruction
        self.window_span = window_span
        self.captions_dir = captions_dir
        self.vqa_dir = vqa_dir
        self.llm_reasons_dir = llm_reasons_dir
       
        self.load_captions = load_captions
        self.save_captions = save_captions
        self.load_vqa = load_vqa
        self.save_vqa = save_vqa
        self.load_summary = load_summary
        self.save_summary = save_summary
        self.load_llm_reasons = load_llm_reasons
        self.save_llm_reasons = save_llm_reasons

        self.max_returned_window_size = max_returned_window_size

        vlm_ModelClass = get_model(self.vlm_name)

        self.vlm_config = self.vlm_config[1:-1].replace(";", ",").replace("#", "=")
        self.vlm = vlm_ModelClass.create_from_arg_string(
           self.vlm_config,
            {
                "batch_size": batch_size,
                "device": self.vlm_device,
            },
        )
        
        if "3B" in self.vlm_config:
            self.vlm_name = self.vlm_name + "_3B"

        if "qwen2_5_vl" not in self.vlm_name:
            self.vlm_fps = None
        else:
            self.vlm_fps = self.vlm.fps

        self.vlm_vqa_max_num_frames = min(self.vlm.max_frames_num, vlm_vqa_max_frames_num)
        self.vlm_vqa_min_num_frames = vlm_vqa_min_frames_num
        self.vlm_vqa_fps = vlm_vqa_fps
        self.random_vqa_candidates = random_vqa_candidates
        self.adjust_num_frames2window = adjust_num_frames2window
        self.vlm_caption_max_num_frames = vlm_caption_max_num_frames
        self.vlm_caption_min_num_frames = vlm_captions_min_num_frames
        self.vlm_caption_fps = vlm_captions_fps

        # self.vlm_caption_prompt = "Provide a detailed caption of this clip."
        vlm_caption_prompt_used = "Provide a short description of this clip that belongs to a long video. 15 words maximum. Be confident of your description, DO NOT PROVIDE INCORRECT INFORMATION."
        vlm_caption_prompt_used = "Provide 5 unique actions or events that could distinguish this clip from the rest of the video. 15 words maximum."
        self.vlm_caption_prompt = vlm_caption_prompt
        self.vlm_key_caption_prompt = "".join([w[0] for w in self.vlm_caption_prompt.split(" ")])
        self.vlm_description_prompt = "Provide a general description of the video."
        self.vlm_caption_max_new_tokens = vlm_caption_max_new_tokens
 
        llm_reasoning_ModelClass = get_model(self.llm_reasoning_name)
        self.llm_reasoning_config = self.llm_reasoning_config[1:-1].replace(";", ",").replace("#", "=")
        self.llm_reasoning = llm_reasoning_ModelClass.create_from_arg_string(
           self.llm_reasoning_config,
            {
                "batch_size": batch_size,
                "device": self.llm_reasoning_device,
                "save_llm_reasons": self.save_llm_reasons,
                "load_llm_reasons": self.load_llm_reasons,
                "llm_reasons_dir": self.llm_reasons_dir,
            },
        )

        self.llm_reasoning_prompt1 = "You are a helpful video question answering assistant. The user provides some captions of the video with a question to be answered."
        self.llm_reasoning_prompt2 = "Identify and return the top5 scenes from the list above that are most likely to contain the visual information needed to answer the question."
        
        self.llm_resp_format = VideoReasoning
        self.llm_return_logprobs = llm_return_logprobs
        self.max_num_vqa_inf = max_num_vqa_inf
        self.frames_sampling_strategy = frames_sampling_strategy   
        self.conf_thres = None
        self.conf_thres_w_options = conf_thres_w_options
        self.conf_thres_wo_options = conf_thres_wo_options
        
        self.min_window_duration2vqa = min_window_duration
        self.min_window_duration2caption = min_window_duration
        self.min_window_duration2reason = 4*min_window_duration

        self.ffmpeg_scene_threshold = ffmpeg_scene_threshold
        self.ffmpeg_min_segments = ffmpeg_min_segments
        self.ffmpeg_max_segments = ffmpeg_max_segments
        self.all_key_timestamps = None
        self.ffmpeg_skip_frames = ffmpeg_skip_frames
        self.ffmpeg_width_res = ffmpeg_width_res
        self.ffmpeg_video_windows = {}

        self.uniform_maxwindow2caption = uniform_maxwindow2caption
        self.uniform_minwindow2caption = uniform_minwindow2caption

        self.curr_filename = None
        self.curr_captions_dict = None
        self.curr_llm_reasons = None
        self.curr_llm_filename = None

        os.makedirs(self.captions_dir, exist_ok=True)
        os.makedirs(self.vqa_dir, exist_ok=True)
        os.makedirs(self.llm_reasons_dir, exist_ok=True)
        
        if not hasattr(self.vlm, "inference"):
            raise AttributeError(f"Class '{self.vlm.__name__}' does not have a function 'inference'. This function should act as generate_until but for a unique question and a video. Refer to qwen-2_5_vl.py as example.")
        if not hasattr(self.llm_reasoning, "get_llm_response"):
            raise AttributeError(f"Class '{self.llm_reasoning.__name__}' does not have a function 'get_llm_response'. This function should act as generate_until but for a unique question. Refer to gpt4v.py as example.")
            
    def save_llm_reasons_func(self, filename, key, value):
        cache_llm_file = os.path.join(self.llm_reasons_dir, f"{filename}.pkl")
        self.curr_llm_reasons[key.encode()] = value.encode()
        with open(cache_llm_file, "wb") as f:
            pickle.dump(self.curr_llm_reasons, f)
    
    def load_llm_reasons_func(self, filename, key):
        if filename != self.curr_llm_filename or self.curr_llm_reasons is None:
            cache_llm_file = os.path.join(self.llm_reasons_dir, f"{filename}.pkl")
            if os.path.isfile(cache_llm_file):
                with open(cache_llm_file, "rb") as f:
                    self.curr_llm_reasons = pickle.load(f)
            else:
                self.curr_llm_reasons = {}
            self.curr_llm_filename = filename

        if key.encode() in self.curr_llm_reasons:
            out = self.curr_llm_reasons[key.encode()].decode()
            if out == "GPT Error":
                print("The saved LLM reasoning is an error.")
                return None
            else:
                return out
        else:
            return None

    def load_llm_reasons_func(self, filename):
        if filename == self.curr_filename and self.curr_llm_reasons is not None:
            return self.curr_llm_reasons
        else:
            os.makedirs(self.llm_reasons_dir, exist_ok=True)
            video_llm_reasons_path = os.path.join(self.dir_save_llm_reasonings, f"{filename}.json")
            # Load existing data if the file exists
            if os.path.exists(video_llm_reasons_path):
                with open(video_llm_reasons_path, 'r') as f:
                    video_llm_reasons = json.load(f)
            else:
                video_llm_reasons = {}
            return video_llm_reasons
    
    def save_summary_func(self, filename, summary):
        captions_dir = os.path.join(self.captions_dir, filename)
        os.makedirs(captions_dir, exist_ok=True)

        if self.frames_sampling_strategy == "ffmpeg_keyframes":
            captions_filename = f'{self.vlm_name}_{self.frames_sampling_strategy}_thres{self.ffmpeg_scene_threshold}_maxseg{self.ffmpeg_max_segments}_minseg{self.ffmpeg_min_segments}_toks{self.vlm_caption_max_new_tokens}_{self.vlm_key_caption_prompt}.json'
        elif self.frames_sampling_strategy == "uniform":
            captions_filename = f'{self.vlm_name}_{self.frames_sampling_strategy}_window{self.window_span}s_fmin{self.vlm_caption_min_num_frames}_fmax{self.vlm_caption_max_num_frames}_toks{self.vlm_caption_max_new_tokens}_{self.vlm_key_caption_prompt}.json'
        else:
            raise NotImplementedError
        
        summary_filename = f'summary_{captions_filename}'
        summary_path = os.path.join(captions_dir, summary_filename)
        with open(summary_path, "w") as f:
            json.dump(summary, f)

    def load_summary_func(self, filename):
        captions_dir = os.path.join(self.captions_dir, filename)
        if self.frames_sampling_strategy == "ffmpeg_keyframes":
            captions_filename = f'{self.vlm_name}_{self.frames_sampling_strategy}_thres{self.ffmpeg_scene_threshold}_maxseg{self.ffmpeg_max_segments}_minseg{self.ffmpeg_min_segments}_toks{self.vlm_caption_max_new_tokens}_{self.vlm_key_caption_prompt}.json'
        elif self.frames_sampling_strategy == "uniform":
            captions_filename = f'{self.vlm_name}_{self.frames_sampling_strategy}_window{self.window_span}s_fmin{self.vlm_caption_min_num_frames}_fmax{self.vlm_caption_max_num_frames}_toks{self.vlm_caption_max_new_tokens}_{self.vlm_key_caption_prompt}.json'
        else:
            raise NotImplementedError

        summary_filename = f'summary_{captions_filename}'
        summary_path = os.path.join(captions_dir, summary_filename)

        if os.path.isfile(summary_path):
            with open(summary_path, "r") as f:
                summary_dict = json.load(f)
        else:
            summary_dict = None
        return summary_dict
    
    def load_captions_func(self, filename):
        if filename == self.curr_filename and self.curr_captions_dict is not None:
            return self.curr_captions_dict
        else:
            captions_dir = os.path.join(self.captions_dir, filename)
            os.makedirs(captions_dir, exist_ok=True)

            if self.frames_sampling_strategy == "ffmpeg_keyframes":
                captions_filename = f'{self.vlm_name}_{self.frames_sampling_strategy}_thres{self.ffmpeg_scene_threshold}_maxseg{self.ffmpeg_max_segments}_minseg{self.ffmpeg_min_segments}_toks{self.vlm_caption_max_new_tokens}_{self.vlm_key_caption_prompt}.json'
            elif self.frames_sampling_strategy == "uniform":
                captions_filename = f'{self.vlm_name}_{self.frames_sampling_strategy}_window{self.window_span}s_fmin{self.vlm_caption_min_num_frames}_fmax{self.vlm_caption_max_num_frames}_toks{self.vlm_caption_max_new_tokens}_{self.vlm_key_caption_prompt}.json'
            else:
                raise NotImplementedError

            captions_filename = os.path.join(captions_dir, captions_filename)
            if os.path.isfile(captions_filename):
                with open(captions_filename, "r") as f:
                    captions_dict = json.load(f)
            else:
                captions_dict = {"captions": {}, "num_inferences": 0, "total_time": 0}
            
            return captions_dict

    def save_captions_func(self, filename, captions):
        captions_dir = os.path.join(self.captions_dir, filename)

        if self.frames_sampling_strategy == "ffmpeg_keyframes":
            captions_filename = f'{self.vlm_name}_{self.frames_sampling_strategy}_thres{self.ffmpeg_scene_threshold}_maxseg{self.ffmpeg_max_segments}_minseg{self.ffmpeg_min_segments}_toks{self.vlm_caption_max_new_tokens}_{self.vlm_key_caption_prompt}.json'
        elif self.frames_sampling_strategy == "uniform":
            captions_filename = f'{self.vlm_name}_{self.frames_sampling_strategy}_window{self.window_span}s_fmin{self.vlm_caption_min_num_frames}_fmax{self.vlm_caption_max_num_frames}_toks{self.vlm_caption_max_new_tokens}_{self.vlm_key_caption_prompt}.json'
        else:
            raise NotImplementedError
        captions_filename = os.path.join(captions_dir, captions_filename)
        with open(captions_filename, "w") as f:
            json.dump(captions, f)
        
        self.curr_captions_dict = captions
        self.curr_filename = filename

    def load_vqa_func(self, filename, question_id, gpt_eval):
        vqa_dir = os.path.join(self.vqa_dir, filename)
        os.makedirs(vqa_dir, exist_ok=True)

        vqa_filename = self.vlm_name + str(question_id)
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

        vqa_filename = self.vlm_name + str(question_id)
        if gpt_eval:
            vqa_filename = f"{vqa_filename}_gpteval"
        vqa_filename = os.path.join(vqa_dir, vqa_filename+".json")

        vqa_dict[key_sampling_frames_info] = output_dict

        with open(vqa_filename, "w") as f:
            json.dump(vqa_dict, f, default=self.numpy_converter)


    def load_image(self, image_path):
        frame_files = [os.path.join(image_path, f) for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]
        frame_files.sort()  # Ensure the frames are sorted if they are named sequentially

        # TODO: Hard CODE: Determine the indices for uniformly sampling 10 frames
        num_frames_to_sample = 10

        total_frames = len(frame_files)

        sampled_indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)

        # Read and store the sampled frames
        video = []
        for idx in sampled_indices:
            frame_path = frame_files[idx]
            try:
                with Image.open(frame_path) as img:
                    # Convert the PIL image to a numpy array if needed
                    # frame = np.array(img.convert('RGB'))
                    frame = img.convert("RGB")
                    video.append(frame)
            except IOError:
                print(f"Failed to read frame at path: {frame_path}")
        return video

    def split_frame_into_tiles(self, frame, num_tiles):
        """
        Splits a frame into a specified number of tiles based on the frame's shape.

        Args:
            frame (numpy.ndarray): The frame to split.
            num_tiles (int): The total number of tiles to create.

        Returns:
            list: A list of tiles, each as a numpy.ndarray.
        """
        height, width, _ = frame.shape

        # Determine the number of rows and columns based on the aspect ratio and num_tiles
        aspect_ratio = width / height
        cols = int((num_tiles * aspect_ratio) ** 0.5)
        rows = num_tiles // cols

        # Adjust rows and columns if needed to match the exact number of tiles
        while rows * cols < num_tiles:
            cols += 1
            if rows * cols > num_tiles:
                rows += 1

        tile_height = height // rows
        tile_width = width // cols

        tiles = []
        for i in range(rows):
            for j in range(cols):
                if len(tiles) >= num_tiles:
                    break
                # Calculate the boundaries for each tile
                start_row = i * tile_height
                end_row = (i + 1) * tile_height if i != rows - 1 else height
                start_col = j * tile_width
                end_col = (j + 1) * tile_width if j != cols - 1 else width

                # Extract the tile
                tile = frame[start_row:end_row, start_col:end_col]
                tiles.append(tile)

        return tiles

    def resize_frame(self, frame, size=(384, 384)):
        """
        Resizes a frame to the given size using PIL.Image.
        """
        image = Image.fromarray(frame)  # Convert NumPy array to PIL Image
        resized_image = image.resize(size, Image.BICUBIC)  # Resize to target size
        return np.array(resized_image)  # Convert back to NumPy array
    

    def load_video(self, video_info, max_num_frames, window_time=None, num_tiles=None):

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


        if num_tiles is not None:
            # For each frame, split into tiles and include the resized original frame and tiles
            all_frames_with_tiles = []
            for frame in frames:
                # Resize the original frame
                resized_frame = self.resize_frame(frame)
                # Split into tiles
                tiles = self.split_frame_into_tiles(frame, num_tiles=num_tiles)
                # Resize each tile
                resized_tiles = [self.resize_frame(tile) for tile in tiles]
                # Append the resized original frame and its resized tiles
                all_frames_with_tiles.append(resized_frame)
                all_frames_with_tiles.extend(resized_tiles)

            frames = np.array(all_frames_with_tiles)  # Convert to a NumPy array

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

        filename = video_file.split("/")[-1].split(".")[0]
        total_frames = len(vr)
        video_time = total_frames / fps
        video_info = {"vr": vr, "fps": fps, "total_frames": total_frames, "total_duration": video_time, "path": video_file, "filename": filename}
        return video_info

    # Function to extract cliptimeinterval sublists and compute probabilities
    def extract_cliptimeintervals(self, logprobs_content):
        cliptimeintervals = []
        current_interval = None

        for token in logprobs_content:
            if token["token"] == "interval":  # Start of a cliptimeinterval
                current_interval = []
            if current_interval is not None:
                current_interval.append(token)
            if token["token"] in ["}}", "}},", "}}."]:  # End of a cliptimeinterval
                if current_interval is not None:
                    cliptimeintervals.append(current_interval)
                current_interval = None

        return cliptimeintervals
    
    def extract_choices_from_question(self, question):
        choices = ["A", "B", "C", "D"]
        if "(E)" in question or "E." in question:
            choices.append("E")
        if "(F)" in question or "F." in question:
            choices.append("F")
        if "(G)" in question or "G." in question:
            choices.append("G")
        if "(H)" in question or "H." in question:
            choices.append("H")
        return choices
        
    def get_confidence_from_outputs(self, question, outputs, choices_in_question=False):
        num_tokens = outputs["num_tokens"]
        tokens = outputs["tokens"]

        if choices_in_question:
            choices = self.extract_choices_from_question(question)
            options_token = [tok for tok in tokens.values() if tok["token"] in choices]
            if len(options_token) != 1 or len(outputs["tokens"]) > 5:
                conf = 0
            else:
                options_token = options_token[0]
                conf = options_token["top5_probs"][0]
        else:
            response_probs = [float(tok["top5_probs"][0]) for tok in tokens.values()]
            conf = math.prod(response_probs) ** (1/len(response_probs))
        
        if isinstance(conf, np.ndarray):
            return round(conf.item(), 3)
        else:
            return round(conf, 3)
    
    def get_numframes_and_numtiles_from_window(self, window_length, max_frames_num):
        if window_length >= 60:
            num_frames = max_frames_num
            num_tiles = None
        elif window_length >= 10:
            num_frames = 12
            num_tiles = 2
        else:
            num_frames = 6
            num_tiles = 6
        num_frames = max_frames_num
        num_tiles = None
        return num_frames, num_tiles
    
    def get_numframes_from_window(self, window_length, max_frames_num, min_frames_num, max_fps, min_fps, mode):
        if mode == "captioning":
            if max_fps is not None and window_length * max_fps <= min_frames_num:
                num_frames = int(window_length * max_fps)
            else:
                num_frames = min(max_frames_num, max(min_frames_num, int(min_fps*window_length)))
        
        elif mode == "vqa":
            fps_dif = max_fps - min_fps
            fps_33 = min_fps + fps_dif * 0.3333
            fps_66 = min_fps + fps_dif * 0.6666

            if window_length <= 5:
                new_fps = max_fps
            elif window_length <= 30:
                new_fps = max_fps - (window_length - 5) * (fps_66 / 25)  # Linear from 2 to 1 fps
            elif window_length <= 60:
                new_fps = fps_33 - (window_length - 30) * (min_fps / 30)  # Linear from 1 to 0.5 fps
            else:
                new_fps = min_fps

            num_frames = min(max_frames_num, max(round(window_length * new_fps), min_frames_num))
        return num_frames

        
    def generate_final_answer_or_keep_exploring(self, video_info, question, responses, summary, conf_thres, gpt_eval=False):
        answer_format = {"final_answer": "xxx", "window_start_seconds": "xxx", "window_end_seconds": "xxx", "confidence": "xxx", "explanation": "xxx"}
        prompt = f"""
            Given a {video_info['total_duration']}-second video with this summary:
            ```
            {summary}. 
            ```
            And the following question:
            ``` 
            {question}
            ``` 
            A Visual Language Model generated these local responses from different time windows, along with confidence scores:
            ```
            {responses}
            ```
            Do **not** simply return the highest-confidence response. Instead, determine the most logical response by considering:
                - Temporal windows used for responses.
                - Captions.
                - Confidence scores.

            Return the final answer in JSON format: {answer_format}.  
            Please return Nan as the final answer in any of the next cases:
            - The best **response has confidence lower than {conf_thres}**.
            - The best response does **not** answer the question.
            - A better answer could be found by exploring more temporal windows.
            """
        
        system_prompt = "You are a helpful assistant designed to output JSON."
        response = self.llm_reasoning.get_llm_response(system_prompt, prompt, video_info["filename"], json_format=True)
        print("Final response:", response)
        try:
            if type(response["response"]["final_answer"]) is str and response["response"]["final_answer"].lower().strip() != "nan":
                if "explanation" in response["response"]:
                    explanation = response["response"]["explanation"]
                else:
                    explanation = "An accurate response has been found."
                return response, explanation
            else:
                if "explanation" in response["response"]:
                    explanation = response["response"]["explanation"]
                else:
                    explanation = "An accurate response has not been found."
                response["response"] = None
                return response, explanation
        except Exception as e:
            print(f"Error parsing response: {response}\n{e}")
            response["response"] = None
            explanation = "The LLM did not return a valid response."
            return response, explanation
        
    def generate_final_answer(self, video_info, question, responses, summary, conf_thres, gpt_eval=False):
        answer_format = {"final_answer": "xxx", "window_start_seconds": "xxx", "window_end_seconds": "xxx", "confidence": "xxx", "explanation": "xxx"}
        prompt = f"""
            Given a {video_info['total_duration']}-second video with this summary:
            ```
            {summary}. 
            ```
            And the following question:
            ``` 
            {question}
            ``` 
            A Visual Language Model generated these local responses from different time windows, along with confidence scores:
            ```
            {responses}
            ```
            Do **not** simply return the highest-confidence response. Instead, determine the most logical response by considering:
                - Temporal windows used for responses.
                - Captions.
                - Confidence scores.

            Return the final answer in JSON format: {answer_format}.  
            """
        
        system_prompt = "You are a helpful assistant designed to output JSON."
        response = self.llm_reasoning.get_llm_response(system_prompt, prompt, video_info["filename"], json_format=True)
        try:
            if type(response["response"]) is dict:
                if type(response["response"]["final_answer"]) is str:
                    return response
                else:
                    response["response"]["final_answer"] = "Nan"
            else:
                response["response"] = {"final_answer": "Nan", "window_start_seconds": 0, "window_end_seconds": video_info["total_duration"], "confidence": 0.0, "explanation": "Nan"}
        except:
            print("Error while generating final answer.")
            response["response"] = {"final_answer": "Nan", "window_start_seconds": 0, "window_end_seconds": video_info["total_duration"], "confidence": 0.0, "explanation": "Nan"}
        return response

    def generate_candidate_windows_to_explore(self, video_info, question, responses, explanation, gpt_eval=False):
        answer_format = {
            "clip1:": {"start_s": "xxx", "end_s": "xxx", "explanation": "xxx"},
            "clip2:": {"start_s": "xxx", "end_s": "xxx", "explanation": "xxx"},
            "clip3:": {"start_s": "xxx", "end_s": "xxx", "explanation": "xxx"},
        }
        prompt = f"""
            Given a {video_info['total_duration']}-second video and the following question whose answer we must search in the video:
            ```
            {question}
            ```
            A Visual Language Model provided these local responses from different time windows:

            ```
            {responses}
            ```
            However, none were considered final due to the following reason:
            ```
            {explanation}
            ```
            Prioritize captions over responses and carefully identify which of these candidate windows (not necessarily 3) worth to continue exploring more deeply.
            Return a list in JSON format: {answer_format}.  
            If no candidate window is close to answer the question, return an empty list.
        """
        
        system_prompt = "You are a helpful assistant designed to output JSON."
        response = self.llm_reasoning.get_llm_response(system_prompt, prompt, video_info["filename"], json_format=True)
        window_candidates, window_explanations = [], []
        try:
            if type(response["response"]) is str and response["response"].lower().strip() == "nan":
                print("No candidate windows to explore found.")
            elif type(response["response"]) is list and len(response["response"]) == 0:
                print("No candidate windows to explore found.")
            else:
                for clip, clip_info in response['response'].items():
                    try:
                        if type(clip_info["start_s"]) is str:
                            window_s = float(clip_info["start_s"].replace("s", ""))
                        else:
                            window_s = float(clip_info["start_s"])
                        if type(clip_info["end_s"]) is str:
                            window_e = float(clip_info["end_s"].replace("s", ""))
                        else:
                            window_e = float(clip_info["end_s"])

                        window_candidates.append([window_s, window_e])
                        window_explanations.append(clip_info["explanation"])
                    except Exception as e:
                        print(f"Error parsing clip info: {clip_info}\n{e}")
                        if type(clip_info) is dict:
                            print("GPT returned a dict inside the dict.")
                            try:
                                for clip, clip_info in clip_info.items():
                                    if type(clip_info["start_s"]) is str:
                                        window_s = float(clip_info["start_s"].replace("s", ""))
                                    else:
                                        window_s = float(clip_info["start_s"])
                                    if type(clip_info["end_s"]) is str:
                                        window_e = float(clip_info["end_s"].replace("s", ""))
                                    else:
                                        window_e = float(clip_info["end_s"])

                                    window_candidates.append([window_s, window_e])
                                    window_explanations.append(clip_info["explanation"])
                            except Exception as e:
                                print(f"Error parsing clip info: {clip_info}\n{e}")
                            print(f"Clip info: {clip_info['clip']}")
        except Exception as e:
            print(f"Error parsing response: {response}\n{e}")
            print("No candidate windows to explore found.")
        return window_candidates, window_explanations, response['tokens_usage']

    def generate_summary_from_captions(self, video_info, captions, gpt_eval=False):
        answer_format = {"summary": "xxx"}
        prompt = f"""
            Given a {video_info['total_duration']}-second video with the following captions from selected clips:
            ```
            {captions}
            ```
            Generate a short but detailed summary outlining the evolution of the main events. Return the summary in JSON format: {answer_format}.
            """
        
        system_prompt = "You are a helpful assistant designed to output JSON."
        response = self.llm_reasoning.get_llm_response(system_prompt, prompt, video_info["filename"], json_format=True)
        return response



    def generate_candidate_windows_to_vqa(self, video_info, question, captions, summary, gpt_eval=False):
        answer_format = {
            "clip1": {"start_s": "xxx", "end_s": "xxx", "explanation": "xxx"},
            "clip2": {"start_s": "xxx", "end_s": "xxx", "explanation": "xxx"},
            "clip3": {"start_s": "xxx", "end_s": "xxx", "explanation": "xxx"},
            "clip4": {"start_s": "xxx", "end_s": "xxx", "explanation": "xxx"},
            "clip5": {"start_s": "xxx", "end_s": "xxx", "explanation": "xxx"},
        }

        prompt = f"""
            Given a {video_info['total_duration']}-second video with the following summary:
            ```
            {summary}. 
            ```
            and captions from specific clips:
            ```
            {captions}
            ```
            Find the most relevant clips (not necessarily 5) that likely contain the answer to this question:
            ``` 
            {question}
            ``` 
            Consider temporal context from both the question and captions. 
            Return the result in JSON format: {answer_format}. Sort them by likelihood of containing the answer.
            """
        
        system_prompt = "You are a helpful assistant designed to output JSON."
        response = self.llm_reasoning.get_llm_response(system_prompt, prompt, video_info["filename"], json_format=True)
        vqa_candidates, vqa_explanations = [], []
        try:
            for clip, clip_info in response['response'].items():
                try:
                    if type(clip_info["start_s"]) is str:
                        window_s = float(clip_info["start_s"].replace("s", ""))
                    else:
                        window_s = float(clip_info["start_s"])
                    if type(clip_info["end_s"]) is str:
                        window_e = float(clip_info["end_s"].replace("s", ""))
                    else:
                        window_e = float(clip_info["end_s"])

                    vqa_candidates.append([window_s, window_e])
                    vqa_explanations.append(clip_info["explanation"])
                except Exception as e:
                    print(f"Error parsing clip info: {clip_info}\n{e}")
                    if type(clip_info) is dict:
                        print("GPT returned a dict inside the dict.")
                        try:
                            for clip, clip_info in clip_info.items():
                                if type(clip_info["start_s"]) is str:
                                    window_s = float(clip_info["start_s"].replace("s", ""))
                                else:
                                    window_s = float(clip_info["start_s"])
                                if type(clip_info["end_s"]) is str:
                                    window_e = float(clip_info["end_s"].replace("s", ""))
                                else:
                                    window_e = float(clip_info["end_s"])

                                vqa_candidates.append([window_s, window_e])
                                vqa_explanations.append(clip_info["explanation"])
                        except Exception as e:
                            print(f"Error parsing clip info: {clip_info}\n{e}")
                            print(f"Clip info: {clip_info['clip']}")
        except Exception as e:
            print(f"Error parsing response: {response}\n{e}")
            print("Failed in video:", video_info["filename"])
        return vqa_candidates, vqa_explanations, response['tokens_usage']



    def filter_timestamps_indices(self, timestamps, min_interval=5):
        """
        Removes timestamps that are closer than `min_interval` seconds to each other
        and returns the indices of the kept timestamps.
        
        Args:
            timestamps (list): List of timestamps in seconds (floats or ints).
            min_interval (int): Minimum allowed interval between timestamps in seconds.
        
        Returns:
            list: Indices of the filtered timestamps.
        """
        # Step 1: Sort timestamps and keep track of original indices
        indexed_timestamps = sorted(enumerate(timestamps), key=lambda x: x[1])
        
        filtered_indices = []
        last_kept = None  # Keep track of the last added timestamp

        # Step 2: Iterate through sorted timestamps
        for idx, ts in indexed_timestamps:
            if last_kept is None or ts - last_kept >= min_interval:
                filtered_indices.append(idx)  # Store original index
                last_kept = ts  # Update last added timestamp

        return sorted(filtered_indices)  # Return indices in original order



    def extract_keyframes_ffmpeg(self, video_path, video_info, start_time, end_time, scene_threshold, skip_frames, width_res):
        scale_filter = f"scale={width_res}:-1"
        command = [
                "ffmpeg", "-ss", f"{round(start_time, 2)}", "-to", f"{round(end_time,2)}", "-i", video_path,
                "-vf", f"{scale_filter},select='not(mod(n,{skip_frames}))*gt(scene,{scene_threshold})',metadata=print",
                "-vsync", "vfr",
                "-f", "null", "-"
            ]
        try:
            process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stderr_output = process.stderr
        except Exception as e:
            print(f"Error running FFmpeg: {e}")
            return None, None, None

        # Extract frame indices and timestamps from FFmpeg output
        frames_info = []
        for line in stderr_output.split("\n"):
            if "pts_time:" in line and "frame:" in line:
                try:
                    # Parse filtered frame index (`n:`)
                    filtered_frame_index = int(line.split("frame:")[1].split()[0])
                    # Parse pts_time
                    pts_time = start_time + float(line.split("pts_time:")[1].split()[0])
                    # Compute global frame index
                    global_frame_index = int(pts_time * video_info["fps"])

                except (IndexError, ValueError) as e:
                    print(f"Error parsing line: {line}\n{e}")

            elif "score" in line:
                scene_score = round(float(line.split("score=")[1]), 3)

                # Append parsed data
                frames_info.append({
                    "filtered_index": filtered_frame_index,  # Index in filtered sequence
                    "global_index": global_frame_index,  # Global index in the video
                    "timestamp": pts_time,  # Timestamp in video timeline
                    "scene_score": scene_score
                })
                # print(line)
        
        return frames_info

    def get_ffmpeg_scene_threshold(self, video_path):
        # Hardcoded scene threshold for specific video categories such as egocentric videos 
        if "Walking_Tours" in video_path:
            return 0.1
        else:
            return 0.3
        
    def extract_windows_ffmpeg(self, video_path, video_info, start, end):
        frames_info = []
        # threshold = self.ffmpeg_scene_threshold
        ffmpeg_max_segments = max(round(video_info['total_duration'] / 60), 5)
        if video_path != self.curr_video_path or self.all_key_timestamps is None:
            ffmpeg_scene_threshold = self.get_ffmpeg_scene_threshold(video_path)
            frames_info = self.extract_keyframes_ffmpeg(video_path, video_info, start, end, ffmpeg_scene_threshold, self.ffmpeg_skip_frames, self.ffmpeg_width_res)
            window_duration = end - start
            start_frame = int(start * video_info["fps"])
            if len(frames_info) < self.ffmpeg_min_segments:
                print(f"Video '{video_path}' has less than {self.ffmpeg_min_segments} keyframes ({len(frames_info)}) within temporal window {[start, end]}. Threshold is too low. We sample uniformly.")
                frames_info = [{
                        "filtered_index": k,  # Index in filtered sequence
                        "global_index": start_frame + int(window_duration*video_info["fps"]*k/self.ffmpeg_min_segments),  # Global index in the video
                        "timestamp": start + (window_duration*k/self.ffmpeg_min_segments),
                        "scene_score": 0.1
                    } for k in range(1, self.ffmpeg_min_segments)]
            else:
                print(f"Video '{video_path}' has {len(frames_info)} keyframes. Threshold is good.")   

            # Extract timestamps and indices
            frame_indices = [frame["global_index"] for frame in frames_info]
            timestamps = [frame["timestamp"] for frame in frames_info]
            scores = [frame["scene_score"] for frame in frames_info]

            idxs_timestamps = self.filter_timestamps_indices(timestamps, min_interval=self.min_window_duration2caption)
            timestamps = [timestamps[i] for i in idxs_timestamps]
            scores = [scores[i] for i in idxs_timestamps]

            # Return the min_segments timestamps with the highest scores
            best_scores_idxs = np.argsort(scores)[::-1]
            self.all_key_timestamps = [round(timestamps[i]) for i in best_scores_idxs]

        key_timestamps = [key_time for key_time in self.all_key_timestamps if key_time >= start and key_time <= end]
        key_timestamps = key_timestamps[:ffmpeg_max_segments]
        key_timestamps.extend([start, end])

        key_timestamps = list(dict.fromkeys(sorted(key_timestamps)))
        windows = [[key_timestamps[i], key_timestamps[i+1]] for i in range(len(key_timestamps) - 1)]
        return windows

    def add_caption_to_outputs(self, video_info, outputs, window):
        if window == [0, video_info["total_duration"]]:
            if "summary" in video_info:
                caption = video_info["summary"]
            else:
                caption = f"Summary: {video_info['total_duration']}-second video"
        else:
            file_captions_dict = self.load_captions_func(video_info["filename"])
            all_captions_dict = file_captions_dict["captions"]
            if f"[{int(window[0])}s-{int(window[1])}s]" in all_captions_dict:
                caption = all_captions_dict[f"[{int(window[0])}s-{int(window[1])}s]"]
            elif f"[{window[0]}s-{window[1]}s]" in all_captions_dict:
                caption = all_captions_dict[f"[{window[0]}s-{window[1]}s]"]
            else:
                print("No caption found for window: ", window, "in video: ", video_info["filename"], "We add the caption of the minute interval that it belongs to.")
                start_w = int(window[0]//60)*60
                minute_interval = [start_w, start_w+60]
                if f"[{minute_interval[0]}s-{minute_interval[1]}s]" in all_captions_dict:
                    caption = all_captions_dict[f"[{minute_interval[0]}s-{minute_interval[1]}s]"]
                elif f"[{float(minute_interval[0])}s-{float(minute_interval[1])}s]" in all_captions_dict:
                    caption = all_captions_dict[f"[{float(minute_interval[0])}s-{float(minute_interval[1])}s]"]
                else:
                    print("No caption found for window: ", minute_interval, "in video: ", video_info["filename"], "Adding empty caption.")
                    caption = ""
        outputs["caption"] = caption
        return outputs

    def compute_smallwindow_span2caption(self, window_duration):
        if window_duration > 2*self.uniform_maxwindow2caption:
            smallwindow_span = self.uniform_maxwindow2caption
        elif window_duration > self.uniform_maxwindow2caption//3:
            smallwindow_span = 4*self.uniform_minwindow2caption
        else:
            smallwindow_span = self.uniform_minwindow2caption 
        return smallwindow_span

    def generate_candidates4captioning(self, video_path, video_info, window_start, window_end, explored_windows, frames_sampling_strategy, times_and_inferences):
        window_duration = window_end - window_start
        t_generate_candidates = time.time()
        if frames_sampling_strategy == "ffmpeg_keyframes":
            window = f"{[window_start, window_end]}"
            if video_path not in self.ffmpeg_video_windows:
                smallwindows = self.extract_windows_ffmpeg(video_path, video_info, window_start, window_end)
                self.ffmpeg_video_windows[video_path] = {window: smallwindows}
            elif window not in self.ffmpeg_video_windows[video_path]:
                smallwindows = self.extract_windows_ffmpeg(video_path, video_info, window_start, window_end)
                self.ffmpeg_video_windows[video_path][window] = smallwindows
            else:
                smallwindows = self.ffmpeg_video_windows[video_path][window]

        elif frames_sampling_strategy == "resnet_keyframes":
            print("TODO: Implement resnet_keyframes")
            raise NotImplementedError

        else:
            if window_end == video_info["total_duration"] and int(window_start) == 0: window_start = 0
            smallwindow_span = self.compute_smallwindow_span2caption(window_duration)
            num_smallwindows = int(window_duration / smallwindow_span)
            smallwindows = [[window_start + k*smallwindow_span, window_start + (k+1)*smallwindow_span] for k in range(num_smallwindows)]
        
        smallwindows = [w for w in smallwindows if w not in explored_windows]
        times_and_inferences["windows_candidates_generation_time"] += time.time() - t_generate_candidates
        return smallwindows
    
    def generate_captions(self, video_path, video_info, window_start, window_end, smallwindows, times_and_inferences, gen_kwargs, return_dict=False):
        window_duration = window_end - window_start
        if window_duration == video_info['total_duration']:
            generic_captions = True
        else:
            generic_captions = False

        filename = video_path.split("/")[-1].split(".")[0]
        gen_kwargs["return_dict_in_generate"], gen_kwargs["output_scores"], gen_kwargs["output_logits"] = False, False, False
        gen_kwargs["max_new_tokens"] = self.vlm_caption_max_new_tokens

        
        if self.load_captions:
            # all_captions_dict = self.captions[filename]["captions"]
            file_captions_dict = self.load_captions_func(filename)
            all_captions_dict = file_captions_dict["captions"]
            smallwindows2caption = [w for w in smallwindows if f"[{w[0]}s-{w[1]}s]" not in all_captions_dict]
        else:
            file_captions_dict = {"captions": {}, "num_inferences": 0, "total_time": 0}
            all_captions_dict = file_captions_dict["captions"]
            smallwindows2caption = smallwindows

        num_inferences = len(smallwindows2caption)
        t_caption_init = time.time()
        for smallwindow in smallwindows2caption:
            print(f"Generating caption for window: {smallwindow} in video: {video_path}")
            smallwindow_s, smallwindow_e = smallwindow[0], smallwindow[1]
            
            if "qwen2_5_vl" not in self.vlm_name:
                sampling_frames_info = {"num_frames": self.vlm_caption_max_num_frames, "num_tiles": None, "window": smallwindow}
            else:
                sampling_frames_info = {"window": smallwindow} 
                if self.adjust_num_frames2window:
                    num_frames = self.get_numframes_from_window(smallwindow_e - smallwindow_s, self.vlm_caption_max_num_frames, self.vlm_caption_min_num_frames, self.vlm_fps, self.vlm_caption_fps, mode="captioning")
                    sampling_frames_info["num_frames"] = num_frames
                    print(f"Sampling {num_frames} frames in window {smallwindow} for Captioning") 

            if self.add_time_instruction:
                caption_prompt = self.add_time_instruction_to_contexts(self.vlm_caption_prompt, video_info, sampling_frames_info)
            else:
                caption_prompt = self.vlm_caption_prompt

            caption = self.vlm.inference(video_info, sampling_frames_info, caption_prompt, gen_kwargs)
            if caption is None:
                smallwindows.remove(smallwindow)
                print("No caption generated for the window: ", smallwindow, "in the video: ", video_path, "of total duration: ", video_info["total_duration"])
                continue
            all_captions_dict[f"[{smallwindow_s}s-{smallwindow_e}s]"] = caption



        caption_time = time.time() - t_caption_init
        file_captions_dict["captions"] = all_captions_dict
        file_captions_dict["total_time"] += caption_time
        file_captions_dict["num_inferences"] += num_inferences
        avg_caption_time = file_captions_dict["total_time"] / file_captions_dict["num_inferences"]

        if self.save_captions:
            self.save_captions_func(filename, file_captions_dict)

        if generic_captions:
            times_and_inferences["num_caption_video_inferences"] += len(smallwindows2caption)
            times_and_inferences["caption_video_time"] += caption_time
        else:
            times_and_inferences["num_caption_question_inferences"] += len(smallwindows)
            times_and_inferences["caption_question_time"] += avg_caption_time * len(smallwindows) 

        captions_window = [f" [{w[0]}s-{w[1]}s]: [" + all_captions_dict[f"[{w[0]}s-{w[1]}s]"].replace("1.", "").replace("2.", "").replace("3.", "").replace("4.", "").replace("5.", "").replace("\n", "") + "] " for w in smallwindows]
        if return_dict:
            captions_window_dict = {f"[{w[0]}s-{w[1]}s]": all_captions_dict[f"[{w[0]}s-{w[1]}s]"] for w in smallwindows}
            return captions_window_dict
        else:
            return captions_window


    def generate_candidates_with_llm_reasoning(self, video_path, video_info, context, window_reason, explored_windows, times_and_inferences, gen_kwargs):
        window_start, window_end = window_reason[0], window_reason[1]
        print("Generating smallwindows for captioning...")
        window_duration = window_end - window_start
        if window_duration == video_info['total_duration']:
            smallwindows = self.generate_candidates4captioning(video_path, video_info, window_start, window_end, explored_windows, self.frames_sampling_strategy, times_and_inferences)
            print("Generating captions for LLM Reasoning in the new smallwindows: ", smallwindows)
            captions_dict = self.generate_captions(video_path, video_info, window_start, window_end, smallwindows, times_and_inferences, gen_kwargs, return_dict=True)
            t_cand = time.time()
            if self.random_vqa_candidates:
                vqa_candidates = random.sample(smallwindows, min(5, len(smallwindows)))
                vqa_explanations = ["Randomly selected candidate."]*len(vqa_candidates)
            else:
                vqa_candidates, vqa_explanations, tokens_usage = self.generate_candidate_windows_to_vqa(video_info, context, captions_dict, video_info['summary'])
                times_and_inferences["llm_tokens_usage"] += tokens_usage
                times_and_inferences["num_llm_inferences"] += 1
                times_and_inferences["llm_reasoning_time"] += time.time() - t_cand
            seen = set()
            unique_new_candidates, unique_new_explanations  = [], []
            for idx, candidate in enumerate(vqa_candidates):
                try:
                    good_candidate, candidate = self.check_candidate_window(candidate, video_info)
                except Exception as e:
                    print(f"Error parsing scene: {candidate}. Skipping this candidate.")
                    print(e)
                    continue # Convert to tuple for hashable type
                if good_candidate:
                    t = tuple(candidate) 
                    if t not in seen:
                        seen.add(t)  # Add the tuple to the seen set
                        unique_new_candidates.append(candidate)  # Append the original list version
                        unique_new_explanations.append(vqa_explanations[idx])
            print("Candidates for VQA: ", unique_new_candidates)
            if len(unique_new_candidates) == 0 and len(smallwindows) > 5:
                print("No valid candidates found for VQA. Sampling random candidates.")
                unique_new_candidates = random.sample(smallwindows, 3)
                unique_new_explanations.extend(["Randomly selected candidates as no valid candidates were found."]*3)
            return unique_new_candidates, unique_new_explanations
        else:
            smallwindows = self.generate_candidates4captioning(video_path, video_info, window_start, window_end, explored_windows, "uniform", times_and_inferences)
            explanations = [f"The window {smallwindow} lies inside a potential segment {window_reason} to explore." for smallwindow in smallwindows]
            print("Candidates for VQA: ", smallwindows)
            return smallwindows, explanations
        
    def check_candidate_window(self, window_candidate, video_info):
        window_candidate_duration = window_candidate[1] - window_candidate[0]
        if window_candidate_duration == video_info["total_duration"]:
            print(f"Window {window_candidate} is the whole video, skipping.")
            return False, None
        elif window_candidate_duration < self.min_window_duration2vqa:
            print(f"Window {window_candidate} is too small for VQA with minimum duration for VQA of {self.min_window_duration2vqa}, skipping")
            return False, None
        elif window_candidate[0] >= video_info["total_duration"] or window_candidate[1] <= 0:
            print(f"Window {window_candidate} is outside the video duration {video_info['total_duration']}, skipping")
            return False, None
        elif window_candidate[1] > video_info["total_duration"] or window_candidate[0] < 0:
            print(f"Window {window_candidate} is outside the video duration {video_info['total_duration']}, adjusting")
            window_candidate = [max(0, window_candidate[0]), min(video_info["total_duration"], window_candidate[1])]
            new_window_candidate_duration = window_candidate[1] - window_candidate[0]
            if new_window_candidate_duration == video_info["total_duration"]:
                print(f"New window {window_candidate} is the whole video, skipping.")
                return False, None
            else:
                return True, window_candidate
        else:
            return True, window_candidate

    
    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        num_vqa_frames_dict = {}
        num_vqa_inf = 0
        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            

            if "return_tempwindow" in gen_kwargs and gen_kwargs["return_tempwindow"]:
                return_tempwindow = True
            else:
                return_tempwindow = False
            if "question_format" in gen_kwargs and gen_kwargs["question_format"] == "oq":
                choices_in_question = False
                gpt_eval = True
                self.conf_thres = self.conf_thres_wo_options
            else:
                choices_in_question = True
                gpt_eval = False
                self.conf_thres = self.conf_thres_w_options


            times_and_inferences = {
                "caption_video_time": 0,
                "caption_question_time": 0,
                "llm_reasoning_time": 0,
                "windows_candidates_generation_time": 0,
                "vqa_time": 0,
                "total_time": 0,
                "num_caption_video_inferences": 0,
                "num_caption_question_inferences": 0,
                "num_vqa_inferences": 0,
                "num_llm_inferences": 0,
                "llm_tokens_usage": 0,
            }

            t_init = time.time()
            visuals = doc_to_visual(self.task_dict[task][split][doc_id])
            video_path = visuals[0]
            print("Video path:", video_path)
            self.curr_video_path = video_path
            video_name = video_path.split("/")[-1].split(".")[0]  # Extract video name

            video_info = self.get_video_info(video_path)

            if "question_id" in self.task_dict[task][split][doc_id]:
                q_id = self.task_dict[task][split][doc_id]["question_id"]
            elif "qid" in self.task_dict[task][split][doc_id]:
                q_id = self.task_dict[task][split][doc_id]["qid"]
            else:
                q_id = None
                print("No question ID found in the task dictionary. Using None.")

            if self.load_vqa:
                question_vqa_dict = self.load_vqa_func(video_name, q_id, "gpteval" in task)
            

            if self.load_summary:
                try:
                    video_summary_dict = self.load_summary_func(video_name)
                except Exception as e:
                    print(f"Error loading video summary: {e}")
                    video_summary_dict = None
            if video_summary_dict is None or not self.load_summary:
                smallwindows2caption = self.generate_candidates4captioning(video_path, video_info,  0, video_info["total_duration"], [], self.frames_sampling_strategy, times_and_inferences)
                captions = self.generate_captions(video_path, video_info, 0, video_info["total_duration"], smallwindows2caption, times_and_inferences, gen_kwargs, return_dict=True)
                t0 = time.time()
                video_summary_dict = self.generate_summary_from_captions(video_info, captions, gpt_eval=choices_in_question)
                times_and_inferences["llm_reasoning_time"] += time.time() - t0
                if self.save_summary:
                    self.save_summary_func(video_name, video_summary_dict)
                times_and_inferences["num_llm_inferences"] += 1
                times_and_inferences["llm_tokens_usage"] += video_summary_dict['tokens_usage']

            video_info["summary"] = video_summary_dict['response']['summary']
            vqa_gen_kwargs = copy.deepcopy(gen_kwargs)
            vqa_gen_kwargs["return_dict_in_generate"] = True
            vqa_gen_kwargs["output_scores"] = True

            initial_window = [0, video_info["total_duration"]]
            candidates = [initial_window]
            candidates_explanation = ["Initial window"]
            candidates2reason = []
            filteredcandidates2reason = []

            best_confidence = 0
            best_outputs, best_window = {
                "response": "None"
            }, initial_window

            best_answer_checked = False
            explanation = None
            curr_best_confidence, curr_best_window_size = 0, 0

            all_responses_dict = {}
            curr_responses_dict = {}
            candidates2reason_responses_dict = {}
            all_outputs = {}

            explored_windows = []
            print(f"Trying to find the answer to the question: {contexts} in video {video_name}")
            while True:
                print("------------------")
                window_candidate = candidates.pop(0)
                candidate_explanation = candidates_explanation.pop(0)
                print(f"Exploring window {window_candidate} in the video with total duration {video_info['total_duration']} since: {candidate_explanation}")
                
                window_candidate_duration = window_candidate[1] - window_candidate[0]
                t_vqa_init = time.time()
               
                if "qwen2_5_vl" not in self.vlm_name:
                    num_frames, num_tiles = self.get_numframes_and_numtiles_from_window(window_candidate_duration, self.vlm_vqa_max_num_frames)
                    sampling_frames_info = {"num_frames": num_frames, "num_tiles": num_tiles, "window": window_candidate}
                    num_vqa_inf += 1
                    
                else:
                    sampling_frames_info = {"window": window_candidate}
                    if self.adjust_num_frames2window:
                        num_frames = self.get_numframes_from_window(window_candidate_duration, self.vlm_vqa_max_num_frames, self.vlm_vqa_min_num_frames, self.vlm_fps, self.vlm_vqa_fps, mode="vqa")
                        sampling_frames_info["num_frames"] = num_frames
                        print(f"Sampling {num_frames} frames in window {window_candidate} for VQA") 
                        if num_frames in num_vqa_frames_dict:
                            num_vqa_frames_dict[num_frames] += 1
                        else:
                            num_vqa_frames_dict[num_frames] = 1

                if self.add_time_instruction:
                    contexts = self.add_time_instruction_to_contexts(contexts, video_info, sampling_frames_info)
                
                key_sampling_frames_info = "".join([f"{key}_{value}" for key, value in sampling_frames_info.items()])
                if video_info["total_duration"] > 600 and window_candidate_duration == video_info["total_duration"]:
                    print("We do not try to find the answer in the whole video. Let us start with the exploration.")
                    outputs = None
                elif self.load_vqa and key_sampling_frames_info in question_vqa_dict:
                    print("Loading VQA inference for window: ", window_candidate)
                    outputs = question_vqa_dict.get(key_sampling_frames_info)
                    times_and_inferences["vqa_time"] += outputs["vqa_time"]
                    if outputs is not None and "caption" not in outputs:
                        outputs = self.add_caption_to_outputs(video_info, outputs, window_candidate)
                else:
                    print("Generating VQA inference for window: ", window_candidate)
                    outputs = self.vlm.inference(video_info, sampling_frames_info, contexts, vqa_gen_kwargs)
                    times_and_inferences["vqa_time"] += time.time() - t_vqa_init
                    if outputs is not None:
                        outputs = self.add_caption_to_outputs(video_info, outputs, window_candidate)
                        if self.save_vqa:
                            outputs["vqa_time"] = times_and_inferences["vqa_time"]
                            self.save_vqa_func(video_name, q_id, "gpteval" in task, key_sampling_frames_info, question_vqa_dict, outputs)

                if outputs is None:
                    print(f"The VLM model returned None for the outputs. Skipping this window: {window_candidate}.")
                else:
                    times_and_inferences["num_vqa_inferences"] += 1
                    confidence = self.get_confidence_from_outputs(contexts, outputs, choices_in_question)
                    
                    response_dict = {"response": outputs['response'], "caption": outputs['caption'], "frames_res": outputs['frames_res'], "confidence": round(confidence, 3)}
                    curr_responses_dict[str(window_candidate)] = response_dict
                    all_outputs[str([float(window_candidate[0]), float(window_candidate[1])])] = outputs
                    print(f"[{times_and_inferences['num_vqa_inferences']}] VQA answer: ", outputs['response'], "with confidence: ", confidence)
                    if confidence > curr_best_confidence:
                        curr_best_outputs = outputs
                        curr_best_confidence = confidence
                        curr_best_window_size = window_candidate[1] - window_candidate[0]
                        
                    if confidence > best_confidence:
                        best_outputs = outputs
                        best_confidence = confidence
                        best_window = window_candidate
                        best_window_size = best_window[1] - best_window[0]
                        best_answer_checked = False
                        best_answer_num_inferences = times_and_inferences["num_vqa_inferences"]
                    
                    if not best_answer_checked and best_confidence > self.conf_thres and best_window_size <= self.max_returned_window_size:
                        print("******************** Confidence threshold surpassed. Possible BP *********************")
        
                        if not self.trust_in_confidence_mode and gpt_eval:
                            print(f"Since there is an answer {best_outputs['response']} with confidence {best_confidence} surpassing the threshold {self.conf_thres}, we will check with lmm reasoning if it is our final answer.")
                            t_reason0 = time.time()
                            final_outputs, explanation = self.generate_final_answer_or_keep_exploring(video_info, contexts, curr_responses_dict, video_info["summary"], self.conf_thres, gpt_eval)
                            times_and_inferences["num_llm_inferences"] += 1
                            times_and_inferences["llm_tokens_usage"] += final_outputs['tokens_usage']
                            times_and_inferences["llm_reasoning_time"] += time.time() - t_reason0
                        else:
                            print(f"Since there is an answer {best_outputs['response']} with confidence {best_confidence} surpassing the threshold {self.conf_thres} in Multiple-Choice question. We break it.")
                            final_outputs = best_outputs

                        if final_outputs["response"] is not None:
                            print(f"Confidence threshold reached after {times_and_inferences['num_vqa_inferences']} inferences, breaking. VLM answer: {best_outputs['response']} with confidence {best_confidence}")
                            print("**************************************************************************************")
                            break
                        else:
                            best_answer_checked = True
                            print("The answer was not good enough. We will continue exploring.")
                            print("**************************************************************************************")
                    elif window_candidate_duration > self.min_window_duration2reason:
                        if not window_candidate_duration > 10: print("Window duration is quite small for splitting it: ", window_candidate)
                        candidates2reason_responses_dict[str(window_candidate)] = response_dict
                        candidates2reason.append(window_candidate)
                
                print("------------------")
                explored_windows.append(window_candidate)

                if len(candidates) == 0:
                    conf_thres = self.conf_thres - 0.1 
                    if gpt_eval and not self.trust_in_confidence_mode and curr_best_confidence > conf_thres and curr_best_window_size <= self.max_returned_window_size:
                        print("******************** Confidence threshold is quite close. Possible BP *********************")
                        print(f"Since there is an answer {curr_best_outputs['response']} with confidence {curr_best_confidence} close to the threshold {self.conf_thres}, we will try to answer with lmm reasoning.")
                        t_reason0 = time.time()
                        final_outputs, explanation = self.generate_final_answer_or_keep_exploring(video_info, contexts, curr_responses_dict, video_info["summary"], self.conf_thres, gpt_eval)
                        times_and_inferences["num_llm_inferences"] += 1
                        times_and_inferences["llm_tokens_usage"] += final_outputs['tokens_usage']
                        times_and_inferences["llm_reasoning_time"] += time.time() - t_reason0
                        if final_outputs["response"] is not None:
                            best_window = [float(final_outputs["response"]["window_start_seconds"]), float(final_outputs["response"]["window_end_seconds"])]
                            if str(best_window) in all_outputs: best_outputs = all_outputs[str(best_window)]
                            else: best_outputs["response"] = final_outputs["response"]["final_answer"]
                            best_window = [float(final_outputs["response"]["window_start_seconds"]), float(final_outputs["response"]["window_end_seconds"])]
                            best_confidence = final_outputs["response"]["confidence"]
                            print(f"Confidence threshold reached after {times_and_inferences['num_vqa_inferences']} inferences, breaking. VLM best answer: {best_outputs['response']} with confidence {best_confidence} /  LLM final answer: {final_outputs['response']['final_answer']} LLM explanation: {final_outputs['response']['explanation']}")
                            print("**************************************************************************************")
                            break

                    all_responses_dict.update(curr_responses_dict)
                    curr_responses_dict = {}
                    curr_best_confidence, curr_best_window_size = 0, 0


                    # Generate new candidates for the next window from LLM reasoning
                    if times_and_inferences["num_vqa_inferences"] > self.max_num_vqa_inf:

                        print("******************END OF EXPLORATION. Generating Final Answer and Breaking.*********************")
                        # if gpt_eval and not self.trust_in_confidence_mode:
                        if not self.trust_in_confidence_mode:
                            print(f"Explored in detail after {times_and_inferences['num_vqa_inferences']} inferences, breaking. Generating answer leveraging LLM reasoning from all answers.")
                            t_reason0 = time.time()
                            final_outputs = self.generate_final_answer(video_info, contexts, all_responses_dict, video_info["summary"], self.conf_thres, gpt_eval)
                            times_and_inferences["num_llm_inferences"] += 1
                            times_and_inferences["llm_tokens_usage"] += final_outputs['tokens_usage']
                            times_and_inferences["llm_reasoning_time"] += time.time() - t_reason0
                            best_outputs["response"] = final_outputs["response"]["final_answer"]
                            try:
                                best_window = [float(final_outputs["response"]["window_start_seconds"]), float(final_outputs["response"]["window_end_seconds"])]
                            except ValueError:
                                best_window = [0, video_info["total_duration"]]
                        else:
                            print(f"Explored in detail after {times_and_inferences['num_vqa_inferences']} inferences, breaking. Taking the best answer as the final answer.")
                        print("***********************************************************************************************************************")
                        break
                    elif len(filteredcandidates2reason) != 0:
                        window_reason = filteredcandidates2reason.pop(0)
                        window_reason_exp = filteredcandidates2reason_exp.pop(0)
                    elif len(filteredcandidates2reason) == 0 and len(candidates2reason) != 0:
                        if candidates2reason[0] == [0, video_info["total_duration"]]:
                            filteredcandidates2reason = candidates2reason
                            filteredcandidates2reason_exp = ["Initial window. Just one candidate to continue with the exploration."]
                        else:
                            if explanation is None: explanation = f"Not approaching the minimum confidence threshold of {self.conf_thres}."
                            print("+++++++++++++++++++ Select the most promising windows to continue exploring +++++++++++++++++++++")
                            filteredcandidates2reason, filteredcandidates2reason_exp, tokens_usage = self.generate_candidate_windows_to_explore(video_info, contexts, candidates2reason_responses_dict, explanation)
                            times_and_inferences["num_llm_inferences"] += 1
                            times_and_inferences["llm_tokens_usage"] += tokens_usage
                        exp_str = " ".join([f"\n {k+1}. {window} since {exp}" for k, (window, exp) in enumerate(zip(filteredcandidates2reason, filteredcandidates2reason_exp))])
                        print("From windows:", candidates2reason, ", the windows chosen as promising to continue exploring are:", exp_str)
                        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                        explanation = None
                        candidates2reason_responses_dict = {}
                        candidates2reason = []
                        if len(filteredcandidates2reason) == 0:
                            window_reason_exp = f"ARRIVED TO THE END OF THE BRANCH. AFTER {times_and_inferences['num_vqa_inferences']} INFERENCES. LET US RE-START FROM THE BEGINNING."
                            window_reason = [0, video_info["total_duration"]]
                        else:
                            window_reason = filteredcandidates2reason.pop(0)
                            window_reason_exp = filteredcandidates2reason_exp.pop(0)
                    else:
                        window_reason_exp = f"ARRIVED TO THE END OF THE BRANCH. AFTER {times_and_inferences['num_vqa_inferences']} INFERENCES. LET US RE-START FROM THE BEGINNING."
                        window_reason = [0, video_info["total_duration"]]
                    print(f"Generating VQA candidates in window {window_reason} since: {window_reason_exp}")
                    candidates, candidates_explanation = self.generate_candidates_with_llm_reasoning(video_path, video_info, contexts, window_reason, explored_windows, times_and_inferences, gen_kwargs)
                    
                    if len(candidates) == 0:
                        print(f"NO CANDIDATES. FAIL? AFTER {times_and_inferences['num_vqa_inferences']} INFERENCES")
                        break
            print("------------------------------------------------------------------------------------------------------------------------------------------------------------------")
            print(f"Response to question {contexts} after {times_and_inferences['num_vqa_inferences']} inferences with confidence {best_confidence} is: {best_outputs['response']}. Answer found in window {best_window} in the inference number {best_answer_num_inferences}")
            t_end = time.time()
            total_time = t_end - t_init
            times_and_inferences["total_time"] = total_time

            if times_and_inferences["num_llm_inferences"] > 0:
                times_and_inferences["llm_tokens_usage"] = times_and_inferences["llm_tokens_usage"] / times_and_inferences["num_llm_inferences"]
            else:
                times_and_inferences["llm_tokens_usage"] = 0

            times_and_inferences["num_caption_inferences"] = times_and_inferences["num_caption_video_inferences"] + times_and_inferences["num_caption_question_inferences"]
            times_and_inferences["caption_time"] = times_and_inferences["caption_video_time"] + times_and_inferences["caption_question_time"]
            times_and_inferences["num_vlm_inferences"] = times_and_inferences["num_caption_inferences"] + times_and_inferences["num_vqa_inferences"]
            times_and_inferences["true_total_time"] = times_and_inferences["caption_time"] + times_and_inferences["vqa_time"] + times_and_inferences["llm_reasoning_time"]
            answer = best_outputs
            answer["temporal_window"] = best_window
            answer["times_and_inferences"] = times_and_inferences
            answer["it_num_answer"] = best_answer_num_inferences

            if return_tempwindow:
                res.append(answer)
            else:
                res.append(answer["response"])

        
            del video_info, outputs, best_outputs
            torch.cuda.empty_cache()
            gc.collect()
            pbar.update(1)
            print("------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        print("The number of VQA inferences for each number of frames: ", num_vqa_frames_dict)
        print("The total number of VQA inferences: ", num_vqa_inf)
        print("The number of inferences per question:", num_vqa_inf/len(res))
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for LLaVAVid")
