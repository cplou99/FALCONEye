import base64
import json
import os
import time
from copy import deepcopy
from io import BytesIO
from typing import List, Tuple, Optional

import numpy as np
import requests as url_requests
import re
import pickle
from accelerate import Accelerator, DistributedType
from tqdm import tqdm
from openai import OpenAI
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

try:
    from decord import VideoReader, cpu
except ImportError:
    pass

from PIL import Image

client = OpenAI()
API_TYPE = os.getenv("API_TYPE", "openai")
NUM_SECONDS_TO_SLEEP = 30
from loguru import logger as eval_logger

if API_TYPE == "openai":
    API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    API_KEY = os.getenv("OPENAI_API_KEY", "")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
elif API_TYPE == "azure":
    API_URL = os.getenv("AZURE_ENDPOINT", "https://api.cognitive.microsoft.com/sts/v1.0/issueToken")
    API_KEY = os.getenv("AZURE_API_KEY", "YOUR_API_KEY")
    headers = {
        "api-key": API_KEY,
        "Content-Type": "application/json",
    }


@register_model("gpt4v")
class GPT4V(lmms):
    def __init__(
        self,
        model_version: str = "gpt-4o",
        modality: str = "video",
        max_frames_num: int = 250,
        timeout: int = 6000,
        continual_mode: bool = False,
        response_persistent_folder: str = None,
        detail: str = "low",
        save_llm_reasons: Optional[bool] = True,
        load_llm_reasons: Optional[bool] = True,
        llm_reasons_dir: Optional[str] = f"{os.path.dirname(os.getcwd())}/features/llm_reasonings",
        **kwargs,
    ) -> None:
        super().__init__()
        # Manually set a image token for GPT4V so that we can search for it
        # and split the text and image
        # Here we just use the same token as llava for convenient
        self.model_version = model_version
        self.modality = modality
        self.max_frames_num = max_frames_num
        self.image_token = "<image>"
        self.timeout = timeout
        self.continual_mode = continual_mode
        self.detail = detail
        self.load_llm_reasons = load_llm_reasons
        self.save_llm_reasons = save_llm_reasons
        self.curr_llm_filename = None
        self.curr_llm_reasons = None
        self.llm_reasons_dir = llm_reasons_dir
        os.makedirs(self.llm_reasons_dir, exist_ok=True)

        if self.continual_mode:
            if response_persistent_folder is None:
                raise ValueError("Continual mode requires a persistent path for the response. Please provide a valid path.")

            os.makedirs(response_persistent_folder, exist_ok=True)
            self.response_persistent_folder = response_persistent_folder
            self.response_persistent_file = os.path.join(self.response_persistent_folder, f"{self.model_version}_response.json")

            if os.path.exists(self.response_persistent_file):
                with open(self.response_persistent_file, "r") as f:
                    self.response_cache = json.load(f)
                self.cache_mode = "resume"
            else:
                self.response_cache = {}
                self.cache_mode = "start"

        accelerator = Accelerator()
        # assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        self.device = self.accelerator.device

    def add_llm_reason_to_curr_dict(self, key, value):
        value = json.dumps(value)
        self.curr_llm_reasons[key.encode()] = value.encode()

    def save_llm_reasons_func(self, filename, key, value):
        cache_llm_file = os.path.join(self.llm_reasons_dir, f"{filename}.pkl")
        self.add_llm_reason_to_curr_dict(key, value)
        with open(cache_llm_file, "wb") as f:
            pickle.dump(self.curr_llm_reasons, f)
            f.flush()  # Flush internal buffers
            os.fsync(f.fileno())  # Force writing to disk
    
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
            return self.curr_llm_reasons[key.encode()].decode()
        else:
            return None
        
    # Function to encode the image
    def encode_image(self, image: Image):
        output_buffer = BytesIO()
        image.save(output_buffer, format="PNG")
        byte_data = output_buffer.getvalue()
        base64_str = base64.b64encode(byte_data).decode("utf-8")
        return base64_str

    # Function to encode the video
    def encode_video(self, video_path, for_get_frames_num):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, for_get_frames_num, dtype=int)

        # Ensure the last frame is included
        if total_frame_num - 1 not in uniform_sampled_frames:
            uniform_sampled_frames = np.append(uniform_sampled_frames, total_frame_num - 1)

        frame_idx = uniform_sampled_frames.tolist()
        frames = vr.get_batch(frame_idx).asnumpy()

        base64_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            output_buffer = BytesIO()
            img.save(output_buffer, format="PNG")
            byte_data = output_buffer.getvalue()
            base64_str = base64.b64encode(byte_data).decode("utf-8")
            base64_frames.append(base64_str)

        return base64_frames

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def inference(self, imgs, contexts, gen_kwargs):
        payload = {"messages": []}
        if API_TYPE == "openai":
            payload["model"] = self.model_version

        response_json = {"role": "user", "content": []}
        # When there is no image token in the context, append the image to the text
        if self.image_token not in contexts:
            payload["messages"].append(deepcopy(response_json))
            payload["messages"][0]["content"].append({"type": "text", "text": contexts})
            for img in imgs:
                payload["messages"][0]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}", "detail": self.detail}})
        else:
            contexts = contexts.split(self.image_token)
            for idx, img in enumerate(imgs):
                payload["messages"].append(deepcopy(response_json))
                payload["messages"][idx]["content"].append({"type": "text", "text": contexts[idx]})
                payload["messages"][idx]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}", "detail": self.detail}})

            # If n image tokens are in the contexts
            # contexts will be splitted into n+1 chunks
            # Manually add it into the payload
            payload["messages"].append(deepcopy(response_json))
            payload["messages"][-1]["content"].append({"type": "text", "text": contexts[-1]})

        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 1024
        if gen_kwargs["max_new_tokens"] > 4096:
            gen_kwargs["max_new_tokens"] = 4096
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 0
        if "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = None
        if "num_beams" not in gen_kwargs:
            gen_kwargs["num_beams"] = 1

        payload["max_tokens"] = gen_kwargs["max_new_tokens"]
        payload["temperature"] = gen_kwargs["temperature"]

        for attempt in range(5):
            try:
                response = url_requests.post(API_URL, headers=headers, json=payload, timeout=self.timeout)
                response_data = response.json()

                response_text = response_data["choices"][0]["message"]["content"].strip()
                break  # If successful, break out of the loop

            except Exception as e:
                try:
                    error_msg = response.json()
                except:
                    error_msg = ""

                eval_logger.info(f"Attempt {attempt + 1} failed with error: {str(e)}.\nReponse: {error_msg}")
                if attempt <= 5:
                    time.sleep(NUM_SECONDS_TO_SLEEP)
                else:  # If this was the last attempt, log and return empty string
                    eval_logger.error(f"All 5 attempts failed. Last error message: {str(e)}.\nResponse: {response.json()}")
                    response_text = ""
        
        response_dict = {"response": response_text,
                         "tokens_usage": response_data["usage"]["prompt_tokens"]}
        
        return response_dict

    def inference_format(self, resp_format, messages, logprobs=False):
        from openai import OpenAI
        client = OpenAI()
        completion = client.beta.chat.completions.parse(
            model=self.model_version,
            messages=messages,
            response_format=resp_format,
            logprobs=logprobs
        )

        return completion


    def parse_json(self, text):
        try:
            # First, try to directly parse the text as JSON
            return json.loads(text)
        except json.JSONDecodeError:
            # If direct parsing fails, use regex to extract JSON
            json_pattern = r"\{.*?\}|\[.*?\]"  # Pattern for JSON objects and arrays

            matches = re.findall(json_pattern, text, re.DOTALL)
            for match in matches:
                try:
                    match = match.replace("'", '"')
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue

            # If no JSON structure is found
            print("No valid JSON found in the text.")
            return None

        

    def get_llm_response(self, system_prompt, prompt, filename, json_format=True):
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": prompt},
        ]

        key = json.dumps([self.model_version, messages])
        if self.load_llm_reasons:
            cached_value = self.load_llm_reasons_func(filename, key)
            if cached_value is not None:
                cached_value = self.parse_json(cached_value)
                print("Get LLM reasoning from cache")
                return cached_value
        
        for _ in range(3):
            try:
                print("GPT4V: Sending request to OpenAI")
                t_llm_init = time.time()
                if json_format:
                    completion = client.chat.completions.create(
                        model=self.model_version,
                        response_format={"type": "json_object"},
                        messages=messages,
                    )
                else:
                    completion = client.chat.completions.create(
                        model=self.model_version, messages=messages
                    )
                response_text = completion.choices[0].message.content
                if json_format:
                    response = self.parse_json(response_text)
                else:
                    response = response_text
                response_dict = {"response": response, "tokens_usage": completion.usage.total_tokens, "time": time.time() - t_llm_init}
                if self.save_llm_reasons:
                    self.save_llm_reasons_func(filename, key, response_dict)
                return response_dict
            except Exception as e:
                print(f"GPT Error: {e}")
                continue
        return "GPT Error"
    
    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            if self.continual_mode is True and self.cache_mode == "resume":
                doc_uuid = f"{task}___{split}___{doc_id}"
                if doc_uuid in self.response_cache:
                    response_text = self.response_cache[doc_uuid]
                    if response_text:
                        res.append(response_text)
                        pbar.update(1)
                        continue

            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            imgs = []  # multiple images or frames for video
            for visual in visuals:
                if self.modality == "image":
                    img = self.encode_image(visual)
                    imgs.append(img)
                elif self.modality == "video":
                    frames = self.encode_video(visual, self.max_frames_num)
                    imgs.extend(frames)

            payload = {"messages": []}
            if API_TYPE == "openai":
                payload["model"] = self.model_version

            response_json = {"role": "user", "content": []}
            # When there is no image token in the context, append the image to the text
            if self.image_token not in contexts:
                payload["messages"].append(deepcopy(response_json))
                payload["messages"][0]["content"].append({"type": "text", "text": contexts})
                for img in imgs:
                    payload["messages"][0]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}", "detail": self.detail}})
            else:
                contexts = contexts.split(self.image_token)
                for idx, img in enumerate(imgs):
                    payload["messages"].append(deepcopy(response_json))
                    payload["messages"][idx]["content"].append({"type": "text", "text": contexts[idx]})
                    payload["messages"][idx]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}", "detail": self.detail}})

                # If n image tokens are in the contexts
                # contexts will be splitted into n+1 chunks
                # Manually add it into the payload
                payload["messages"].append(deepcopy(response_json))
                payload["messages"][-1]["content"].append({"type": "text", "text": contexts[-1]})

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if gen_kwargs["max_new_tokens"] > 4096:
                gen_kwargs["max_new_tokens"] = 4096
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            payload["max_tokens"] = gen_kwargs["max_new_tokens"]
            payload["temperature"] = gen_kwargs["temperature"]

            for attempt in range(5):
                try:
                    response = url_requests.post(API_URL, headers=headers, json=payload, timeout=self.timeout)
                    response_data = response.json()

                    response_text = response_data["choices"][0]["message"]["content"].strip()
                    break  # If successful, break out of the loop

                except Exception as e:
                    try:
                        error_msg = response.json()
                    except:
                        error_msg = ""

                    eval_logger.info(f"Attempt {attempt + 1} failed with error: {str(e)}.\nReponse: {error_msg}")
                    if attempt <= 5:
                        time.sleep(NUM_SECONDS_TO_SLEEP)
                    else:  # If this was the last attempt, log and return empty string
                        eval_logger.error(f"All 5 attempts failed. Last error message: {str(e)}.\nResponse: {response.json()}")
                        response_text = ""
            print("Response text: ", response_text)
            print("Tokens usage: ", response_data["usage"]["total_tokens"])
            res.append(response_text)
            pbar.update(1)

            if self.continual_mode is True:  # Cache the response
                doc_uuid = f"{task}___{split}___{doc_id}"
                self.response_cache[doc_uuid] = response_text
                with open(self.response_persistent_file, "w") as f:
                    json.dump(self.response_cache, f)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for GPT4V")

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        assert False, "GPT4V not support"
