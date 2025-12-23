import io
import json
import os
import pickle
import time
from typing import List, Tuple

import datasets
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

try:
    from google import genai

    NUM_SECONDS_TO_SLEEP = 30
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

except Exception as e:
    eval_logger.error(f"Error importing generativeai: {str(e)}")
    genai = None


@register_model("gemini_api")
class GeminiAPI(lmms):
    def __init__(
        self,
        model_version: str = "gemini-2.5-flash",
        # modality: str = "image",
        timeout: int = 120,
        continual_mode: bool = False,
        response_persistent_folder: str = "./logs/gemini_persistent_folder",
        # We will cache the Gemini API response in this path and use it for future requests
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_version = model_version
        self.timeout = timeout
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.continual_mode = continual_mode

        # if self.continual_mode and response_persistent_folder is None:
        #     raise ValueError("Continual mode requires a persistent path for the response. We will cache the Gemini API response in this path and use it for future requests. Please provide a valid path.")
        if self.continual_mode:
            self.response_persistent_folder = response_persistent_folder
            if not os.path.exists(self.response_persistent_folder):
                os.makedirs(self.response_persistent_folder)
            self.response_persistent_file = os.path.join(self.response_persistent_folder, f"{self.model_version}_response.json")
        else:
            self.response_persistent_folder = ""
            self.response_persistent_file = ""

        if os.path.exists(self.response_persistent_file):
            with open(self.response_persistent_file, "r") as f:
                self.response_cache = json.load(f)
            self.cache_mode = "resume"
        else:
            self.response_cache = {}
            self.cache_mode = "start"

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            assert self.continual_mode is False, "Continual mode is not supported with distributed inference."
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

        # self.modality = modality

        self.video_pool = []

    def free_video(self):
        for video in self.video_pool:
            video.delete()
        self.video_pool = []

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def get_image_size(self, image):
        # Create a BytesIO object to store the image bytes
        img_byte_array = io.BytesIO()

        # Save the image to the BytesIO object
        image.save(img_byte_array, format="PNG")

        # Get the size of the BytesIO object
        img_size = img_byte_array.tell()

        return img_size

    def encode_video(self, video_path):
        uploaded_obj = genai.upload_file(path=video_path)
        time.sleep(5)
        self.video_pool.append(uploaded_obj)
        return uploaded_obj

    def encode_audio(self, audio):
        audio_io = io.BytesIO()
        sf.write(audio_io, audio["array"], audio["sampling_rate"], format="WAV")
        return genai.upload_file(audio_io, mime_type="audio/wav")

    def convert_modality(self, images):
        for idx, img in enumerate(images):
            if isinstance(img, dict) and "sampling_rate" in img:  # audio
                audio = self.encode_audio(img)
                images[idx] = audio
            elif isinstance(img, str):  # video
                try:
                    images[idx] = self.encode_video(img)
                except Exception as e:
                    eval_logger.error(f"Error converting video: {str(e)}")
        return images

    def add_llm_reason_to_curr_dict(self, key, value):
        value = json.dumps(value)
        self.curr_llm_reasons[key.encode()] = value.encode()

    def save_llm_reasons_func(self, filename, key, value):
        cache_llm_file = os.path.join(self.response_persistent_folder, f"{filename}.pkl")
        if not hasattr(self, "curr_llm_reasons") or self.curr_llm_reasons is None:
            self.curr_llm_reasons = {}
        self.add_llm_reason_to_curr_dict(key, value)
        with open(cache_llm_file, "wb") as f:
            pickle.dump(self.curr_llm_reasons, f)
            f.flush()
            os.fsync(f.fileno())

    def load_llm_reasons_func(self, filename, key):
        cache_llm_file = os.path.join(self.response_persistent_folder, f"{filename}.pkl")
        if not hasattr(self, "curr_llm_filename") or filename != self.curr_llm_filename or not hasattr(self, "curr_llm_reasons") or self.curr_llm_reasons is None:
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

    def parse_json(self, text):
        import re

        # Remove leading/trailing whitespace and unescape newlines
        text = text.strip()
        # Remove triple backticks and possible "json" markers
        text = re.sub(r"^```json|^```|```$", "", text, flags=re.MULTILINE).strip()
        # Replace escaped newlines with real newlines
        text = text.replace("\\n", "\n")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Fallback: extract the largest JSON object or array
            json_pattern = r"(\{.*\}|\[.*\])"
            matches = re.findall(json_pattern, text, re.DOTALL)
            for match in matches:
                try:
                    match = match.replace("'", '"')
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
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
        if self.continual_mode and hasattr(self, "load_llm_reasons_func") and self.load_llm_reasons_func(filename, key) is not None:
            cached_value = self.load_llm_reasons_func(filename, key)
            if cached_value is not None:
                cached_value = self.parse_json(cached_value)
                print("Get LLM reasoning from cache")
                return cached_value
        for _ in range(3):
            try:
                print("GeminiAPI: Sending request to Gemini")
                t_llm_init = time.time()
                # Gemini API expects a single string prompt, so we concatenate
                full_prompt = f"{system_prompt}\n{prompt}"
                response = self.client.models.generate_content(model=self.model_version, contents=full_prompt)
                response_text = response.text
                if json_format:
                    response_parsed = self.parse_json(response_text)
                else:
                    response_parsed = response_text
                response_dict = {"response": response_parsed, "tokens_usage": response.usage_metadata.total_token_count, "time": time.time() - t_llm_init}  # Gemini API may not provide token usage
                if hasattr(self, "save_llm_reasons_func"):
                    self.save_llm_reasons_func(filename, key, response_dict)
                return response_dict
            except Exception as e:
                print(f"Gemini Error: {e}")
                continue
        return "Gemini Error"

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        def get_uuid(task, split, doc_id):
            return f"{task}___{split}___{doc_id}"

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            if self.continual_mode and self.cache_mode == "resume":
                doc_uuid = get_uuid(task, split, doc_id)
                if doc_uuid in self.response_cache:
                    content = self.response_cache[doc_uuid]
                    if content:
                        res.append(content)
                        pbar.update(1)
                        continue

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0

            config = genai.GenerationConfig(
                max_output_tokens=gen_kwargs["max_new_tokens"],
                temperature=gen_kwargs["temperature"],
            )

            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            visuals = self.convert_modality(visuals)

            message = [contexts] + visuals

            for attempt in range(5):
                try:
                    content = self.model.generate_content(
                        message,
                        generation_config=config,
                        safety_settings={
                            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                        },
                    )
                    content = content.text
                    break
                except Exception as e:
                    eval_logger.info(f"Attempt {attempt + 1} failed with error: {str(e)}")
                    if isinstance(e, ValueError):
                        try:
                            eval_logger.info(f"Prompt feed_back: {content.prompt_feedback}")
                            content = ""
                            break
                        except Exception:
                            pass
                    if attempt < 5 - 1:  # If we have retries left, sleep and then continue to next attempt
                        time.sleep(NUM_SECONDS_TO_SLEEP)
                    else:  # If this was the last attempt, log and return empty
                        eval_logger.error(f"All 5 attempts failed. Last error message: {str(e)}")
                        content = ""
            res.append(content)
            pbar.update(1)

            self.free_video()

            if self.continual_mode is True:  # Cache the response
                doc_uuid = get_uuid(task, split, doc_id)
                self.response_cache[doc_uuid] = content
                with open(self.response_persistent_file, "w") as f:
                    json.dump(self.response_cache, f)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for Gemini API")

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        assert False, "Gemini API not support"
