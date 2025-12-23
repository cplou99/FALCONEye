import base64
from io import BytesIO
from typing import List, Optional, Tuple, Union

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import load_video_decord

try:
    from lmms_eval.models.simple.qwen_vl_utils_local.vision_process import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


@register_model("qwen2_5_vl")
class Qwen2_5_VL(lmms):
    """
    Qwen2_VL Model
    "https://github.com/QwenLM/Qwen2-VL"
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        modality: Optional[str] = "video",
        use_cache=True,
        use_flash_attention_2: Optional[bool] = True,
        max_pixels: int = 768 * 28 * 28,
        min_pixels: int = 16 * 28 * 28,
        total_pixels: int = 17280 * 28 * 28,
        res_width: int = None,
        res_height: int = None,
        nframes: int = None,
        max_frames_num: int = 768,
        fps: int = 2.0,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        
        self.torch_dtype = torch.bfloat16
        if use_flash_attention_2:
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                pretrained,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map,
                attn_implementation="flash_attention_2",
            ).eval()
        else:
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(pretrained, torch_dtype=self.torch_dtype, device_map=self.device_map).eval()
        self.processor = AutoProcessor.from_pretrained(pretrained)
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.nframes = nframes
        self.fps = fps
        self.max_frames_num = max_frames_num
        self.res_width = res_width
        self.res_height = res_height
        self.total_pixels = total_pixels
        self.modality = modality
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)

        self._config = self.model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if type(self.max_pixels) == str:
            self.max_pixels = eval(self.max_pixels)
        if type(self.min_pixels) == str:
            self.min_pixels = eval(self.min_pixels)
        if type(self.total_pixels) == str:
            self.total_pixels = eval(self.total_pixels)
            
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._word_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2_VL")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            visuals = doc_to_visual(self.task_dict[task][split][doc_id])
            video_path = visuals[0]
            try:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": [
                            {"type": "text", "text": contexts},
                        ]
                    },
                ]

                if self.modality == "video":
                    messages[1]["content"].append({"video": video_path, "total_pixels": self.total_pixels, "min_pixels": self.min_pixels})
                elif self.modality == "image":
                    messages[1]["content"].append({"image": video_path, "total_pixels": self.total_pixels, "min_pixels": self.min_pixels})
                elif self.modality == "blind":
                    print("Blind mode")
                else:
                    raise ValueError("Modality not supported")
                
                if self.nframes is not None:
                    messages[1]["content"][1]["nframes"] = self.nframes
                if  self.max_pixels is not None:
                    messages[1]["content"][1]["max_pixels"] = self.max_pixels
                if self.res_width is not None:
                    messages[1]["content"][1]["resized_width"] = self.res_width
                if self.res_height is not None:
                    messages[1]["content"][1]["resized_height"] = self.res_height
                
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True)
            except Exception as e:
                # import pdb;pdb.set_trace()
                eval_logger.info(f"{e}")
                eval_logger.info(f"Video {visuals} can not load, check the source")
                video_path = "\n".join(visuals)
                res.append(f"Video {video_path} can not load, check the source")
                pbar.update(1)
                continue

            fps_inputs = video_kwargs['fps']
            inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, fps=fps_inputs, padding=True, return_tensors="pt")

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 128
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            if "return_dict_in_generate" not in gen_kwargs:
                gen_kwargs["return_dict_in_generate"] = False
            if "output_scores" not in gen_kwargs:
                gen_kwargs["output_scores"] = False
            if "output_logits" not in gen_kwargs:
                gen_kwargs["output_logits"] = False

            output_ids = self.model.generate(
                **inputs,
                do_sample=True if gen_kwargs["temperature"] > 0 else False,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
                return_dict_in_generate=gen_kwargs["return_dict_in_generate"],
                output_scores=gen_kwargs["output_scores"],
                output_logits=gen_kwargs["output_logits"]
            )

            
            if gen_kwargs["return_dict_in_generate"]:
                scores = torch.stack(output_ids.scores).reshape(len(output_ids.scores), -1).transpose(0, 1).cpu()
                generated_ids = [out_ids[len(input_ids):] for input_ids, out_ids in zip(inputs.input_ids,  output_ids.sequences)][0].cpu()
                num_tokens = generated_ids.shape[-1]
                scores = scores.reshape(-1, scores.shape[0], scores.shape[-1])
                scores = torch.nn.functional.log_softmax(scores, dim=1)
                scores = scores.reshape(-1, scores.shape[-1]).numpy()
                probs = np.exp(scores)

                # print("Response without skipping special tokens:", self.tokenizer.batch_decode(output_ids.sequences, skip_special_tokens=False)[0].strip())
                # print("Number of tokens:", output_ids.sequences.shape[-1])
                tokens_dict = {}
                for i in range(num_tokens):
                    out_token = self.tokenizer.decode(generated_ids[i].item())
                    tokens_dict[i] = {'token': out_token}
                    # print(f"Token [{i}]: {out_token}")
                for i in range(num_tokens):
                    # print(f"Top 5 tokens for token at pos {i}")
                    # print("| token | token string | log probability | probability |")
                    top5_token_list, top5_prob_list = [], []
                    for tok_id in np.argsort(scores[:, i]).tolist()[::-1][:5]:
                        tok = self.tokenizer.decode(tok_id)
                        score = scores[tok_id, i]
                        prob = np.exp(score)
                        top5_token_list.append(tok)
                        top5_prob_list.append(prob)
                        # print(f"| {tok_id:5d} | {tok:8s} | {score:.3f} | {prob:.2%}")
                    tokens_dict[i]['top5_tokens'] = top5_token_list
                    tokens_dict[i]['top5_probs'] = top5_prob_list
                    tokens_dict[i]['avg_prob'] = np.mean(probs[:, i])
                    tokens_dict[i]['std_prob'] = np.std(probs[:, i])
        
                output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                output_text = "".join(output_text)
                output_dict = {
                    "response": output_text,
                    "num_tokens": num_tokens,
                    "tokens": tokens_dict,
                    "frames_res": list(video_inputs[0].shape),
                }
                print("Response in dict:", output_text)
                res.append(output_dict)
                del scores
            else:       
                generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
                output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                output_text = "".join(output_text)
                print("Response:", output_text)
                res.append(output_text)
            del output_ids, inputs
            torch.cuda.empty_cache()
            pbar.update(1)

        pbar.close()
        return res

    def inference(self, video_info, sampling_frames_info, context, gen_kwargs):
        video_path = video_info["path"]
        
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                        {"type": "text", "text": context},
                        {"video": video_info["path"]},
                    ]
                },
            ]
            # if 'vr' in video_info:
                # messages[1]["content"][1]["vr"] = video_info['vr']
            if 'num_frames' in sampling_frames_info:
                messages[1]["content"][1]["nframes"] = sampling_frames_info['num_frames']
            if self.min_pixels is not None:
                messages[1]["content"][1]["min_pixels"] = self.min_pixels
            if  self.max_pixels is not None:
                messages[1]["content"][1]["max_pixels"] = self.max_pixels
            if  self.total_pixels is not None:
                messages[1]["content"][1]["total_pixels"] = self.total_pixels
            if 'res_width' in sampling_frames_info:
                messages[1]["content"][1]["resized_width"] = sampling_frames_info['res_width']
            if 'res_height' in sampling_frames_info:
                messages[1]["content"][1]["resized_height"] = sampling_frames_info['res_height']
            if "window" in sampling_frames_info:
                messages[1]["content"][1]["video_start"] = sampling_frames_info["window"][0]
                messages[1]["content"][1]["video_end"] = sampling_frames_info["window"][1]
            if "frames_idxs" in sampling_frames_info:
                messages[1]["content"][1]["frames_idxs"] = sampling_frames_info["frames_idxs"]

            
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True)
        except Exception as e:
            # import pdb;pdb.set_trace()
            eval_logger.info(f"{e}")
            eval_logger.info(f"Video {video_path} can not load, check the source")
            return None


        fps_inputs = video_kwargs['fps']
        print("video input:", video_inputs[0].shape)
        num_frames, _, resized_height, resized_width = video_inputs[0].shape
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, fps=fps_inputs, padding=True, return_tensors="pt")

        if self.device_map == "auto":
            inputs = inputs.to("cuda")
        else:
            inputs = inputs.to(self.device)

        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 128
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 0
        if "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = None
        if "num_beams" not in gen_kwargs:
            gen_kwargs["num_beams"] = 1
        if "return_dict_in_generate" not in gen_kwargs:
            gen_kwargs["return_dict_in_generate"] = False
        if "output_scores" not in gen_kwargs:
            gen_kwargs["output_scores"] = False
        if "output_logits" not in gen_kwargs:
            gen_kwargs["output_logits"] = False

        output_ids = self.model.generate(
            **inputs,
            do_sample=True if gen_kwargs["temperature"] > 0 else False,
            temperature=gen_kwargs["temperature"],
            top_p=gen_kwargs["top_p"],
            num_beams=gen_kwargs["num_beams"],
            max_new_tokens=gen_kwargs["max_new_tokens"],
            use_cache=self.use_cache,
            return_dict_in_generate=gen_kwargs["return_dict_in_generate"],
            output_scores=gen_kwargs["output_scores"],
            output_logits=gen_kwargs["output_logits"]
        )

        if gen_kwargs["return_dict_in_generate"]:
            scores = torch.stack(output_ids.scores).reshape(len(output_ids.scores), -1).transpose(0, 1).cpu()
            generated_ids = [out_ids[len(input_ids):] for input_ids, out_ids in zip(inputs.input_ids,  output_ids.sequences)][0].cpu()
            num_tokens = generated_ids.shape[-1]
            scores = scores.reshape(-1, scores.shape[0], scores.shape[-1])
            scores = torch.nn.functional.log_softmax(scores, dim=1)
            scores = scores.reshape(-1, scores.shape[-1]).numpy()
            probs = np.exp(scores)

            # print("Response without skipping special tokens:", self.tokenizer.batch_decode(output_ids.sequences, skip_special_tokens=False)[0].strip())
            # print("Number of tokens:", output_ids.sequences.shape[-1])
            tokens_dict = {}
            for i in range(num_tokens):
                out_token = self.tokenizer.decode(generated_ids[i].item())
                tokens_dict[i] = {'token': out_token}
                # print(f"Token [{i}]: {out_token}")
            for i in range(num_tokens):
                # print(f"Top 5 tokens for token at pos {i}")
                # print("| token | token string | log probability | probability |")
                top5_token_list, top5_prob_list = [], []
                for tok_id in np.argsort(scores[:, i]).tolist()[::-1][:5]:
                    tok = self.tokenizer.decode(tok_id)
                    score = scores[tok_id, i]
                    prob = np.exp(score)
                    top5_token_list.append(tok)
                    top5_prob_list.append(prob)
                    # print(f"| {tok_id:5d} | {tok:8s} | {score:.3f} | {prob:.2%}")
                tokens_dict[i]['top5_tokens'] = top5_token_list
                tokens_dict[i]['top5_probs'] = top5_prob_list
                tokens_dict[i]['avg_prob'] = np.mean(probs[:, i])
                tokens_dict[i]['std_prob'] = np.std(probs[:, i])
    
            output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            output_text = "".join(output_text)
            output_dict = {
                "response": output_text,
                "num_tokens": num_tokens,
                "tokens": tokens_dict,
                "frames_res": list(video_inputs[0].shape),
            }
            res = output_dict
            del scores
        else:       
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
            output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            output_text = "".join(output_text)
            res = output_text
        del output_ids, inputs, image_inputs, video_inputs
        torch.cuda.empty_cache()
        return res
    
    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
