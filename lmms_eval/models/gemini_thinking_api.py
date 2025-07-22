# NOTE: also added use of asyncio for concurrency
# NOTE: this code uses the Vertex AI API as opposed to the Gemini API

import io
import json
import os
import time
from typing import List, Tuple

from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm.asyncio import tqdm as tqdm_async

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

import hashlib
import asyncio

try:
    from google import genai
    from google.genai.types import Part, GenerateContentConfig
    from google.api_core.exceptions import ResourceExhausted

    # Load the GCS URI map
    with open("../gcs_uri_map.json", "r") as f:
        GCS_URI_MAP = json.load(f)

    USING_VERTEX_API = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", None)
    assert USING_VERTEX_API, "Should be using Vertex AI API."
    print(f"\nUsing Vertex API? {USING_VERTEX_API}\n")

    GOOGLE_API_KEY = None # hard-coded this for now

except Exception as e:
    eval_logger.error(f"Error importing required libraries or loading URI map: {str(e)}")
    genai = None
    GCS_URI_MAP = None

@register_model("gemini_thinking_api")
class GeminiThinkingAPI(lmms):
    def __init__(
        self,
        model_version: str = "gemini-2.5-pro",
        modality: str = "image",
        timeout: int = 120,
        continual_mode: bool = False,
        response_persistent_folder: str = None,
        **kwargs,
    ) -> None:
        super().__init__()

        if genai is None:
            raise ImportError("`google-genai` library not available. Cannot initialize client.")

        self.client = genai.Client()
        self.model_version = model_version

        self.timeout = timeout
        self.generation_config = GenerateContentConfig(
            temperature=0,
            top_p=1,
            top_k=1,
            max_output_tokens=8192,
        )

        self.continual_mode = continual_mode
        if self.continual_mode and response_persistent_folder is None:
            raise ValueError("Continual mode requires a persistent path for the response. Please provide a valid path.")
        self.response_persistent_folder = response_persistent_folder
        if self.continual_mode:
            if not os.path.exists(self.response_persistent_folder):
                os.makedirs(self.response_persistent_folder)
            self.response_persistent_file = os.path.join(self.response_persistent_folder, f"{self.model_version}_response.json")
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
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided."
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
        self.modality = modality

        self.queries_per_second = 20
        self.semaphore = asyncio.Semaphore(self.queries_per_second)
        
        self.cache_lock = asyncio.Lock()
    
    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def get_uri_for_path(self, local_path):
        return GCS_URI_MAP.get(local_path, None)

    def encode_video(self, gcs_video_uri):
        return Part.from_uri(file_uri=gcs_video_uri, mime_type="video/mp4")

    def encode_image(self, gcs_image_uri):
        return Part.from_uri(gcs_image_uri, mime_type="image/jpeg")

    def convert_video(self, visuals):
        converted_items = []
        for visual_path in visuals:
            if isinstance(visual_path, str):
                gcs_uri = self.get_uri_for_path(visual_path)
                if gcs_uri:
                    if self.modality == "video":
                        converted_items.append(self.encode_video(gcs_uri))
                    elif self.modality == "image":
                        converted_items.append(self.encode_image(gcs_uri))
                else:
                    eval_logger.warning(f"No GCS URI found for local path: {visual_path}. Skipping.")
        return converted_items

    def get_uuid(self, task, split, doc_id):
        return f"{task}___{split}___{doc_id}"

    async def _process_request_async(self, request: Instance, pbar: tqdm_async) -> str:
        contexts, gen_kwargs, doc_to_visual, doc_id, task, split = request.args
        doc_uuid = self.get_uuid(task, split, doc_id)
        
        if self.continual_mode and self.cache_mode == "resume":
            if doc_uuid in self.response_cache:
                content = self.response_cache[doc_uuid]
                if content:
                    pbar.update(1)
                    return content

        visuals_raw = [doc_to_visual(self.task_dict[task][split][doc_id])]
        
        visuals = []
        if visuals_raw != [None]:
            visuals = self.flatten(visuals_raw)
            visuals = self.convert_video(visuals)

        contents = []
        if visuals:
            contents.extend(visuals)
        
        if isinstance(contexts, list):
            contents.extend(contexts)
        else:
            contents.append(contexts)

        content = "" # ensure content is never None
        
        async with self.semaphore:
            for attempt in range(5):
                try:
                    response = await self.client.aio.models.generate_content(
                        model=self.model_version,
                        contents=contents,
                        config=self.generation_config,
                        # The 'timeout' parameter is not supported by the async client method.
                        # timeout=self.timeout
                    )
                    if hasattr(response, 'text') and response.text is not None:
                        content = response.text
                        break
                except Exception as e:
                    eval_logger.info(f"Attempt {attempt + 1} failed with error: {str(e)}")
                    if isinstance(e, ResourceExhausted):
                        eval_logger.error("Quota exceeded, sleeping for 60 seconds...")
                        content = ""
                        await asyncio.sleep(60)
                        break
                    elif "prompt_feedback" in str(e) or "safety" in str(e):
                        eval_logger.info(f"Content was blocked due to safety settings: {str(e)}")
                        content = ""
                        break
                    
                    if attempt < 5 - 1:
                        await asyncio.sleep(1)
                    else:
                        eval_logger.error(f"All 5 attempts failed. Last error message: {str(e)}")
                        content = ""
        
        if self.continual_mode:
            async with self.cache_lock:
                self.response_cache[doc_uuid] = content
                with open(self.response_persistent_file, "w") as f:
                    json.dump(self.response_cache, f)

        pbar.update(1)
        print(f"Response:\n{content}")
        return content

    def generate_until(self, requests) -> List[str]:
        """
        Synchronous wrapper for the async generation process.
        This allows the evaluation framework to call the method without being async-aware.
        """

        async def _async_wrapper():
            pbar = tqdm_async(total=len(requests), disable=(self._rank != 0), desc="Model Responding")
            tasks = [self._process_request_async(req, pbar) for req in requests]
            res = await asyncio.gather(*tasks)
            pbar.close()
            print(res)
            return res
        
        return asyncio.run(_async_wrapper())

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        assert False, "Gemini API does not support loglikelihood."