import asyncio
import torch
import os
import sys
from typing import List, Any, Callable, Dict
try:
    from diffusers import KolorsPipeline, HunyuanDiTPipeline, StableDiffusion3Pipeline, CogView3PlusPipeline, DiffusionPipeline
except ImportError:
    pass
    #print("\033[94mDiffusers module is not installed, skipping related imports.\033[0m")

from .factories import ModelRequestFactory, RequestHandlerFactory
from tqdm.asyncio import tqdm_asyncio
import asyncio
import yaml
from .utils.tools import retry_on_failure,retry_on_failure_async

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)



def load_config(config_file):
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

class ModelService:
    def __init__(self, request_type='llm', handler_type='api', model_name=None, config_path=os.path.join(PROJECT_ROOT, 'src/config/config.yaml'), **kwargs):
        self.request_type = request_type
        self.handler_type = handler_type
        config=load_config(config_path)
        MODEL_NAME_MAPPINGS=config['MODEL_NAME_MAPPINGS']
        self.model_name = MODEL_NAME_MAPPINGS.get(model_name, model_name)
        self.config_path = config_path
        self.request_factory = ModelRequestFactory()
        self.handler_factory = RequestHandlerFactory()
        self.kwargs = kwargs
        self.pipe = self._initialize_pipeline()

    def _initialize_pipeline(self):
        if self.request_type == 't2i' and self.handler_type == 'local':
            if self.model_name == "HunyuanDiT":
                pipe = HunyuanDiTPipeline.from_pretrained(
                    "Tencent-Hunyuan/HunyuanDiT-Diffusers", torch_dtype=torch.float16
                )
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                except:
                    print("xformers not available for HunyuanDiT")
                pipe.enable_attention_slicing(1)
                return pipe
            elif self.model_name == "kolors":
                pipe = KolorsPipeline.from_pretrained(
                    "Kwai-Kolors/Kolors-diffusers", torch_dtype=torch.float16, variant="fp16"
                )
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                except:
                    print("xformers not available for Kolors")
                pipe.enable_attention_slicing(1)
                return pipe
            elif self.model_name == 'sd-3.5-large':
                pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16)
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                except:
                    print("xformers not available for SD 3.5 Large")
                pipe.enable_attention_slicing(1)
                return pipe
            elif self.model_name == 'sd-3.5-large-turbo':
                pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large-turbo", torch_dtype=torch.bfloat16)
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                except:
                    print("xformers not available for SD 3.5 Large Turbo")
                pipe.enable_attention_slicing(1)
                return pipe
            elif self.model_name == 'stable-diffusion-xl-base-1.0':
                pipe = DiffusionPipeline.from_pretrained(
                        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
                ).to("cuda")
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                except:
                    print("xformers not available for SDXL")
                pipe.enable_attention_slicing(1)
                return pipe
            elif self.model_name == 'stable-diffusion-3-medium':
                pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                except:
                    print("xformers not available for SD 3 Medium")
                pipe.enable_attention_slicing(1)
                return pipe
            elif self.model_name == "cogView-3-plus":
                pipe = CogView3PlusPipeline.from_pretrained("THUDM/CogView3-Plus-3B", torch_dtype=torch.float16)
                # Enable it to reduce GPU memory usage
                pipe.enable_model_cpu_offload()
                pipe.vae.enable_slicing()
                pipe.vae.enable_tiling()
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                except:
                    print("xformers not available for CogView3-Plus")
                pipe.enable_attention_slicing(1)
                return pipe
            elif self.model_name == "playground-v2.5":
                pipe = DiffusionPipeline.from_pretrained(
                    "playgroundai/playground-v2.5-1024px-aesthetic",
                    torch_dtype=torch.float16,
                    variant="fp16",
                ).to("cuda")
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                except:
                    print("xformers not available for Playground v2.5")
                pipe.enable_attention_slicing(1)
                return pipe
            elif self.model_name == "FLUX.1-dev":
                pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                except:
                    print("xformers not available for FLUX.1-dev")
                pipe.enable_attention_slicing(1)
                return pipe
        return None

    def _format_messages(self, conversation_history):
        formatted_messages = ""
        for message in conversation_history:
            role = message["role"]
            content = message["content"]
            formatted_messages += f"{role}: {content}\n\n"
        return formatted_messages

    def _process_single(self, prompt, **kwargs):
        if "system_prompt" in kwargs:
            system_prompt = kwargs.pop("system_prompt")
            conversation_history = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            prompt = self._format_messages(conversation_history)
        request = self.request_factory.create_request(self.request_type, self.model_name, prompt, **self.kwargs, **kwargs)
        handler = self.handler_factory.create_handler(self.request_type, self.handler_type, self.model_name, self.config_path, self.pipe)
        return request.send_request(handler)

    async def _process_single_async(self, prompt, **kwargs):
        if "system_prompt" in kwargs:
            system_prompt = kwargs.pop("system_prompt")
            conversation_history = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            prompt = self._format_messages(conversation_history)
        request = await self.request_factory.create_request_async(self.request_type, self.model_name, prompt, **self.kwargs, **kwargs)
        handler = await self.handler_factory.create_handler_async(self.request_type, self.handler_type, self.model_name, self.config_path)
        return await request.send_request_async(handler)

    def _process_multiturn(self, prompts):
        if self.request_type == "t2i":
            request = self.request_factory.create_request(self.request_type, self.model_name, prompts, **self.kwargs)
            handler = self.handler_factory.create_handler(self.request_type, self.handler_type, self.model_name, self.config_path, self.pipe)
            return request.send_request(handler)
        conversation_history = []
        responses = []
        for prompt in prompts:
            conversation_history.append({"role": "user", "content": prompt})
            messages = self._format_messages(conversation_history)
            request = self.request_factory.create_request(
                self.request_type, self.model_name, messages, **self.kwargs
            )
            handler = self.handler_factory.create_handler(
                self.request_type, self.handler_type, self.model_name, self.config_path
            )
            response = request.send_request(handler)
            conversation_history.append({"role": "assistant", "content": response})
            responses.append(response)
        return responses

    async def _process_multiturn_async(self, prompts):
        conversation_history = []
        responses = []
        for prompt in prompts:
            conversation_history.append({"role": "user", "content": prompt})
            messages = self._format_messages(conversation_history)
            request = await self.request_factory.create_request_async(
                self.request_type, self.model_name, messages, **self.kwargs
            )
            handler = await self.handler_factory.create_handler_async(
                self.request_type, self.handler_type, self.model_name, self.config_path
            )
            response = await request.send_request_async(handler)
            conversation_history.append({"role": "assistant", "content": response})
            responses.append(response)
        return responses
    
    @retry_on_failure(max_retries=3, delay=1, backoff=1.1)
    def process(self, prompt, **kwargs):
        if isinstance(prompt, str):
            return self._process_single(prompt, **kwargs)
        elif isinstance(prompt, list):
            return self._process_multiturn(prompt)
        else:
            raise ValueError("Prompt must be a string or a list of strings.")

    @retry_on_failure_async(max_retries=1, delay=1, backoff=1.1)
    async def process_async(self, prompt, **kwargs):
        if isinstance(prompt, str):
            return await self._process_single_async(prompt, **kwargs)
        elif isinstance(prompt, list):
            return await self._process_multiturn_async(prompt)
        else:
            raise ValueError("Prompt must be a string or a list of strings.")


async def apply_function_concurrently(
    func: Callable[..., Dict[str, Any]],
    elements: List[Dict[str, Any]],
    max_concurrency: int
) -> List[Dict[str, Any]]:
    semaphore = asyncio.Semaphore(max_concurrency)
    results = [None] * len(elements)

    async def bound_function(index: int, element: Dict[str, Any]):
        async with semaphore:
            result = await func(**element)
            results[index] = result

    tasks = [bound_function(index, element) for index, element in enumerate(elements)]
    await tqdm_asyncio.gather(*tasks)
    return results