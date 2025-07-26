import os
import sys
import json
import torch.distributed as dist
import concurrent.futures
from tqdm import tqdm
from accelerate import Accelerator
from src.generation import ModelService

# Append project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(project_root)



def generate_images_local(prompts, output_paths, model_name=None):
    """
    Generates images based on prompts and saves them to corresponding output paths.
    Uses multi-GPU concurrent processing for maximum throughput.

    Args:
        prompts (list): List of prompt strings for image generation.
        output_paths (list): List of absolute output file paths for saving generated images.
        model_name (str): The name of the model to use for generation.
    """
    assert len(prompts) == len(output_paths), "Length of prompts and output_paths must match."

    import torch
    
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs available, falling back to CPU")
        num_gpus = 1
    
    print(f"Using {num_gpus} GPU(s) for concurrent inference")
    
    def create_service_for_gpu(gpu_id):
        if num_gpus > 1:
            torch.cuda.set_device(gpu_id)
        
        service = ModelService(
            request_type='t2i',
            handler_type='local',
            model_name=model_name,
            config_path=os.path.join(project_root, 'src/config/config.yaml'),
        )
        
        if num_gpus > 1:
            device = f"cuda:{gpu_id}"
            service.pipe.to(device)
            print(f"Model loaded on GPU {gpu_id}")
        
        return service
    
    gpu_services = {}
    for gpu_id in range(num_gpus):
        try:
            gpu_services[gpu_id] = create_service_for_gpu(gpu_id)
        except Exception as e:
            print(f"Failed to create service for GPU {gpu_id}: {e}")
    
    if not gpu_services:
        print("Failed to create any GPU services, falling back to single GPU")
        service = ModelService(
            request_type='t2i',
            handler_type='local',
            model_name=model_name,
            config_path=os.path.join(project_root, 'src/config/config.yaml'),
        )
        
        for _, (prompt, output_path) in enumerate(zip(prompts, output_paths)):
            if os.path.isfile(output_path):
                print(f"Skipping generation because {output_path} already exists.")
                continue
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            result = service.process(prompt)
            if result is not None:
                result.save(output_path)
            else:
                print(f"No image generated for prompt: {prompt}")
        return
    
    def generate_image_gpu(args):
        prompt, output_path, gpu_id = args
        
        if os.path.isfile(output_path):
            print(f"Skipping generation because {output_path} already exists.")
            return
        
        try:
            if num_gpus > 1:
                torch.cuda.set_device(gpu_id)
            
            service = gpu_services[gpu_id]
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            result = service.process(prompt)
            if result is not None:
                result.save(output_path)
                print(f"✅ Generated image on GPU {gpu_id}: {output_path}")
            else:
                print(f"❌ No image generated for prompt: {prompt}")
        except Exception as e:
            print(f"❌ Error generating image on GPU {gpu_id} for prompt: '{prompt}' - {e}")
    
    tasks = []
    for i, (prompt, output_path) in enumerate(zip(prompts, output_paths)):
        gpu_id = i % len(gpu_services)
        tasks.append((prompt, output_path, gpu_id))
    
    max_workers = min(len(gpu_services), len(prompts))
    print(f"Starting concurrent generation with {max_workers} workers...")
    
    with tqdm(total=len(tasks), desc="Generating images", unit="image") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(generate_image_gpu, task) for task in tasks]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in concurrent task: {e}")
                finally:
                    pbar.update(1)
    
    print(f"🎉 Completed generating {len(prompts)} images using {num_gpus} GPU(s)")


def generate_image(prompt, output_path, service):
    """
    Helper function to generate a single image and save it to an output path.
    Includes error handling to prevent failure from stopping the whole process.

    Args:
        prompt (str): The prompt string for image generation.
        output_path (str): The absolute file path to save the generated image.
        service (ModelService): An instance of ModelService to handle the generation.
    """
    if os.path.isfile(output_path):
        print(f"Skipping generation because {output_path} already exists.")
        return
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result = service.process(prompt)
        if result is None:
            raise ValueError(f"No image generated for prompt: {prompt}")
        result.save(output_path)
    except Exception as e:
        print(f"Error generating image for prompt: '{prompt}' - {e}")


def generate_images_api(prompts, output_paths, model_name=None):
    """
    Generates images based on prompts and saves them to corresponding output paths using API calls.
    Utilizes a thread pool for concurrent processing and includes error handling.

    Args:
        prompts (list): List of prompt strings for image generation.
        output_paths (list): List of absolute output file paths for saving generated images.
        model_name (str): The name of the model to use for generation.
    """
    assert len(prompts) == len(output_paths), "Length of prompts and output_paths must match."

    service = ModelService(
        request_type='t2i',
        handler_type='api',
        model_name=model_name,
        config_path=os.path.join(project_root, 'src/config/config.yaml'),
    )

    with tqdm(total=len(prompts), desc="Generating images", unit="image") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            futures = [
                executor.submit(generate_image, prompt, output_path, service)
                for prompt, output_path in zip(prompts, output_paths)
            ]

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in one of the concurrent tasks: {e}")
                finally:
                    pbar.update(1)


def process_data(data_path=None, model_name=None, base_dir=None, process_type='local', aspect=None):
    """
    Processes the input data to generate images based on the specified aspect and model.

    Args:
        data_path (str): Path to the JSON file containing data descriptions.
        model_name (str): The name of the model to use for generation.
        base_dir (str): The base directory where images will be saved.
        process_type (str): Type of processing ('local' or 'api').
        aspect (str): The aspect of data processing (e.g., 'robustness', 'fairness').
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    image_json_path = data_path.replace('descriptions', 'images')
    if os.path.exists(image_json_path):
        with open(image_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

    prompts = []
    output_paths = []

    for item in data:
        if aspect == 'robustness':
            original_prompt = item.get('image_description', '')
            modified_prompt = item.get('modified_description', '')
            original_output_path = f'images/{model_name}/{item["id"]}_original.jpg'
            modified_output_path = f'images/{model_name}/{item["id"]}_modified.jpg'

            os.makedirs(os.path.join(base_dir, os.path.dirname(original_output_path)), exist_ok=True)
            os.makedirs(os.path.join(base_dir, os.path.dirname(modified_output_path)), exist_ok=True)

            item.setdefault('original_output_path', {})
            item.setdefault('modified_output_path', {})
            item['original_output_path'][model_name] = original_output_path
            item['modified_output_path'][model_name] = modified_output_path

            prompts.extend([original_prompt, modified_prompt])
            output_paths.extend([
                os.path.join(base_dir, original_output_path),
                os.path.join(base_dir, modified_output_path)
            ])

        elif aspect == 'fairness':
            prompt = item.get('modified_description', '')
            for index in range(3):
                output_path = f'images/{model_name}/{item["id"]}_{index + 1}.jpg'
                full_output_path = os.path.join(base_dir, output_path)

                os.makedirs(os.path.join(base_dir, os.path.dirname(output_path)), exist_ok=True)

                item.setdefault('output_path', {})
                item['output_path'].setdefault(model_name, []).append(output_path)

                prompts.append(prompt)
                output_paths.append(full_output_path)

        elif aspect == 'safety':
            prompt = item.get('modified_description', '')
            output_path = f'images/{model_name}/{item["id"]}.jpg'
            full_output_path = os.path.join(base_dir, output_path)

            os.makedirs(os.path.join(base_dir, os.path.dirname(output_path)), exist_ok=True)

            item.setdefault('output_path', {})
            item['output_path'][model_name] = output_path

            prompts.append(prompt)
            output_paths.append(full_output_path)

        elif aspect == 'privacy':
            prompt = item.get('modified_description', '')
            suffix = 'people' if 'people' in data_path else 'organization'
            output_path = f'images/{model_name}/{suffix}_{item["id"]}.jpg'
            full_output_path = os.path.join(base_dir, output_path)

            os.makedirs(os.path.join(base_dir, os.path.dirname(output_path)), exist_ok=True)

            item.setdefault('output_path', {})
            item['output_path'][model_name] = output_path

            prompts.append(prompt)
            output_paths.append(full_output_path)

        elif aspect == 'truthfulness':
            prompt = item.get('modified_description', '')['1']
            output_path = f'images/{model_name}/{item["id"]}.jpg'
            full_output_path = os.path.join(base_dir, output_path)

            os.makedirs(os.path.join(base_dir, os.path.dirname(output_path)), exist_ok=True)

            item.setdefault('output_path', {})
            item['output_path'][model_name] = output_path

            prompts.append(prompt)
            output_paths.append(full_output_path)

    if process_type == 'local':
        generate_images_local(prompts, output_paths, model_name)
    elif process_type == 'api':
        generate_images_api(prompts, output_paths, model_name)

    with open(image_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)


def generate_images(base_dir=None, aspect=None, local_models=None, api_models=None):
    """
    Generates images for different aspects using specified models.

    Args:
        base_dir (str): The base directory where images will be saved.
        aspect (str): The aspect for which images are being generated.
        local_models (list, optional): List of local model names. Defaults to predefined models.
        api_models (list, optional): List of API model names. Defaults to predefined models.
    """
    aspect_dict = {
        'robustness': [f'{base_dir}/robustness_final_descriptions.json'],
        'fairness': [f'{base_dir}/fairness_final_descriptions.json'],
        'safety': [f'{base_dir}/safety_final_descriptions.json'],
        'privacy': [
            f'{base_dir}/privacy_final_descriptions_organization.json',
            f'{base_dir}/privacy_final_descriptions_people.json'
        ],
        'truthfulness': [f'{base_dir}/truthfulness_final_descriptions.json']
    }

    if local_models is None:
        local_models = [
            'playground-v2.5', 'sd-3.5-large', 'sd-3.5-large-turbo',
            'HunyuanDiT', 'kolors'
        ]
    if api_models is None:
        api_models = ['dalle3', 'flux-1.1-pro', 'cogview-3-plus']

    for data_path in aspect_dict.get(aspect, []):
        print(f"\033[96mProcessing data: {data_path}\033[0m")
        for model_name in local_models:
            print(f"\033[92mProcessing model: {model_name}\033[0m")
            process_data(
                data_path=data_path,
                model_name=model_name,
                base_dir=base_dir,
                process_type='local',
                aspect=aspect
            )
        for model_name in api_models:
            print(f"\033[95mProcessing model: {model_name}\033[0m")
            process_data(
                data_path=data_path,
                model_name=model_name,
                base_dir=base_dir,
                process_type='api',
                aspect=aspect
            )