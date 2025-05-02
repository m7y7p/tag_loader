import folder_paths
import re
from pathlib import Path
import requests
import os
from typing import List, Dict, Any
from server import PromptServer
from aiohttp import web
import base64

# Load environment variables
BACKBLAZE_KEY_ID = os.environ.get("BACKBLAZE_KEY_ID")
BACKBLAZE_APPLICATION_KEY = os.environ.get("BACKBLAZE_APPLICATION_KEY")
BACKBLAZE_BUCKET_NAME = os.environ.get("BACKBLAZE_BUCKET_NAME")

class WanVideoLoraTagLoader:
    def __init__(self):
        self.tag_pattern = r"<lora:(\d+)(?::([\d\.]+))?>"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
            },
            "optional": {
                "prev_lora":("WANVIDLORA", {"default": None, "tooltip": "For loading multiple LoRAs"}),
                "blocks":("SELECTEDBLOCKS", ),
                "low_mem_load": ("BOOLEAN", {"default": False, "tooltip": "Load with less VRAM usage"}),
            }
        }

    RETURN_TYPES = ("WANVIDLORA", "STRING")
    RETURN_NAMES = ("lora", "text")
    FUNCTION = "parse_and_load"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Parses <lora:model_id:strength> tags in the prompt and loads WAN-compatible LoRAs from a private Backblaze bucket."

    def parse_and_load(self, text, prev_lora=None, blocks=None, low_mem_load=False):
        founds = re.findall(self.tag_pattern, text)
        print(f"[WanVideoLoraTagLoader] Found LoRA tags: {founds}")

        loras_list = []

        if prev_lora is not None:
            loras_list.extend(prev_lora)

        lora_files = folder_paths.get_filename_list("loras")
        lora_files_lower = [f.lower() for f in lora_files]
        lora_dir = folder_paths.get_folder_paths("loras")[0]

        for model_id, strength in founds:
            model_filename = f"{model_id}.safetensors"
            model_path = Path(lora_dir) / model_filename

            if not model_path.exists():
                print(f"[WanVideoLoraTagLoader] {model_filename} not found. Downloading from Backblaze...")
                success = self.download_lora_from_backblaze(model_id, model_path)
                if not success:
                    print(f"[WanVideoLoraTagLoader] Failed to download {model_id} from Backblaze")
                    continue

            try:
                weight = float(strength) if strength else 1.0
            except ValueError:
                weight = 1.0

            lora_entry = {
                "path": str(model_path),
                "strength": weight,
                "name": model_id,
                "blocks": blocks,
                "low_mem_load": low_mem_load,
            }
            print(f"[WanVideoLoraTagLoader] Using LoRA: {lora_entry}")
            loras_list.append(lora_entry)

        cleaned_text = re.sub(self.tag_pattern, "", text).strip()
        return (loras_list, cleaned_text)

    def download_lora_from_backblaze(self, model_id: str, save_path: Path) -> bool:
        try:
            if not BACKBLAZE_KEY_ID or not BACKBLAZE_APPLICATION_KEY or not BACKBLAZE_BUCKET_NAME:
                raise RuntimeError("Missing Backblaze credentials or bucket name")

            # Step 1: Authorize account
            auth_str = f"{BACKBLAZE_KEY_ID}:{BACKBLAZE_APPLICATION_KEY}"
            auth_encoded = base64.b64encode(auth_str.encode()).decode()
            headers = {"Authorization": f"Basic {auth_encoded}"}

            auth_response = requests.get("https://api.backblazeb2.com/b2api/v3/b2_authorize_account", headers=headers)
            auth_response.raise_for_status()
            auth_data = auth_response.json()

            api_url = auth_data['apiInfo']['storageApi']['apiUrl']
            download_url = auth_data['apiInfo']['storageApi']['downloadUrl']
            account_auth_token = auth_data['authorizationToken']
            account_id = auth_data['accountId']

            # Step 2: Get bucket ID
            buckets_url = f"{api_url}/b2api/v3/b2_list_buckets"
            buckets_payload = {"accountId": account_id, "bucketName": BACKBLAZE_BUCKET_NAME}
            buckets_headers = {"Authorization": account_auth_token}

            buckets_resp = requests.post(buckets_url, json=buckets_payload, headers=buckets_headers)
            buckets_resp.raise_for_status()
            bucket_id = buckets_resp.json()['buckets'][0]['bucketId']

            # Step 3: Get download authorization
            authz_url = f"{api_url}/b2api/v3/b2_get_download_authorization"
            authz_payload = {
                "bucketId": bucket_id,
                "fileNamePrefix": f"{model_id}.safetensors",
                "validDurationInSeconds": 3600
            }
            authz_resp = requests.post(authz_url, json=authz_payload, headers=buckets_headers)
            authz_resp.raise_for_status()
            download_auth_token = authz_resp.json()['authorizationToken']

            # Step 4: Download file with token
            file_url = f"{download_url}/file/{BACKBLAZE_BUCKET_NAME}/{model_id}.safetensors?Authorization={download_auth_token}"
            with requests.get(file_url, stream=True) as resp:
                resp.raise_for_status()
                with open(save_path, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f"[WanVideoLoraTagLoader] Downloaded {model_id}.safetensors to {save_path}")
            return True

        except Exception as e:
            print(f"[WanVideoLoraTagLoader] Exception downloading {model_id}: {e}")
            return False

class WanVideoDynamicTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "t5": ("WANTEXTENCODER",),
                "positive_prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True})
            },
            "optional": {
                "override_positive": ("STRING", {"multiline": True, "default": ""}),
                "override_negative": ("STRING", {"multiline": True, "default": ""}),
                "force_offload": ("BOOLEAN", {"default": True}),
                "model_to_offload": ("WANVIDEOMODEL", {"tooltip": "Model to move to offload_device before encoding"})
            }
        }

    RETURN_TYPES = ("WANVIDEOTEXTEMBEDS", )
    RETURN_NAMES = ("text_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Encodes text prompts into text embeddings. Supports prompt overrides via input nodes."

    def process(self, t5, positive_prompt, negative_prompt, override_positive="", override_negative="", force_offload=True, model_to_offload=None):
        import torch
        import comfy.model_management as mm
        import comfy.utils as log

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        if model_to_offload is not None:
            log.info(f"Moving video model to {offload_device}")
            model_to_offload.model.to(offload_device)
            mm.soft_empty_cache()

        encoder = t5["model"]
        dtype = t5["dtype"]

        if override_positive.strip():
            positive_prompt = override_positive
        if override_negative.strip():
            negative_prompt = override_negative

        positive_prompts_raw = [p.strip() for p in positive_prompt.split('|')]
        positive_prompts = []
        all_weights = []

        for p in positive_prompts_raw:
            cleaned_prompt, weights = self.parse_prompt_weights(p)
            positive_prompts.append(cleaned_prompt)
            all_weights.append(weights)

        encoder.model.to(device)

        with torch.autocast(device_type=mm.get_autocast_device(device), dtype=dtype, enabled=True):
            context = encoder(positive_prompts, device)
            context_null = encoder([negative_prompt], device)

            for i, weights in enumerate(all_weights):
                for text, weight in weights.items():
                    log.info(f"Applying weight {weight} to prompt: {text}")
                    if len(weights) > 0:
                        context[i] = context[i] * weight

        if force_offload:
            encoder.model.to(offload_device)
            mm.soft_empty_cache()

        return ({"prompt_embeds": context, "negative_prompt_embeds": context_null},)

    def parse_prompt_weights(self, prompt):
        import re
        pattern = r'\((.*?):([\d\.]+)\)'
        matches = re.findall(pattern, prompt)
        cleaned_prompt = prompt
        weights = {}

        for match in matches:
            text, weight = match
            orig_text = f"({text}:{weight})"
            cleaned_prompt = cleaned_prompt.replace(orig_text, text)
            weights[text] = float(weight)

        return cleaned_prompt, weights

NODE_CLASS_MAPPINGS = {
    "WanVideoLoraTagLoader": WanVideoLoraTagLoader,
    "WanVideoDynamicTextEncode": WanVideoDynamicTextEncode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoLoraTagLoader": "WanVideo Auto-Load LoRA Tags",
    "WanVideoDynamicTextEncode": "WanVideo Dynamic Text Encoder"
}
