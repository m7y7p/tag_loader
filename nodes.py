import folder_paths
import re
from pathlib import Path
import requests
import os
from typing import List, Dict, Any
from server import PromptServer
from aiohttp import web
from urllib.parse import quote_plus
import base64
import time

# Load environment variables
BACKBLAZE_BASE_URL = os.environ.get("BACKBLAZE_BASE_URL")
BACKBLAZE_KEY_ID = os.environ.get("BACKBLAZE_KEY_ID")
BACKBLAZE_APPLICATION_KEY = os.environ.get("BACKBLAZE_APPLICATION_KEY")

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
            if not BACKBLAZE_KEY_ID or not BACKBLAZE_APPLICATION_KEY:
                raise RuntimeError("Missing Backblaze credentials (BACKBLAZE_KEY_ID and BACKBLAZE_APPLICATION_KEY must be set)")

            # Build authorization header
            auth_str = f"{BACKBLAZE_KEY_ID}:{BACKBLAZE_APPLICATION_KEY}"
            b64_auth = base64.b64encode(auth_str.encode()).decode()

            headers = {
                "Authorization": f"Basic {b64_auth}"
            }

            url = f"{BACKBLAZE_BASE_URL}/{model_id}.safetensors"
            with requests.get(url, stream=True, headers=headers) as resp:
                if resp.status_code != 200:
                    print(f"[WanVideoLoraTagLoader] HTTP {resp.status_code} for {url}")
                    return False
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
