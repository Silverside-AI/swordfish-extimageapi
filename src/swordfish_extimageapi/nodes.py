import base64
import os
from io import BytesIO
from inspect import cleandoc

import numpy as np
import torch
from PIL import Image

from google import genai
from google.genai import types


def _comfyui_image_to_pil(image: np.ndarray) -> Image.Image:
    """Convert ComfyUI image tensor [N, H, W, C] to PIL Image (first frame, RGB)."""
    if hasattr(image, "numpy"):
        image = image.numpy()
    frame = np.asarray(image[0])
    if frame.dtype in (np.float32, np.float64):
        if frame.max() <= 1.0:
            frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
        else:
            frame = frame.clip(0, 255).astype(np.uint8)
    else:
        frame = frame.astype(np.uint8)
    if frame.shape[-1] == 4:
        return Image.fromarray(frame, mode="RGBA").convert("RGB")
    return Image.fromarray(frame, mode="RGB")


def _pil_to_comfyui_image(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL Image to ComfyUI image tensor [1, H, W, C] float 0-1."""
    arr = np.array(pil_image.convert("RGB"), dtype=np.float32) / 255.0
    return arr[np.newaxis, ...]


class SwordfishImageAPI:
    """
    A Swordfish Image API node

    Class methods
    -------------
    INPUT_TYPES (dict):
        Tell the main program input parameters of nodes.
    IS_CHANGED:
        optional method to control when the node is re executed.

    Attributes
    ----------
    RETURN_TYPES (`tuple`):
        The type of each element in the output tulple.
    RETURN_NAMES (`tuple`):
        Optional: The name of each output in the output tulple.
    FUNCTION (`str`):
        The name of the entry-point method. For example, if `FUNCTION = "execute"` then it will run Example().execute()
    OUTPUT_NODE ([`bool`]):
        If this node is an output node that outputs a result/image from the graph. The SaveImage node is an example.
        The backend iterates on these output nodes and tries to execute all their parents if their parent graph is properly connected.
        Assumed to be False if not present.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    execute(s) -> tuple || None:
        The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
        For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        """
            Return a dictionary which contains config for all input fields.
            Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
            Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
            The type can be a list for selection.

            Returns: `dict`:
                - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                - Value input_fields (`dict`): Contains input fields config:
                    * Key field_name (`string`): Name of a entry-point method's argument
                    * Value field_config (`tuple`):
                        + First value is a string indicate the type of field or a list for selection.
                        + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {
            "required": {
                "image": ("IMAGE", { "tooltip": "This is an image"}),
                "prompt": ("STRING", { "tooltip": "This is a prompt", "multiline": True}),
                "model": (["gemini-2.5-flash-image"],),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 2147483647,
                    "step": 1,
                    "display": "number"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("image_output_name",)
    DESCRIPTION = cleandoc(__doc__)
    FUNCTION = "call_api"

    #OUTPUT_NODE = False
    #OUTPUT_TOOLTIPS = ("",) # Tooltips for the output node

    CATEGORY = "Swordfish"

    def call_api(self, image, prompt, model, seed):
       
        api_key = os.environ.get(" ") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Gemini API key not set. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable."
            )
        client = genai.Client(api_key=api_key)
        pil_input = _comfyui_image_to_pil(image)
        seed_clamped = min(max(0, int(seed)), 2147483647)
        config = types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            seed=seed_clamped,
            image_config=types.ImageConfig(aspect_ratio="1:1")
        )
        try:
            response = client.models.generate_content(
                model=model,
                contents=[prompt, pil_input],
                config=config,
            )
        except Exception as e:
            raise RuntimeError(f"Gemini API call failed: {e}") from e
        out_tensor = None
        for part in response.parts:
            if part.inline_data is not None:
                blob = part.inline_data
                raw = (
                    blob.data
                    if isinstance(blob.data, bytes)
                    else base64.b64decode(blob.data)
                )
                pil_image = Image.open(BytesIO(raw)).convert("RGB")
                out_tensor = _pil_to_comfyui_image(pil_image)
                break
        if out_tensor is None:
            raise RuntimeError("Gemini API returned no image in the response.")
        return (torch.from_numpy(out_tensor),)

    """
        The node will always be re executed if any of the inputs change but
        this method can be used to force the node to execute again even when the inputs don't change.
        You can make this node return a number or a string. This value will be compared to the one returned the last time the node was
        executed, if it is different the node will be executed again.
        This method is used in the core repo for the LoadImage node where they return the image hash as a string, if the image hash
        changes between executions the LoadImage node is executed again.
    """
    #@classmethod
    #def IS_CHANGED(s, image, string_field, int_field, float_field, print_to_screen):
    #    return ""


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "SwordfishImageAPI": SwordfishImageAPI
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "SwordfishImageAPI": "Swordfish Image API"
}
