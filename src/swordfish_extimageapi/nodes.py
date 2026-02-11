import base64
import logging
import os
from collections import deque
from io import BytesIO
from inspect import cleandoc

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


def _dilate_mask(mask: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Perform morphological dilation using convolution.
    Dilation: pixel becomes 1 if ANY neighbor (where kernel=1) is 1.
    
    Args:
        mask: [B, C, H, W] tensor
        kernel: [1, 1, kH, kW] convolution kernel
    """
    padding = kernel.shape[-1] // 2
    output = F.conv2d(mask, kernel, padding=padding)
    return (output > 0).float()


def _erode_mask(mask: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Perform morphological erosion using convolution.
    Erosion: pixel remains 1 only if ALL neighbors (where kernel=1) are 1.
    
    Args:
        mask: [B, C, H, W] tensor
        kernel: [1, 1, kH, kW] convolution kernel
    """
    padding = kernel.shape[-1] // 2
    kernel_sum = kernel.sum()
    output = F.conv2d(mask, kernel, padding=padding)
    return (output >= kernel_sum).float()


def _binary_fill_holes(mask_np: np.ndarray) -> np.ndarray:
    """
    Fill holes in a binary mask using flood fill from border.
    Any region of 0s not connected to the border is considered a hole and filled with 1.
    
    Args:
        mask_np: 2D numpy array (H, W) with values 0 or 1
    
    Returns:
        Filled mask as float32 numpy array
    """
    h, w = mask_np.shape
    binary = (mask_np > 0).astype(np.uint8)
    
    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = binary
    
    visited = np.zeros((h + 2, w + 2), dtype=bool)
    queue = deque()
    
    for i in range(h + 2):
        if padded[i, 0] == 0 and not visited[i, 0]:
            queue.append((i, 0))
            visited[i, 0] = True
        if padded[i, w + 1] == 0 and not visited[i, w + 1]:
            queue.append((i, w + 1))
            visited[i, w + 1] = True
    
    for j in range(w + 2):
        if padded[0, j] == 0 and not visited[0, j]:
            queue.append((0, j))
            visited[0, j] = True
        if padded[h + 1, j] == 0 and not visited[h + 1, j]:
            queue.append((h + 1, j))
            visited[h + 1, j] = True
    
    while queue:
        y, x = queue.popleft()
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h + 2 and 0 <= nx < w + 2:
                if not visited[ny, nx] and padded[ny, nx] == 0:
                    visited[ny, nx] = True
                    queue.append((ny, nx))
    
    inner_visited = visited[1:-1, 1:-1]
    inner_binary = binary.astype(bool)
    result = (inner_binary | ~inner_visited).astype(np.float32)
    
    return result


def _tensor_to_pil_mask(tensor: torch.Tensor) -> Image.Image:
    """Convert a single mask tensor [H, W] to PIL Image (grayscale)."""
    arr = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def _pil_to_tensor_mask(pil_image: Image.Image) -> torch.Tensor:
    """Convert PIL Image (grayscale) to mask tensor [1, H, W]."""
    arr = np.array(pil_image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


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
                "model": (["gemini-2.5-flash-image", "gemini-3-pro-image-preview"],{"default": "gemini-2.5-flash-image"}),
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
       
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Gemini API key not set. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable."
            )
        client = genai.Client(api_key=api_key)
        pil_input = _comfyui_image_to_pil(image)
        seed_clamped = min(max(0, int(seed)), 2147483647)
        config = types.GenerateContentConfig(
            response_modalities=["IMAGE"],
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
        for part in (response.parts or []):
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
            reason = ""
            if response.candidates and len(response.candidates) > 0:
                fr = getattr(response.candidates[0], "finish_reason", None)
                if fr is not None:
                    reason = f" (finish_reason={fr}). Try a different prompt or image."
            if not reason:
                reason = " in the response."
            raise RuntimeError(f"Gemini API returned no image{reason}")
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


MAX_RESOLUTION = 16384


class SwordfishGrowMaskWithBlur:
    """
    Expands or contracts a mask with optional blur, interpolation, and hole filling.
    Custom implementation without kornia or scipy dependencies.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "expand": ("INT", {"default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1}),
                "incremental_expandrate": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "tapered_corners": ("BOOLEAN", {"default": True}),
                "flip_input": ("BOOLEAN", {"default": False}),
                "blur_radius": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100, "step": 0.1}),
                "lerp_alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "decay_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "fill_holes": ("BOOLEAN", {"default": False}),
            },
        }

    CATEGORY = "Swordfish/masking"
    RETURN_TYPES = ("MASK", "MASK",)
    RETURN_NAMES = ("mask", "mask_inverted",)
    FUNCTION = "expand_mask"
    DESCRIPTION = """
Expands or contracts a mask with optional blur and interpolation effects.

Parameters:
- mask: Input mask or mask batch
- expand: Pixels to expand (positive) or contract (negative)
- incremental_expandrate: Increase expand amount per frame
- tapered_corners: Use cross-shaped kernel (True) or square kernel (False)
- flip_input: Invert input mask before processing
- blur_radius: Gaussian blur radius (0 = no blur)
- lerp_alpha: Interpolation alpha between frames (1.0 = no interpolation)
- decay_factor: Decay factor for frame-to-frame accumulation (1.0 = no decay)
- fill_holes: Fill holes in the mask (slower)
"""

    def expand_mask(self, mask, expand, tapered_corners, flip_input, blur_radius,
                    incremental_expandrate, lerp_alpha, decay_factor, fill_holes=False):
        device = mask.device
        logger.info(
            "expand_mask started shape=%s expand=%s tapered_corners=%s blur_radius=%s fill_holes=%s",
            tuple(mask.shape), expand, tapered_corners, blur_radius, fill_holes,
        )

        if flip_input:
            mask = 1.0 - mask
            logger.debug("expand_mask: flipped input mask")

        grow_mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
        num_frames = grow_mask.shape[0]
        logger.debug("expand_mask: processing %s frame(s)", num_frames)

        out = []
        previous_output = None
        current_expand = expand

        if tapered_corners:
            kernel_data = torch.tensor([[0, 1, 0],
                                        [1, 1, 1],
                                        [0, 1, 0]], dtype=torch.float32)
        else:
            kernel_data = torch.tensor([[1, 1, 1],
                                        [1, 1, 1],
                                        [1, 1, 1]], dtype=torch.float32)
        
        kernel = kernel_data.unsqueeze(0).unsqueeze(0)
        logger.debug("expand_mask: kernel tapered_corners=%s", tapered_corners)

        for i, m in enumerate(grow_mask):
            output = m.unsqueeze(0).unsqueeze(0)
            kernel_device = kernel.to(output.device)
            
            steps = abs(round(current_expand))
            if steps > 0:
                op = "erode" if current_expand < 0 else "dilate"
                logger.debug("expand_mask: frame %s/%s %s steps=%s", i + 1, num_frames, op, steps)
                for _ in range(steps):
                    if current_expand < 0:
                        output = _erode_mask(output, kernel_device)
                    else:
                        output = _dilate_mask(output, kernel_device)

            output = output.squeeze(0).squeeze(0)

            if current_expand < 0:
                current_expand -= abs(incremental_expandrate)
            else:
                current_expand += abs(incremental_expandrate)
            
            if fill_holes:
                binary_mask = output > 0
                output_np = binary_mask.cpu().numpy()
                filled = _binary_fill_holes(output_np)
                output = torch.from_numpy(filled).to(device)
            
            if lerp_alpha < 1.0 and previous_output is not None:
                output = lerp_alpha * output + (1 - lerp_alpha) * previous_output
            
            if decay_factor < 1.0 and previous_output is not None:
                output = output + decay_factor * previous_output
                max_val = output.max()
                if max_val > 0:
                    output = output / max_val
            
            previous_output = output
            out.append(output.cpu())

        if blur_radius > 0:
            logger.debug("expand_mask: applying GaussianBlur radius=%s", blur_radius)
            blurred_out = []
            for tensor in out:
                pil_image = _tensor_to_pil_mask(tensor)
                pil_image = pil_image.filter(ImageFilter.GaussianBlur(blur_radius))
                blurred_tensor = _pil_to_tensor_mask(pil_image)
                blurred_out.append(blurred_tensor)
            result = torch.cat(blurred_out, dim=0)
            logger.info("expand_mask done result_shape=%s (blurred)", tuple(result.shape))
            return (result, 1.0 - result)
        else:
            result = torch.stack(out, dim=0)
            logger.info("expand_mask done result_shape=%s", tuple(result.shape))
            return (result, 1.0 - result)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "SwordfishImageAPI": SwordfishImageAPI,
    "SwordfishGrowMaskWithBlur": SwordfishGrowMaskWithBlur,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "SwordfishImageAPI": "Swordfish Image API",
    "SwordfishGrowMaskWithBlur": "Swordfish Grow Mask With Blur",
}
