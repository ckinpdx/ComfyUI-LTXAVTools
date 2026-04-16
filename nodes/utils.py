import os
import uuid
import numpy as np
import folder_paths
from PIL import Image


class PreviewImagePassthrough:
    """
    Displays a preview of the input image and passes it through unchanged.
    Useful inside loops where terminal PreviewImage nodes don't refresh per iteration.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("image",)
    FUNCTION      = "preview"
    OUTPUT_NODE   = True
    CATEGORY      = "LTXAVTools/utils"

    def preview(self, image):
        tmp_dir = folder_paths.get_temp_directory()
        results = []

        for i in range(image.shape[0]):
            arr = (image[i].numpy() * 255).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
            filename = f"preview_{uuid.uuid4().hex[:12]}.png"
            path = os.path.join(tmp_dir, filename)
            img.save(path)
            results.append({
                "filename": filename,
                "subfolder": "",
                "type": "temp",
            })

        return {"ui": {"images": results}, "result": (image,)}


NODE_CLASS_MAPPINGS = {
    "PreviewImagePassthrough": PreviewImagePassthrough,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PreviewImagePassthrough": "Preview Image Passthrough",
}
