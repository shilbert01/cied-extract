"""
OCR engine wrapping PaddleOCR-VL-1.6 via HuggingFace Transformers.

The model is downloaded from HuggingFace on first use (~1.8 GB).
GPU (CUDA) is used if available; falls back to CPU automatically.
"""
import os
import torch
from PIL import Image
from pathlib import Path
from typing import Optional
import pdf2image


MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.6"

TASK_PROMPTS = {
    "ocr":     "OCR:",
    "table":   "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart":   "Chart Recognition:",
    "spotting":"Spotting:",
    "seal":    "Seal Recognition:",
}


class OCREngine:
    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        self.model_id = model_id
        # FORCE_DEVICE env var lets docker-compose.cpu.yml override without CLI flag
        self.device = device or os.environ.get("FORCE_DEVICE") or \
            ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        from transformers import AutoProcessor, AutoModelForImageTextToText

        print(f"Loading {self.model_id} on {self.device} …")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            dtype=dtype,
            cache_dir=self.cache_dir,
        ).to(self.device).eval()
        self._processor = AutoProcessor.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
        )
        print("Model ready.")

    def ocr_image(self, image: Image.Image, task: str = "ocr", max_new_tokens: int = 1280) -> str:
        self._load()
        prompt = TASK_PROMPTS.get(task, "OCR:")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": prompt},
                ],
            }
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                repetition_penalty=1.2,
            )

        # Decode only the newly generated tokens
        result = self._processor.decode(
            outputs[0][inputs["input_ids"].shape[-1]: -1],
            skip_special_tokens=True,
        )
        return result

    def ocr_pdf(
        self,
        pdf_path: str | Path,
        dpi: int = 200,
        task: str = "ocr",
        max_new_tokens: int = 1280,
    ) -> list[str]:
        """Convert every page of a PDF to an image, then run OCR on each page."""
        images = pdf2image.convert_from_path(str(pdf_path), dpi=dpi)
        results = []
        for i, img in enumerate(images, start=1):
            print(f"  OCR page {i}/{len(images)} …")
            text = self.ocr_image(img, task=task, max_new_tokens=max_new_tokens)
            results.append(text)
        return results
