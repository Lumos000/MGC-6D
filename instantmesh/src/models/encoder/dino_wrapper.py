# Copyright (c) 2023, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os

import torch.nn as nn
from transformers import ViTImageProcessor
from einops import rearrange
from .dino import ViTModel


class DinoWrapper(nn.Module):
    """
    Dino v1 wrapper using huggingface transformer implementation.
    """
    def __init__(self, model_name: str, freeze: bool = True):
        super().__init__()
        self.model, self.processor = self._build_dino(model_name)
        self.camera_embedder = nn.Sequential(
            nn.Linear(16, self.model.config.hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size, bias=True)
        )
        if freeze:
            self._freeze()

    def forward(self, image, camera):
        # image: [B, N, C, H, W]
        # camera: [B, N, D]
        # RGB image with [0,1] scale and properly sized
        if image.ndim == 5:
            image = rearrange(image, 'b n c h w -> (b n) c h w')
        dtype = image.dtype
        inputs = self.processor(
            images=image.float(), 
            return_tensors="pt", 
            do_rescale=False, 
            do_resize=False,
        ).to(self.model.device).to(dtype)
        # embed camera
        camera_embeddings = self.camera_embedder(camera)
        camera_embeddings = rearrange(camera_embeddings, 'b n d -> (b n) d')
        embeddings = camera_embeddings
        # This resampling of positional embedding uses bicubic interpolation
        outputs = self.model(**inputs, adaln_input=embeddings, interpolate_pos_encoding=True)
        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states

    def _freeze(self):
        print("======== Freezing DinoWrapper ========")
        self.model.eval()
        for name, param in self.model.named_parameters():
            param.requires_grad = False

    @staticmethod
    def _build_dino(model_name: str):
        """Load DINO from a verified local snapshot only.

        ``MGC6D_DINO_LOCAL_PATH`` may override the configured value.  When the
        configuration contains a Hugging Face repository id, resolve it from
        the existing cache with ``local_files_only=True``.  This prevents an
        accidental network request halfway through a long anchor run while
        keeping the formal launchers self-contained on a prepared machine.
        """
        local_path = os.environ.get("MGC6D_DINO_LOCAL_PATH", "").strip()
        resolved = local_path or model_name
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        expanded = os.path.expanduser(resolved)
        if os.path.exists(expanded):
            resolved = os.path.realpath(expanded)
        else:
            try:
                from huggingface_hub import snapshot_download

                resolved = snapshot_download(
                    repo_id=resolved,
                    local_files_only=True,
                )
            except Exception as exc:
                raise FileNotFoundError(
                    "InstantMesh DINO must be available offline. Set "
                    "MGC6D_DINO_LOCAL_PATH to a local facebook/dino-vitb16 "
                    f"snapshot (configured value: {model_name!r})."
                ) from exc
        model = ViTModel.from_pretrained(
            resolved,
            add_pooling_layer=False,
            local_files_only=True,
        )
        processor = ViTImageProcessor.from_pretrained(
            resolved,
            local_files_only=True,
        )
        return model, processor
