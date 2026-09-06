# -*- coding: utf-8 -*-
"""
Optional SAM silhouette backend (segment-anything checkpoint or HF transformers
mask-generation pipeline). Holds the loaded model between calls.

Settings are read through the injected ``settings`` object (QSettings or any
object with ``.value(key, default)``), so this module stays QGIS-free.
"""

import os

import cv2
import numpy as np

from ...log import log_exception
from .segment import detect_center_circle_mask, mask_selection_score, select_primary_component, smooth_mask_edges


class SamBackend:
    def __init__(self, settings):
        self.settings = settings
        self._sam_model = None
        self._sam_cache_key = None
        self._sam_hf_generator = None
        self._sam_hf_cache_key = None

    def configured(self):
        """True when settings point at a usable SAM model (HF id or checkpoint)."""
        model_type = str(self.settings.value('ArcheoGlyph/sam_model_type', 'vit_b')).strip() or "vit_b"
        if model_type.lower().startswith("hf:"):
            return True
        checkpoint = str(self.settings.value('ArcheoGlyph/sam_checkpoint_path', '')).strip()
        return bool(checkpoint) and os.path.exists(checkpoint)

    def get_mask(self, bgr_img):
        """
        Optional SAM backend.
        Supports:
        - SAM1 (segment-anything + local checkpoint)
        - SAM2.1/SAM3 (transformers mask-generation via HF model ID)
        """
        model_type = str(self.settings.value('ArcheoGlyph/sam_model_type', 'vit_b')).strip() or "vit_b"
        if model_type.lower().startswith("hf:"):
            model_id = model_type[3:].strip()
            if not model_id:
                model_id = "facebook/sam2.1-hiera-small"
            return self._get_mask_hf(bgr_img, model_id)

        checkpoint = str(self.settings.value('ArcheoGlyph/sam_checkpoint_path', '')).strip()
        if not checkpoint:
            return None

        if not os.path.exists(checkpoint):
            return None

        try:
            import torch
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        except Exception as e:
            log_exception("SAM 1 needs torch and segment-anything", e)
            return None

        device = "cuda" if torch.cuda.is_available() else "cpu"
        cache_key = (checkpoint, model_type, device)

        try:
            if self._sam_model is None or self._sam_cache_key != cache_key:
                if model_type not in sam_model_registry:
                    return None
                sam = sam_model_registry[model_type](checkpoint=checkpoint)
                sam.to(device=device)
                self._sam_model = sam
                self._sam_cache_key = cache_key

            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            mask_generator = SamAutomaticMaskGenerator(
                self._sam_model,
                points_per_side=24,
                pred_iou_thresh=0.88,
                stability_score_thresh=0.92,
                min_mask_region_area=400,
            )
            masks = mask_generator.generate(rgb_img)
            if not masks:
                return None

            h, w = bgr_img.shape[:2]
            cx, cy = w * 0.5, h * 0.5
            best_mask = None
            best_score = -1.0

            for item in masks:
                seg = item.get("segmentation")
                area = float(item.get("area", 0))
                if seg is None or area <= 0:
                    continue

                ys, xs = np.where(seg)
                if len(xs) == 0:
                    continue

                if area < (h * w * 0.015):
                    continue

                mx = float(np.mean(xs))
                my = float(np.mean(ys))
                center_dist = ((mx - cx) ** 2 + (my - cy) ** 2) / max(1.0, (w * w + h * h))

                bbox = item.get("bbox", [0, 0, w, h])
                bbox_area = max(1.0, float(bbox[2]) * float(bbox[3]))
                fill_ratio = area / bbox_area

                score = area * (1.0 - min(center_dist, 0.95)) * (0.65 + 0.35 * min(fill_ratio, 1.0))
                if score > best_score:
                    best_score = score
                    best_mask = seg

            if best_mask is None:
                return None

            target_mask = np.zeros((h, w), dtype=np.uint8)
            target_mask[best_mask] = 255

            kernel = np.ones((3, 3), np.uint8)
            target_mask = cv2.morphologyEx(target_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            target_mask = cv2.morphologyEx(target_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
            target_mask = select_primary_component(target_mask)
            target_mask = smooth_mask_edges(target_mask)
            return target_mask
        except Exception as e:
            log_exception("SAM 1 segmentation failed", e)
            return None

    def _get_mask_hf(self, bgr_img, model_id):
        """
        HF/Transformers SAM2.1/SAM3 path.
        Expects model_id like 'facebook/sam2.1-hiera-small' or 'facebook/sam3-hiera-large'.
        """
        model_id = (model_id or "").strip()
        if not model_id:
            return None

        try:
            import torch
            from PIL import Image
            from transformers import pipeline
        except Exception as e:
            log_exception("SAM 2/3 needs torch, Pillow and transformers", e)
            return None

        device = 0 if torch.cuda.is_available() else -1
        cache_key = (model_id, device)
        try:
            if self._sam_hf_generator is None or self._sam_hf_cache_key != cache_key:
                self._sam_hf_generator = pipeline(
                    task="mask-generation",
                    model=model_id,
                    device=device,
                )
                self._sam_hf_cache_key = cache_key
        except Exception as e:
            log_exception(f"Could not load the SAM model {model_id}", e)
            self._sam_hf_generator = None
            self._sam_hf_cache_key = None
            return None

        try:
            rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            output = self._sam_hf_generator(image)
        except Exception as e:
            log_exception("SAM 2/3 segmentation failed", e)
            return None

        # Pipeline usually returns a list with one dict per image.
        payload = output
        if isinstance(output, list) and output:
            payload = output[0]
        if not isinstance(payload, dict):
            return None

        raw_masks = payload.get("masks", None)
        raw_scores = payload.get("scores", None)

        def _as_mask_list(value):
            if value is None:
                return []
            if isinstance(value, (list, tuple)):
                return list(value)
            try:
                if hasattr(value, "detach"):
                    arr_v = value.detach().cpu().numpy()
                else:
                    arr_v = np.asarray(value)
            except Exception as e:
                # SAM then contributes nothing and Auto Trace falls back
                # silently, which reads as "SAM did not help" rather than
                # "SAM could not be read".
                log_exception("Could not read the masks SAM returned", e)
                return []
            if arr_v is None:
                return []
            if arr_v.ndim == 2:
                return [arr_v]
            if arr_v.ndim >= 3:
                return [arr_v[i] for i in range(arr_v.shape[0])]
            return []

        def _as_score_list(value):
            if value is None:
                return []
            if isinstance(value, (list, tuple)):
                out = []
                for sv in value:
                    try:
                        out.append(float(sv))
                    except Exception:
                        continue
                return out
            try:
                if hasattr(value, "detach"):
                    arr_s = value.detach().cpu().numpy()
                else:
                    arr_s = np.asarray(value)
            except Exception as e:
                log_exception("Could not read the mask scores SAM returned", e)
                return []
            if arr_s is None:
                return []
            arr_s = np.asarray(arr_s).reshape(-1)
            out = []
            for sv in arr_s:
                try:
                    out.append(float(sv))
                except Exception:
                    continue
            return out

        masks = _as_mask_list(raw_masks)
        scores = _as_score_list(raw_scores)
        if len(masks) == 0:
            return None

        h, w = bgr_img.shape[:2]
        total = float(max(1, h * w))
        circle_mask = None
        circle_area = 0.0
        try:
            gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
            circle_mask = detect_center_circle_mask(gray)
            if circle_mask is not None:
                circle_area = float(np.count_nonzero(circle_mask))
                if circle_area < (total * 0.04):
                    circle_mask = None
                    circle_area = 0.0
        except Exception:
            circle_mask = None
            circle_area = 0.0

        best = None
        best_score = -1.0

        for idx, mask_item in enumerate(masks):
            arr = None
            try:
                if hasattr(mask_item, "detach"):
                    arr = mask_item.detach().cpu().numpy()
                else:
                    arr = np.asarray(mask_item)
            except Exception:
                continue

            if arr is None:
                continue
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim == 3 and arr.shape[-1] == 1:
                arr = arr[:, :, 0]
            if arr.ndim != 2:
                continue

            if arr.shape[0] != h or arr.shape[1] != w:
                arr = cv2.resize(arr.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)

            if arr.dtype == np.bool_:
                m = arr.astype(np.uint8) * 255
            else:
                thr = 0.5 if float(np.max(arr)) <= 1.0 else 127.0
                m = (arr > thr).astype(np.uint8) * 255

            if int(np.count_nonzero(m)) < max(80, int(h * w * 0.002)):
                continue

            m = select_primary_component(m)
            m = smooth_mask_edges(m)

            m_area = float(np.count_nonzero(m))
            if circle_mask is not None and circle_area > 0.0:
                overlap_circle = float(np.count_nonzero(cv2.bitwise_and(m, circle_mask))) / max(1.0, circle_area)
                # Reject tiny center fragments when a dominant round silhouette is detected.
                if m_area < (circle_area * 0.12) and overlap_circle < 0.20:
                    continue
            else:
                overlap_circle = 0.0

            quality = mask_selection_score(bgr_img, m)
            if idx < len(scores):
                try:
                    quality += (0.06 * float(scores[idx]))
                except Exception:
                    pass
            if circle_mask is not None and circle_area > 0.0:
                area_ratio_to_circle = m_area / max(1.0, circle_area)
                area_match = max(0.0, 1.0 - abs(area_ratio_to_circle - 1.0))
                quality += (0.14 * overlap_circle) + (0.08 * area_match)
            if quality > best_score:
                best_score = quality
                best = m

        return best
