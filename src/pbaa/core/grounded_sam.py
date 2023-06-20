from json import dump
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import supervision as sv
import torch
import torchvision
import wget
from groundingdino.config.GroundingDINO_SwinT_OGC import __file__ as grounding_dino_config_path
from groundingdino.util.inference import Model
from loguru import logger
from segment_anything import SamPredictor, sam_model_registry

HQ = False
GROUNDING_DINO_CHECKPOINT_PATH = Path("groundingdino_swint_ogc.pth")
GROUNDING_DINO_DOWNLOAD_URL = (
    "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
)

if HQ:
    SAM_CHECKPOINT_PATH = Path("sam_hq_vit_h.pth")
    SAM_DOWNLOAD_URL = "https://blueclairvoyancestorage.blob.core.windows.net/package/sam_hq_vit_h.pth"
else:
    SAM_CHECKPOINT_PATH = Path("sam_vit_h_4b8939.pth")
    SAM_DOWNLOAD_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"


class PBAA:
    def __init__(self):
        self.model_init()
        self.grounding_dino_model = None
        self.sam_predictor = None
        self.box_annotator = sv.BoxAnnotator()
        self.mask_annotator = sv.MaskAnnotator()

    def __call__(
        self, src, _prompt, annot_format=None, box_threshold=0.25, nms_threshold=0.8, save=None, output_dir="outputs"
    ):
        return self.inference(src, _prompt, annot_format, box_threshold, nms_threshold, save, output_dir)

    @staticmethod
    def model_init():
        # GroundingDINO config and checkpoint
        if not GROUNDING_DINO_CHECKPOINT_PATH.exists():
            logger.warning("GROUNDING_DINO_CHECKPOINT doesn't exist")
            logger.info("Start download")
            wget.download(GROUNDING_DINO_DOWNLOAD_URL)
        if not SAM_CHECKPOINT_PATH.exists():
            logger.warning("SAM_CHECKPOINT doesn't exist")
            logger.info("Start download")
            wget.download(SAM_DOWNLOAD_URL)

    def segment(self, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        self.sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = self.sam_predictor.predict(box=box, multimask_output=True)
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)

    def inference(
        self, src, prompt, annot_format=None, box_threshold=0.25, nms_threshold=0.8, save=None, output_dir="outputs"
    ):
        if (src is None) or (not prompt):
            s = ", ".join(["src" if src is None else "", "prompt" if prompt is None else ""])
            logger.warning(f"{s} required")
            return None, None, None, None

        dst = Path(output_dir)
        dst.mkdir(parents=True, exist_ok=True)

        annotated_frame, annotated_image, annotated_mask, json_data, detections = None, None, None, {}, []

        if annot_format is None:
            annot_format = ["Rectangle", "Polygon", "Mask"]
        if save is None:
            save = False
        rectangle, polygon, mask = (i in annot_format for i in ["Rectangle", "Polygon", "Mask"])
        extension = ".jpg"

        if type(src) == str:
            image = cv2.imread(src)
            src = Path(src)
            output = dst / src.stem
            extension = src.suffix
        else:
            image = src
            output = dst / str(uuid4())

        if type(prompt) == str:
            __prompt = prompt.split(",")
            prompt = {}
            for p in __prompt:
                split_prompt = p.split(":")
                if len(split_prompt) == 1:
                    v = split_prompt[0].strip()
                    k = v
                else:
                    k, v = map(str.strip, p.split(":"))
                prompt[k] = v

        # Predict classes and hyper-param for GroundingDINO
        _prompt = [*map(str.lower, prompt.keys())]

        if any([rectangle, polygon, mask]):
            # Building GroundingDINO inference model
            if self.grounding_dino_model is None:
                self.grounding_dino_model = Model(
                    model_config_path=grounding_dino_config_path, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH
                )

            # detect objects
            detections = self.grounding_dino_model.predict_with_classes(
                image=image, classes=_prompt, box_threshold=box_threshold, text_threshold=box_threshold
            )

            nms_idx = (
                torchvision.ops.nms(
                    torch.from_numpy(detections.xyxy), torch.from_numpy(detections.confidence), nms_threshold
                )
                .numpy()
                .tolist()
            )

            detections.xyxy = detections.xyxy[nms_idx]
            detections.confidence = detections.confidence[nms_idx]
            detections.class_id = detections.class_id[nms_idx]

        if rectangle:
            # annotate image with detections
            labels = [f"{prompt[_prompt[class_id]]} {confidence:0.2f}" for _, _, confidence, class_id, _ in detections]
            annotated_frame = self.box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

        if any([polygon, mask]):
            # Building SAM Model and SAM Predictor
            if self.sam_predictor is None:
                self.sam_predictor = SamPredictor(
                    sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
                )

            # convert detections to masks
            detections.mask = self.segment(image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB), xyxy=detections.xyxy)

            polys = []
            for _mask in detections.mask:
                canvas = np.zeros((image.shape[:2]), dtype=np.uint8)
                canvas[_mask] = 255
                poly, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                _polys = []
                for _poly in poly:
                    approx = cv2.approxPolyDP(_poly, cv2.arcLength(_poly, True) * 0.001, True)
                    _polys.append(approx.squeeze().astype(int).tolist())
                polys.append(_polys)
            detections.__setattr__("poly", polys)

            if mask:
                mask_canvas = np.zeros_like(image, dtype=np.uint8)
                annotated_mask = self.mask_annotator.annotate(scene=mask_canvas, detections=detections)

            if polygon:
                labels = [
                    f"{prompt[_prompt[class_id]]} {confidence:0.2f}" for _, _, confidence, class_id, _ in detections
                ]

                annotated_image = self.mask_annotator.annotate(scene=image.copy(), detections=detections)
                annotated_image = self.box_annotator.annotate(
                    scene=annotated_image, detections=detections, labels=labels
                )

        json_data["filename"] = str(output)
        json_data["prompt"] = prompt
        for idx in range(len(detections)):
            json_data[str(idx)] = {}
            target = json_data[str(idx)]
            target["class"] = prompt[_prompt[detections.class_id[idx]]]
            target["conf"] = round(float(detections.confidence[idx]), 4)
            if rectangle or mask:
                target["rectangle"] = [*map(int, detections.xyxy[idx])]
            if polygon:
                target["polygon"] = detections.poly[idx]

        if save:
            save_images(output, annotated_frame, annotated_image, annotated_mask, extension)
            save_json(output, json_data)

        return annotated_frame, annotated_image, annotated_mask, json_data


def save_images(name, det, seg, mask, extension):
    if det is not None:
        cv2.imwrite(f"{name}_det{extension}", det)
    if seg is not None:
        cv2.imwrite(f"{name}_seg{extension}", seg)
    if mask is not None:
        cv2.imwrite(f"{name}_mask{extension}", mask)


def save_json(name, json_data):
    with open(f"{name}.json", "w") as f:
        dump(json_data, f, indent=4, ensure_ascii=False)
