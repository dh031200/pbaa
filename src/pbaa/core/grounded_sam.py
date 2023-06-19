from json import dump
from pathlib import Path

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

if HQ:
    SAM_CHECKPOINT_PATH = Path("sam_hq_vit_h.pth")
    url = "https://blueclairvoyancestorage.blob.core.windows.net/package/sam_hq_vit_h.pth"
else:
    SAM_CHECKPOINT_PATH = Path("sam_vit_h_4b8939.pth")
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"


def model_init():
    # GroundingDINO config and checkpoint
    if not GROUNDING_DINO_CHECKPOINT_PATH.exists():
        logger.warning("GROUNDING_DINO_CHECKPOINT doesn't exist")
        logger.info("Start download")
        wget.download(
            "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
        )
    if not SAM_CHECKPOINT_PATH.exists():
        logger.warning("SAM_CHECKPOINT_PATH doesn't exist")
        logger.info("Start download")
        wget.download(url)


def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(box=box, multimask_output=True)
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


def inference(_src, _prompt, box_threshold=0.25, nms_threshold=0.8, output_dir="."):
    src = Path(_src)
    dst = Path(output_dir)
    dst.mkdir(parents=True, exist_ok=True)

    # Building GroundingDINO inference model
    grounding_dino_model = Model(
        model_config_path=grounding_dino_config_path, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH
    )

    # Building SAM Model and SAM Predictor
    sam_predictor = SamPredictor(sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH))

    # Predict classes and hyper-param for GroundingDINO
    prompt = [*map(str.lower, _prompt.keys())]

    # load image
    image = cv2.imread(_src)

    # detect objects
    detections = grounding_dino_model.predict_with_classes(
        image=image, classes=prompt, box_threshold=box_threshold, text_threshold=box_threshold
    )

    # annotate image with detections
    box_annotator = sv.BoxAnnotator()
    labels = [f"{_prompt[prompt[class_id]]} {confidence:0.2f}" for _, _, confidence, class_id, _ in detections]
    annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

    nms_idx = (
        torchvision.ops.nms(torch.from_numpy(detections.xyxy), torch.from_numpy(detections.confidence), nms_threshold)
        .numpy()
        .tolist()
    )

    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]

    # Prompting SAM with detected boxes
    # convert detections to masks
    detections.mask = segment(
        sam_predictor=sam_predictor, image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB), xyxy=detections.xyxy
    )

    confidences = [round(i, 4) for i in detections.confidence.astype(float)]
    boxes = detections.xyxy.astype(int).tolist()
    polys = []
    for mask in detections.mask:
        canvas = np.zeros((image.shape[:2]), dtype=np.uint8)
        canvas[mask] = 255
        poly, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        _polys = []
        for _poly in poly:
            _polys.append(_poly.squeeze().astype(int).tolist())
        polys.append(_polys)

    mask_canvas = np.zeros_like(image, dtype=np.uint8)

    # annotate image with detections
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    labels = [f"{_prompt[prompt[class_id]]} {confidence:0.2f}" for _, _, confidence, class_id, _ in detections]
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_mask = mask_annotator.annotate(scene=mask_canvas, detections=detections)
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

    class_id = detections.class_id.astype(int).tolist()
    json_data = {}
    for idx, (_id, conf, box, poly) in enumerate(zip(class_id, confidences, boxes, polys)):
        json_data[idx] = {"cls": _prompt[prompt[_id]], "conf": conf, "box": box, "poly": poly}

    with open(f"{dst / src.stem}.json", "w") as f:
        dump(json_data, f, indent=4, ensure_ascii=False)

    # save images
    cv2.imwrite(f"{dst / src.stem}_det.jpg", annotated_frame)
    cv2.imwrite(f"{dst / src.stem}_seg.jpg", annotated_image)
    cv2.imwrite(f"{dst / src.stem}_mask.jpg", annotated_mask)
