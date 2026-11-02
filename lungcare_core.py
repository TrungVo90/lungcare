from dataclasses import asdict, dataclass
import json
import re
from typing import Any, Callable, Sequence
from urllib.request import Request, urlopen

import cv2
import numpy as np

try:
    from .retrieval_core import LABEL_TO_TOPIC, RELATED_TOPICS, HybridRetriever, build_retrieval_plan
except ImportError: 
    from retrieval_core import LABEL_TO_TOPIC, RELATED_TOPICS, HybridRetriever, build_retrieval_plan


FINDING_DESCRIPTIONS = {
    "Cardiomegaly": "a wider-than-expected heart outline",
    "Nodule": "a small, rounded, whiter-looking region in the lung",
    "Diffuse Nodule": "several small, rounded, whiter-looking regions in the lung",
    "Consolidation": "a denser, whiter-looking region in the lung",
    "Effusion": "a denser-looking area near the lower edge of a lung",
    "Mass": "a relatively large, whiter-looking region in the lung",
    "Pneumothorax": "a more transparent-looking area near the outer edge of a lung",
    "Fibrosis": "a line-like pattern in the lung",
    "Pleural Thickening": "an area where the lung edge looks thicker",
    "Fracture": "a change along the outline of a bone",
    "Emphysema": "an area of lung that looks more transparent or over-expanded",
    "Atelectasis": "an area of lung that looks less expanded",
    "Calcification": "an especially bright area that may contain calcium",
}

URGENT_PHRASES = (
    "severe trouble breathing", "severe shortness of breath", "cannot breathe",
    "can't breathe", "chest pain", "fainting", "fainted", "coughing blood",
    "coughing up blood", "blue lips", "sudden difficulty breathing",
    "gasping for breath", "new confusion", "getting worse quickly", "worsening quickly",
)
BREATHING_CONCERNS = (
    "shortness of breath", "short of breath", "trouble breathing",
    "difficulty breathing", "breathless", "dyspnea", "dyspnoea",
)


@dataclass(frozen=True)
class ImageTransform:
    width: int
    height: int
    scale: float
    pad_x: int
    pad_y: int
    target_size: int


def preprocess_xray(image: np.ndarray, target_size: int = 1024) -> tuple[np.ndarray, ImageTransform]:
    if image is None or image.size == 0:
        raise ValueError("The X-ray image could not be decoded.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.astype(np.uint8)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    height, width = enhanced.shape[:2]
    scale = min(target_size / width, target_size / height)
    new_width, new_height = max(1, round(width * scale)), max(1, round(height * scale))
    resized = cv2.resize(enhanced, (new_width, new_height), interpolation=cv2.INTER_AREA)
    pad_x, pad_y = (target_size - new_width) // 2, (target_size - new_height) // 2
    output = np.zeros((target_size, target_size), dtype=np.uint8)
    output[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = resized
    return output, ImageTransform(width, height, scale, pad_x, pad_y, target_size)


def restore_points(points: np.ndarray, transform: ImageTransform) -> list[list[float]]:
    restored = []
    for x, y in points:
        sx = min(max((float(x) - transform.pad_x) / transform.scale, 0.0), transform.width - 1.0)
        sy = min(max((float(y) - transform.pad_y) / transform.scale, 0.0), transform.height - 1.0)
        restored.append([round(sx, 2), round(sy, 2)])
    return restored


class YOLOSegmenter:
    def __init__(self, checkpoint: str, device: str = "0", confidence: float = 0.25, iou: float = 0.45):
        from ultralytics import YOLO
        self.model = YOLO(checkpoint)
        self.device, self.confidence, self.iou = device, confidence, iou

    def predict(self, image_path: str) -> dict[str, Any]:
        source = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        processed, transform = preprocess_xray(source)
        result = self.model.predict(
            source=processed, imgsz=1024, conf=self.confidence, iou=self.iou,
            max_det=12, device=self.device, retina_masks=True, verbose=False,
        )[0]
        regions = []
        if result.boxes is not None and result.masks is not None:
            for index, box in enumerate(result.boxes):
                if index >= len(result.masks.xy) or len(result.masks.xy[index]) < 3:
                    continue
                xyxy = box.xyxy[0].detach().cpu().numpy()
                corners = restore_points(np.array([[xyxy[0], xyxy[1]], [xyxy[2], xyxy[3]]]), transform)
                regions.append({
                    "label": str(result.names[int(box.cls.item())]),
                    "score": round(float(box.conf.item()), 4),
                    "box": [corners[0][0], corners[0][1], corners[1][0], corners[1][1]],
                    "polygon": restore_points(np.asarray(result.masks.xy[index]), transform),
                })
        return {"regions": regions, "width": transform.width, "height": transform.height}


def preflight_safety(question: str) -> dict[str, Any] | None:
    text = question.lower()
    if any(phrase in text for phrase in URGENT_PHRASES):
        return {
            "mode": "defer_urgent",
            "message": "The symptoms described may require urgent medical care.",
            "next_steps": [
                "Go to the nearest appropriate medical facility or contact local emergency services now.",
                "Do not wait for this demo to analyse the X-ray before seeking help.",
            ],
            "pipeline_bypassed": True,
        }
    return None


def _display_location(box: Sequence[float], width: float, height: float) -> str:
    x1, y1, x2, y2 = map(float, box)
    center_x, center_y = (x1 + x2) / (2 * width), (y1 + y2) / (2 * height)
    box_width = (x2 - x1) / width
    horizontal = "center" if box_width >= 0.35 else "left" if center_x < 0.4 else "right" if center_x > 0.6 else "center"
    vertical = "upper" if center_y < 0.5 else "lower"
    return f"{vertical}-{horizontal} part of the displayed image"


def select_question_regions(question: str, segmentation: dict[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    regions = segmentation["regions"]
    labels = [region["label"] for region in regions]
    plan = build_retrieval_plan(question, labels)
    selected: list[dict[str, Any]] = []
    for label in plan.selected_image_labels:
        matches = [region for region in regions if region["label"] == label and region not in selected]
        if matches:
            selected.append(max(matches, key=lambda region: region["score"]))
        if len(selected) == 2:
            break
    for number, region in enumerate(selected, 1):
        region["area"] = number
        region["description"] = FINDING_DESCRIPTIONS.get(region["label"], "an area selected for closer review")
        region["location"] = _display_location(region["box"], segmentation["width"], segmentation["height"])
    return plan, selected


def safety_guidance(question: str, has_regions: bool) -> dict[str, Any]:
    text = question.lower()
    if any(phrase in text for phrase in BREATHING_CONCERNS):
        return {
            "mode": "prompt_assessment",
            "reason": "Breathing symptoms require assessment based on the person's current condition.",
            "next_steps": [
                "Arrange prompt medical assessment for new, persistent, or worsening shortness of breath.",
                "Seek emergency care now for severe or sudden breathing difficulty, chest pain, fainting, coughing blood, blue lips, or confusion.",
            ],
        }
    return {
        "mode": "explain_with_review" if has_regions else "defer_localization",
        "reason": "A highlighted image cannot establish a diagnosis or urgency by itself.",
        "next_steps": [
            "Ask a doctor or radiologist to review the original X-ray together with symptoms and medical history.",
            "Seek urgent medical care for severe trouble breathing, chest pain, fainting, coughing blood, or symptoms that worsen quickly.",
        ],
    }


def verified_explanation(selected: Sequence[dict[str, Any]], evidence: Sequence[dict[str, Any]]) -> dict[str, str]:
    if not selected:
        observation = "No image area matching the question passed the system's display checks."
        meaning = "This does not mean the X-ray is normal or rule out the concern."
    else:
        details = [
            f"Area {r['area']} is in the {r['location']} and appears as {r['description']}"
            for r in selected
        ]
        observation = ". ".join(details) + "."
        meaning = (
            "These are computer-selected visual differences, not diagnoses. "
            "They do not confirm a condition or identify its cause."
            if len(selected) > 1 else
            "This is a computer-selected visual difference, not a diagnosis. It does not confirm a condition or identify its cause."
        )
    if evidence:
        noun, verb = ("article", "provides") if len(evidence) == 1 else ("articles", "provide")
        meaning += f" The related medical {noun} {verb} general background only and cannot confirm what is present on this X-ray."
    return {"observation": observation, "meaning": meaning}


def validate_generation(generated: dict[str, Any], verified: dict[str, str], safety: dict[str, Any]) -> dict[str, Any]:
    meaning = re.sub(r"\s+", " ", str(generated.get("meaning", ""))).strip()
    steps = [re.sub(r"\s+", " ", str(x)).strip() for x in generated.get("next_steps", [])]
    if not 40 <= len(meaning) <= 900 or len(steps) != len(safety["next_steps"]):
        raise ValueError("Invalid generated response structure.")
    forbidden = (
        "confirmed diagnosis", "diagnosis is", "proves that", "confirms that",
        "safe to wait", "not urgent", "no need to see", "take this medication",
        "might be a", "may be a", "could be a",
    )
    combined = f"{meaning} {' '.join(steps)}".lower()
    if any(phrase in combined for phrase in forbidden):
        raise ValueError("Generated text introduced an unsupported or unsafe claim.")
    if not any(marker in meaning.lower() for marker in ("not a diagnosis", "does not confirm", "do not confirm", "cannot confirm")):
        raise ValueError("Generated text removed the required uncertainty statement.")
    if not any(role in steps[0].lower() for role in ("doctor", "radiologist", "clinician", "medical assessment")):
        raise ValueError("Generated text removed clinician review.")
    if not any(term in steps[-1].lower() for term in ("urgent", "emergency")):
        raise ValueError("Generated text removed urgent escalation.")
    return {"observation": verified["observation"], "meaning": meaning, "next_steps": steps}


def qwen_rewrite(
    endpoint: str,
    model: str,
    question: str,
    verified: dict[str, str],
    safety: dict[str, Any],
    evidence: Sequence[dict[str, Any]],
    timeout: float = 60,
) -> dict[str, Any]:
    source_titles = [item["title"] for item in evidence]
    prompt = f"""Rewrite VERIFIED_MEANING and APPROVED_NEXT_STEPS in concise caregiver-friendly English.
Return JSON only: {{"meaning":"...","next_steps":["...","..."]}}.
Do not add findings, diagnoses, causes, treatments, urgency judgments, or patient facts.
Highlighted regions are computer-selected appearances, not confirmed abnormalities.
Related articles are background only and cannot establish what is on this X-ray.
Keep all approved actions, in order, including clinician review and urgent escalation.

QUESTION: {question}
VERIFIED_MEANING: {verified['meaning']}
RELATED_ARTICLE_TITLES: {json.dumps(source_titles)}
APPROVED_NEXT_STEPS: {json.dumps(safety['next_steps'])}
"""
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": "Preserve supplied evidence and medical-safety constraints exactly."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 320,
        "response_format": {"type": "json_object"},
    }).encode()
    request = Request(endpoint, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode())
    generated = json.loads(payload["choices"][0]["message"]["content"])
    return validate_generation(generated, verified, safety)


def validate_consistency(result: dict[str, Any]) -> None:
    areas = result.get("areas", [])
    evidence = result.get("evidence", [])
    if len(areas) > 2 or [x.get("area") for x in areas] != list(range(1, len(areas) + 1)):
        raise ValueError("Region consistency failed.")
    if any(not item.get("title") or not item.get("focus_topics") for item in evidence):
        raise ValueError("Evidence provenance/relevance consistency failed.")
    if result["safety"]["mode"] == "defer_urgent" and not result["safety"].get("pipeline_bypassed"):
        raise ValueError("Urgent safety routing was not preserved.")
    if re.search(r"\b(?:article|source) (?:confirms|proves|verifies)\b", result.get("answer", ""), re.I):
        raise ValueError("The answer overstates retrieved evidence.")


class LungCAREPipeline:
    def __init__(
        self,
        segmenter: YOLOSegmenter,
        retriever: HybridRetriever,
        *,
        qwen_endpoint: str = "",
        qwen_model: str = "qwen2.5:7b",
    ) -> None:
        self.segmenter, self.retriever = segmenter, retriever
        self.qwen_endpoint, self.qwen_model = qwen_endpoint, qwen_model

    def run(self, image_path: str, question: str) -> dict[str, Any]:
        urgent = preflight_safety(question)
        if urgent:
            result = {"status": "red_alert", "question": question, "image_path": image_path, "safety": urgent}
            validate_consistency(result)
            return result

        segmentation = self.segmenter.predict(image_path)
        plan, areas = select_question_regions(question, segmentation)
        allow_disease_specific = bool(DISEASE_REQUEST.search(question))
        evidence = self.retriever.retrieve(plan, top_k=3, allow_disease_specific=allow_disease_specific)
        safety = safety_guidance(question, bool(areas))
        verified = verified_explanation(areas, evidence)
        sections = {**verified, "next_steps": safety["next_steps"]}
        provider = "deterministic verified fallback"
        if self.qwen_endpoint:
            try:
                sections = qwen_rewrite(
                    self.qwen_endpoint, self.qwen_model, question, verified, safety, evidence
                )
                provider = f"{self.qwen_model} + deterministic safety validation"
            except Exception:
                pass 

        result = {
            "status": "complete",
            "question": question,
            "retrieval_plan": asdict(plan),
            "areas": areas,
            "evidence": evidence,
            "explanation": sections,
            "answer": f"{sections['observation']} {sections['meaning']}",
            "safety": {**safety, "next_steps": sections["next_steps"]},
            "synthesis_provider": provider,
        }
        validate_consistency(result)
        return result


DISEASE_REQUEST = re.compile(
    r"\b(?:cancer|carcinoma|malignan(?:t|cy)|tumou?rs?|sarcoma|lymphoma|metasta(?:sis|ses|tic))\b",
    re.I,
)

