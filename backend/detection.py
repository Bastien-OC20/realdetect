import torch
from transformers import DetrForObjectDetection, DetrFeatureExtractor
from PIL import Image

# Initialisation du modèle et du processeur
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50", use_fast=True)

def detect_objects(image, confidence_threshold=0.9):
    """
    Détecte les objets dans une image.

    Args:
        image (PIL.Image): L'image à analyser.
        confidence_threshold (float): Seuil de confiance pour afficher une détection.

    Returns:
        list: Liste des détections avec labels, scores et bounding boxes.
    """
    if not isinstance(image, Image.Image):
        raise ValueError("L'image doit être un objet PIL.Image")

    inputs = feature_extractor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)

    target_sizes = torch.tensor([list(image.size)[::-1]]).to(device)
    results = feature_extractor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]


    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if score > confidence_threshold:
            detections.append({
                "label": model.config.id2label[label.item()],
                "score": round(score.item(), 3),
                "bbox": [round(i, 2) for i in box.tolist()]
            })
    return detections
