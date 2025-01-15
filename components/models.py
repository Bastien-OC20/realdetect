from transformers import DetrForObjectDetection, DetrImageProcessor, pipeline
import torch

def initialize_object_detection():
    """Initialise le modèle de détection d'objets."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obj_processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-101-dc5')
    obj_model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-101-dc5').to(device)
    return obj_processor, obj_model, device


def initialize_image_captioning():
    """Initialise le pipeline pour la génération de descriptions d'images."""
    caption_pipeline = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning", device=0 if torch.cuda.is_available() else -1)
    return caption_pipeline
