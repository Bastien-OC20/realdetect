import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image

# Initialisation du modèle et du processeur
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

def generate_caption(image):
    """
    Génère une légende pour une image.

    Args:
        image (PIL.Image): L'image à analyser.

    Returns:
        str: Légende générée.
    """
    inputs = feature_extractor(images=image, return_tensors="pt").to(device)
    outputs = model.generate(inputs["pixel_values"])
    caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return caption
