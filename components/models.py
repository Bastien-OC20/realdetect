import os
import cv2
import numpy as np
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection, VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, MarianMTModel, MarianTokenizer

# Initialisation des modèles
object_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101")
object_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101")

caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
caption_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
caption_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Initialisation du modèle de traduction
translation_model_name = "Helsinki-NLP/opus-mt-en-fr"
translation_model = MarianMTModel.from_pretrained(translation_model_name)
translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)

# Détection d'objets
def detect_objects(image):
    inputs = object_processor(images=image, return_tensors="pt")
    outputs = object_model(**inputs)
    results = object_processor.post_process_object_detection(
        outputs, threshold=0.9, target_sizes=[image.size]
    )[0]
    objects = [
        {"label": object_model.config.id2label[label.item()], "box": box.tolist()}
        for label, box in zip(results["labels"], results["boxes"])
    ]
    return objects

# Génération de descriptions
def generate_caption(image):
    inputs = caption_processor(images=image, return_tensors="pt")
    outputs = caption_model.generate(**inputs)
    caption = caption_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return caption

# Traduction de la description
def translate_caption(caption, target_language="fr"):
    inputs = translation_tokenizer(caption, return_tensors="pt", padding=True)
    translated_outputs = translation_model.generate(**inputs)
    translated_caption = translation_tokenizer.decode(translated_outputs[0], skip_special_tokens=True)
    return translated_caption

# Dessiner les boîtes de détection
def draw_detections(frame, objects):
    for obj in objects:
        label = obj["label"]
        box = obj["box"]
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame
