import cv2
import gradio as gr
from transformers import DetrImageProcessor, DetrForObjectDetection, VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from PIL import Image
import torch

# Charger les modèles
detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
image_to_text_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

def detect_objects(image):
    # Préparer l'image
    inputs = detr_processor(images=image, return_tensors="pt")
    outputs = detr_model(**inputs)
    # Extraire les objets détectés
    logits = outputs.logits
    boxes = outputs.pred_boxes
    labels = torch.argmax(logits, dim=-1)
    objects = [detr_processor.decode([label.item()]) for label in labels[0]]
    return objects

def generate_caption(image):
    # Préparer l'image pour le modèle image-to-text
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    generated_ids = image_to_text_model.generate(pixel_values)
    description = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return description

def process_image(image):
    objects = detect_objects(image)
    caption = generate_caption(image)
    return objects, caption

def live_video():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convertir pour OpenCV
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        objects, caption = process_image(Image.fromarray(rgb_frame))
        for obj in objects:
            print(obj)  # À améliorer pour annoter dans l'interface vidéo
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Interface Gradio
def gradio_interface(image):
    image = Image.fromarray(image)
    objects, caption = process_image(image)
    return {"Objects": objects, "Caption": caption}

interface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Image(type="numpy"),
    outputs=[gr.JSON()],
    live=True,
)

if __name__ == "__main__":
    interface.launch()
