from transformers import DetrForObjectDetection, DetrImageProcessor, pipeline
import cv2
from PIL import Image
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


def process_frame(frame, obj_processor, obj_model, device, confidence_threshold=0.9):
    """Traite une image pour la détection d'objets."""
    # Convertir l'image en format PIL
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Préparer l'image pour le modèle
    inputs = obj_processor(images=image, return_tensors="pt").to(device)

    # Inférence
    with torch.no_grad():
        outputs = obj_model(**inputs)

    # Post-traitement
    target_sizes = torch.tensor([image.size[::-1]]).to(device)
    results = obj_processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

    # Ajouter les boîtes de délimitation au frame
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if score > confidence_threshold:
            box = [round(i, 2) for i in box.tolist()]
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            text = f"{obj_model.config.id2label[label.item()]}: {round(score.item(), 2)}"
            cv2.putText(frame, text, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame, image


def main():
    """Programme principal pour la détection d'objets et la génération de descriptions."""
    # Initialisation des modèles
    obj_processor, obj_model, device = initialize_object_detection()
    caption_pipeline = initialize_image_captioning()

    cap = cv2.VideoCapture(0)  # Utilise la webcam (ID 0)

    if not cap.isOpened():
        print("Erreur : Impossible d'accéder à la webcam.")
        return

    print("Appuyez sur 'q' pour quitter.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur : Impossible de lire l'image de la webcam.")
            break

        # Traiter l'image pour la détection d'objets
        frame, image = process_frame(frame, obj_processor, obj_model, device)

        # Générer une description de l'image
        description = caption_pipeline(image)[0]['generated_text']

        # Afficher la description sur l'image
        cv2.putText(frame, f"Description: {description}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Afficher l'image avec les annotations
        cv2.imshow('Webcam Object Detection and Captioning', frame)

        # Quitter avec 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libérer les ressources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
