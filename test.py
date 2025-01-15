from transformers import DetrForObjectDetection, DetrFeatureExtractor
import cv2
from PIL import Image
import torch
import random


# Charger le modèle et l'extracteur de fonctionnalités
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")

# Initialiser la capture vidéo
cap = cv2.VideoCapture(0)

while True:
    # Capturer une image de la webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir l'image en format PIL
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Préparer l'image
    inputs = feature_extractor(images=image, return_tensors="pt")

    # Effectuer la détection d'objets
    outputs = model(**inputs)

    # Post-traitement des résultats
    target_sizes = torch.tensor([image.size[::-1]])
    results = feature_extractor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

    # Dessiner les boîtes de délimitation sur l'image
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if score > 0.9:  # Seuil de confiance
            box = [round(i, 2) for i in box.tolist()]
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
            cv2.putText(frame, f"{model.config.id2label[label.item()]}: {round(score.item(), 3)}", 
                        (int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Afficher l'image avec les boîtes de délimitation
    cv2.imshow('Webcam Object Detection', frame)

    # Quitter la boucle si 'q' est pressé
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la capture vidéo et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()