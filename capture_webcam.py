import cv2
from components.processing import process_frame
from components.models import initialize_object_detection, initialize_image_captioning, draw_detections
from PIL import Image

current_caption = ""
current_labels = []

def generate_video_stream():
    """Génère un flux vidéo MJPEG."""
    global current_caption, current_labels
    obj_processor, obj_model, device = initialize_object_detection()
    caption_pipeline = initialize_image_captioning()

    reverse= cv2.flip(image, 1)
    cap = reverse.VideoCapture(0)

    if not cap.isOpened():
        print("Erreur : Impossible d'accéder à la webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Retourner l'image horizontalement
        frame = cv2.flip(frame, 1)

        # Processus de détection et génération de description
        frame, image, labels = process_frame(frame, obj_processor, obj_model, device)
        current_caption = caption_pipeline(image)[0]['generated_text']
        current_labels = labels

        # Dessiner les boîtes de détection
        frame_with_detections = draw_detections(frame, labels)

        # Encode l'image en JPEG
        ret, jpeg = cv2.imencode('.jpg', frame_with_detections)
        if not ret:
            continue

        # Convertir l'image encodée en bytes
        frame_bytes = jpeg.tobytes()

        # Générer le flux MJPEG
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

    cap.release()

def get_current_caption():
    """Retourne la légende actuelle."""
    return current_caption

def get_current_labels():
    """Retourne les étiquettes actuelles."""
    return current_labels
