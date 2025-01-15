import cv2
from components.processing import process_frame
from components.models import initialize_object_detection, initialize_image_captioning
from PIL import Image

def generate_video_stream():
    """Génère un flux vidéo MJPEG pour Flask."""
    obj_processor, obj_model, device = initialize_object_detection()
    caption_pipeline = initialize_image_captioning()

    
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erreur : Impossible d'accéder à la webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        # Processus de détection et génération de description
        frame, image = process_frame(frame, obj_processor, obj_model, device)
        description = caption_pipeline(image)[0]['generated_text']

        # Ajout de la description à l'image
        cv2.putText(frame, f"Description: {description}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Encode la frame en JPEG et envoie à Flask
        ret, jpeg = cv2.imencode('.jpg', frame)
        if ret:
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()
