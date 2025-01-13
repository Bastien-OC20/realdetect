import cv2
from PIL import Image
from time import sleep
from backend.detection import detect_objects
from backend.image_to_text import generate_caption
from threading import Thread
import numpy as np

webcam_active = True
frame = None

def start_webcam():
    global cap, frame,webcam_active
    webcam_active = True
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Essayer avec DSHOW

    if not cap.isOpened():
        print("Erreur avec DSHOW, tentative avec VFW.")
        cap = cv2.VideoCapture(0, cv2.CAP_VFW)  # Essayer avec VFW
    
    if not cap.isOpened():
        print("Erreur avec VFW, tentative avec MSMF.")
        cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
        
    if not cap.isOpened():
        print("Erreur avec VFW, tentative avec MSMF.")
        cap = cv2.VideoCapture(0)# Essayer avec MSMF

    if not cap.isOpened():
        print("Erreur lors de l'ouverture de la webcam.")
        return

    print("Webcam activée.")

    while webcam_active:
        ret, frame_temp = cap.read()
        if ret:
            # Convertit le frame au format RGB
            frame = cv2.cvtColor(frame_temp, cv2.COLOR_BGR2RGB)
        else:
            print("Erreur lors de la lecture de la caméra.")
            break

        sleep(0.03)  # Pause de 30ms pour éviter une consommation CPU excessive

    # Libère la capture de la caméra après avoir quitté la boucle
    cap.release()
    print("Webcam désactivée.")

def stop_webcam():
    global webcam_active
    webcam_active = False
    if cap:
        cap.release()
        print("Webcam désactivée.")

def get_frame():
    global frame
    if cap and webcam_active:
        ret, frame_temp = cap.read()
        if ret:
            frame = cv2.cvtColor(frame_temp, cv2.COLOR_BGR2RGB)  # Convertir en RGB pour Gradio
            return frame
    return None


def update_video():
    frame = get_frame()  # Cette fonction doit retourner l'image sous forme de tableau ou d'image lisible par Gradio
    if frame is not None:
        detections = detect_objects(Image.fromarray(np.uint8(frame)))
        return frame, detections
    return None, None

# Fonction pour traiter l'image et générer les légendes
def process_image():
    frame = get_frame()
    if frame is not None:
        # Convertir le frame en objet PIL.Image
        image = Image.fromarray(np.uint8(frame))
        detections = detect_objects(image)
        caption = generate_caption(image)
        return frame, detections, caption
    return None, None, None

def start_webcam_thread():
    # Démarrer la webcam dans un thread séparé
    print("Démarrage du thread de la webcam")
    Thread(target=start_webcam, daemon=True).start()
