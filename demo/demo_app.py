import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gradio as gr
from threading import Thread
from backend.utils import start_webcam, stop_webcam, get_frame, start_webcam_thread, process_image, update_video

import time



with gr.Blocks() as demo:
    with gr.Row():
        start_button = gr.Button("Activer Webcam")
        stop_button = gr.Button("Désactiver Webcam")
        capture_button = gr.Button("Prendre une Photo")

    video_output = gr.Image(label="Flux vidéo en direct")
    detection_output = gr.Text(label="Détections")
    caption_output = gr.Text(label="Légende générée")

    # Le bouton de démarrage de la webcam appelle la fonction `start_webcam_thread`
    start_button.click(start_webcam_thread, inputs=[], outputs=[])
    stop_button.click(stop_webcam, inputs=[], outputs=[])
    capture_button.click(process_image, inputs=[], outputs=[video_output, detection_output, caption_output])

    # Mise à jour du flux vidéo en boucle
    def update_loop():
        while True:
            frame, detections = update_video()
            if frame is not None:
                # Retourner l'image et les détections à Gradio pour qu'il mette à jour l'interface
                video_output.update(value=frame)  # Correction ici : Utiliser `update()` sur l'output
                detection_output.update(value=detections)  # De même pour `detection_output`
            time.sleep(0.1)

    # Lancer la boucle de mise à jour dans un thread pour ne pas bloquer l'interface
    Thread(target=update_loop, daemon=True).start()

demo.launch()