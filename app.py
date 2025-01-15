from flask import Flask, render_template, Response
from components.webcam import generate_video_stream

app = Flask(__name__)

@app.route('/')
def index():
    """Page d'accueil pour afficher la webcam."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Génère le flux vidéo pour la webcam."""
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # Démarre le serveur Flask pour diffuser la vidéo en temps réel
    app.run(host='0.0.0.0', port=5000, debug=True)
