from flask import Flask, render_template, Response, jsonify
from components.webcam import generate_video_stream, get_current_caption, get_current_labels

app = Flask(__name__)

@app.route('/')
def index():
    """Page d'accueil pour afficher la webcam."""
    return render_template('index.html', caption=get_current_caption(), labels=get_current_labels())

@app.route('/video_feed')
def video_feed():
    """Génère le flux vidéo pour la webcam."""
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/caption')
def caption():
    """Retourne la légende générée pour l'image actuelle."""
    return jsonify({'caption': get_current_caption()})

@app.route('/labels')
def labels():
    """Retourne les étiquettes générées pour l'image actuelle."""
    return jsonify({'labels': get_current_labels()})

if __name__ == "__main__":
    # Démarre le serveur Flask pour diffuser la vidéo en temps réel
    app.run(host='0.0.0.0', port=5000, debug=True)
