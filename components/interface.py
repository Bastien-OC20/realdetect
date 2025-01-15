import taipy as tp

def start_gui():
    """Lance l'interface Taipy pour afficher les résultats."""
    page = """
    <h1>Webcam Object Detection and Captioning</h1>
    <img src="http://127.0.0.1:5000/video_feed" alt="Current Webcam Feed"/>
    """
    tp.Gui(page=page).run()
