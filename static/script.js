const startWebcamButton = document.getElementById('start-webcam');
const stopWebcamButton = document.getElementById('stop-webcam');
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const captureButton = document.getElementById('capture');
const outputImage = document.getElementById('output-image');
const captionResult = document.getElementById('caption-result');
let stream;

// Démarrer la webcam lorsque le bouton est cliqué
startWebcamButton.addEventListener('click', () => {
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(mediaStream => {
            stream = mediaStream;
            video.srcObject = stream;
            video.style.display = 'block';
            captureButton.style.display = 'block';
            stopWebcamButton.style.display = 'block';
            startWebcamButton.style.display = 'none';
        })
        .catch(err => {
            console.error("Erreur : Impossible d'accéder à la webcam", err);
        });
});

// Arrêter la webcam lorsque le bouton est cliqué
stopWebcamButton.addEventListener('click', () => {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        video.style.display = 'none';
        captureButton.style.display = 'none';
        stopWebcamButton.style.display = 'none';
        startWebcamButton.style.display = 'block';
    }
});

canvas.width = 320; // Optimisation
canvas.height = 240;

// Capturer l'image
captureButton.addEventListener('click', () => {
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(blob => {
        const formData = new FormData();
        formData.append('file', blob, 'captured_image.jpg');

        // Envoyer l'image au serveur
        fetch('http://127.0.0.1:8000/process', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())
            .then(data => displayResults(data))
            .catch(error => console.error('Erreur:', error));
    }, 'image/jpeg');
});

// Fonction pour afficher les résultats
function displayResults(data) {
    const objects = data.objects;
    const caption = data.caption;

    // Afficher les boîtes de détection sur le canvas
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    objects.forEach(obj => {
        const [x, y, w, h] = obj.box;
        context.strokeStyle = 'red';
        context.lineWidth = 2;
        context.strokeRect(x, y, w, h);
        context.font = '16px Arial';
        context.fillStyle = 'red';
        context.fillText(obj.label, x, y - 5);
    });

    // Mettre à jour les résultats dans l'interface
    captionResult.textContent = `Description : ${caption}`;
    outputImage.src = canvas.toDataURL('image/jpeg');
    outputImage.style.display = 'block';
}
