const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const captureButton = document.getElementById('capture');
const outputImage = document.getElementById('output-image');
const captionResult = document.getElementById('caption-result');

// Accéder à la webcam
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    })
    .catch(err => {
        console.error("Erreur : Impossible d'accéder à la webcam", err);
    });

canvas.width = 320; // Réduire la taille pour un traitement plus rapide
canvas.height = 240;

// Capturer l'image
captureButton.addEventListener('click', () => {
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(blob => {
        const formData = new FormData();
        formData.append('file', blob, 'captured_image.jpg');

        // Envoyer l'image au serveur
        fetch('http://127.0.0.1:8000/process', { // Utiliser l'endpoint correct
            method: 'POST',
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                // Afficher les objets détectés et l'image annotée
                displayResults(data);
            })
            .catch(error => {
                console.error('Erreur:', error);
            });
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
