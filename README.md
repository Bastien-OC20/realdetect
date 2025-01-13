# RealDetect - Détection d'objets en temps réel avec webcam

RealDetect est une application de détection d'objets en temps réel utilisant une webcam, qui génère des légendes basées sur les objets détectés dans les images capturées. Le projet utilise OpenCV pour la capture vidéo, des modèles pré-entraînés de Hugging Face pour la détection d'objets et la génération de légendes, ainsi que Gradio pour l'interface utilisateur. Il permet de visualiser en temps réel le flux de la webcam, d'effectuer des détections d'objets et de générer des descriptions textuelles.

## Prérequis

Avant de commencer, assurez-vous d'avoir installé Python 3.8 ou supérieur.

### Cloner le projet

Clonez ce dépôt sur votre machine locale en utilisant la commande suivante :

```bash
git clone https://github.com/votre-utilisateur/real_detect.git
```

Accédez ensuite au dossier du projet :

```bash
cd real_detect
```

### Installation et Configuration de l'Environnement

#### 1. Créer un environnement virtuel

Sur macOS ou Linux :

```bash
python3 -m venv env
source env/bin/activate
```

Sur Windows :

```bash
python -m venv env
.\env\Scripts\activate
```

#### 2. Installer les dépendances

Une fois l'environnement virtuel activé, installez les dépendances du projet en utilisant pip :

```bash
pip install -r requirements.txt
```

Si vous n'avez pas encore créé le fichier requirements.txt, voici un exemple de contenu que vous pouvez inclure :

```text
opencv-python
gradio
torch
transformers
numpy
```

#### 3. Démarrer l'application

Une fois les dépendances installées, vous pouvez démarrer l'application en lançant le fichier demo_app.py :

```bash
python demo/demo_app.py
```

Cela lancera l'interface Gradio dans votre navigateur et vous pourrez commencer à utiliser l'application.

### 4. Arrêter l'application

Pour arrêter l'application, vous pouvez simplement fermer la fenêtre du terminal ou appuyer sur Ctrl+C.

## Fonctionnalités

- Activer/Désactiver Webcam : Démarre ou arrête la capture vidéo en temps réel depuis la webcam.
- Prendre une Photo : Capture une image à partir du flux vidéo et effectue des détections d'objets ainsi que la génération de légendes.
- Affichage du flux vidéo en direct : Le flux de la webcam est affiché en temps réel avec les objets détectés superposés dessus.

## Technologies utilisées

- OpenCV : Pour la gestion de la capture vidéo et du traitement d'images.
- Gradio : Pour créer l'interface graphique permettant d'interagir avec l'application.
- Transformers (Hugging Face) : Pour la détection d'objets et la génération de légendes à partir des images.
- Threading : Pour gérer la capture vidéo en temps réel sans bloquer l'interface.

## Dépannage

Si vous rencontrez des problèmes avec la webcam, voici quelques pistes de dépannage :

- Vérifiez que votre caméra est bien connectée et accessible via OpenCV.
- Sur Windows, essayez d'utiliser le backend cv2.CAP_DSHOW si cv2.CAP_VFW ne fonctionne pas.
- Assurez-vous que votre carte graphique est compatible si vous utilisez des modèles de deep learning lourds.

## Contributions

Les contributions sont les bienvenues ! Si vous avez des améliorations ou des corrections de bugs à proposer, n'hésitez pas à soumettre une pull request.

## License

Ce projet est sous la licence MIT. Consultez le fichier LICENSE pour plus de détails.
