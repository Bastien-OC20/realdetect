from PIL import Image
import torch
import cv2

def process_frame(frame, obj_processor, obj_model, device, confidence_threshold=0.9):
    """Traite une image pour la détection d'objets."""
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = obj_processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = obj_model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]]).to(device)
    results = obj_processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

    labels = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if score > confidence_threshold:
            box = [round(i, 2) for i in box.tolist()]
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            text = f"{obj_model.config.id2label[label.item()]}: {round(score.item(), 2)}"
            labels.append(text)

    return frame, image, labels
