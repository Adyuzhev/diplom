from ultralytics import YOLO

def YOLOv8m_analise(image):
    model = YOLO("./models/YOLOv8m.pt")
    results = model.predict(image)
    image_with_bbox = results[0].plot()
    predcited_labels = results[0].names[int(results[0].boxes.cls)]
    return image_with_bbox, predcited_labels

def YOLOv8m_cls_analise(image):
    model = YOLO("./models/YOLOv8m-cls.pt")
    results = model.predict(image)
    image_with_labels = results[0].plot()
    predcited_labels = results[0].names[int(results[0].probs.top1)]
    return image_with_labels, predcited_labels