# Object Detection

Progressive implementations of a YOLO v3 detector using Keras, NumPy and OpenCV.
Each numbered file contains the preceding functionality and adds one operation.

| File | Added operation |
| --- | --- |
| `0-yolo.py` | Model, class-name and threshold initialization |
| `1-yolo.py` | Decode model outputs into original-image bounding boxes |
| `2-yolo.py` | Filter boxes by object confidence and class probability |
| `3-yolo.py` | Apply non-max suppression independently within each class |
| `4-yolo.py` | Load images with their paths |
| `5-yolo.py` | Resize and normalize images for the model |
| `6-yolo.py` | Draw and optionally save labeled detections |
| `7-yolo.py` | Run the complete detection and display pipeline |

Boxes use `(x1, y1, x2, y2)` coordinates in the original image. Non-max suppression
orders results by class, then by descending score. Images retain OpenCV's BGR color
order. Press `s` in an image window to save it under `detections`; other keys close
that window without saving.

The model and class-name files are supplied to the constructor. Target environment:
Python 3.9, TensorFlow 2.15, NumPy 1.25.2 and OpenCV 4.9.0.
