import os
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

tests = os.listdir("data/test")

model = YOLO("runs/train/weights/best.pt")

for test in tests:
    img_path = os.path.join("data/test", test)
    results = model(img_path)
    
    plt.imshow(cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB))
    plt.show()