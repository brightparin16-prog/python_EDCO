import cv2
import numpy as np


paper = np.ones((600, 800, 3), dtype=np.uint8) * 255


cv2.rectangle(paper, (100, 150), (300, 250), (0, 255, 0), -1)


cv2.imwrite("test.png", paper)


