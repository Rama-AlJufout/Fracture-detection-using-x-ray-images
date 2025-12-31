# arrow_utils.py
import cv2
import numpy as np

def draw_single_arrow(image, cam):
    h, w, _ = image.shape

    cam = cv2.resize(cam, (w, h))
    y, x = np.unravel_index(np.argmax(cam), cam.shape)

    start = (x, max(y - 50, 0))
    end = (x, y)

    cv2.arrowedLine(
        image,
        start,
        end,
        color=(0, 0, 255),
        thickness=3,
        tipLength=0.35
    )

    return image
