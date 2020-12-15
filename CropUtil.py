import numpy as np
from mss import mss
import time
import keyboard

class CropUtil:
    # capture_rate :: num_frames per second
    def __init__(self, capture_rate):
        self.cap_rate = capture_rate

    def capture(self):
        sct = mss()
        all_images = []
        reference_time = time.time()
        # Keep capturing until 'q' is pressed
        while True:
            # Capture entire screen
            # Make sure to capture images at a particular rate (cap_rate)
            if time.time() - reference_time >= self.cap_rate:
                img = sct.grab(sct.monitors[1])
                img_np = np.array(img)  # Convert to numpy
                all_images.append(img_np)   # Append images (numpy arrays) onto a python list
                reference_time = time.time()
                print(reference_time)
            # Break
            if keyboard.read_key() == "q":
                break

        return all_images
