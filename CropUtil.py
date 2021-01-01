import mss
import mss.tools
import time
import sys, signal
import os


def signal_handler(signal, frame):
    print("\nprogram exiting gracefully")
    sys.exit(0)


class CropUtil:
    # capture_rate :: num_frames per second
    def __init__(self, base_dir, capture_rate):
        self.cap_rate = capture_rate
        self.path = base_dir
        self.image_path = os.path.join(self.path, "images")

        if os.path.exists(self.image_path ):
            os.system("rm -R %s" % self.image_path )
        os.mkdir(self.image_path )

    # Take screen caps at self.cap_rate frames per sec
    # Return the cropped images
    def capture(self):
        sct = mss.mss()
        reference_time = time.time()
        frame_count = 0

        while True:
            # Capture entire screen
            # Make sure to capture images at a particular rate (cap_rate)
            if time.time() - reference_time >= self.cap_rate:
                png_path = os.path.join(self.image_path, 'frame_%s.png' % str(frame_count).zfill(5))
                sct_img = sct.grab(sct.monitors[1])
                mss.tools.to_png(sct_img.rgb, sct_img.size, output=png_path)
                frame_count += 1
                print("Frame number:", frame_count)
                reference_time = time.time()

            # Break when ^C is pressed
            signal.signal(signal.SIGINT, signal_handler)
