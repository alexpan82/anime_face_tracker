import cv2
import os.path
import glob

class Detect:
    def __init__(self, base_dir, xml_path="./lbpcascade_animeface.xml"):
        # Path to captured images
        self.path = base_dir
        self.input_path = os.path.join(self.path, "image")
        # Path to cascade xml with trained parameters
        self.xml_path = xml_path

        # Path to output images
        self.out_path = os.path.join(self.path, "cascade-image")

        if os.path.exists(self.out_path ):
            os.system("rm -R %s" % self.out_path )
        os.mkdir(self.out_path )

    def FindFace(self):
        if not os.path.isfile(self.xml_path):
            raise RuntimeError("%s: not found" % self.xml_path)

        img_list = sorted(glob.glob(os.path.join(self.input_path, "*png")))
        total_imgs = len(img_list)
        cascade = cv2.CascadeClassifier(self.xml_path)

        for n, img in enumerate(img_list):
            image = cv2.imread(img, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            faces = cascade.detectMultiScale(gray,
                                             # detector options
                                             scaleFactor=1.1,
                                             minNeighbors=5,
                                             minSize=(24, 24))
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Export png file
            png_path = os.path.join(self.out_path, 'frame_%s.png' % str(n).zfill(5))

            cv2.imwrite(png_path, image)
            print("Frame number:", n, "/", total_imgs)
