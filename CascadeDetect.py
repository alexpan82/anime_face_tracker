import cv2
import os.path
import glob
import math
import multiprocessing
import time


def detect_face(img_list, cascade, out_path):
    # img_list is a list of png files with the following naming convention: frame_[FRAME NUMBER].png
    # Output from CropUtil.capture
    # cascade is the trained cv2.CascadeClassifer object
    # out_path is the output directory
    print(len(img_list), out_path)

    for img in img_list:
        image = cv2.imread(img, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = cascade.detectMultiScale(gray,
                                         # detector options
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(24, 24))
        for m, (x, y, w, h) in enumerate(faces):
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            face_coord = expand_face([y, h, x, w], image.shape, 1.69)
            new_y1, new_y2, new_x1, new_x2 = face_coord

            # Export cropped png file(s)
            # Make output name the same as the input name, except with an additional modifier m
            png_path = os.path.join(out_path, '%s_%s.png' % (img.split('/')[-1].split('.')[0], str(m)))
            # Resize all imgs for input to Classify.py/ipynb
            resize = cv2.resize(image[new_y1:new_y2, new_x1:new_x2], (160, 160))
            cv2.imwrite(png_path, resize)

def expand_face(bound_box, img_size, expand):
    # Expand symmetrically and keep within confines of image
    # bound_box = [y,h,x,w]
    # img_size = img.shape
    # expand = expand the area by some coefficient
    org_h = float(bound_box[1])
    org_w = float(bound_box[3])
    org_area = org_h * org_w
    new_area = float(expand) * org_area
    ratio = org_h / org_w

    # Find new h and w
    new_w = math.sqrt(new_area/ratio)
    new_h = new_area / new_w

    # Now return new x, y
    org_y = float(bound_box[0])
    org_x = float(bound_box[2])
    new_y = org_y - ((new_h - org_h) / 2.)
    new_x = org_x - ((new_w - org_w) / 2.)

    # Now fit within confines of img dimensions
    y_bound = float(img_size[0])
    x_bound = float(img_size[1])
    if (new_y + new_h) > y_bound:
        new_h = y_bound - new_y
    if new_y < 0:
        new_y = 0
    if (new_x + new_w) > x_bound:
        new_w = x_bound - new_x
    if new_x < 0:
        new_x = 0

    return list(map(lambda x: int(x), (new_y, new_y + new_h, new_x, new_x + new_w)))


# Divide list into n lists of approx equal size
def n_even_chunks(input_list, n):
    final = []
    """Yield n as even chunks as possible from l."""
    last = 0
    for i in range(1, n + 1):
        cur = int(round(i * (len(input_list) / n)))
        final.append(input_list[last:cur])
        last = cur
    return final


class Detect:
    def __init__(self, base_dir, xml_path="./lbpcascade_animeface.xml", threads=4):
        # Path to captured images
        self.path = base_dir
        self.input_path = os.path.join(self.path, "image")
        # Path to cascade xml with trained parameters
        self.xml_path = xml_path

        # Path to output images
        self.out_path = os.path.join(self.path, "cascade-image")
        self.out_path_pngs = os.path.join(self.out_path, "*.png")
        if os.path.exists(self.out_path):
            os.system("rm %s" % self.out_path_pngs)
        else:
            os.mkdir(self.out_path)

        # Number of threads
        self.threads = int(threads)

    def find_face(self):
        start_time = time.time()
        if not os.path.isfile(self.xml_path):
            raise RuntimeError("%s: not found" % self.xml_path)

        cascade = cv2.CascadeClassifier(self.xml_path)
        total_img_list = sorted(glob.glob(os.path.join(self.input_path, "*png")))

        # Divide img_list into even sized lists for multiprocessing
        split_imgs = n_even_chunks(total_img_list, self.threads)
        jobs = []
        for img_list in split_imgs:
            p = multiprocessing.Process(target=detect_face, args=(img_list, cascade, self.out_path,))
            jobs.append(p)
            p.start()

        for i in jobs:
            i.join()
        print(time.time()-start_time)
