import cv2
import os.path
import glob
import math


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

    return list(map(lambda x: int(x), [new_y, new_y + new_h, new_x, new_x + new_w]))


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
        os.mkdir(self.out_path)

    def find_face(self):
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
            for m, (x, y, w, h) in enumerate(faces):
                # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                face_coord = expand_face([y, h, x, w], image.shape, 1.5)
                new_y1 = face_coord[0]
                new_y2 = face_coord[1]
                new_x1 = face_coord[2]
                new_x2 = face_coord[3]
                # cv2.imshow('hi', image[new_y1:new_y2, new_x1:new_x2])
                # cv2.waitKey(0)

                # Export cropped png file(s)
                png_path = os.path.join(self.out_path, 'frame_%s_%s.png' % (str(n).zfill(5), str(m)))
                cv2.imwrite(png_path, image[new_y1:new_y2, new_x1:new_x2])

            print("Frame number:", n+1, "/", total_imgs)
