import sys
sys.path.insert(1, "../sulcus")

import Otolith
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import os
import shutil



def remove_bg(input, t_val=122, sigma=1):
    img = (input/256).astype(np.uint8)

    _,gray = cv2.threshold(img, t_val, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Filter out really small contours
    cnts_areas = [cv2.contourArea(c) for c in cnts]

    filtered_cnts = []
    for i in range(len(cnts_areas)):
        if cnts_areas[i] > 10:
            filtered_cnts.append(cnts[i])

    # Generate mask
    mask = np.zeros(gray.shape, dtype=np.uint8)
    mask = cv2.drawContours(mask, filtered_cnts, -1, 255, cv2.FILLED)

    kernel_size = 5
    kernel  = np.ones((kernel_size,kernel_size), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    # mask = cv2.erode(mask, kernel, iterations=1)
    # mask = cv2.GaussianBlur(mask, (0,0), sigma)

    mask = mask / 255
    mask = 1 - mask

    output = input.copy()
    output = (output - (output * mask)).astype(np.uint16)
    # output = cv2.GaussianBlur(mask, (0,0), sigma).astype(np.uint16)

    return output

    # plt.imshow(output)
    # plt.show()

    # cv2.imwrite("/home/steven/scriptie/code/sulcus/tost/img_0000.tif", output)


if __name__ == "__main__":
    args = sys.argv[1:]
    input_folder = str(args[0])
    basename = os.path.basename(input_folder)

    output_folder = args[1]

    folder = "%s/%s_no_bg" % (output_folder, basename)

    if not os.path.exists(folder):
        os.makedirs(folder)
    # delete folder if it already exists
    else:
        shutil.rmtree(folder)
        os.makedirs(folder)

    t_val = 100
    file_format = "tif"
    sigma = 2

    files = sorted(glob.glob(input_folder + "/*." + file_format))

    first = len(files) - 1
    last = 0

    for z in range(len(files)):
        print("Slice %s..." % z)
        input = cv2.imread(files[z], cv2.IMREAD_UNCHANGED)
        output = remove_bg(input, t_val=t_val)

        p = str(f'{z:04}')

        # plt.imshow(output)
        # plt.show()

        cv2.imwrite("%s/img_%s.tif" % (folder, p), output)
