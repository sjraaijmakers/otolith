import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt


def remove_bg(img):
    # sran = (np.min(img), np.max(img))

    _, gray = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)

    cnts, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    filtered_cnts = []

    for i in range(len(cnts)):
        if cv2.contourArea(cnts[i]) > 5:
            filtered_cnts.append(cnts[i])

    # Generate mask
    mask = np.ones(gray.shape)
    mask = cv2.drawContours(mask, filtered_cnts, -1, 0, cv2.FILLED)

    output = img.copy()

    output[mask.astype(np.bool)] = 0

    return output

if __name__ == "__main__":
    input_folder =  "/home/steven/inputs/scans/original/otoF75_0.5"
    t_val = 120
    file_format = "tif"
    sigma = 2

    files = sorted(glob.glob(input_folder + "/*." + file_format))

    first = len(files) - 1
    last = 0

    for z in range(len(files)):
        input = cv2.imread(files[z], cv2.IMREAD_UNCHANGED)

        img = (input/256).astype(np.uint8)
        _,gray = cv2.threshold(img, t_val, 255, cv2.THRESH_BINARY)

        cnts, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        cnts_areas = [cv2.contourArea(c) for c in cnts]

        print("%s: %s" % (z, cnts_areas))

        filtered_cnts = []
        for i in range(len(cnts_areas)):
            if cnts_areas[i] > 1:
                filtered_cnts.append(cnts[i])

        # Generate mask
        mask = np.zeros(gray.shape, dtype=np.uint8)
        mask = cv2.drawContours(mask, filtered_cnts, -1, 255, cv2.FILLED)
        mask = cv2.GaussianBlur(mask, (0,0), sigma)
        mask = mask / 255
        mask = 1 - mask

        output = input.copy()

        output = (output - (output * mask)).astype(np.uint16)

        # plt.imshow(input)
        # plt.show()

        cv2.imwrite("/home/steven/inputs/scans/no_bg/otoF75_0.5_no_bg/img_" + str(f'{z:04}') + ".tif", output)
