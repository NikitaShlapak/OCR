import numpy as np
import cv2
import matplotlib.pyplot as plt



def form_sample(x, y, w, h, letter_crop, out_size):
    size_max = max(w, h)
    letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)

    if w > h:
        # Enlarge image top-bottom
        # ------
        # ======
        # ------
        y_pos = size_max // 2 - h // 2
        letter_square[y_pos:y_pos + h, 0:w] = letter_crop[:h, :w]
    elif w < h:
        # Enlarge image left-right
        # --||--
        x_pos = size_max // 2 - w // 2
        letter_square[0:h, x_pos:x_pos + w] = letter_crop[:h, :w]
    else:
        letter_square = letter_crop

    # Resize letter to 28x28 and add letter and its X-coordinate
    return (x,y), (w,h), cv2.resize(letter_square, (out_size, out_size), interpolation=cv2.INTER_AREA)


def letters_extract(img, out_size=28, scale_factor=3, erode_core=3, num_erosions=1,split_trashold=200, verbose=False):
    letters = []

    img = cv2.resize(img, (img.shape[1] * scale_factor, img.shape[0] * scale_factor), interpolation=cv2.INTER_LINEAR)
    thresh = cv2.adaptiveThreshold(img, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)
    img_erode = cv2.erode(thresh, np.ones((erode_core, erode_core), np.uint8), iterations=num_erosions)
    # Get contours
    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    hir = hierarchy[0, :, 3]

    inner_conts = []

    for idx, contour in enumerate(contours):
        if hir[idx] == 0:
            (x, y, w, h) = cv2.boundingRect(contour)

            inner_conts.append(contour)

            if (w > img.shape[1] * scale_factor / split_trashold) or ((w-h>25)and(h>img.shape[0]//3*scale_factor)):
                letter_crop1 = img[y:y+h, x:x + w // 2]
                letter_crop2 = img[y:y+h, x + w // 2:x + w]

                # Resize letter canvas to square
                letters.append(form_sample(x, y, w // 2, h, letter_crop1, out_size))
                letters.append(form_sample(x + w // 2, y, w // 2, h, letter_crop2, out_size))
            else:

                letter_crop = img[y:y+h, x:x + w]

                # Resize letter canvas to square
                letters.append(form_sample(x, y, w, h, letter_crop, out_size))

    # Sort array in place by X-coordinate
    letters.sort(key=lambda x: x[0][0], reverse=False)
    if verbose:
        fig, ax = plt.subplots(ncols=1, nrows=3, figsize=(20,5))
        ax[0].imshow(img, cmap='gray')
        ax[1].imshow(img_erode, cmap='gray')
        ax[2].imshow(cv2.drawContours(img, inner_conts, -1, (0,0,0), 3), cmap='gray')
        for axs in ax:
            axs.set_xticks([])
            axs.set_yticks([])
    print(f"Found {len(inner_conts)} contours")
    return letters


