from utils.letter_extraction import form_sample
import cv2
import numpy as np
import matplotlib.pyplot as plt


def word_extract(img, out_size=0,
                 scale_factor=8,
                 erode_core=5,
                 num_erosions=7,
                 split_trashold=200,
                 split=False,
                 verbose=False):
    letters = []

    img = cv2.resize(img, (img.shape[1] * scale_factor, img.shape[0] * scale_factor), interpolation=cv2.INTER_LINEAR)
    original = img.copy()

    img = cv2.medianBlur(img, 9)
    blured = img.copy()

    thresh = cv2.adaptiveThreshold(img, 255, 1, cv2.THRESH_BINARY, 21, 5)

    img_erode = cv2.erode(thresh, np.ones((erode_core, erode_core), np.uint8), iterations=num_erosions)
    # Get contours
    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    print('contours:', len(contours))

    hir = hierarchy[0, :, 3]

    inner_conts = []
    sizes = []
    for idx, contour in enumerate(contours):
        if hir[idx] == 0:
            (x, y, w, h) = cv2.boundingRect(contour)

            sizes.append([h, w])

            inner_conts.append(contour)

            if split and w > img.shape[1] * scale_factor / split_trashold:
                letter_crop1 = img[y:y + h, x:x + w // 2]
                letter_crop2 = img[y:y + h, x + w // 2:x + w]

                # Resize letter canvas to square
                letters.append(((x, y), w // 2, h, letter_crop1))
                letters.append(((x + w // 2, y), w // 2, h, letter_crop2))
            else:

                letter_crop = img[y:y + h, x:x + w]

                # Resize letter canvas to square
                letters.append(((x, y), w, h, letter_crop))

    # Sort array in place by X-coordinate
    letters.sort(key=lambda x: x[0][0], reverse=False)

    if len(letters) > 1:
        drop_list = []
        for i in range(0, len(letters)):

            x0, y1 = letters[i][0]
            x1 = (x0 + letters[i][1]) / 2
            h1 = letters[i][2]
            for j in range(0, len(letters)):
                if i == j:
                    continue
                x2, y2 = letters[j][0]
                h2 = letters[j][2]

                if h1 + h2 >= img.shape[0]:
                    continue

                if abs(x2 - x1) < 5 and abs(y2 - y1) > 10 and not j in drop_list:
                    if y2 < y1:
                        top_let = letters[j][-1]
                        bot_let = letters[i][-1]
                    else:
                        top_let = letters[i][-1]
                        bot_let = letters[j][-1]

                    width = max(top_let.shape[1], bot_let.shape[1])
                    high = top_let.shape[0] + bot_let.shape[0]

                    merged = np.ones((high, width), dtype=np.uint8) * 255

                    print(top_let.shape, bot_let.shape, merged.shape)

                    x_pos_top = width // 2 - top_let.shape[1] // 2
                    x_pos_bot = width // 2 - bot_let.shape[1] // 2

                    print(x_pos_top, x_pos_bot)

                    merged[:top_let.shape[0], x_pos_top:x_pos_top + top_let.shape[1]] = top_let[:, :]
                    merged[top_let.shape[0]:, x_pos_bot:x_pos_bot + bot_let.shape[1]] = bot_let[:, :]

                    letters[i] = ((x0, y1), merged.shape[1], merged.shape[0], merged)
                    drop_list.append(j)

        for i in range(len(letters)):
            if i in drop_list:
                continue
            let_w, let_h = letters[i][1:3]
            if let_h * let_w < 1000:
                # print(f'{i} small')
                drop_list.append(i)

    if verbose:
        fig, ax = plt.subplots(ncols=1, nrows=5, figsize=(20, 5))
        ax[0].imshow(original, cmap='gray')
        ax[0].set_title('Original Image')
        ax[1].imshow(blured, cmap='gray')
        ax[1].set_title('Blured Image')
        ax[2].imshow(thresh, cmap='gray')
        ax[2].set_title('Tresholde Image')
        ax[3].imshow(img_erode, cmap='gray')
        ax[3].set_title('Eroded Image')
        ax[4].imshow(cv2.drawContours(img, inner_conts, -1, (0, 0, 0), 3), cmap='gray')
        ax[4].set_title('Contours of Image')
        for axs in ax:
            axs.set_xticks([])
            axs.set_yticks([])
    print(f"Found {len(inner_conts)} contours")

    output = []
    for i in range(len(letters)):
        if i in drop_list:
            continue
        letter = letters[i]
        output.append(form_sample(*letter[0], *letter[1:], out_size))
    return output


def slicing_window_transform(img, window_width_factor=0.5, window_step=1, output_size=28):
    windows = []

    h, w = img.shape

    if 1 - h / w < 0.1:
        windows = [cv2.resize(img, (output_size,) * 2, interpolation=cv2.INTER_AREA)]
    else:
        x = 0
        window_w = int(window_width_factor * h)
        while x < w - window_w:

            size_max = max(window_w, h)
            patch_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)

            letter_crop = img[:, x:x + window_w]

            if window_w > h:
                # Enlarge image top-bottom
                # ------
                # ======
                # ------
                y_pos = size_max // 2 - h // 2
                patch_square[y_pos:y_pos + h, :window_w] = letter_crop[:h, :window_w]
            elif window_w < h:
                # Enlarge image left-right
                # --||--
                x_pos = size_max // 2 - window_w // 2
                patch_square[:h, x_pos:x_pos + window_w] = letter_crop[:h, :window_w]
            else:
                patch_square = letter_crop

            windows.append(cv2.resize(patch_square, (output_size,) * 2, interpolation=cv2.INTER_AREA))
            x += int(window_w * window_step)
    print(f'Split to {len(windows)} images')
    return windows
