import random

import cv2
import imutils
import numpy as np


class BaseAugmenter():
    def __init__(self, chanse=0.5):
        self.chance = chanse

    def __call__(self, img, *args, **kwargs):
        augmented = False
        if random.random() <= self.chance:
            try:
                img = self._apply(img, *args, **kwargs)
                augmented = True
            except Exception as e:
                print(e)

        return img, augmented

    def _apply(self, img, *args, **kwargs):
        return img


class NoiseAugmenter(BaseAugmenter):
    def _apply(self, letter, max=30):
        noise = np.random.normal(size=(letter.shape)) * max
        return letter + noise


class FlipAugmenter(BaseAugmenter):

    def _apply(self, letter, mode='horizontal'):
        if mode == 'horizontal':
            return np.flip(letter, axis=0)
        elif mode == 'vertical':
            return np.flip(letter, axis=1)
        elif mode == 'random':
            axis = random.random() < 0.5
            return np.flip(letter, axis=int(axis))


class RotationAugmenter(BaseAugmenter):

    def _apply(self, letter, angle=60):
        image = np.zeros_like(letter) + 255 - letter
        rotated = imutils.rotate_bound(image, angle)
        rotated = np.zeros_like(rotated) + 255 - rotated
        rotated = cv2.resize(rotated, (letter.shape[1], letter.shape[0]))
        return rotated


class BlurAugmenter(BaseAugmenter):

    def _apply(self, letter, kernel_size=3):
        return cv2.medianBlur(letter, kernel_size)


class MultipleAugmenter():
    NOISE_CHANCE = 0.1
    NOISE_HARDNESS = 20

    FLIP_CHANCE = 0.3

    ROTATE_CHANCE = 0.15

    BLUR_CHANCE = 0.1
    BLUR_HARD_CHANCE = 0.05

    images = []
    captions = []

    def __init__(self, images, captions):
        self.images_orig = images
        self.captions_orig = captions
        self.augmenters = [
            FlipAugmenter(self.FLIP_CHANCE),
            RotationAugmenter(self.ROTATE_CHANCE),
            BlurAugmenter(self.BLUR_CHANCE),
            BlurAugmenter(self.BLUR_HARD_CHANCE),

        ]
        self.noise = (NoiseAugmenter(self.NOISE_CHANCE), self.NOISE_HARDNESS)

        self.augment_args = [
            ['random'],
            [],
            [3],
            [5],

        ]

    def random_augment(self, rounds=1, seed=42, save_last=False):
        random.seed(seed)

        images = self.images_orig
        captions = self.captions_orig

        for _ in range(rounds):
            print(len(images))
            num_images = len(images)
            for n in range(num_images):
                image = images[n]
                caption = captions[n]

                for n, aug in enumerate(self.augmenters):
                    aug_args = self.augment_args[n]
                    if not aug_args:
                        aug_args = [random.randint(-15, 15)]
                    image, augmented = aug(image, *aug_args)
                    if augmented:
                        images.append(image)
                        captions.append(caption)

            if save_last:
                images = images[num_images:]
                captions = captions[num_images:]

        self.images = images
        self.captions = captions
        self._apply_noise()
        self._normalize()

    def _apply_noise(self):
        for n, im in enumerate(self.images):
            noise_augmenter, hardeness = self.noise
            image, augmented = noise_augmenter(im, hardeness)
            if augmented:
                self.images[n] = image

    def _normalize(self):
        for n, im in enumerate(self.images):
            im = np.array(im)
            self.images[n] = im / im.max()