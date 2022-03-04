#  Copyright Université de Rouen Normandie (1), INSA Rouen (2),
#  tutelles du laboratoire LITIS (1 et 2)
#  contributors :
#  - Denis Coquenet
#
#
#  This software is a computer program written in XXX whose purpose is XXX.
#
#  This software is governed by the CeCILL-C license under French law and
#  abiding by the rules of distribution of free software.  You can  use,
#  modify and/ or redistribute the software under the terms of the CeCILL-C
#  license as circulated by CEA, CNRS and INRIA at the following URL
#  "http://www.cecill.info".
#
#  As a counterpart to the access to the source code and  rights to copy,
#  modify and redistribute granted by the license, users are provided only
#  with a limited warranty  and the software's author,  the holder of the
#  economic rights,  and the successive licensors  have only  limited
#  liability.
#
#  In this respect, the user's attention is drawn to the risks associated
#  with loading,  using,  modifying and/or developing or reproducing the
#  software by the user in light of its specific status of free software,
#  that may mean  that it is complicated to manipulate,  and  that  also
#  therefore means  that it is reserved for developers  and  experienced
#  professionals having in-depth computer knowledge. Users are therefore
#  encouraged to load and test the software's suitability as regards their
#  requirements in conditions enabling the security of their systems and/or
#  data to be ensured and,  more generally, to use and operate it in the
#  same conditions as regards security.
#
#  The fact that you are presently reading this means that you have had
#  knowledge of the CeCILL-C license and that you accept its terms.

import numpy as np
from numpy import random
from PIL import Image, ImageOps
from cv2 import erode, dilate, normalize
import cv2
import math
from basic.utils import randint, rand_uniform, rand
from torchvision.transforms import RandomPerspective, RandomCrop, ColorJitter, GaussianBlur, RandomRotation
from torchvision.transforms.functional import InterpolationMode

"""
Each transform class defined here takes as input a PIL Image and returns the modified PIL Image
"""


class SignFlipping:
    """
    Color inversion
    """

    def __init__(self):
        pass

    def __call__(self, x):
        return ImageOps.invert(x)


class DPIAdjusting:
    """
    Resolution modification
    """

    def __init__(self, factor, preserve_ratio):
        self.factor = factor

    def __call__(self, x):
        w, h = x.size
        return x.resize((int(np.ceil(w * self.factor)), int(np.ceil(h * self.factor))), Image.BILINEAR)


class Dilation:
    """
    OCR: stroke width increasing
    """

    def __init__(self, kernel, iterations):
        self.kernel = np.ones(kernel, np.uint8)
        self.iterations = iterations

    def __call__(self, x):
        return Image.fromarray(dilate(np.array(x), self.kernel, iterations=self.iterations))


class Erosion:
    """
    OCR: stroke width decreasing
    """

    def __init__(self, kernel, iterations):
        self.kernel = np.ones(kernel, np.uint8)
        self.iterations = iterations

    def __call__(self, x):
        return Image.fromarray(erode(np.array(x), self.kernel, iterations=self.iterations))


class GaussianNoise:
    """
    Add Gaussian Noise
    """

    def __init__(self, std):
        self.std = std

    def __call__(self, x):
        x_np = np.array(x)
        mean, std = np.mean(x_np), np.std(x_np)
        std = math.copysign(max(abs(std), 0.000001), std)
        min_, max_ = np.min(x_np,), np.max(x_np)
        normal_noise = np.random.randn(*x_np.shape)
        if len(x_np.shape) == 3 and x_np.shape[2] == 3 and np.all(x_np[:, :, 0] == x_np[:, :, 1]) and np.all(x_np[:, :, 0] == x_np[:, :, 2]):
            normal_noise[:, :, 1] = normal_noise[:, :, 2] = normal_noise[:, :, 0]
        x_np = ((x_np-mean)/std + normal_noise*self.std) * std + mean
        x_np = normalize(x_np, x_np, max_, min_, cv2.NORM_MINMAX)

        return Image.fromarray(x_np.astype(np.uint8))


class Sharpen:
    """
    Add Gaussian Noise
    """

    def __init__(self, alpha, strength):
        self.alpha = alpha
        self.strength = strength

    def __call__(self, x):
        x_np = np.array(x)
        id_matrix = np.array([[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0]]
                             )
        effect_matrix = np.array([[1, 1, 1],
                                  [1, -(8+self.strength), 1],
                                  [1, 1, 1]]
                                 )
        kernel = (1 - self.alpha) * id_matrix - self.alpha * effect_matrix
        kernel = np.expand_dims(kernel, axis=2)
        kernel = np.concatenate([kernel, kernel, kernel], axis=2)
        sharpened = cv2.filter2D(x_np, -1, kernel=kernel[:, :, 0])
        return Image.fromarray(sharpened.astype(np.uint8))


class ZoomRatio:
    """
        Crop by ratio
        Preserve dimensions if keep_dim = True (= zoom)
    """

    def __init__(self, ratio_h, ratio_w, keep_dim=True):
        self.ratio_w = ratio_w
        self.ratio_h = ratio_h
        self.keep_dim = keep_dim

    def __call__(self, x):
        w, h = x.size
        x = RandomCrop((int(h * self.ratio_h), int(w * self.ratio_w)))(x)
        if self.keep_dim:
            x = x.resize((w, h), Image.BILINEAR)
        return x


class ElasticDistortion:

    def __init__(self, kernel_size=(7, 7), sigma=5, alpha=1):

        self.kernel_size = kernel_size
        self.sigma = sigma
        self.alpha = alpha

    def __call__(self, x):
        x_np = np.array(x)

        h, w = x_np.shape[:2]

        dx = np.random.uniform(-1, 1, (h, w))
        dy = np.random.uniform(-1, 1, (h, w))

        x_gauss = cv2.GaussianBlur(dx, self.kernel_size, self.sigma)
        y_gauss = cv2.GaussianBlur(dy, self.kernel_size, self.sigma)

        n = np.sqrt(x_gauss**2 + y_gauss**2)

        nd_x = self.alpha * x_gauss / n
        nd_y = self.alpha * y_gauss / n

        ind_y, ind_x = np.indices((h, w), dtype=np.float32)

        map_x = nd_x + ind_x
        map_x = map_x.reshape(h, w).astype(np.float32)
        map_y = nd_y + ind_y
        map_y = map_y.reshape(h, w).astype(np.float32)

        dst = cv2.remap(x_np, map_x, map_y, cv2.INTER_LINEAR)
        return Image.fromarray(dst.astype(np.uint8))


class Tightening:
    """
    Reduce interline spacing
    """

    def __init__(self, color=255, remove_proba=0.75):
        self.color = color
        self.remove_proba = remove_proba

    def __call__(self, x):
        x_np = np.array(x)
        interline_indices = [np.all(line == 255) for line in x_np]
        indices_to_removed = np.logical_and(np.random.choice([True, False], size=len(x_np), replace=True, p=[self.remove_proba, 1-self.remove_proba]), interline_indices)
        new_x = x_np[np.logical_not(indices_to_removed)]
        return Image.fromarray(new_x.astype(np.uint8))


def get_list_augmenters(img, aug_configs, fill_value):
    """
    Randomly select a list of data augmentation techniques to used based on aug_configs
    """
    augmenters = list()
    for aug_config in aug_configs:
        if rand() > aug_config["proba"]:
            continue
        if aug_config["type"] == "dpi":
            valid_factor = False
            while not valid_factor:
                factor = rand_uniform(aug_config["min_factor"], aug_config["max_factor"])
                valid_factor = not (("max_width" in aug_config and factor*img.size[0] > aug_config["max_width"]) or \
                               ("max_height" in aug_config and factor * img.size[1] > aug_config["max_height"]) or \
                               ("min_width" in aug_config and factor*img.size[0] < aug_config["min_width"]) or \
                               ("min_height" in aug_config and factor * img.size[1] < aug_config["min_height"]))
            augmenters.append(DPIAdjusting(factor, preserve_ratio=aug_config["preserve_ratio"]))

        elif aug_config["type"] == "zoom_ratio":
            ratio_h = rand_uniform(aug_config["min_ratio_h"], aug_config["max_ratio_h"])
            ratio_w = rand_uniform(aug_config["min_ratio_w"], aug_config["max_ratio_w"])
            augmenters.append(ZoomRatio(ratio_h=ratio_h, ratio_w=ratio_w, keep_dim=aug_config["keep_dim"]))

        elif aug_config["type"] == "perspective":
            scale = rand_uniform(aug_config["min_factor"], aug_config["max_factor"])
            augmenters.append(RandomPerspective(distortion_scale=scale, p=1, interpolation=InterpolationMode.BILINEAR, fill=fill_value))

        elif aug_config["type"] == "elastic_distortion":
            kernel_size = randint(aug_config["min_kernel_size"], aug_config["max_kernel_size"]) // 2 * 2 + 1
            sigma = rand_uniform(aug_config["min_sigma"], aug_config["max_sigma"])
            alpha= rand_uniform(aug_config["min_alpha"], aug_config["max_alpha"])
            augmenters.append(ElasticDistortion(kernel_size=(kernel_size, kernel_size), sigma=sigma, alpha=alpha))

        elif aug_config["type"] == "dilation_erosion":
            kernel_h = randint(aug_config["min_kernel"], aug_config["max_kernel"] + 1)
            kernel_w = randint(aug_config["min_kernel"], aug_config["max_kernel"] + 1)
            if randint(0, 2) == 0:
                augmenters.append(Erosion((kernel_w, kernel_h), aug_config["iterations"]))
            else:
                augmenters.append(Dilation((kernel_w, kernel_h), aug_config["iterations"]))

        elif aug_config["type"] == "color_jittering":
            augmenters.append(ColorJitter(contrast=aug_config["factor_contrast"],
                              brightness=aug_config["factor_brightness"],
                              saturation=aug_config["factor_saturation"],
                              hue=aug_config["factor_hue"],
                              ))

        elif aug_config["type"] == "gaussian_blur":
            max_kernel_h = min(aug_config["max_kernel"], img.size[1])
            max_kernel_w = min(aug_config["max_kernel"], img.size[0])
            kernel_h = randint(aug_config["min_kernel"], max_kernel_h + 1) // 2 * 2 + 1
            kernel_w = randint(aug_config["min_kernel"], max_kernel_w + 1) // 2 * 2 + 1
            sigma = rand_uniform(aug_config["min_sigma"], aug_config["max_sigma"])
            augmenters.append(GaussianBlur(kernel_size=(kernel_w, kernel_h), sigma=sigma))

        elif aug_config["type"] == "gaussian_noise":
            augmenters.append(GaussianNoise(std=aug_config["std"]))

        elif aug_config["type"] == "sharpen":
            alpha = rand_uniform(aug_config["min_alpha"], aug_config["max_alpha"])
            strength = rand_uniform(aug_config["min_strength"], aug_config["max_strength"])
            augmenters.append(Sharpen(alpha=alpha, strength=strength))

        else:
            print("Error - unknown augmentor: {}".format(aug_config["type"]))
            exit(-1)

    return augmenters


def apply_data_augmentation(img, da_config):
    """
    Apply data augmentation strategy on input image
    """
    applied_da = list()
    if da_config["proba"] != 1 and rand() > da_config["proba"]:
        return img, applied_da

    # Convert to PIL Image
    img = img[:, :, 0] if img.shape[2] == 1 else img
    img = Image.fromarray(img)

    fill_value = da_config["fill_value"] if "fill_value" in da_config else 255
    augmenters = get_list_augmenters(img, da_config["augmentations"], fill_value=fill_value)
    if da_config["order"] == "random":
        random.shuffle(augmenters)

    for augmenter in augmenters:
        img = augmenter(img)
        applied_da.append(type(augmenter).__name__)

    # convert to numpy array
    img = np.array(img)
    img = np.expand_dims(img, axis=2) if len(img.shape) == 2 else img
    return img, applied_da


def apply_transform(img, transform):
    """
    Apply data augmentation technique on input image
    """
    img = img[:, :, 0] if img.shape[2] == 1 else img
    img = Image.fromarray(img)
    img = transform(img)
    img = np.array(img)
    return np.expand_dims(img, axis=2) if len(img.shape) == 2 else img


def line_aug_config(proba_use_da, p):
    return {
        "order": "random",
        "proba": proba_use_da,
        "augmentations": [
            {
                "type": "dpi",
                "proba": p,
                "min_factor": 0.5,
                "max_factor": 1.5,
                "preserve_ratio": True,
            },
            {
                "type": "perspective",
                "proba": p,
                "min_factor": 0,
                "max_factor": 0.4,
            },
            {
                "type": "elastic_distortion",
                "proba": p,
                "min_alpha": 0.5,
                "max_alpha": 1,
                "min_sigma": 1,
                "max_sigma": 10,
                "min_kernel_size": 3,
                "max_kernel_size": 9,
            },
            {
                "type": "dilation_erosion",
                "proba": p,
                "min_kernel": 1,
                "max_kernel": 3,
                "iterations": 1,
            },
            {
                "type": "color_jittering",
                "proba": p,
                "factor_hue": 0.2,
                "factor_brightness": 0.4,
                "factor_contrast": 0.4,
                "factor_saturation": 0.4,
            },
            {
                "type": "gaussian_blur",
                "proba": p,
                "min_kernel": 3,
                "max_kernel": 5,
                "min_sigma": 3,
                "max_sigma": 5,
            },
            {
                "type": "gaussian_noise",
                "proba": p,
                "std": 0.5,
            },
            {
                "type": "sharpen",
                "proba": p,
                "min_alpha": 0,
                "max_alpha": 1,
                "min_strength": 0,
                "max_strength": 1,
            },
            {
                "type": "zoom_ratio",
                "proba": p,
                "min_ratio_h": 0.8,
                "max_ratio_h": 1,
                "min_ratio_w": 0.99,
                "max_ratio_w": 1,
                "keep_dim": True
            },
        ]
    }


def aug_config(proba_use_da, p):
    return {
        "order": "random",
        "proba": proba_use_da,
        "augmentations": [
            {
                "type": "dpi",
                "proba": p,
                "min_factor": 0.75,
                "max_factor": 1,
                "preserve_ratio": True,
            },
            {
                "type": "perspective",
                "proba": p,
                "min_factor": 0,
                "max_factor": 0.4,
            },
            {
                "type": "elastic_distortion",
                "proba": p,
                "min_alpha": 0.5,
                "max_alpha": 1,
                "min_sigma": 1,
                "max_sigma": 10,
                "min_kernel_size": 3,
                "max_kernel_size": 9,
            },
            {
                "type": "dilation_erosion",
                "proba": p,
                "min_kernel": 1,
                "max_kernel": 3,
                "iterations": 1,
            },
            {
                "type": "color_jittering",
                "proba": p,
                "factor_hue": 0.2,
                "factor_brightness": 0.4,
                "factor_contrast": 0.4,
                "factor_saturation": 0.4,
            },
            {
                "type": "gaussian_blur",
                "proba": p,
                "min_kernel": 3,
                "max_kernel": 5,
                "min_sigma": 3,
                "max_sigma": 5,
            },
            {
                "type": "gaussian_noise",
                "proba": p,
                "std": 0.5,
            },
            {
                "type": "sharpen",
                "proba": p,
                "min_alpha": 0,
                "max_alpha": 1,
                "min_strength": 0,
                "max_strength": 1,
            },
        ]
    }
