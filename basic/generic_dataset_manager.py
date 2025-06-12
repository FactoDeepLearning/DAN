#  Copyright Université de Rouen Normandie (1), INSA Rouen (2),
#  tutelles du laboratoire LITIS (1 et 2)
#  contributors :
#  - Denis Coquenet
#
#  This software is a computer program written in Python whose purpose is 
#  to recognize text and layout from full-page images with end-to-end deep neural networks.
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

import torch
import random
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from basic.transforms import apply_data_augmentation
from Datasets.dataset_formatters.utils_dataset import natural_sort
import os
import numpy as np
import pickle
from PIL import Image
import cv2


class DatasetManager:

    def __init__(self, params):
        self.params = params
        self.dataset_class = params["dataset_class"]
        self.img_padding_value = params["config"]["padding_value"]

        self.my_collate_function = None

        self.train_dataset = None
        self.valid_datasets = dict()
        self.test_datasets = dict()

        self.train_loader = None
        self.valid_loaders = dict()
        self.test_loaders = dict()

        self.train_sampler = None
        self.valid_samplers = dict()
        self.test_samplers = dict()

        self.generator = torch.Generator()
        self.generator.manual_seed(0)

        self.batch_size = {
            "train": self.params["batch_size"],
            "valid": self.params["valid_batch_size"] if "valid_batch_size" in self.params else self.params["batch_size"],
            "test": self.params["test_batch_size"] if "test_batch_size" in self.params else 1,
        }

    def apply_specific_treatment_after_dataset_loading(self, dataset):
        raise NotImplementedError

    def load_datasets(self):
        """
        Load training and validation datasets
        """
        self.train_dataset = self.dataset_class(self.params, "train", self.params["train"]["name"], self.get_paths_and_sets(self.params["train"]["datasets"]))
        self.params["config"]["mean"], self.params["config"]["std"] = self.train_dataset.compute_std_mean()

        self.my_collate_function = self.train_dataset.collate_function(self.params["config"])
        self.apply_specific_treatment_after_dataset_loading(self.train_dataset)

        for custom_name in self.params["valid"].keys():
            self.valid_datasets[custom_name] = self.dataset_class(self.params, "valid", custom_name, self.get_paths_and_sets(self.params["valid"][custom_name]))
            self.apply_specific_treatment_after_dataset_loading(self.valid_datasets[custom_name])

    def load_ddp_samplers(self):
        """
        Load training and validation data samplers
        """
        if self.params["use_ddp"]:
            self.train_sampler = DistributedSampler(self.train_dataset, num_replicas=self.params["num_gpu"], rank=self.params["ddp_rank"], shuffle=True)
            for custom_name in self.valid_datasets.keys():
                self.valid_samplers[custom_name] = DistributedSampler(self.valid_datasets[custom_name], num_replicas=self.params["num_gpu"], rank=self.params["ddp_rank"], shuffle=False)
        else:
            for custom_name in self.valid_datasets.keys():
                self.valid_samplers[custom_name] = None

    def load_dataloaders(self):
        """
        Load training and validation data loaders
        """
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.batch_size["train"],
                                       shuffle=True if self.train_sampler is None else False,
                                       drop_last=False,
                                       batch_sampler=self.train_sampler,
                                       sampler=self.train_sampler,
                                       num_workers=self.params["num_gpu"]*self.params["worker_per_gpu"],
                                       pin_memory=True,
                                       collate_fn=self.my_collate_function,
                                       worker_init_fn=self.seed_worker,
                                       generator=self.generator)

        for key in self.valid_datasets.keys():
            self.valid_loaders[key] = DataLoader(self.valid_datasets[key],
                                                 batch_size=self.batch_size["valid"],
                                                 sampler=self.valid_samplers[key],
                                                 batch_sampler=self.valid_samplers[key],
                                                 shuffle=False,
                                                 num_workers=self.params["num_gpu"]*self.params["worker_per_gpu"],
                                                 pin_memory=True,
                                                 drop_last=False,
                                                 collate_fn=self.my_collate_function,
                                                 worker_init_fn=self.seed_worker,
                                                 generator=self.generator)

    @staticmethod
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def generate_test_loader(self, custom_name, sets_list):
        """
        Load test dataset, data sampler and data loader
        """
        if custom_name in self.test_loaders.keys():
            return
        paths_and_sets = list()
        for set_info in sets_list:
            paths_and_sets.append({
                "path": self.params["datasets"][set_info[0]],
                "set_name": set_info[1]
            })
        self.test_datasets[custom_name] = self.dataset_class(self.params, "test", custom_name, paths_and_sets)
        self.apply_specific_treatment_after_dataset_loading(self.test_datasets[custom_name])
        if self.params["use_ddp"]:
            self.test_samplers[custom_name] = DistributedSampler(self.test_datasets[custom_name], num_replicas=self.params["num_gpu"], rank=self.params["ddp_rank"], shuffle=False)
        else:
            self.test_samplers[custom_name] = None
        self.test_loaders[custom_name] = DataLoader(self.test_datasets[custom_name],
                                                    batch_size=self.batch_size["test"],
                                                    sampler=self.test_samplers[custom_name],
                                                    shuffle=False,
                                                    num_workers=self.params["num_gpu"]*self.params["worker_per_gpu"],
                                                    pin_memory=True,
                                                    drop_last=False,
                                                    collate_fn=self.my_collate_function,
                                                    worker_init_fn=self.seed_worker,
                                                    generator=self.generator)

    def remove_test_dataset(self, custom_name):
        del self.test_datasets[custom_name]
        del self.test_samplers[custom_name]
        del self.test_loaders[custom_name]

    def remove_valid_dataset(self, custom_name):
        del self.valid_datasets[custom_name]
        del self.valid_samplers[custom_name]
        del self.valid_loaders[custom_name]

    def remove_train_dataset(self):
        self.train_dataset = None
        self.train_sampler = None
        self.train_loader = None

    def remove_all_datasets(self):
        self.remove_train_dataset()
        for name in list(self.valid_datasets.keys()):
            self.remove_valid_dataset(name)
        for name in list(self.test_datasets.keys()):
            self.remove_test_dataset(name)

    def get_paths_and_sets(self, dataset_names_folds):
        paths_and_sets = list()
        for dataset_name, fold in dataset_names_folds:
            path = self.params["datasets"][dataset_name]
            paths_and_sets.append({
                "path": path,
                "set_name": fold
            })
        return paths_and_sets


class GenericDataset(Dataset):
    """
    Main class to handle dataset loading
    """

    def __init__(self, params, set_name, custom_name, paths_and_sets):
        self.params = params
        self.name = custom_name
        self.set_name = set_name
        self.mean = np.array(params["config"]["mean"]) if "mean" in params["config"].keys() else None
        self.std = np.array(params["config"]["std"]) if "std" in params["config"].keys() else None

        self.load_in_memory = self.params["config"]["load_in_memory"] if "load_in_memory" in self.params["config"] else True

        self.samples = self.load_samples(paths_and_sets, load_in_memory=self.load_in_memory)

        if self.load_in_memory:
            self.apply_preprocessing(params["config"]["preprocessings"])

        self.padding_value = params["config"]["padding_value"]
        if self.padding_value == "mean":
            if self.mean is None:
                _, _ = self.compute_std_mean()
            self.padding_value = self.mean
            self.params["config"]["padding_value"] = self.padding_value

        self.curriculum_config = None
        self.training_info = None

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def load_image(path):
        with Image.open(path) as pil_img:
            img = np.array(pil_img)
            ## grayscale images
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=2)
        return img

    @staticmethod
    def load_samples(paths_and_sets, load_in_memory=True):
        """
        Load images and labels
        """
        samples = list()
        for path_and_set in paths_and_sets:
            path = path_and_set["path"]
            set_name = path_and_set["set_name"]
            with open(os.path.join(path, "labels.pkl"), "rb") as f:
                info = pickle.load(f)
                gt = info["ground_truth"][set_name]
                for filename in natural_sort(gt.keys()):
                    name = os.path.join(os.path.basename(path), set_name, filename)
                    full_path = os.path.join(path, set_name, filename)
                    if isinstance(gt[filename], dict) and "text" in gt[filename]:
                        label = gt[filename]["text"]
                    else:
                        label = gt[filename]
                    samples.append({
                        "name": name,
                        "label": label,
                        "unchanged_label": label,
                        "path": full_path,
                        "nb_cols": 1 if "nb_cols" not in gt[filename] else gt[filename]["nb_cols"]
                    })
                    if load_in_memory:
                        samples[-1]["img"] = GenericDataset.load_image(full_path)
                    if type(gt[filename]) is dict:
                        if "lines" in gt[filename].keys():
                            samples[-1]["raw_line_seg_label"] = gt[filename]["lines"]
                        if "paragraphs" in gt[filename].keys():
                            samples[-1]["paragraphs_label"] = gt[filename]["paragraphs"]
                        if "pages" in gt[filename].keys():
                            samples[-1]["pages_label"] = gt[filename]["pages"]
        return samples

    def apply_preprocessing(self, preprocessings):
        for i in range(len(self.samples)):
            self.samples[i] = apply_preprocessing(self.samples[i], preprocessings)

    def compute_std_mean(self):
        """
        Compute cumulated variance and mean of whole dataset
        """
        if self.mean is not None and self.std is not None:
            return self.mean, self.std
        if not self.load_in_memory:
            sample = self.samples[0].copy()
            sample["img"] = self.get_sample_img(0)
            img = apply_preprocessing(sample, self.params["config"]["preprocessings"])["img"]
        else:
            img = self.get_sample_img(0)
        _, _, c = img.shape
        sum = np.zeros((c,))
        nb_pixels = 0

        for i in range(len(self.samples)):
            if not self.load_in_memory:
                sample = self.samples[i].copy()
                sample["img"] = self.get_sample_img(i)
                img = apply_preprocessing(sample, self.params["config"]["preprocessings"])["img"]
            else:
                img = self.get_sample_img(i)
            sum += np.sum(img, axis=(0, 1))
            nb_pixels += np.prod(img.shape[:2])
        mean = sum / nb_pixels
        diff = np.zeros((c,))
        for i in range(len(self.samples)):
            if not self.load_in_memory:
                sample = self.samples[i].copy()
                sample["img"] = self.get_sample_img(i)
                img = apply_preprocessing(sample, self.params["config"]["preprocessings"])["img"]
            else:
                img = self.get_sample_img(i)
            diff += [np.sum((img[:, :, k] - mean[k]) ** 2) for k in range(c)]
        std = np.sqrt(diff / nb_pixels)

        self.mean = mean
        self.std = std
        return mean, std

    def apply_data_augmentation(self, img):
        """
        Apply data augmentation strategy on the input image
        """
        augs = [self.params["config"][key] if key in self.params["config"].keys() else None for key in ["augmentation", "valid_augmentation", "test_augmentation"]]
        for aug, set_name in zip(augs, ["train", "valid", "test"]):
            if aug and self.set_name == set_name:
                return apply_data_augmentation(img, aug)
        return img, list()

    def get_sample_img(self, i):
        """
        Get image by index
        """
        if self.load_in_memory:
            return self.samples[i]["img"]
        else:
            return GenericDataset.load_image(self.samples[i]["path"])

    def denormalize(self, img):
        """
        Get original image, before normalization
        """
        return img * self.std + self.mean


def apply_preprocessing(sample, preprocessings):
    """
    Apply preprocessings on each sample
    """
    resize_ratio = [1, 1]
    img = sample["img"]
    for preprocessing in preprocessings:

        if preprocessing["type"] == "dpi":
            ratio = preprocessing["target"] / preprocessing["source"]
            temp_img = img
            h, w, c = temp_img.shape
            temp_img = cv2.resize(temp_img, (int(np.ceil(w * ratio)), int(np.ceil(h * ratio))))
            if len(temp_img.shape) == 2:
                temp_img = np.expand_dims(temp_img, axis=2)
            img = temp_img

            resize_ratio = [ratio, ratio]

        if preprocessing["type"] == "to_grayscaled":
            temp_img = img
            h, w, c = temp_img.shape
            if c == 3:
                img = np.expand_dims(
                    0.2125 * temp_img[:, :, 0] + 0.7154 * temp_img[:, :, 1] + 0.0721 * temp_img[:, :, 2],
                    axis=2).astype(np.uint8)

        if preprocessing["type"] == "to_RGB":
            temp_img = img
            h, w, c = temp_img.shape
            if c == 1:
                img = np.concatenate([temp_img, temp_img, temp_img], axis=2)

        if preprocessing["type"] == "resize":
            keep_ratio = preprocessing["keep_ratio"]
            max_h, max_w = preprocessing["max_height"], preprocessing["max_width"]
            temp_img = img
            h, w, c = temp_img.shape

            ratio_h = max_h / h if max_h else 1
            ratio_w = max_w / w if max_w else 1
            if keep_ratio:
                ratio_h = ratio_w = min(ratio_w, ratio_h)
            new_h = min(max_h, int(h * ratio_h))
            new_w = min(max_w, int(w * ratio_w))
            temp_img = cv2.resize(temp_img, (new_w, new_h))
            if len(temp_img.shape) == 2:
                temp_img = np.expand_dims(temp_img, axis=2)

            img = temp_img
            resize_ratio = [ratio_h, ratio_w]

        if preprocessing["type"] == "fixed_height":
            new_h = preprocessing["height"]
            temp_img = img
            h, w, c = temp_img.shape
            ratio = new_h / h
            temp_img = cv2.resize(temp_img, (int(w*ratio), new_h))
            if len(temp_img.shape) == 2:
                temp_img = np.expand_dims(temp_img, axis=2)
            img = temp_img
            resize_ratio = [ratio, ratio]
    if resize_ratio != [1, 1] and "raw_line_seg_label" in sample:
        for li in range(len(sample["raw_line_seg_label"])):
            for side, ratio in zip((["bottom", "top"], ["right", "left"]), resize_ratio):
                for s in side:
                    sample["raw_line_seg_label"][li][s] = sample["raw_line_seg_label"][li][s] * ratio

    sample["img"] = img
    sample["resize_ratio"] = resize_ratio
    return sample

