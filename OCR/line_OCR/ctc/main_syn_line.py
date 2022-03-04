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

import os
import sys
from os.path import dirname
DOSSIER_COURRANT = dirname(os.path.abspath(__file__))
ROOT_FOLDER = dirname(dirname(dirname(DOSSIER_COURRANT)))
sys.path.append(ROOT_FOLDER)
from OCR.line_OCR.ctc.trainer_line_ctc import TrainerLineCTC
from OCR.line_OCR.ctc.models_line_ctc import Decoder
from basic.models import FCN_Encoder
from torch.optim import Adam
from basic.transforms import line_aug_config
from basic.scheduler import exponential_dropout_scheduler, exponential_scheduler
from OCR.ocr_dataset_manager import OCRDataset, OCRDatasetManager
import torch.multiprocessing as mp
import torch
import numpy as np
import random


def train_and_test(rank, params):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    params["training_params"]["ddp_rank"] = rank
    model = TrainerLineCTC(params)

    model.generate_syn_line_dataset("READ_2016_syn_line")  # ["RIMES_syn_line", "READ_2016_syn_line"]


def main():
    dataset_name = "READ_2016"  # ["RIMES", "READ_2016"]
    dataset_level = "page"
    params = {
        "dataset_params": {
            "dataset_manager": OCRDatasetManager,
            "dataset_class": OCRDataset,
            "datasets": {
                dataset_name: "../../../Datasets/formatted/{}_{}".format(dataset_name, dataset_level),
            },
            "train": {
                "name": "{}-train".format(dataset_name),
                "datasets": [(dataset_name, "train"), ],
            },
            "valid": {
                "{}-valid".format(dataset_name): [(dataset_name, "valid"), ],
            },
            "config": {
                "load_in_memory": False,  # Load all images in CPU memory
                "worker_per_gpu": 4,
                "width_divisor": 8,  # Image width will be divided by 8
                "height_divisor": 32,  # Image height will be divided by 32
                "padding_value": 0,  # Image padding value
                "padding_token": 1000,  # Label padding value (None: default value is chosen)
                "padding_mode": "br",  # Padding at bottom and right
                "charset_mode": "CTC",  # add blank token
                "constraints": [],  # Padding for CTC requirements if necessary
                "normalize": True,  # Normalize with mean and variance of training dataset
                "preprocessings": [],
                # Augmentation techniques to use at training time
                "augmentation": line_aug_config(0.9, 0.1),
                #
                "synthetic_data": {
                    "mode": "line_hw_to_printed",
                    "init_proba": 1,
                    "end_proba": 1,
                    "num_steps_proba": 1e5,
                    "proba_scheduler_function": exponential_scheduler,
                    "config": {
                        "background_color_default": (255, 255, 255),
                        "background_color_eps": 15,
                        "text_color_default": (0, 0, 0),
                        "text_color_eps": 15,
                        "font_size_min": 30,
                        "font_size_max": 50,
                        "color_mode": "RGB",
                        "padding_left_ratio_min": 0.02,
                        "padding_left_ratio_max": 0.1,
                        "padding_right_ratio_min": 0.02,
                        "padding_right_ratio_max": 0.1,
                        "padding_top_ratio_min": 0.02,
                        "padding_top_ratio_max": 0.2,
                        "padding_bottom_ratio_min": 0.02,
                        "padding_bottom_ratio_max": 0.2,
                    },
                },
            }
        },

        "model_params": {
            # Model classes to use for each module
            "models": {
                "encoder": FCN_Encoder,
                "decoder": Decoder,
            },
            "transfer_learning": None,
            "input_channels": 3,  # 1 for grayscale images, 3 for RGB ones (or grayscale as RGB)
            "enc_size": 256,
            "dropout_scheduler": {
                "function": exponential_dropout_scheduler,
                "T": 5e4,
            },
            "dropout": 0.5,
        },

        "training_params": {
            "output_folder": "FCN_Encoder_read_syn_line_all_pad_max_cursive",  # folder names for logs and weigths
            "max_nb_epochs": 10000,  # max number of epochs for the training
            "max_training_time": 3600 * 24 * 1.9,  # max training time limit (in seconds)
            "load_epoch": "last",  # ["best", "last"], to load weights from best epoch or last trained epoch
            "interval_save_weights": None,  # None: keep best and last only
            "use_ddp": False,  # Use DistributedDataParallel
            "use_amp": True,  # Enable automatic mix-precision
            "nb_gpu": torch.cuda.device_count(),
            "batch_size": 1,  # mini-batch size per GPU
            "optimizers": {
                "all": {
                    "class": Adam,
                    "args": {
                        "lr": 0.0001,
                        "amsgrad": False,
                    }
                }
            },
            "lr_schedulers": None,
            "eval_on_valid": True,  # Whether to eval and logs metrics on validation set during training or not
            "eval_on_valid_interval": 2,  # Interval (in epochs) to evaluate during training
            "focus_metric": "cer",  # Metrics to focus on to determine best epoch
            "expected_metric_value": "low",  # ["high", "low"] What is best for the focus metric value
            "set_name_focus_metric": "{}-valid".format(dataset_name),
            "train_metrics": ["loss_ctc", "cer", "wer"],  # Metrics name for training
            "eval_metrics": ["loss_ctc", "cer", "wer"],  # Metrics name for evaluation on validation set during training
            "force_cpu": False,  # True for debug purposes to run on cpu only
        },
    }

    if params["training_params"]["use_ddp"] and not params["training_params"]["force_cpu"]:
        mp.spawn(train_and_test, args=(params,), nprocs=params["training_params"]["nb_gpu"])
    else:
        train_and_test(0, params)


if __name__ == "__main__":
    main()