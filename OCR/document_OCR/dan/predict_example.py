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

import os.path

import torch
from torch.optim import Adam
from PIL import Image
import numpy as np

from basic.models import FCN_Encoder
from OCR.document_OCR.dan.models_dan import GlobalHTADecoder
from OCR.document_OCR.dan.trainer_dan import Manager
from basic.utils import pad_images
from basic.metric_manager import keep_all_but_tokens


class FakeDataset:

    def __init__(self, charset):
        self.charset = charset

        self.tokens = {
            "end": len(self.charset),
            "start": len(self.charset) + 1,
            "pad": len(self.charset) + 2,
        }


def get_params(weight_path):
    return {
        "dataset_params": {
            "charset": None,
        },
        "model_params": {
            "models": {
                "encoder": FCN_Encoder,
                "decoder": GlobalHTADecoder,
            },
            # "transfer_learning": None,
            "transfer_learning": {
                # model_name: [state_dict_name, checkpoint_path, learnable, strict]
                "encoder": ["encoder", weight_path, True, True],
                "decoder": ["decoder", weight_path, True, False],
            },
            "transfered_charset": True,  # Transfer learning of the decision layer based on charset of the line HTR model
            "additional_tokens": 1,  # for decision layer = [<eot>, ], only for transfered charset

            "input_channels": 3,  # number of channels of input image
            "dropout": 0.5,  # dropout rate for encoder
            "enc_dim": 256,  # dimension of extracted features
            "nb_layers": 5,  # encoder
            "h_max": 500,  # maximum height for encoder output (for 2D positional embedding)
            "w_max": 1000,  # maximum width for encoder output (for 2D positional embedding)
            "l_max": 15000,  # max predicted sequence (for 1D positional embedding)
            "dec_num_layers": 8,  # number of transformer decoder layers
            "dec_num_heads": 4,  # number of heads in transformer decoder layers
            "dec_res_dropout": 0.1,  # dropout in transformer decoder layers
            "dec_pred_dropout": 0.1,  # dropout rate before decision layer
            "dec_att_dropout": 0.1,  # dropout rate in multi head attention
            "dec_dim_feedforward": 256,  # number of dimension for feedforward layer in transformer decoder layers
            "use_2d_pe": True,  # use 2D positional embedding
            "use_1d_pe": True,  # use 1D positional embedding
            "use_lstm": False,
            "attention_win": 100,  # length of attention window
        },

        "training_params": {
            "output_folder": "dan_rimes_page",  # folder name for checkpoint and results
            "max_nb_epochs": 50000,  # maximum number of epochs before to stop
            "max_training_time": 3600 * 24 * 1.9,  # maximum time before to stop (in seconds)
            "load_epoch": "last",  # ["best", "last"]: last to continue training, best to evaluate
            "interval_save_weights": None,  # None: keep best and last only
            "batch_size": 1,  # mini-batch size for training
            "valid_batch_size": 4,  # mini-batch size for valdiation
            "use_ddp": False,  # Use DistributedDataParallel
            "ddp_port": "20027",
            "use_amp": True,  # Enable automatic mix-precision
            "nb_gpu": torch.cuda.device_count(),
            "ddp_rank": 0,
            "lr_schedulers": None,  # Learning rate schedulers
            "eval_on_valid": True,  # Whether to eval and logs metrics on validation set during training or not
            "eval_on_valid_interval": 5,  # Interval (in epochs) to evaluate during training
            "focus_metric": "cer",  # Metrics to focus on to determine best epoch
            "expected_metric_value": "low",  # ["high", "low"] What is best for the focus metric value
            "eval_metrics": ["cer", "wer", "map_cer"],  # Metrics name for evaluation on validation set during training
            "force_cpu": True,  # True for debug purposes
            "max_char_prediction": 3000,  # max number of token prediction
            # Keep teacher forcing rate to 20% during whole training
            "teacher_forcing_scheduler": {
                "min_error_rate": 0.2,
                "max_error_rate": 0.2,
                "total_num_steps": 5e4
            },
            "optimizers": {
                "all": {
                    "class": Adam,
                    "args": {
                        "lr": 0.0001,
                        "amsgrad": False,
                    }
                },
            },
        },
    }


def predict(model_path, img_paths):
    params = get_params(model_path)
    checkpoint = torch.load(model_path, map_location="cpu")
    charset = checkpoint["charset"]

    manager = Manager(params)
    manager.params["model_params"]["vocab_size"] = len(charset)
    manager.load_model()
    for model_name in manager.models.keys():
        manager.models[model_name].eval()
    manager.dataset = FakeDataset(charset)

    # format images
    imgs = [np.array(Image.open(img_path)) for img_path in img_paths]
    imgs = [np.expand_dims(img, axis=2) if len(img.shape)==2 else img for img in imgs]
    imgs = [np.concatenate([img, img, img], axis=2) if img.shape[2] == 1 else img for img in imgs]
    shapes = [img.shape[:2] for img in imgs]
    reduced_shapes = [[shape[0]//32, shape[1]//8] for shape in shapes]
    imgs_positions = [([0, shape[0]], [0, shape[1]]) for shape in shapes]
    imgs = pad_images(imgs, padding_value=0, padding_mode="br")
    imgs = torch.tensor(imgs).float().permute(0, 3, 1, 2)

    batch_data = {
        "imgs": imgs,
        "imgs_reduced_shape": reduced_shapes,
        "imgs_position": imgs_positions,
        "raw_labels": None,
    }

    with torch.no_grad():
        res = manager.evaluate_batch(batch_data, metric_names = [])
    prediction = res["str_x"]
    layout_tokens = "".join(['Ⓑ', 'Ⓞ', 'Ⓟ', 'Ⓡ', 'Ⓢ', 'Ⓦ', 'Ⓨ', "Ⓐ", "Ⓝ", 'ⓑ', 'ⓞ', 'ⓟ', 'ⓡ', 'ⓢ', 'ⓦ', 'ⓨ', "ⓐ", "ⓝ"])
    prediction = [keep_all_but_tokens(x, layout_tokens) for x in prediction]
    print(prediction)


if __name__ == "__main__":

    model_path = "outputs/dan_rimes_page/checkpoints/dan_rimes_page.pt"
    img_paths = ["../../../test.png", "../../../test2.png"]  # CHANGE WITH YOUR IMAGES PATH
    predict(model_path, img_paths)

