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

from basic.metric_manager import MetricManager
from OCR.ocr_manager import OCRManager
from OCR.ocr_utils import LM_ind_to_str
import torch
from torch.cuda.amp import autocast
from torch.nn import CTCLoss
import re
import time


class TrainerLineCTC(OCRManager):

    def __init__(self, params):
        super(TrainerLineCTC, self).__init__(params)

    def train_batch(self, batch_data, metric_names):
        """
        Forward and backward pass for training
        """
        x = batch_data["imgs"].to(self.device)
        y = batch_data["labels"].to(self.device)
        x_reduced_len = [s[1] for s in batch_data["imgs_reduced_shape"]]
        y_len = batch_data["labels_len"]

        loss_ctc = CTCLoss(blank=self.dataset.tokens["blank"])
        self.zero_optimizers()

        with autocast(enabled=self.params["training_params"]["use_amp"]):
            x = self.models["encoder"](x)
            global_pred = self.models["decoder"](x)

        loss = loss_ctc(global_pred.permute(2, 0, 1), y, x_reduced_len, y_len)

        self.backward_loss(loss)

        self.step_optimizers()
        pred = torch.argmax(global_pred, dim=1).cpu().numpy()

        values = {
            "nb_samples": len(batch_data["raw_labels"]),
            "loss_ctc": loss.item(),
            "str_x": self.pred_to_str(pred, x_reduced_len),
            "str_y": batch_data["raw_labels"]
        }

        return values

    def evaluate_batch(self, batch_data, metric_names):
        """
        Forward pass only for validation and test
        """
        x = batch_data["imgs"].to(self.device)
        y = batch_data["labels"].to(self.device)
        x_reduced_len = [s[1] for s in batch_data["imgs_reduced_shape"]]
        y_len = batch_data["labels_len"]

        loss_ctc = CTCLoss(blank=self.dataset.tokens["blank"])

        start_time = time.time()
        with autocast(enabled=self.params["training_params"]["use_amp"]):
            x = self.models["encoder"](x)
            global_pred = self.models["decoder"](x)

        loss = loss_ctc(global_pred.permute(2, 0, 1), y, x_reduced_len, y_len)
        pred = torch.argmax(global_pred, dim=1).cpu().numpy()
        str_x = self.pred_to_str(pred, x_reduced_len)

        process_time =time.time() - start_time

        values = {
            "nb_samples": len(batch_data["raw_labels"]),
            "loss_ctc": loss.item(),
            "str_x": str_x,
            "str_y": batch_data["raw_labels"],
            "time": process_time
        }
        return values

    def ctc_remove_successives_identical_ind(self, ind):
        res = []
        for i in ind:
            if res and res[-1] == i:
                continue
            res.append(i)
        return res

    def pred_to_str(self, pred, pred_len):
        """
        convert prediction tokens to string
        """
        ind_x = [pred[i][:pred_len[i]] for i in range(pred.shape[0])]
        ind_x = [self.ctc_remove_successives_identical_ind(t) for t in ind_x]
        str_x = [LM_ind_to_str(self.dataset.charset, t, oov_symbol="") for t in ind_x]
        str_x = [re.sub("( )+", ' ', t).strip(" ") for t in str_x]
        return str_x
