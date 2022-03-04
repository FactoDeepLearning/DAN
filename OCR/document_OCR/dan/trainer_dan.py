from OCR.ocr_manager import OCRManager
from torch.nn import CrossEntropyLoss
import torch
from OCR.ocr_utils import LM_ind_to_str
import numpy as np
from torch.cuda.amp import autocast
import time


class Manager(OCRManager):

    def __init__(self, params):
        super(Manager, self).__init__(params)

    def load_save_info(self, info_dict):
        if "curriculum_config" in info_dict.keys():
            if self.dataset.train_dataset is not None:
                self.dataset.train_dataset.curriculum_config = info_dict["curriculum_config"]

    def add_save_info(self, info_dict):
        info_dict["curriculum_config"] = self.dataset.train_dataset.curriculum_config
        return info_dict

    def get_init_hidden(self, batch_size):
        num_layers = 1
        hidden_size = self.params["model_params"]["enc_dim"]
        return torch.zeros(num_layers, batch_size, hidden_size), torch.zeros(num_layers, batch_size, hidden_size)

    def apply_teacher_forcing(self, y, y_len, error_rate):
        y_error = y.clone()
        for b in range(len(y_len)):
            for i in range(1, y_len[b]):
                if np.random.rand() < error_rate and y[b][i] != self.dataset.tokens["pad"]:
                    y_error[b][i] = np.random.randint(0, len(self.dataset.charset)+2)
        return y_error, y_len

    def train_batch(self, batch_data, metric_names):
        # Add loss weight for <eot> token
        weights = None
        if self.params["model_params"]["weight_end_of_prediction"]:
            vocab_size = self.params["model_params"]["vocab_size"] + 1
            weights = torch.ones((vocab_size, ), device=self.device)
            weights[self.dataset.tokens["end"]] = 10
        loss_func = CrossEntropyLoss(weight=weights, ignore_index=self.dataset.tokens["pad"])

        sum_loss = 0
        x = batch_data["imgs"].to(self.device)
        y = batch_data["labels"].to(self.device)
        reduced_size = [s[:2] for s in batch_data["imgs_reduced_shape"]]
        y_len = batch_data["labels_len"]

        # add errors in teacher forcing
        if "teacher_forcing_error_rate" in self.params["training_params"] and self.params["training_params"]["teacher_forcing_error_rate"] is not None:
            error_rate = self.params["training_params"]["teacher_forcing_error_rate"]
            simulated_y_pred, y_len = self.apply_teacher_forcing(y, y_len, error_rate)
        elif "teacher_forcing_scheduler" in self.params["training_params"]:
            error_rate = self.params["training_params"]["teacher_forcing_scheduler"]["min_error_rate"] + min(self.latest_step, self.params["training_params"]["teacher_forcing_scheduler"]["total_num_steps"]) * (self.params["training_params"]["teacher_forcing_scheduler"]["max_error_rate"]-self.params["training_params"]["teacher_forcing_scheduler"]["min_error_rate"]) / self.params["training_params"]["teacher_forcing_scheduler"]["total_num_steps"]
            simulated_y_pred, y_len = self.apply_teacher_forcing(y, y_len, error_rate)
        else:
            simulated_y_pred = y

        with autocast(enabled=self.params["training_params"]["use_amp"]):
            hidden_emb = None
            hidden_predict = None
            cache = None

            raw_features = self.models["encoder"](x)
            features_size = raw_features.size()
            b, c, h, w = features_size

            pos_features = self.models["decoder"].features_updater.get_pos_features(raw_features)
            features = torch.flatten(pos_features, start_dim=2, end_dim=3).permute(2, 0, 1)
            enhanced_features = pos_features
            enhanced_features = torch.flatten(enhanced_features, start_dim=2, end_dim=3).permute(2, 0, 1)
            output, pred, hidden_emb, hidden_predict, cache, weights = self.models["decoder"](features, enhanced_features,
                                                                               simulated_y_pred[:, :-1],
                                                                               reduced_size,
                                                                               [max(y_len) for _ in range(b)],
                                                                               features_size,
                                                                               start=0, hidden_emb=hidden_emb,
                                                                               hidden_predict=hidden_predict,
                                                                               cache=cache,
                                                                               keep_all_weights=True)

            loss_ce = loss_func(pred, y[:, 1:])
            sum_loss += loss_ce
            with autocast(enabled=False):
                self.backward_loss(sum_loss)
                self.step_optimizers()
                self.zero_optimizers()
            predicted_tokens = torch.argmax(pred, dim=1).detach().cpu().numpy()
            predicted_tokens = [predicted_tokens[i, :y_len[i]] for i in range(b)]
            str_x = [LM_ind_to_str(self.dataset.charset, t, oov_symbol="") for t in predicted_tokens]

        values = {
            "nb_samples": b,
            "str_y": batch_data["raw_labels"],
            "str_x": str_x,
            "loss": sum_loss.item(),
            "loss_ce": loss_ce.item(),
            "syn_max_lines": self.dataset.train_dataset.get_syn_max_lines() if self.params["dataset_params"]["config"]["synthetic_data"] else 0,
        }

        return values

    def evaluate_batch(self, batch_data, metric_names):
        x = batch_data["imgs"].to(self.device)
        reduced_size = [s[:2] for s in batch_data["imgs_reduced_shape"]]

        max_chars = self.params["training_params"]["max_char_prediction"]

        start_time = time.time()
        with autocast(enabled=self.params["training_params"]["use_amp"]):
            b = x.size(0)
            reached_end = torch.zeros((b, ), dtype=torch.bool, device=self.device)
            prediction_len = torch.zeros((b, ), dtype=torch.int, device=self.device)
            predicted_tokens = torch.ones((b, 1), dtype=torch.long, device=self.device) * self.dataset.tokens["start"]
            predicted_tokens_len = torch.ones((b, ), dtype=torch.int, device=self.device)

            whole_output = list()
            confidence_scores = list()
            cache = None
            hidden_predict = None
            hidden_emb = None
            if b > 1:
                features_list = list()
                for i in range(b):
                    pos = batch_data["imgs_position"]
                    features_list.append(self.models["encoder"](x[i:i+1, :, pos[i][0][0]:pos[i][0][1], pos[i][1][0]:pos[i][1][1]]))
                max_height = max([f.size(2) for f in features_list])
                max_width = max([f.size(3) for f in features_list])
                features = torch.zeros((b, features_list[0].size(1), max_height, max_width), device=self.device, dtype=features_list[0].dtype)
                for i in range(b):
                    features[i, :, :features_list[i].size(2), :features_list[i].size(3)] = features_list[i]
            else:
                features = self.models["encoder"](x)
            features_size = features.size()
            coverage_vector = torch.zeros((features.size(0), 1, features.size(2), features.size(3)), device=self.device)
            pos_features = self.models["decoder"].features_updater.get_pos_features(features)
            features = torch.flatten(pos_features, start_dim=2, end_dim=3).permute(2, 0, 1)
            enhanced_features = pos_features
            enhanced_features = torch.flatten(enhanced_features, start_dim=2, end_dim=3).permute(2, 0, 1)

            for i in range(0, max_chars):
                output, pred, hidden_emb, hidden_predict, cache, weights = self.models["decoder"](features, enhanced_features, predicted_tokens, reduced_size, predicted_tokens_len, features_size, start=0, hidden_emb=hidden_emb, hidden_predict=hidden_predict, cache=cache, num_pred=1)
                whole_output.append(output)
                confidence_scores.append(torch.max(torch.softmax(pred[:, :], dim=1), dim=1).values)
                coverage_vector = torch.clamp(coverage_vector + weights, 0, 1)
                predicted_tokens = torch.cat([predicted_tokens, torch.argmax(pred[:, :, -1], dim=1, keepdim=True)], dim=1)
                reached_end = torch.logical_or(reached_end, torch.eq(predicted_tokens[:, -1], self.dataset.tokens["end"]))
                predicted_tokens_len += 1

                prediction_len[reached_end == False] = i
                if torch.all(reached_end):
                    break

            confidence_scores = torch.cat(confidence_scores, dim=1).cpu().detach().numpy()
            predicted_tokens = predicted_tokens[:, 1:]
            prediction_len[torch.eq(reached_end, False)] = max_chars - 1
            predicted_tokens = [predicted_tokens[i, :prediction_len[i]] for i in range(b)]
            confidence_scores = [confidence_scores[i, :prediction_len[i]].tolist() for i in range(b)]
            str_x = [LM_ind_to_str(self.dataset.charset, t, oov_symbol="") for t in predicted_tokens]

        process_time = time.time() - start_time

        values = {
            "nb_samples": b,
            "str_y": batch_data["raw_labels"],
            "str_x": str_x,
            "confidence_score": confidence_scores,
            "time": process_time,
        }
        return values
