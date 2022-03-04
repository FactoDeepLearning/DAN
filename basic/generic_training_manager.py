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

import torch
import os
import sys
import copy
import json
import torch.distributed as dist
import torch.multiprocessing as mp
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.nn.init import kaiming_uniform_
from tqdm import tqdm
from time import time
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler
from basic.metric_manager import MetricManager
from basic.scheduler import DropoutScheduler
from datetime import date


class GenericTrainingManager:

    def __init__(self, params):
        self.type = None
        self.is_master = False
        self.params = params
        self.dropout_scheduler = None
        self.models = {}
        self.begin_time = None
        self.dataset = None
        self.dataset_name = list(self.params["dataset_params"]["datasets"].values())[0]
        self.paths = None
        self.latest_step = 0
        self.latest_epoch = -1
        self.latest_batch = 0
        self.total_batch = 0
        self.grad_acc_step = 0
        self.latest_train_metrics = dict()
        self.latest_valid_metrics = dict()
        self.curriculum_info = dict()
        self.curriculum_info["latest_valid_metrics"] = dict()
        self.phase = None
        self.max_mem_usage_by_epoch = list()
        self.losses = list()
        self.lr_values = list()

        self.scaler = None

        self.optimizers = dict()
        self.optimizers_named_params_by_group = dict()
        self.lr_schedulers = dict()
        self.best = None
        self.writer = None
        self.metric_manager = dict()

        self.init_hardware_config()
        self.init_paths()
        self.load_dataset()
        self.params["model_params"]["use_amp"] = self.params["training_params"]["use_amp"]

    def init_paths(self):
        """
        Create output folders for results and checkpoints
        """
        output_path = os.path.join("outputs", self.params["training_params"]["output_folder"])
        os.makedirs(output_path, exist_ok=True)
        checkpoints_path = os.path.join(output_path, "checkpoints")
        os.makedirs(checkpoints_path, exist_ok=True)
        results_path = os.path.join(output_path, "results")
        os.makedirs(results_path, exist_ok=True)

        self.paths = {
            "results": results_path,
            "checkpoints": checkpoints_path,
            "output_folder": output_path
        }

    def load_dataset(self):
        """
        Load datasets, data samplers and data loaders
        """
        self.params["dataset_params"]["use_ddp"] = self.params["training_params"]["use_ddp"]
        self.params["dataset_params"]["batch_size"] = self.params["training_params"]["batch_size"]
        if "valid_batch_size" in self.params["training_params"]:
            self.params["dataset_params"]["valid_batch_size"] = self.params["training_params"]["valid_batch_size"]
        if "test_batch_size" in self.params["training_params"]:
            self.params["dataset_params"]["test_batch_size"] = self.params["training_params"]["test_batch_size"]
        self.params["dataset_params"]["num_gpu"] = self.params["training_params"]["nb_gpu"]
        self.params["dataset_params"]["worker_per_gpu"] = 4 if "worker_per_gpu" not in self.params["dataset_params"] else self.params["dataset_params"]["worker_per_gpu"]
        self.dataset = self.params["dataset_params"]["dataset_manager"](self.params["dataset_params"])
        self.dataset.load_datasets()
        self.dataset.load_ddp_samplers()
        self.dataset.load_dataloaders()

    def init_hardware_config(self):
        # Debug mode
        if self.params["training_params"]["force_cpu"]:
            self.params["training_params"]["use_ddp"] = False
            self.params["training_params"]["use_amp"] = False
        # Manage Distributed Data Parallel & GPU usage
        self.manual_seed = 1111 if "manual_seed" not in self.params["training_params"].keys() else \
        self.params["training_params"]["manual_seed"]
        self.ddp_config = {
            "master": self.params["training_params"]["use_ddp"] and self.params["training_params"]["ddp_rank"] == 0,
            "address": "localhost" if "ddp_addr" not in self.params["training_params"].keys() else self.params["training_params"]["ddp_addr"],
            "port": "11111" if "ddp_port" not in self.params["training_params"].keys() else self.params["training_params"]["ddp_port"],
            "backend": "nccl" if "ddp_backend" not in self.params["training_params"].keys() else self.params["training_params"]["ddp_backend"],
            "rank": self.params["training_params"]["ddp_rank"],
        }
        self.is_master = self.ddp_config["master"] or not self.params["training_params"]["use_ddp"]
        if self.params["training_params"]["force_cpu"]:
            self.device = "cpu"
        else:
            if self.params["training_params"]["use_ddp"]:
                self.device = torch.device(self.ddp_config["rank"])
                self.params["dataset_params"]["ddp_rank"] = self.ddp_config["rank"]
                self.launch_ddp()
            else:
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.params["model_params"]["device"] = self.device.type
        # Print GPU info
        # global
        if (self.params["training_params"]["use_ddp"] and self.ddp_config["master"]) or not self.params["training_params"]["use_ddp"]:
            print("##################")
            print("Available GPUS: {}".format(self.params["training_params"]["nb_gpu"]))
            for i in range(self.params["training_params"]["nb_gpu"]):
                print("Rank {}: {} {}".format(i, torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i)))
            print("##################")
        # local
        print("Local GPU:")
        if self.device != "cpu":
            print("Rank {}: {} {}".format(self.params["training_params"]["ddp_rank"], torch.cuda.get_device_name(), torch.cuda.get_device_properties(self.device)))
        else:
            print("WORKING ON CPU !\n")
        print("##################")

    def load_model(self, reset_optimizer=False, strict=True):
        """
        Load model weights from scratch or from checkpoints
        """
        # Instantiate Model
        for model_name in self.params["model_params"]["models"].keys():
            self.models[model_name] = self.params["model_params"]["models"][model_name](self.params["model_params"])
            self.models[model_name].to(self.device)  # To GPU or CPU
            # make the model compatible with Distributed Data Parallel if used
            if self.params["training_params"]["use_ddp"]:
                self.models[model_name] = DDP(self.models[model_name], [self.ddp_config["rank"]])

        # Handle curriculum dropout
        if "dropout_scheduler" in self.params["model_params"]:
            func = self.params["model_params"]["dropout_scheduler"]["function"]
            T = self.params["model_params"]["dropout_scheduler"]["T"]
            self.dropout_scheduler = DropoutScheduler(self.models, func, T)

        self.scaler = GradScaler(enabled=self.params["training_params"]["use_amp"])

        # Check if checkpoint exists
        checkpoint = self.get_checkpoint()
        if checkpoint is not None:
            self.load_existing_model(checkpoint, strict=strict)
        else:
            self.init_new_model()

        self.load_optimizers(checkpoint, reset_optimizer=reset_optimizer)

        if self.is_master:
            print("LOADED EPOCH: {}\n".format(self.latest_epoch), flush=True)

    def get_checkpoint(self):
        """
        Seek if checkpoint exist, return None otherwise
        """
        if self.params["training_params"]["load_epoch"] in ("best", "last"):
            for filename in os.listdir(self.paths["checkpoints"]):
                if self.params["training_params"]["load_epoch"] in filename:
                    return torch.load(os.path.join(self.paths["checkpoints"], filename))
        return None

    def load_existing_model(self, checkpoint, strict=True):
        """
        Load information and weights from previous training
        """
        self.load_save_info(checkpoint)
        self.latest_epoch = checkpoint["epoch"]
        if "step" in checkpoint:
            self.latest_step = checkpoint["step"]
        self.best = checkpoint["best"]
        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        # Load model weights from past training
        for model_name in self.models.keys():
            self.models[model_name].load_state_dict(checkpoint["{}_state_dict".format(model_name)], strict=strict)

    def init_new_model(self):
        """
        Initialize model
        """
        # Specific weights initialization if exists
        for model_name in self.models.keys():
            try:
                self.models[model_name].init_weights()
            except:
                pass

        # Handle transfer learning instructions
        if self.params["model_params"]["transfer_learning"]:
            # Iterates over models
            for model_name in self.params["model_params"]["transfer_learning"].keys():
                state_dict_name, path, learnable, strict = self.params["model_params"]["transfer_learning"][model_name]
                # Loading pretrained weights file
                checkpoint = torch.load(path)
                try:
                    # Load pretrained weights for model
                    self.models[model_name].load_state_dict(checkpoint["{}_state_dict".format(state_dict_name)], strict=strict)
                    print("transfered weights for {}".format(state_dict_name), flush=True)
                except RuntimeError as e:
                    print(e, flush=True)
                    # if error, try to load each parts of the model (useful if only few layers are different)
                    for key in checkpoint["{}_state_dict".format(state_dict_name)].keys():
                        try:
                            # for pre-training of decision layer
                            if "end_conv" in key and "transfered_charset" in self.params["model_params"]:
                                self.adapt_decision_layer_to_old_charset(model_name, key, checkpoint, state_dict_name)
                            else:
                                self.models[model_name].load_state_dict(
                                    {key: checkpoint["{}_state_dict".format(state_dict_name)][key]}, strict=False)
                        except RuntimeError as e:
                            ## exception when adding linebreak token from pretraining
                                print(e, flush=True)
                # Set parameters no trainable
                if not learnable:
                    self.set_model_learnable(self.models[model_name], False)

    def adapt_decision_layer_to_old_charset(self, model_name, key, checkpoint, state_dict_name):
        """
        Transfer learning of the decision learning in case of close charsets between pre-training and training
        """
        pretrained_chars = list()
        weights = checkpoint["{}_state_dict".format(state_dict_name)][key]
        new_size = list(weights.size())
        new_size[0] = len(self.dataset.charset) + self.params["model_params"]["additional_tokens"]
        new_weights = torch.zeros(new_size, device=weights.device, dtype=weights.dtype)
        old_charset = checkpoint["charset"] if "charset" in checkpoint else self.params["model_params"]["old_charset"]
        if not "bias" in key:
            kaiming_uniform_(new_weights, nonlinearity="relu")
        for i, c in enumerate(self.dataset.charset):
            if c in old_charset:
                new_weights[i] = weights[old_charset.index(c)]
                pretrained_chars.append(c)
        if "transfered_charset_last_is_ctc_blank" in self.params["model_params"] and self.params["model_params"]["transfered_charset_last_is_ctc_blank"]:
            new_weights[-1] = weights[-1]
            pretrained_chars.append("<blank>")
        checkpoint["{}_state_dict".format(state_dict_name)][key] = new_weights
        self.models[model_name].load_state_dict({key: checkpoint["{}_state_dict".format(state_dict_name)][key]}, strict=False)
        print("Pretrained chars for {} ({}): {}".format(key, len(pretrained_chars), pretrained_chars))

    def load_optimizers(self, checkpoint, reset_optimizer=False):
        """
        Load the optimizer of each model
        """
        for model_name in self.models.keys():
            new_params = dict()
            if checkpoint and "optimizer_named_params_{}".format(model_name) in checkpoint:
                self.optimizers_named_params_by_group[model_name] = checkpoint["optimizer_named_params_{}".format(model_name)]
                # for progressively growing models
                for name, param in self.models[model_name].named_parameters():
                    existing = False
                    for gr in self.optimizers_named_params_by_group[model_name]:
                        if name in gr:
                            gr[name] = param
                            existing = True
                            break
                    if not existing:
                        new_params.update({name: param})
            else:
                self.optimizers_named_params_by_group[model_name] = [dict(), ]
                self.optimizers_named_params_by_group[model_name][0].update(self.models[model_name].named_parameters())

            # Instantiate optimizer
            self.reset_optimizer(model_name)

            # Handle learning rate schedulers
            if "lr_schedulers" in self.params["training_params"] and self.params["training_params"]["lr_schedulers"]:
                key = "all" if "all" in self.params["training_params"]["lr_schedulers"] else model_name
                if key in self.params["training_params"]["lr_schedulers"]:
                    self.lr_schedulers[model_name] = self.params["training_params"]["lr_schedulers"][key]["class"]\
                        (self.optimizers[model_name], **self.params["training_params"]["lr_schedulers"][key]["args"])

            # Load optimizer state from past training
            if checkpoint and not reset_optimizer:
                self.optimizers[model_name].load_state_dict(checkpoint["optimizer_{}_state_dict".format(model_name)])
                # Load optimizer scheduler config from past training if used
                if "lr_schedulers" in self.params["training_params"] and self.params["training_params"]["lr_schedulers"] \
                        and "lr_scheduler_{}_state_dict".format(model_name) in checkpoint.keys():
                    self.lr_schedulers[model_name].load_state_dict(checkpoint["lr_scheduler_{}_state_dict".format(model_name)])

            # for progressively growing models, keeping learning rate
            if checkpoint and new_params:
                self.optimizers_named_params_by_group[model_name].append(new_params)
                self.optimizers[model_name].add_param_group({"params": list(new_params.values())})

    @staticmethod
    def set_model_learnable(model, learnable=True):
        for p in list(model.parameters()):
            p.requires_grad = learnable

    def save_model(self, epoch, name, keep_weights=False):
        """
        Save model weights and training info for curriculum learning or learning rate for instance
        """
        if not self.is_master:
            return
        to_del = []
        for filename in os.listdir(self.paths["checkpoints"]):
            if name in filename:
                to_del.append(os.path.join(self.paths["checkpoints"], filename))
        path = os.path.join(self.paths["checkpoints"], "{}_{}.pt".format(name, epoch))
        content = {
            'optimizers_named_params': self.optimizers_named_params_by_group,
            'epoch': epoch,
            'step': self.latest_step,
            "scaler_state_dict": self.scaler.state_dict(),
            'best': self.best,
            "charset": self.dataset.charset
        }
        for model_name in self.optimizers:
            content['optimizer_{}_state_dict'.format(model_name)] = self.optimizers[model_name].state_dict()
        for model_name in self.lr_schedulers:
            content["lr_scheduler_{}_state_dict".format(model_name)] = self.lr_schedulers[model_name].state_dict()
        content = self.add_save_info(content)
        for model_name in self.models.keys():
            content["{}_state_dict".format(model_name)] = self.models[model_name].state_dict()
        torch.save(content, path)
        if not keep_weights:
            for path_to_del in to_del:
                if path_to_del != path:
                    os.remove(path_to_del)

    def reset_optimizers(self):
        """
        Reset learning rate of all optimizers
        """
        for model_name in self.models.keys():
            self.reset_optimizer(model_name)

    def reset_optimizer(self, model_name):
        """
        Reset optimizer learning rate for given model
        """
        params = list(self.optimizers_named_params_by_group[model_name][0].values())
        key = "all" if "all" in self.params["training_params"]["optimizers"] else model_name
        self.optimizers[model_name] = self.params["training_params"]["optimizers"][key]["class"](params, **self.params["training_params"]["optimizers"][key]["args"])
        for i in range(1, len(self.optimizers_named_params_by_group[model_name])):
            self.optimizers[model_name].add_param_group({"params": list(self.optimizers_named_params_by_group[model_name][i].values())})

    def save_params(self):
        """
        Output text file containing a summary of all hyperparameters chosen for the training
        """
        def compute_nb_params(module):
            return sum([np.prod(p.size()) for p in list(module.parameters())])

        def class_to_str_dict(my_dict):
            for key in my_dict.keys():
                if callable(my_dict[key]):
                    my_dict[key] = my_dict[key].__name__
                elif isinstance(my_dict[key], np.ndarray):
                    my_dict[key] = my_dict[key].tolist()
                elif isinstance(my_dict[key], dict):
                    my_dict[key] = class_to_str_dict(my_dict[key])
            return my_dict

        path = os.path.join(self.paths["results"], "params")
        if os.path.isfile(path):
            return
        params = copy.deepcopy(self.params)
        params = class_to_str_dict(params)
        params["date"] = date.today().strftime("%d/%m/%Y")
        total_params = 0
        for model_name in self.models.keys():
            current_params = compute_nb_params(self.models[model_name])
            params["model_params"]["models"][model_name] = [params["model_params"]["models"][model_name], "{:,}".format(current_params)]
            total_params += current_params
        params["model_params"]["total_params"] = "{:,}".format(total_params)

        params["hardware"] = dict()
        if self.device != "cpu":
            for i in range(self.params["training_params"]["nb_gpu"]):
                params["hardware"][str(i)] = "{} {}".format(torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i))
        else:
            params["hardware"]["0"] = "CPU"
        params["software"] = {
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        }
        with open(path, 'w') as f:
            json.dump(params, f, indent=4)

    def backward_loss(self, loss, retain_graph=False):
        self.scaler.scale(loss).backward(retain_graph=retain_graph)

    def step_optimizers(self, increment_step=True, names=None):
        for model_name in self.optimizers:
            if names and model_name not in names:
                continue
            if "gradient_clipping" in self.params["training_params"] and model_name in self.params["training_params"]["gradient_clipping"]["models"]:
                self.scaler.unscale_(self.optimizers[model_name])
                torch.nn.utils.clip_grad_norm_(self.models[model_name].parameters(), self.params["training_params"]["gradient_clipping"]["max"])
            self.scaler.step(self.optimizers[model_name])
        self.scaler.update()
        self.latest_step += 1

    def zero_optimizers(self, set_to_none=True):
        for model_name in self.optimizers:
            self.zero_optimizer(model_name, set_to_none)

    def zero_optimizer(self, model_name, set_to_none=True):
        self.optimizers[model_name].zero_grad(set_to_none=set_to_none)

    def train(self):
        """
        Main training loop
        """
        # init tensorboard file and output param summary file
        if self.is_master:
            self.writer = SummaryWriter(self.paths["results"])
            self.save_params()
        # init variables
        self.begin_time = time()
        focus_metric_name = self.params["training_params"]["focus_metric"]
        nb_epochs = self.params["training_params"]["max_nb_epochs"]
        interval_save_weights = self.params["training_params"]["interval_save_weights"]
        metric_names = self.params["training_params"]["train_metrics"]

        display_values = None
        # init curriculum learning
        if "curriculum_learning" in self.params["training_params"].keys() and self.params["training_params"]["curriculum_learning"]:
            self.init_curriculum()
        # perform epochs
        for num_epoch in range(self.latest_epoch+1, nb_epochs):
            self.dataset.train_dataset.training_info = {
                "epoch": self.latest_epoch,
                "step": self.latest_step
            }
            self.phase = "train"
            # Check maximum training time stop condition
            if self.params["training_params"]["max_training_time"] and time() - self.begin_time > self.params["training_params"]["max_training_time"]:
                break
            # set models trainable
            for model_name in self.models.keys():
                self.models[model_name].train()
            self.latest_epoch = num_epoch
            if self.dataset.train_dataset.curriculum_config:
                self.dataset.train_dataset.curriculum_config["epoch"] = self.latest_epoch
            # init epoch metrics values
            self.metric_manager["train"] = MetricManager(metric_names=metric_names, dataset_name=self.dataset_name)

            with tqdm(total=len(self.dataset.train_loader.dataset)) as pbar:
                pbar.set_description("EPOCH {}/{}".format(num_epoch, nb_epochs))
                # iterates over mini-batch data
                for ind_batch, batch_data in enumerate(self.dataset.train_loader):
                    self.latest_batch = ind_batch + 1
                    self.total_batch += 1
                    # train on batch data and compute metrics
                    batch_values = self.train_batch(batch_data, metric_names)
                    batch_metrics = self.metric_manager["train"].compute_metrics(batch_values, metric_names)
                    batch_metrics["names"] = batch_data["names"]
                    batch_metrics["ids"] = batch_data["ids"]
                    # Merge metrics if Distributed Data Parallel is used
                    if self.params["training_params"]["use_ddp"]:
                        batch_metrics = self.merge_ddp_metrics(batch_metrics)
                    # Update learning rate via scheduler if one is used
                    if self.params["training_params"]["lr_schedulers"]:
                        for model_name in self.models:
                            key = "all" if "all" in self.params["training_params"]["lr_schedulers"] else model_name
                            if model_name in self.lr_schedulers and ind_batch % self.params["training_params"]["lr_schedulers"][key]["step_interval"] == 0:
                                self.lr_schedulers[model_name].step(len(batch_metrics["names"]))
                                if "lr" in metric_names:
                                    self.writer.add_scalar("lr_{}".format(model_name), self.lr_schedulers[model_name].lr, self.lr_schedulers[model_name].step_num)
                    # Update dropout scheduler if used
                    if self.dropout_scheduler:
                        self.dropout_scheduler.step(len(batch_metrics["names"]))
                        self.dropout_scheduler.update_dropout_rate()

                    # Add batch metrics values to epoch metrics values
                    self.metric_manager["train"].update_metrics(batch_metrics)
                    display_values = self.metric_manager["train"].get_display_values()
                    pbar.set_postfix(values=str(display_values))
                    pbar.update(len(batch_data["names"]))

            # log metrics in tensorboard file
            if self.is_master:
                for key in display_values.keys():
                    self.writer.add_scalar('{}_{}'.format(self.params["dataset_params"]["train"]["name"], key), display_values[key], num_epoch)
            self.latest_train_metrics = display_values

            # evaluate and compute metrics for valid sets
            if self.params["training_params"]["eval_on_valid"] and num_epoch % self.params["training_params"]["eval_on_valid_interval"] == 0:
                for valid_set_name in self.dataset.valid_loaders.keys():
                    # evaluate set and compute metrics
                    eval_values = self.evaluate(valid_set_name)
                    self.latest_valid_metrics = eval_values
                    # log valid metrics in tensorboard file
                    if self.is_master:
                        for key in eval_values.keys():
                            self.writer.add_scalar('{}_{}'.format(valid_set_name, key), eval_values[key], num_epoch)
                        if valid_set_name == self.params["training_params"]["set_name_focus_metric"] and (self.best is None or \
                                (eval_values[focus_metric_name] <= self.best and self.params["training_params"]["expected_metric_value"] == "low") or\
                                (eval_values[focus_metric_name] >= self.best and self.params["training_params"]["expected_metric_value"] == "high")):
                            self.save_model(epoch=num_epoch, name="best")
                            self.best = eval_values[focus_metric_name]

            # Handle curriculum learning update
            if self.dataset.train_dataset.curriculum_config:
                self.check_and_update_curriculum()

            if "curriculum_model" in self.params["model_params"] and self.params["model_params"]["curriculum_model"]:
                self.update_curriculum_model()

            # save model weights
            if self.is_master:
                self.save_model(epoch=num_epoch, name="last")
                if interval_save_weights and num_epoch % interval_save_weights == 0:
                    self.save_model(epoch=num_epoch, name="weigths", keep_weights=True)
                self.writer.flush()

    def evaluate(self, set_name, **kwargs):
        """
        Main loop for validation
        """
        self.phase = "eval"
        loader = self.dataset.valid_loaders[set_name]
        # Set models in eval mode
        for model_name in self.models.keys():
            self.models[model_name].eval()
        metric_names = self.params["training_params"]["eval_metrics"]
        display_values = None

        # initialize epoch metrics
        self.metric_manager[set_name] = MetricManager(metric_names, dataset_name=self.dataset_name)
        with tqdm(total=len(loader.dataset)) as pbar:
            pbar.set_description("Evaluation E{}".format(self.latest_epoch))
            with torch.no_grad():
                # iterate over batch data
                for ind_batch, batch_data in enumerate(loader):
                    self.latest_batch = ind_batch + 1
                    # eval batch data and compute metrics
                    batch_values = self.evaluate_batch(batch_data, metric_names)
                    batch_metrics = self.metric_manager[set_name].compute_metrics(batch_values, metric_names)
                    batch_metrics["names"] = batch_data["names"]
                    batch_metrics["ids"] = batch_data["ids"]
                    # merge metrics values if Distributed Data Parallel is used
                    if self.params["training_params"]["use_ddp"]:
                        batch_metrics = self.merge_ddp_metrics(batch_metrics)

                    # add batch metrics to epoch metrics
                    self.metric_manager[set_name].update_metrics(batch_metrics)
                    display_values = self.metric_manager[set_name].get_display_values()

                    pbar.set_postfix(values=str(display_values))
                    pbar.update(len(batch_data["names"]))
        if "cer_by_nb_cols" in metric_names:
            self.log_cer_by_nb_cols(set_name)
        return display_values

    def predict(self, custom_name, sets_list, metric_names, output=False):
        """
        Main loop for evaluation
        """
        self.phase = "predict"
        metric_names = metric_names.copy()
        self.dataset.generate_test_loader(custom_name, sets_list)
        loader = self.dataset.test_loaders[custom_name]
        # Set models in eval mode
        for model_name in self.models.keys():
            self.models[model_name].eval()

        # initialize epoch metrics
        self.metric_manager[custom_name] = MetricManager(metric_names, self.dataset_name)

        with tqdm(total=len(loader.dataset)) as pbar:
            pbar.set_description("Prediction")
            with torch.no_grad():
                for ind_batch, batch_data in enumerate(loader):
                    # iterates over batch data
                    self.latest_batch = ind_batch + 1
                    # eval batch data and compute metrics
                    batch_values = self.evaluate_batch(batch_data, metric_names)
                    batch_metrics = self.metric_manager[custom_name].compute_metrics(batch_values, metric_names)
                    batch_metrics["names"] = batch_data["names"]
                    batch_metrics["ids"] = batch_data["ids"]
                    # merge batch metrics if Distributed Data Parallel is used
                    if self.params["training_params"]["use_ddp"]:
                        batch_metrics = self.merge_ddp_metrics(batch_metrics)

                    # add batch metrics to epoch metrics
                    self.metric_manager[custom_name].update_metrics(batch_metrics)
                    display_values = self.metric_manager[custom_name].get_display_values()

                    pbar.set_postfix(values=str(display_values))
                    pbar.update(len(batch_data["names"]))

        self.dataset.remove_test_dataset(custom_name)
        # output metrics values if requested
        if output:
            if "pred" in metric_names:
                self.output_pred(custom_name)
            metrics = self.metric_manager[custom_name].get_display_values(output=True)
            path = os.path.join(self.paths["results"], "predict_{}_{}.txt".format(custom_name, self.latest_epoch))
            with open(path, "w") as f:
                for metric_name in metrics.keys():
                    f.write("{}: {}\n".format(metric_name, metrics[metric_name]))

    def output_pred(self, name):
        path = os.path.join(self.paths["results"], "pred_{}_{}.txt".format(name, self.latest_epoch))
        pred = "\n".join(self.metric_manager[name].get("pred"))
        with open(path, "w") as f:
            f.write(pred)

    def launch_ddp(self):
        """
        Initialize Distributed Data Parallel system
        """
        mp.set_start_method('fork', force=True)
        os.environ['MASTER_ADDR'] = self.ddp_config["address"]
        os.environ['MASTER_PORT'] = str(self.ddp_config["port"])
        dist.init_process_group(self.ddp_config["backend"], rank=self.ddp_config["rank"], world_size=self.params["training_params"]["nb_gpu"])
        torch.cuda.set_device(self.ddp_config["rank"])
        random.seed(self.manual_seed)
        np.random.seed(self.manual_seed)
        torch.manual_seed(self.manual_seed)
        torch.cuda.manual_seed(self.manual_seed)

    def merge_ddp_metrics(self, metrics):
        """
        Merge metrics when Distributed Data Parallel is used
        """
        for metric_name in metrics.keys():
            if metric_name in ["edit_words", "nb_words", "edit_chars", "nb_chars", "edit_chars_force_len",
                               "edit_chars_curr", "nb_chars_curr", "ids"]:
                metrics[metric_name] = self.cat_ddp_metric(metrics[metric_name])
            elif metric_name in ["nb_samples", "loss", "loss_ce", "loss_ctc", "loss_ce_end"]:
                metrics[metric_name] = self.sum_ddp_metric(metrics[metric_name], average=False)
        return metrics

    def sum_ddp_metric(self, metric, average=False):
        """
        Sum metrics for Distributed Data Parallel
        """
        sum = torch.tensor(metric[0]).to(self.device)
        dist.all_reduce(sum, op=dist.ReduceOp.SUM)
        if average:
            sum.true_divide(dist.get_world_size())
        return [sum.item(), ]

    def cat_ddp_metric(self, metric):
        """
        Concatenate metrics for Distributed Data Parallel
        """
        tensor = torch.tensor(metric).unsqueeze(0).to(self.device)
        res = [torch.zeros(tensor.size()).long().to(self.device) for _ in range(dist.get_world_size())]
        dist.all_gather(res, tensor)
        return list(torch.cat(res, dim=0).flatten().cpu().numpy())

    @staticmethod
    def cleanup():
        dist.destroy_process_group()

    def train_batch(self, batch_data, metric_names):
        raise NotImplementedError

    def evaluate_batch(self, batch_data, metric_names):
        raise NotImplementedError

    def init_curriculum(self):
        raise NotImplementedError

    def update_curriculum(self):
        raise NotImplementedError

    def add_checkpoint_info(self, load_mode="last", **kwargs):
        for filename in os.listdir(self.paths["checkpoints"]):
            if load_mode in filename:
                checkpoint_path = os.path.join(self.paths["checkpoints"], filename)
                checkpoint = torch.load(checkpoint_path)
                for key in kwargs.keys():
                    checkpoint[key] = kwargs[key]
                torch.save(checkpoint, checkpoint_path)
            return
        self.save_model(self.latest_epoch, "last")

    def load_save_info(self, info_dict):
        """
        Load curriculum info from saved model info
        """
        if "curriculum_config" in info_dict.keys():
            self.dataset.train_dataset.curriculum_config = info_dict["curriculum_config"]

    def add_save_info(self, info_dict):
        """
        Add curriculum info to model info to be saved
        """
        info_dict["curriculum_config"] = self.dataset.train_dataset.curriculum_config
        return info_dict