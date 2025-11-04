import os
import shutil
import os.path as path
import time
import logging

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_

MASTER_NODE = 0
logging.basicConfig(level=logging.INFO)

def get_logger(logger_name, file_name):
    logger = logging.getLogger(logger_name)
    consoleHandler = logging.StreamHandler()
    fileHandler = logging.FileHandler(filename=file_name)

    logger.setLevel(logging.DEBUG)
    consoleHandler.setLevel(logging.DEBUG)
    fileHandler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s %(name)s:%(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    consoleHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)

    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)
    return logger

class StepLRScheduleWrapper:
    def __init__(self, named_params, lr, optim_conf, logger, min_lr=1e-8, max_grad_norm=-1, step_num=5, decay=0.9):
        self.lr_step = 0
        self.lr = lr
        self.min_lr = min_lr
        self.logger = logger
        self.max_grad_norm = max_grad_norm
        params = [p for n, p in named_params if p.requires_grad]
        optimizer_group_params = [{'params': params}]
        if "weight_decay" in optim_conf:
            optim_conf["weight_decay"] = float(optim_conf["weight_decay"])
        self.optimizer = optim.AdamW(optimizer_group_params, self.lr, **optim_conf)
        self.cur_lr = self.lr
        self.step_num = step_num
        self.decay = decay

    def get_learning_rate(self):
        return self.cur_lr

    def adjust_learning_rate(self, lr):
        for param in self.optimizer.param_groups:
            param['lr'] = lr

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        if self.max_grad_norm > 0:
            for group in self.optimizer.param_groups:
                clip_grad_norm_(group['params'], self.max_grad_norm)
        self.optimizer.step()

    def addStep_adjustLR(self, epoch):
        new_lr = self.lr * (self.decay ** ((epoch + 1) // self.step_num))
        self.adjust_learning_rate(new_lr)
        self.cur_lr = new_lr

    def state_dict(self):
        state_dict = self.optimizer.state_dict()
        state_dict.update({
            'lr_step': self.lr_step,
            'cur_lr': self.lr,
            'wrapper_name': self.__class__.__name__
        })
        return state_dict

    def load_state_dict(self, state_dict):
        assert state_dict['wrapper_name'] == self.__class__.__name__
        self.lr_step = state_dict.pop('lr_step')
        self.lr = state_dict.pop('cur_lr')
        state_dict.pop('wrapper_name')
        self.optimizer.load_state_dict(state_dict)
        self.adjust_learning_rate(self.lr)

class MetricStat:
    def __init__(self, tags):
        self.tags = tags
        self.total_count = [0] * len(tags)
        self.total_sum = [0.0] * len(tags)
        self.log_count = [0] * len(tags)
        self.log_sum = [0.0] * len(tags)

    def update_stat(self, metrics, counts):
        for i, (m, c) in enumerate(zip(metrics, counts)):
            self.log_count[i] += c
            self.log_sum[i] += m

    def log_stat(self):
        avg = []
        for i in range(len(self.tags)):
            avg_val = 0.0 if self.log_count[i] == 0 else self.log_sum[i] / self.log_count[i]
            avg.append(avg_val)
            self.total_sum[i] += self.log_sum[i]
            self.total_count[i] += self.log_count[i]
            self.log_sum[i] = 0.0
            self.log_count[i] = 0
        return avg

    def summary_stat(self):
        avg = []
        for i in range(len(self.tags)):
            self.total_sum[i] += self.log_sum[i]
            self.total_count[i] += self.log_count[i]
            avg_val = 0.0 if self.total_count[i] == 0 else self.total_sum[i] / self.total_count[i]
            avg.append(avg_val)
        return avg

    def reset(self):
        self.total_sum = [0.0] * len(self.tags)
        self.total_count = [0] * len(self.tags)
        self.log_sum = [0.0] * len(self.tags)
        self.log_count = [0] * len(self.tags)

class Trainer:
    def __init__(self, model, output_dir, init_model=None, device="cpu", **train_config):
        assert device in ["cpu", "gpu", "ddp"]
        if device == "gpu" and not torch.cuda.is_available():
            raise RuntimeError("GPU requested but CUDA is not available.")
        self.device = torch.device("cuda:0" if device == "gpu" else "cpu")

        self.train_conf = train_config
        self.output_dir = path.abspath(output_dir)
        os.makedirs(path.join(self.output_dir, "log"), exist_ok=True)

        self.logger = get_logger(f"logger.{path.split(self.output_dir)[-1]}", path.join(self.output_dir, "log", "train.log"))

        self.model = model.to(self.device)
        if init_model is None:
            self.logger.info("Random initialize model")
        else:
            self.model.load_state_dict(torch.load(init_model, map_location='cpu'))
            self.logger.info(f"Initialize model from: {init_model}")

        num_param = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"model proto: {self.model._get_name()}, model_size: {num_param / 1e6:.6f} M")

        self.final_model = path.join(self.output_dir, "model.final")
        tensorboard_dir = path.join(self.output_dir, "tensorboard", "rank")
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.tensorboard_writer = SummaryWriter(log_dir=tensorboard_dir)

        self.train_metric = MetricStat(self.model.metric_tags)
        self.valid_metric = MetricStat(self.model.metric_tags)

        named_params = self.model.named_parameters()
        lr = self.train_conf.get('lr', 1e-4)
        optim_conf = self.train_conf.get('optim_conf', {})
        schedule_conf = self.train_conf.get('schedule_conf', {})
        self.optimizer = StepLRScheduleWrapper(named_params, lr, optim_conf, self.logger, **schedule_conf)

        chkpt_path = path.join(self.output_dir, "chkpt")
        self.chkpt_path = chkpt_path
        if path.isfile(chkpt_path):
            chkpt = torch.load(chkpt_path, map_location="cpu")
            self.start_epoch = chkpt["epoch"]
            self.best_model = chkpt["best_model"]
            self.global_step = chkpt["global_step"]
            self.best_valid_loss = chkpt["best_valid_loss"]
            self.recent_models = chkpt["recent_models"]
            self.optimizer.load_state_dict(chkpt["optim"])
            self.model.load_state_dict(torch.load(self.recent_models[-1], map_location="cpu"))
            self.logger.info(f"Loading checkpoint from {chkpt_path}, Current lr: {self.optimizer.get_learning_rate()}")
        else:
            self.start_epoch = 1
            self.best_model = path.join(self.output_dir, "model.epoch-0.step-0")
            self.global_step = 0
            self.best_valid_loss = float('inf')
            self.recent_models = [self.best_model]
            torch.save(self.model.state_dict(), self.best_model)
            self.save_chkpt(self.start_epoch)

        self.stop_step = 0
        self.num_trained = 0
        self.train_dataset = None
        self.valid_dataset = None

    def save_chkpt(self, epoch):
        torch.save({
            'epoch': epoch,
            'best_model': self.best_model,
            'best_valid_loss': self.best_valid_loss,
            'recent_models': self.recent_models,
            'global_step': self.global_step,
            'optim': self.optimizer.state_dict()
        }, self.chkpt_path)

    def save_model_state(self, epoch):
        model_path = path.join(self.output_dir, f"model.epoch-{epoch}.step-{self.global_step}")
        torch.save(self.model.state_dict(), model_path)
        self.recent_models.append(model_path)
        num_recent_models = self.train_conf.get('num_recent_models', -1)
        if num_recent_models > 0 and len(self.recent_models) > num_recent_models:
            os.remove(self.recent_models.pop(0))

    def should_early_stop(self):
        return self.train_conf.get("early_stop_count", 0) > 0 and self.stop_step >= self.train_conf["early_stop_count"]

    def train_one_epoch(self, epoch):
        self.logger.info(f"Epoch {epoch} start, lr {self.optimizer.get_learning_rate():.8f}")
        log_period = self.train_conf.get('log_period', 10)
        accum_grad = self.train_conf.get('accum_grad', 1)
        self.model.train()
        hidden = None
        start_time = time.time()

        for batch_data in self.train_loader:
            data, target, data_lens, target_lens = [d.to(self.device) for d in batch_data]
            batch_size = data.size(0)
            self.global_step += 1
            res = self.model(data, data_lens, hidden=hidden)
            hidden = res.get("hidden", None)
            loss, metrics, counts = self.model.cal_loss(res, target, epoch=epoch)
            loss = loss / accum_grad
            loss.backward()
            self.train_metric.update_stat(metrics, counts)
            if self.global_step % accum_grad == 0:
                self.optimizer.addStep_adjustLR(epoch)
                self.optimizer.step()
                self.optimizer.zero_grad()
            self.num_trained += batch_size

            if self.global_step % log_period == 0:
                elapsed = time.time() - start_time
                avg_stat = self.train_metric.log_stat()
                for tag, stat in zip(self.train_metric.tags, avg_stat):
                    self.tensorboard_writer.add_scalar(f"train/{tag}", stat, self.global_step)
                self.tensorboard_writer.add_scalar("train/lr", self.optimizer.get_learning_rate(), self.global_step)
                self.logger.info(f"Epoch: {epoch}, Trained: {self.num_trained}, " +
                                 ", ".join([f"{tag}: {stat:.6f}" for tag, stat in zip(self.train_metric.tags, avg_stat)]) +
                                 f", lr: {self.optimizer.get_learning_rate():.8f}, time: {elapsed:.1f}s")
                start_time = time.time()

        train_stat = self.train_metric.summary_stat()
        self.train_metric.reset()
        self.logger.info(f"Epoch {epoch} Done, " +
                         ", ".join([f"{tag}: {stat:.6f}" for tag, stat in zip(self.train_metric.tags, train_stat)]))

        if self.valid_loader is not None:
            self.valid(epoch)
        else:
            self.save_model_state(epoch)
            shutil.copyfile(self.recent_models[-1], self.best_model)
        self.save_chkpt(epoch + 1)

    def valid(self, epoch):
        self.logger.info("Start Validation")
        self.model.eval()
        hidden = None
        for batch_data in self.valid_loader:
            data, target, data_lens, target_lens = [d.to(self.device) for d in batch_data]
            with torch.no_grad():
                res = self.model(data, data_lens, hidden=hidden) if hidden else self.model(data, data_lens)
                hidden = getattr(res, "hidden", None)
                loss, metrics, counts = self.model.cal_loss(res, target, epoch=epoch)
                self.valid_metric.update_stat(metrics, counts)

        valid_stat = self.valid_metric.summary_stat()
        self.logger.info("Validation Done, " +
                         ", ".join([f"{tag}: {stat:.6f}" for tag, stat in zip(self.valid_metric.tags, valid_stat)]))
        self.valid_metric.reset()

        self.save_model_state(epoch)
        valid_loss = valid_stat[0]
        if valid_loss < self.best_valid_loss:
            self.best_valid_loss = valid_loss
            shutil.copyfile(self.recent_models[-1], self.best_model)
            self.logger.info(f"New best_valid_loss: {valid_loss:.6f}, saving best model")
            self.stop_step = 0
        else:
            self.stop_step += 1
        self.model.train()

    def fit(self, train_dataset, val_dataset, test_dataset=None, test_batch_size=None):
        self.train_dataset = train_dataset
        self.valid_dataset = val_dataset
        self.train_loader = DataLoader(train_dataset, batch_size=self.train_conf["batch_size"],
                                       drop_last=self.train_conf.get("drop_last", False), num_workers=0)
        self.valid_loader = None
        if val_dataset is not None:
            self.valid_loader = DataLoader(val_dataset, batch_size=self.train_conf["batch_size"],
                                           drop_last=self.train_conf.get("drop_last", False), num_workers=0)
        self.test_dataset = test_dataset
        self.test_batch_size = test_batch_size or self.train_conf["batch_size"]

        for epoch in range(self.start_epoch, self.train_conf["max_epochs"] + 1):
            if self.should_early_stop():
                self.logger.info("Early stopping")
                break
            self.train_dataset.set_epoch(epoch)
            self.train_one_epoch(epoch)

        self.logger.info("Training finished")
        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)
        shutil.copyfile(self.best_model, self.final_model)
