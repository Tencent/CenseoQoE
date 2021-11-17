# -*- coding: utf-8 -*-
# @Time    : 2021/5/13 11:34 上午
# @Author  : shaoguowen
# @Email   : shaoguowen@tencent.com
# @FileName: base_trainer.py
# @Software: PyCharm

import os
import datetime
import shutil
import torch
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
from scipy import stats
import numpy as np
from tqdm import tqdm

import models
import datasets
import losses
from utils import logger, visualize, utils


class CenseoBaseTrainer:
    """
    一个trainer基类，后续有必要可以继承这个基类写自己的trainer
    """

    def __init__(self, config, config_path):
        self.config_path = config_path
        self.config = config
        self.init_exp_dir()
        self.init_vis()
        self.init_log()

        # 判断是否使用多GPU训练
        if torch.cuda.device_count() > 1 and self.config.trainer.gpu_num > 1:
            self.multi_gpu_train = True
        else:
            self.multi_gpu_train = False

        self.gpu_num = min(self.config.trainer.gpu_num, torch.cuda.device_count())

        # 是否使用 SWA
        self.use_swa = self.config.model.get("swa", False)

        self.best_plcc = -1
        self.best_srocc = -1
        self.best_epoch = -1

    def init_exp_dir(self):
        """
        初始化实验的路径
        :return:
        """
        date_str = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
        self.store_name = os.path.basename(self.config_path) + "-" + date_str
        self.exp_dir = os.path.join(self.config.trainer.store_dir, self.store_name)
        os.makedirs(self.exp_dir, exist_ok=True)
        shutil.copy(src=self.config_path, dst=self.exp_dir)

    def init_vis(self):
        """
        初始化visdom可视化
        :return:
        """
        if self.config.trainer.get("vis", False):
            self.vis = visualize.Visualizer(env=self.store_name, port=self.config.trainer.vis.port,
                                            server=self.config.trainer.vis.server)
        else:
            self.vis = None

    def init_log(self):
        """
        初始化logger
        :return:
        """
        self.logger_train = logger.get_logger(self.exp_dir, "train.log")
        self.logger_train.info('train log at: {}'.format(self.exp_dir))

    def init_trainer(self):
        """
        初始化optimizer, lr_scheduler和loss等
        :return:
        """
        # 多GPU训练情况下，获取model参数略有不同
        if self.multi_gpu_train:
            self.optimizer = getattr(optim, self.config.trainer.optimizer.optimizer_name)(
                self.model.module.parameters(), **self.config.trainer.optimizer.kwargs)
        else:
            self.optimizer = getattr(optim, self.config.trainer.optimizer.optimizer_name)(
                self.model.parameters(), **self.config.trainer.optimizer.kwargs)
        self.lr_sched = getattr(optim.lr_scheduler,
                                self.config.trainer.lr_scheduler.lr_scheduler_name)(
            self.optimizer, **self.config.trainer.lr_scheduler.kwargs)
        self.loss_fn = getattr(losses, self.config.trainer.loss.loss_name)(**self.config.trainer.loss.kwargs)
        if self.use_swa:
            self.logger_train.info("Use SWA model!")
            self.swa_start = self.config.model.swa.get("swa_start", int(0.75 * self.config.trainer.epochs))
            self.swa_scheduler = SWALR(self.optimizer, self.config.model.swa.get("swa_lr", 1e-3),
                                       anneal_epochs=self.config.model.swa.get("anneal_epochs", 10))

    def prepare_data(self):
        """
        数据初始化, 获取dataloader
        :return:
        """
        self.logger_train.info('==> Start preparing data')
        trainset = getattr(datasets, self.config.data.dataset_name)(self.config.data, self.logger_train, mode='train')
        self.logger_train.info('len of train data:%d' % len(trainset))
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.config.data.batch_size, shuffle=True,
                                                       drop_last=False,
                                                       num_workers=self.config.data.get("num_workers", 4))

        valset = getattr(datasets, self.config.data.dataset_name)(self.config.data, self.logger_train, mode='val')
        self.logger_train.info('len of val data:%d' % len(valset))
        self.valloader = torch.utils.data.DataLoader(valset, batch_size=self.gpu_num, shuffle=False,
                                                     drop_last=False,
                                                     num_workers=self.config.data.get("num_workers", 4))
        self.logger_train.info('==> Successfully preparing data')

    def init_model(self):
        """
        初始化模型
        :return:
        """
        self.logger_train.info('==>Start building model')
        self.model = getattr(models, self.config.model.model_name)(self.config.model)
        if self.config.trainer.resume_from:
            assert os.path.exists(self.config.trainer.resume_from)
            self.logger_train.info('resume from:%s' % self.config.trainer.resume_from)
            net_dict = self.model.state_dict()
            pretrained_dict = torch.load(self.config.trainer.resume_from)
            tmp1 = set(list(net_dict.keys()))
            pretrained_dict_tmp = {k: v for k, v in pretrained_dict.items() if
                                   k in net_dict and v.size() == net_dict[k].size()}
            tmp3 = set(list(pretrained_dict_tmp.keys()))
            self.logger_train.info(tmp1 - tmp3)  # 为空才说明是完全匹配加载
            net_dict.update(pretrained_dict_tmp)
            self.model.load_state_dict(net_dict)
        self.logger_train.info('==> Successfully building model')

        # 是否使用多GPU训练
        if self.multi_gpu_train:
            self.model = torch.nn.DataParallel(self.model)
            self.model = self.model.cuda()
        else:
            self.model = self.model.cuda()

        # 是否使用SWA 来做模型平均，使得模型的泛化能力更强, ref:https://github.com/timgaripov/swa
        if self.use_swa:
            if self.multi_gpu_train:
                self.swa_model = AveragedModel(self.model.module)
            else:
                self.swa_model = AveragedModel(self.model)

    def train(self, epoch=0):
        """
        训练
        :return:
        """
        self.model.train()
        train_loss = 0
        mos_preds_all = []  # 用于记录预测的mos分
        mos_gts_all = []  # 用于记录真实的mos分
        total = 0
        is_plot = True
        lr_current = utils.get_lr(self.optimizer)
        if self.vis:
            self.vis.plot("lr", lr_current)
        for imgs, img_labels in tqdm(self.trainloader):
            imgs, img_labels = imgs.cuda(), img_labels.cuda()
            b, n, _, _, _ = imgs.size()
            preds = self.model(imgs).view(b, n)
            preds = preds.mean(1)
            self.optimizer.zero_grad()
            loss = self.loss_fn(preds, img_labels)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            mos_preds_all.append(preds.cpu().view(-1).data.numpy())
            mos_gts_all.append(img_labels.cpu().view(-1).data.numpy())
            total += 1
            if is_plot and self.vis:
                is_plot = False
                if self.config.model.fr:
                    utils.plot_image(self.vis, torch.cat([imgs[:, 0, :3, ...], imgs[:, 0, 3:, ...]], -1), preds,
                                     img_labels, "train")
                else:
                    utils.plot_image(self.vis, imgs[:, 0, ...], preds, img_labels, "train")
        train_loss = train_loss / total
        train_srocc = stats.spearmanr(np.concatenate(mos_preds_all), np.concatenate(mos_gts_all))[0]
        train_plcc = stats.pearsonr(np.concatenate(mos_preds_all), np.concatenate(mos_gts_all))[0]
        if self.vis:
            self.vis.img("train_corr",
                         utils.plot_corr(np.concatenate(mos_preds_all), np.concatenate(mos_gts_all), train_srocc,
                                         train_plcc), "train_corr")
            self.vis.plot("train srocc", train_srocc)
            self.vis.plot("train plcc", train_plcc)
        self.logger_train.info(
            'epoch %d, train_loss = %.3f, train_srocc = %.3f, train_plcc = %.3f' % (
                epoch, train_loss, train_srocc, train_plcc))

    def val(self, epoch=0):
        """
        模型验证
        :param epoch:
        :return:
        """
        self.model.eval()
        with torch.no_grad():
            test_loss = 0
            mos_preds_all = []  # 用于记录预测的mos分
            mos_gts_all = []  # 用于记录真实的mos分
            total = 0
            is_plot = True
            for imgs, img_labels in tqdm(self.valloader):
                imgs, img_labels = imgs.cuda(), img_labels.cuda()
                b, n, _, _, _ = imgs.size()
                preds = self.model(imgs).view(b, n)
                preds = preds.mean(1)
                loss = self.loss_fn(preds, img_labels)
                mos_preds_all.append(preds.cpu().view(-1).data.numpy())
                mos_gts_all.append(img_labels.cpu().view(-1).data.numpy())
                test_loss += loss.item()
                total += 1
                if is_plot and self.vis:
                    is_plot = True
                    if self.config.model.fr:
                        utils.plot_image(self.vis, torch.cat([imgs[:, 0, :3, ...], imgs[:, 0, 3:, ...]], -1), preds,
                                         img_labels, "val")
                    else:
                        utils.plot_image(self.vis, imgs[:, 0, ...], preds, img_labels, "val")
            test_loss = test_loss / total
            test_srocc = stats.spearmanr(np.concatenate(mos_preds_all), np.concatenate(mos_gts_all))[0]
            test_plcc = stats.pearsonr(np.concatenate(mos_preds_all), np.concatenate(mos_gts_all))[0]
            if self.vis:
                self.vis.img("val_corr",
                             utils.plot_corr(np.concatenate(mos_preds_all), np.concatenate(mos_gts_all), test_srocc,
                                             test_plcc),
                             "val_corr")
                self.vis.plot("val srocc", test_srocc)
                self.vis.plot("val plcc", test_plcc)
            self.logger_train.info(
                'epoch %d, val_loss = %.3f, test_srocc = %.3f, test_plcc = %.3f' % (
                    epoch, test_loss, test_srocc, test_plcc))

            # 记录SROCC最好的epoch
            if test_srocc > self.best_srocc:
                self.best_srocc = test_srocc
                self.best_plcc = test_plcc
                self.best_epoch = epoch
                self.logger_train.info('saving model at %s' % os.path.join(self.exp_dir, 'model_best.pth'))
                if self.multi_gpu_train:
                    torch.save(self.model.module.state_dict(), os.path.join(self.exp_dir, 'model_best.pth'))
                else:
                    torch.save(self.model.state_dict(), os.path.join(self.exp_dir, 'model_best.pth'))

    def run(self):
        """
        训练主函数入口
        :return:
        """
        self.prepare_data()
        self.init_model()
        self.init_trainer()

        for epoch in range(self.config.trainer.epochs):
            self.train(epoch)
            if self.use_swa and epoch >= self.swa_start:
                if self.multi_gpu_train:
                    self.swa_model.update_parameters(self.model.module)
                else:
                    self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()
            else:
                self.lr_sched.step()
            if epoch % self.config.trainer.eval_model_inter == 0:
                self.val(epoch)
            if epoch % self.config.trainer.save_model_inter == 0:
                self.logger_train.info('saving model at %s' % os.path.join(self.exp_dir, 'model_%03d.pth' % epoch))
                if self.multi_gpu_train:
                    torch.save(self.model.module.state_dict(), os.path.join(self.exp_dir, 'model_%03d.pth' % epoch))
                else:
                    torch.save(self.model.state_dict(), os.path.join(self.exp_dir, 'model_%03d.pth' % epoch))
        self.logger_train.info('saving model at %s' % os.path.join(self.exp_dir, 'model_final.pth'))
        if self.multi_gpu_train:
            torch.save(self.model.module.state_dict(), os.path.join(self.exp_dir, 'model_final.pth'))
        else:
            torch.save(self.model.state_dict(), os.path.join(self.exp_dir, 'model_final.pth'))
        if self.use_swa:
            torch.optim.swa_utils.update_bn(self.trainloader, self.swa_model, device=torch.device("cuda:0"))
            # Update bn statistics for the swa_model at the end
            self.logger_train.info('saving SWA model at %s' % os.path.join(self.exp_dir, 'model_swa.pth'))
            torch.save(self.swa_model.module.state_dict(), os.path.join(self.exp_dir, 'model_swa.pth'))
        self.logger_train.info(
            'best_epoch %d, best_srocc = %.3f, best_plcc = %.3f' % (
                self.best_epoch, self.best_srocc, self.best_plcc))
