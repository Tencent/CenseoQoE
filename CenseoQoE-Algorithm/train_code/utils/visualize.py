import time
import visdom
import numpy as np
import cv2
from .utils import UnNormalizer


class Visualizer(object):
    """
    visdom可视化
    """

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, use_incoming_socket=False, **kwargs)
        self._vis_kw = kwargs

        # e.g.('loss',23) the 23th value of loss
        self.index = {}
        self.log_text = ''
        self.unorm = UnNormalizer(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def reinit(self, env='default', **kwargs):
        """
        重新初始化
        :param env:
        :param kwargs:
        :return:
        """
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        """
        画多个scalar类型
        :param d:
        :return:
        """
        for k, v in d.items():
            if v is not None:
                self.plot(k, v)

    def img_many(self, d):
        """
        画多个图片
        :param d:
        :return:
        """
        for k, v in d.items():
            self.img(k, v)

    def plot(self, name, y, **kwargs):
        """
        画图
        :param name:
        :param y:
        :param kwargs:
        :return:
        """
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def img(self, name, img_, title, **kwargs):
        """
        画图片
        :param name:
        :param img_:
        :param title:
        :param kwargs:
        :return:
        """
        self.vis.image(img_,
                       win=name,
                       opts=dict(title=title),
                       **kwargs
                       )

    def img_tensor(self, name, img_, **kwargs):
        """
        从img tensor画图
        :param name:
        :param img_:
        :param kwargs:
        :return:
        """
        img_ = self.unorm(img_.squeeze(0)).numpy()
        img_ = np.clip(img_ * 255, 0, 255).astype(np.uint8)
        img_ = np.transpose(img_, [1, 2, 0])
        r, g, b = cv2.split(img_)
        img_ = cv2.merge([b, g, r])
        img_ = np.transpose(img_, [2, 0, 1])
        self.vis.image(img_,
                       win=name,
                       opts=dict(title=name),
                       **kwargs
                       )

    def log(self, info, win='log_text'):
        """
        log
        :param info:
        :param win:
        :return:
        """
        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'),
            info=info))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        return getattr(self.vis, name)

    def state_dict(self):
        return {
            'index': self.index,
            'vis_kw': self._vis_kw,
            'log_text': self.log_text,
            'env': self.vis.env
        }

    def load_state_dict(self, d):
        self.vis = visdom.Visdom(env=d.get('env', self.vis.env), **(self.d.get('vis_kw')))
        self.log_text = d.get('log_text', '')
        self.index = d.get('index', dict())
        return self
