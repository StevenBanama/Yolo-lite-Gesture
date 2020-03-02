#coding=utf-8
import numpy as np
import cv2
import time



def cost(func):
    def __wrap__(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print("-------------", func.__name__, end - start)
        return result

    return __wrap__
    

class GMM:

    def __init__(self, height, width, model_per_pixl, channel=3):
        self.k = model_per_pixl
        self.channel = channel
        self.height, self.width = height, width
        self.means = np.zeros([height, width, model_per_pixl, channel], np.float64)
        self.variance = np.ones([height, width, model_per_pixl, channel])
        self.omega = np.ones([height, width, model_per_pixl]) / model_per_pixl  # model_wight 
        self.rol = np.zeros([height, width, model_per_pixl]) 

        self.lr = self.alpha = 0.05
        self.init_weight = 0.1
        self.max_var = 255
        self.count = 0

    def norm_weight(self):
        self.omega = self.omega /np.sum(self.omega, axis=-1)[..., np.newaxis]

    @cost
    def pdf(self, img):
        # 多维高斯分布
        exp = -0.5 * np.sum(np.power(img - self.means, 2) / self.variance, axis=-1)
        c = np.power(2 * np.pi * np.sum(self.variance, axis=-1), self.channel / 2)
        return 1 / c * np.exp(exp)

    @cost
    def resorted(self):
        sort_idx = np.argsort(-self.omega, axis=-1)
        for r in range(self.height):
            for c in range(self.width):
                perm = sort_idx[r, c]  # 降序
                self.omega[r, c] = self.omega[r, c, perm]
                self.means[r, c] = self.means[r, c, perm]
                self.variance[r, c] = self.variance[r, c, perm]

    @cost
    def background_mask(self, T=0.3):
        cum_weight = np.cumsum(self.omega, axis=-1)
        return cum_weight < T


    @cost
    def fit(self, img):
        new_img = np.tile(img[:,:, np.newaxis,:], [1, 1, self.k, 1])
        self.rol = self.alpha * self.pdf(new_img)

        update_var = (new_img - self.means) ** 2
        bgd_delta = np.sqrt(np.sum(update_var / self.variance, axis=-1))
        bgd_mask = bgd_delta < 2.5
        result = np.any(bgd_mask & self.background_mask(), axis=-1)
        self.count += 1
        
        self.alpha = 0.1 if self.count < 10 else 0.0001
        print(self.alpha)

        # update omega
        # min_dis_index = np.argmin(bgd_delta, axis=-1)
        for r in range(self.height):
            for c in range(self.width):
                min_dis_idx = None
                for idx, bgd in enumerate(bgd_delta[r][c]):
                    if bgd:
                         min_dis_idx = idx
                min_weight_idx = -1
                self.omega[r, c] = (1 - self.alpha) * self.omega[r, c]
                if min_dis_idx is not None and bgd_mask[r, c, min_dis_idx]:  # 最小的高斯 小于2.5
                    self.omega[r, c, min_dis_idx] += self.alpha  # update weight
                    self.means[r, c, min_dis_idx] += self.rol[r, c, min_dis_idx] * (img[r, c] - self.means[r, c, min_dis_idx])
                    self.variance[r, c, min_dis_idx] += self.rol[r, c, min_dis_idx] * (update_var[r, c, min_dis_idx] - self.variance[r, c, min_dis_idx])

                else:  # 都不是backgound
                    self.omega[r, c, min_weight_idx] = self.init_weight
                    self.means[r, c, min_weight_idx] = img[r, c]
                    self.variance[r, c, min_weight_idx] = self.max_var

        self.norm_weight()
        self.resorted()
        return result

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 300)
    cap.set(4, 300)
    _, img = cap.read()
    height, width, channel = img.shape
    start = time.time()
    gm = GMM(height, width, 5, channel)
    while(True):
        _, img = cap.read()
        mask = gm.fit(img)
        print(mask.shape, img.shape)
        mask = cv2.medianBlur(mask.astype(np.uint8), 5)
        img[mask.astype(np.bool)] = (255, 255, 255)
        cv2.imshow("test", img)
        cv2.waitKey(100)
main()
