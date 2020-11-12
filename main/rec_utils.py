import cv2
import random
import itertools
import numpy as np
from skimage.measure import compare_mse


def random_drop(img, ws, ratio):
    h, w = img.shape[:2]
    drop_map = np.zeros((h, w))
    img_drop = img.copy()
    for i, j in itertools.product(range(h // ws), range(w // ws)):
        drop_index = random.sample(range(ws ** 2), np.int(np.round(ws ** 2 * ratio)))
        for d in drop_index:
            x, y = i * ws + d % ws, j * ws + d // ws
            img_drop[x, y], drop_map[x, y] = 0, 1

    return img_drop, drop_map


def image_recovery(img_drop, drop_map, ws, delta, mu):
    h, w = img_drop.shape[:2]
    img_rec = img_drop.copy()
    for i, j in itertools.product(range(h // ws), range(w // ws)):
        img1 = img_drop[i * ws:(i + 1) * ws, j * ws:(j + 1) * ws].copy()
        img2 = img_drop[i * ws:(i + 1) * ws, j * ws:(j + 1) * ws].copy()
        for k, l in itertools.product(range(ws), range(ws)):
            if drop_map[i * ws + k, j * ws + l] == 1:
                img1[k, l] += delta
                img2[k, l] -= delta
                dct1 = cv2.dct(img1)
                dct2 = cv2.dct(img2)
                Y1 = np.sum(np.abs(dct1))
                Y2 = np.sum(np.abs(dct2))
                grad = (Y1 - Y2) / (2 * delta)
                img_rec[i * ws + k, j * ws + l] -= mu * grad

    return img_rec


def recovery_iter(img_drop, drop_map, ws, delta, mu, eps=1e-05, max_iters=100):
    max_mse = 0
    gamma = 0.01
    img_rec_0 = img_drop.copy()
    img_rec = np.zeros(img_drop.shape)
    for i in range(max_iters):
        for c in range(3):
            img_rec[:, :, c] = image_recovery(img_rec_0[:, :, c], drop_map, ws, delta, mu)
        img_rec = np.clip(img_rec, 0, 1)
        mse = compare_mse(img_rec_0, img_rec)
        if mse > max_mse:
            max_mse = mse
        elif mse < gamma * max_mse:
            delta /= 10
            mu /= 10
            if mse < eps:
                break
            max_mse = mse

        img_rec_0 = img_rec.copy()

    return img_rec


def accuracy(pred, real):
    return np.sum(np.argmax(pred, 1) == np.argmax(real, 1)) / pred.shape[0]


def linf_distortion(img1, img2):
    if len(img1.shape) == 4:
        n = img1.shape[0]
        l = np.mean(np.max(np.abs(img1.reshape((n, -1)) - img2.reshape((n, -1))), axis=1), axis=0)
    else:
        l = np.max(np.abs(img1 - img2))
            
    return l


def l2_distortion(img1, img2):
    if len(img1.shape) == 4:
        n = img1.shape[0]
        l = np.mean(np.sqrt(np.sum((img1.reshape((n, -1)) - img2.reshape((n, -1))) ** 2, axis=1)), axis=0)
    else:
        l = np.sqrt(np.sum(img1 - img2) ** 2)
    
    return l
