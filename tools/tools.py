import cv2
import numpy as np
import paddle
import paddle.nn.functional as F

def process(img, model, color=(0x40, 0x16, 0x66)):
    h, w, _ = img.shape
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
    data = img.astype('float32').transpose([2, 0, 1])
    data = paddle.unsqueeze(paddle.to_tensor(data), axis=0)

    predict = paddle.squeeze(paddle.argmax(F.softmax(model(data)), axis=-1))
    predict = predict.numpy().astype('int64')

    hair = np.ones(predict.shape) * 13
    mask = np.expand_dims(np.array(predict == hair).astype('int64'), axis=-1)

    img = recolor(img, mask, color)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    return img

def change_v(v, mask, target):
    v_mean = np.sum(v * mask) / np.sum(mask)
    alpha = target / v_mean
    x = v / 255
    x = 1 - (1 - x) ** alpha
    v[:] = x * 255

def recolor(img, mask, color=(0x40, 0x66, 0x66)):
    color = np.array(color, dtype='uint8', ndmin=3)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color_hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    img_hsv[..., 0] = color_hsv[..., 0]
    change_v(img_hsv[..., 2:], mask, color_hsv[..., 2:])
    change_v(img_hsv[..., 1:2], mask, color_hsv[..., 1:2])
    hair = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR) * mask
    origin = img * (1 - mask)
    img = (origin + hair).astype(np.uint8)

    return img