import cv2
import paddle
from model import Unet
from tools.tools import process

def predict():
    img = cv2.imread('./predict/test.jpg')

    model = Unet()
    model_state_dict = paddle.load('./checkpoints/Unet.pdparams')
    model.load_dict(model_state_dict)

    img = process(img, model, color=(0x00, 0x40, 0x66))
    cv2.imwrite('./predict/predict.jpg', img)
    cv2.imshow('predict.jpg', img)
    cv2.waitKey(0)

if __name__ == '__main__':
    predict()