import paddle
import paddle.nn as nn
from model import DataLoader
from model import Unet

def train():
    train_loader = DataLoader()

    model = Unet()
    model.train()
    optim = paddle.optimizer.Adam(parameters=model.parameters(), weight_decay=0.001)

    epoch_num = 10
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epoch_num):
        # train
        for batch_id, data in enumerate(train_loader()):
            inputs = paddle.to_tensor(data[0])
            labels = paddle.to_tensor(data[1])

            predicts = model(inputs)
            loss = loss_fn(predicts, labels)

            if batch_id % 10 == 0:
                print("train: epoch: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, loss.numpy()))

            loss.backward()
            optim.step()
            optim.clear_grad()

        if epoch % 2 == 0:
            paddle.save(model.state_dict(), './checkpoints/Unet.pdparams')
            paddle.save(optim.state_dict(), './checkpoints/Unet.pdopt')


if __name__ == '__main__':
    train()