import torch
import torch.nn as nn
from LaterFusion import LaterFusion
import torch.optim as optim
import Config
import LoadData
import time


device = Config.device
total_train_step = 0
total_test_step = 0
epoch = 10
test_data_size = 482
start_time = time.time()


class FinalClassification(nn.Module):
    def __init__(self, dim):
        super(FinalClassification, self).__init__()
        self.dim = dim
        self.fc1 = nn.Linear(dim, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        return out

# a = torch.randn(32, 128)
# classifier = FinalClassification(128)
# out = classifier(a)
# print(out)

if __name__ == '__main__':

    model = LaterFusion().to(device)


    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    for i in range(epoch):
        print("-----------------第{}轮训练开始-------------------".format(i+1))
        # 训练
        model.train()
        for input_ids, atten_mask, image_feature, token_type_ids, label, group, id in LoadData.train_loader:
            input_ids, atten_mask, image_feature, token_type_ids, label = input_ids.to(device), atten_mask.to(device), \
                                                                          image_feature.to(device), token_type_ids.to(
                device), label

            optimizer.zero_grad()
            pred, dim = model(label, input_ids, atten_mask, token_type_ids, image_feature)
            # print("pred", pred.shape)
            classifier = FinalClassification(dim).to(device)
            pred = classifier(pred)
            # print(group.shape)
            group = group.to(device)
            loss = loss_fn(pred, group)

            loss.backward(retain_graph=True)
            optimizer.step()

            total_train_step += 1
            if total_train_step % 80 == 0:
                end_time = time.time()
                print(end_time - start_time)
                print("训练次数：{}， Loss：{}".format(total_train_step, loss.item()))


        # 测试
        total_test_loss = 0
        total_accuracy = 0

        model.eval()
        with torch.no_grad():
            for input_ids, atten_mask, image_feature, token_type_ids, label, group, id in LoadData.test_loader:
                input_ids, atten_mask, image_feature, token_type_ids, label = input_ids.to(device), atten_mask.to(device), \
                                                                              image_feature.to(device), \
                                                                              token_type_ids.to(device), label

                outputs, dim = model(label, input_ids, atten_mask, token_type_ids, image_feature)
                classifier = FinalClassification(dim).to(device)

                outputs = classifier(outputs)
                group = group.to(device)
                loss = loss_fn(outputs, group)
                total_test_loss += loss.item()

                accuracy = (outputs.argmax(1) == group).sum()
                total_accuracy += accuracy

        print("整体测试集上的Loss：{}".format(total_test_loss))
        print("整体测试集上的Accuracy：{}".format(total_accuracy / test_data_size))

        ExpLR.step()

    torch.save(model, "../data/model/model.pth")
    torch.save(model.state_dict(), "../data/model/model_para.pth")
    print("模型已保存")


