import os
import sys
import json
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
from model import GooLeNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device".format(device))

    data_transform = {"train": transforms.Compose([transforms.RandomResizedCrop(224),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                      "val": transforms.Compose([transforms.Resize((224, 224)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                      }
    data_root = os.path.abspath(os.path.join(os.getcwd(), '../..'))  # get data root path
    image_path = os.path.join(data_root,'data_set','flower_data')
    assert os.path.exists(image_path) , "{} path is not exists.".format(image_path)

    # 返回datasets型 即（data,label）
    # 文件名就是类别名，从上到下自动创建标签
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path,'train'),
                                         transforms=data_transform['train'])
    train_num = len(train_dataset)

    # 返回一个字典
    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    # 将键和值翻转
    cla_dict = dict((val,key) for key, val in flower_list.items())
    # write dict into json file 将python数据结构转为json
    # indent=4 缩进格式
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size =32
    # os.cpu_count获得系统中cpu的数量
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print("Using {} dataloader workers every process".format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'val'),
                                            transforms=data_transform['val'])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=nw)
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    net = GooLeNet(num_classes=5, aux_logits=True, init_weights=True)
    # 如果要使用官方的预训练权重，注意是将权重载入官方的模型，不是我们自己实现的模型
    # 官方的模型中使用了bn层以及改了一些参数，不能混用
    # import torchvision
    # net = torchvision.models.googlenet(num_classes=5)
    # model_dict = net.state_dict()
    # # 预训练权重下载地址: https://download.pytorch.org/models/googlenet-1378be20.pth
    # pretrain_model = torch.load("googlenet.pth")
    # del_list = ["aux1.fc2.weight", "aux1.fc2.bias",111111`
    #             "aux2.fc2.weight", "aux2.fc2.bias",
    #             "fc.weight", "fc.bias"]
    # pretrain_dict = {k: v for k, v in pretrain_model.items() if k not in del_list}
    # model_dict.update(pretrain_dict)
    # net.load_state_dict(model_dict)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0003)

    epochs = 30
    best_acc = 0.0
    save_path = './gooLeNet.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        #   tqdm里可以把一个迭代器作为参数， file=sys.stdout 可以将输出信息呈现默认颜色
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits, aux_logits1, aux_logits2 = net(images.to(device))
            loss0 = loss_function(logits, labels.to(device))
            loss1 = loss_function(aux_logits1, labels.to(device))
            loss2 = loss_function(aux_logits2, labels.to(device))
            # 论文中所给的权重
            loss = loss0 + 0.3 * loss1 + 0.3 * loss2
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            train_bar.desc = "train epoch [{}/{}] loss:{:.3f}".format(epoch+1, epochs, loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad:
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # torch.max()[0] 返回最大值的每个数
                # torch.max()[1] 返回最大值的索引坐标
                predict_y = torch.max(outputs, dim=1)[1]
                # torch.eq()对两个Tensor进行逐行元素比较，相同返回True，不同返回False
                acc += torch.eq(predict_y, val_labels).sum().item()
        val_accurate = acc / val_num
        print('[epoch %d] train_loss = %.3f val_accurate = %.3f' %
              (epoch+1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            # state_dict()是python字典对象将每一层的对应参数建立映射关系
            # 如model每一层的weights和bias等
            torch.save(net.state_dict(), save_path)

    print('Finished Training')

if __name__ == '__main__':
    main()


