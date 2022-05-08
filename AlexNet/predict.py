import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import AlexNet

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([transforms.Resize((224,224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5),(0.5,0.5,0.5))])

    # load image
    img_path = "./tulip.png"
    assert os.path.exists(img_path), "file: '{}' does not exist.".format(img_path)
    img = Image.open(img_path)

    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class)indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path),"file: '{}' does not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = AlexNet(num_classes=5).to(device)

    #load model weights
    weights_path = "./AlexNet.pth"
    assert os.path.exists(weights_path),"file: '{}' does not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))
    # 关闭dropout
    model.eval()
    with torch.no_grad():
        # predict class
        # squeeze 返回一个Tensor 把大小为1的所有张量都删除:把Batch 压缩掉
        # .cpu() 把变量放到cpu上
        output = torch.squeeze(model(img.to(device))).cpu()
        # dim=0对每一行归一化 dim=1对每一列归一化
        predict = torch.softmax(output, dim=0)
        # .argmax得到最大值的索引序号 返回的是Tensor
        #  .numpy() 将Tensor转成 numpy
        predict_cla = torch.argmax(predict).numpy()

    print_res = "calss: {}  probablity:{:.3}".format(class_indict[str(predict_cla)],
                                                      predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   probablity: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))


    plt.show()


if __name__ == '__main__':
    main()
