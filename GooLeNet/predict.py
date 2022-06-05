import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import GooLeNet

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # load image
    image_path = "./tuilp.jpg"
    assert os.path.exists(image_path), "file {} is not exists".format(image_path)
    img = Image.open(image_path)
    plt.imshow(img)
    # [N, C, W, H]
    img = data_transform(img)
    # expand branch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), '{} is not exists.'.format(json_path)
    with open(json_path, 'r') as f:
        class_indict = json.load(f)

    # create model
    model = GooLeNet(num_classes=5, aux_logits=False).to(device)
    # load model path
    weights_path = './GooLeNet.pth'
    assert os.path.exists(weights_path), '{} is noe exists.'.format(weights_path)
    # strict=True 严格按照参数一一对应 但实例化没有辅助分类区 不用严格对照
    missing_keys, unexpected_keys = model.load_state_dict(torch.load(weights_path, map_location=device),
                                                          strict=False)

    model.eval()
    with torch.no_grad:  # 不跟踪变量的损失梯度
        # predict class
        # 图片通过model进行正向传播，得到输出后，将batch维度压缩掉，得到输出结果
        # torch.nn.Module.cpu() 将所有的参数和缓存转移到cpu上 返回自身
        output = torch.squeeze(model(img.to(device))).cpu()
        # dim=0 在每列上进行softmax dim=1 在每行上进行进行softmax
        predict = torch.softmax(output, dim=0)
        # 返回指定维度最大值的序号
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class :{} prob:{:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range (len(predict)):
        print('class:{:10}  prob{.3f}'.format(class_indict[str[i]],
                                              predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()

