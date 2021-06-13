import os, torch, csv
from Utils import Utils, Model
from torchvision import transforms

def load_data():
    images = []
    path = r'test'
    image_n = os.listdir(path)
    transform = transforms.Compose([
                transforms.Resize((100,100)),
                transforms.ToTensor()
                ])

    for img in image_n:
        img_p = os.path.join(path,img)
        img = Utils.pil_loader(img_p)
        img = transform(img)
        images.append(img)
    images = torch.stack(images)
    return images, image_n


def load_net(m_name,c):
    model = Model.My_Model(c)
    model = torch.nn.DataParallel(model).to(torch.device("cuda"))
    model.load_state_dict(torch.load(r'./Models/' + m_name))
    model.eval()
    return model


def test_net(net, data, image_n):
    preds = net(data)

    is_apples = torch.argmax(preds,dim=1) 

    lines = []
    percent = r'%'
    for is_apple, name, prob in zip(is_apples, image_n, preds.data[:,0]):
        if is_apple == 0:
            line = 'Image %s contains an apple with probability of %.1f %s' %(name, prob*100, percent)
            # print(line)
            lines.append(line)
        else:
            line = 'I dont think image %s contains an apple' %(name)
            # print(line)
            lines.append(line)

    return lines

def write_preds(preds):

    f = open('Predictions.csv','a')
    with f:
        writer = csv.writer(f, lineterminator='\n')
        for row in preds:
            writer.writerow([str(row)])
            print (row)
    return

def main():
    names = ['Apple_Classifier.pth']
    classes = [17]
    data, image_n = load_data()
    for name, c in zip(names,classes): 
        net = load_net(name, c)
        print('Predicting on %s' %(name))
        preds = test_net(net, data, image_n)
        write_preds(preds)
    
    return


if __name__ == '__main__':
    main()
