import torch.optim as optim
from Utils import Model
import torchvision as tv
import torch.utils.data as data
from Utils import Utils
import torch

if __name__ == '__main__':
    transform = tv.transforms.Compose([
                    tv.transforms.Resize((100,100)),
                    tv.transforms.RandomHorizontalFlip(),
                    tv.transforms.ToTensor()
                    ])
    im_reader = Utils.pil_loader
    batch_s = 2000
    model = Model.My_Model()
    device = torch.device('cuda')
    model = torch.nn.DataParallel(model).to(device)
    parameters = list(model.parameters())
    dataset = tv.datasets.ImageFolder(r'D:\Fruits\archive', transform=transform,loader=im_reader)
    train_data_loader = data.DataLoader(dataset, batch_size=batch_s, shuffle=True,  num_workers=1, drop_last=True)
    criterion = torch.nn.CrossEntropyLoss() # weight= torch.tensor((.3,.7)).to(device)
    optimizer = optim.Adam(parameters, lr=0.003)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    
    print('Training...')
    for epochs in range(30):

        cm_loss = 0.0
        cm_acc = 0.0
        batches = 0

        for i, data in enumerate(train_data_loader, 0):
            image, label = data
            a_lbl = label.to(device) #torch.eq(label.to(device),torch.zeros((batch_s)).to(device)).long()
            image = image.to(device)
            output = model(image)
            b_loss = criterion(output, a_lbl)
                        
            optimizer.zero_grad()
            b_loss.backward()
            optimizer.step()
            cm_loss +=  b_loss.item()
            batches+=1
            preds = torch.argmax(output,dim=1)

            correct = torch.eq(preds,a_lbl).int()
            cm_acc += torch.sum(correct).data/batch_s

        
        print('Epoch: %d, acc: %.2f, loss: %.3f' %(epochs, cm_acc /batches, cm_loss / batches))
        lr_scheduler.step()


    print('Finished Training')
    
    PATH = r'./Models/Apple_Classifier_d.1_e50.pth'
    torch.save(model.state_dict(), PATH)


    
    # dataloader = 

    # model = Model.My_Model()


    # trainLoop

    # loss = 
    # optimizer




