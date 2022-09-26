import argparse
from statistics import mean

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from mnist_net import MNISTNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(net, optimizer, writer, loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        running_loss = []
        t = tqdm(loader)
        for x, y in t:
            x, y = x.to(device), y.to(device)
            outputs = net(x)
            loss = criterion(outputs, y)
            running_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.set_description(f'training loss: {mean(running_loss)}')
        writer.add_scalar('training loss', mean(running_loss), epoch)

def test(model, dataloader):
    test_corrects = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x).argmax(1)
            test_corrects += y_hat.eq(y).sum().item()
            total += y.size(0)
    return test_corrects / total

if __name__=='__main__':

  parser = argparse.ArgumentParser()
  
  parser.add_argument('--exp_name', type=str, default = 'MNIST', help='experiment name')
  parser.add_argument('--batch_size',type=int, default = 100, help='batch size')
  parser.add_argument('--lr',type=float, default = 1e-4, help='learning rate')
  parser.add_argument('--epochs',type=int, default = 10, help='Number of epochs')
  
  args = parser.parse_args()
  exp_name = args.exp_name
  epochs = args.epochs
  batch_size = args.batch_size
  lr = args.lr
  
  writer = SummaryWriter(f'runs/MNIST')

  # transforms
  transform = transforms.Compose(
      [transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5,))])

  # datasets
 # trainset = torchvision.datasets.MNIST('./data', download=True, train=True, transform=transform)
  trainset_ = torchvision.datasets.MNIST('./data', download=True, train=True, transform=transform)
  indices = torch.arange(9000)
  trainset = torch.utils.data.Subset(trainset_, indices)

#  testset = torchvision.datasets.MNIST('./data', download=True, train=False, transform=transform)
  testset_ = torchvision.datasets.MNIST('./data', download=True, train=False, transform=transform)
  indices = torch.arange(1000)
  testset = torch.utils.data.Subset(testset_, indices)

  # dataloaders
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
  testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
  
  net = MNISTNet()
  # setting net on device(GPU if available, else CPU)
  net = net.to(device)
  optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

  train(net, optimizer, writer, loader=trainloader, epochs=10)
  test_acc = test(model=net, dataloader=testloader)
  print(f'Test accuracy:{test_acc}')
  torch.save(net.state_dict(), "mnist_net.pth")

  #add embeddings to tensorboard
  perm = torch.randperm(len(trainset_.data))
  images, labels = trainset_.data[perm][:256], trainset_.targets[perm][:256]
  images = images.unsqueeze(1).float().to(device)
  with torch.no_grad():
    embeddings = net.get_features(images)
    writer.add_embedding(embeddings,
                  metadata=labels,
                  label_img=images, global_step=1)

  # save networks computational graph in tensorboard
  writer.add_graph(net, images)
  # save a dataset sample in tensorboard
  img_grid = torchvision.utils.make_grid(images[:64])
  writer.add_image('mnist_images', img_grid)