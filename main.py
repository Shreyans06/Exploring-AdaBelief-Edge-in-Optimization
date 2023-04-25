import os
import pickle
from torch import optim, nn
from optimizer import AdaBelief
from models import VGG
import torch
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialize_optimizer(inp_model, optimizer='SGD'):
    if optimizer == 'Adam':
        return optim.Adam(inp_model.parameters(), lr=0.001)
    elif optimizer == 'SGD':
        return optim.SGD(inp_model.parameters(), lr=0.001, momentum=0.9)
    elif optimizer == 'AdaBelief':
        return AdaBelief(inp_model.parameters())


def build_model(model_type):
    network = None
    if model_type == "VGG":
        VGG11 = [64, "MP", 128, "MP", 256, 256, "MP", 512, 512, "MP", 512, 512, "MP"]
        network = VGG(VGG11).to(device)

    if device == 'cuda':
        network = torch.nn.DataParallel(network)

    return network


def cross_entropy_loss_function():
    return nn.CrossEntropyLoss()


def get_data(batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    cifar_train_data = torchvision.datasets.CIFAR10(root='./data/', train=True,
                                                    download=True, transform=transform_train)

    cifar_test_data = torchvision.datasets.CIFAR10(root='./data/', train=False,
                                                   download=True, transform=transform_test)

    cifar_train_loader = DataLoader(cifar_train_data, batch_size=batch_size, shuffle=True)
    cifar_test_loader = DataLoader(cifar_test_data, shuffle=False, batch_size=batch_size)

    return cifar_train_loader, cifar_test_loader


def test(net, test_data, criterion):
    correct = 0
    total = 0
    test_loss = 0

    net.eval()

    with torch.no_grad():
        for data in test_data:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test accuracy {accuracy}%")

    return accuracy, test_loss


def train(net, epoch, train_data, optimizer, criterion):
    net.train()
    correct = 0
    total = 0
    train_loss = 0.0

    print('\nEpoch: %d' % epoch)

    for i, data in enumerate(train_data):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Training accuracy {accuracy}%")

    return accuracy, train_loss


def main(dataset, model_architecture, init_optimizer):
    train_loader, test_loader = get_data()

    net = build_model(model_architecture)
    criterion = cross_entropy_loss_function()
    optimizer = initialize_optimizer(net, init_optimizer)

    start = 1
    end = 200
    best_acc = 0
    train_accuracies = []
    test_accuracies = []
    train_loss_trends = []
    test_loss_trends = []

    for epoch in range(start, end+1):

        train_acc, train_loss = train(net, epoch, train_loader, optimizer, criterion)
        test_acc, test_loss = test(net, test_loader, criterion)

        if test_acc > best_acc:
            state = {
                'net': net.state_dict(),
                'acc': test_acc,
                'epoch': epoch,
            }
            file_path = os.path.join(os.getcwd() + "/Best_trained_models/"+f"{dataset}_{model_architecture}_{init_optimizer}.pt" )
            torch.save(state,file_path)
            best_acc = test_acc

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        train_loss_trends.append(train_loss)
        test_loss_trends.append(test_loss)

    pickle.dump({'train_acc': train_accuracies, 'test_acc': test_accuracies, 'train_loss': train_loss_trends,
                 'test_loss': test_loss_trends}, open(
        os.path.join(os.getcwd() + "/Plot_curves",  f"{dataset}_{model_architecture}_{init_optimizer}.p"),"wb"))


main("CIFAR-10", "VGG", "Adam")
