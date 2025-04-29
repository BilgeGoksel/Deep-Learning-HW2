import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 100 == 99:
            print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.4f}')

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy

def main():
    # Cihaz seçimi
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # CIFAR-10 için dönüşümler
    transform_cifar = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Veri seti
    train_dataset = datasets.CIFAR10(root='C:/Users/HP/Desktop/DERSLER/Derin Öğrenme Lab/cifar-100-python', train=True, download=True, transform=transform_cifar)
    test_dataset = datasets.CIFAR10(root='C:/Users/HP/Desktop/DERSLER/Derin Öğrenme Lab/cifar-100-python', train=False, download=True, transform=transform_cifar)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # VGG16 Modeli
    model = models.vgg16(pretrained=False)
    model.classifier[6] = nn.Linear(4096, 10)  # 10 sınıf için çıktıyı ayarlıyoruz
    model = model.to(device)

    # Optimizer ve Loss
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Eğitim döngüsü
    num_epochs = 5
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}")
        train(model, device, train_loader, optimizer, criterion, epoch)
        test(model, device, test_loader, criterion)

if __name__ == '__main__':
    main()
