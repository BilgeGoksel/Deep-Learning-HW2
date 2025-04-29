import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Cihaz seçimi 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MNIST veri seti için dönüşümler
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST için ortalama ve standart sapma
])

# Eğitim ve test veri setlerini indirme
train_dataset = datasets.MNIST(root="C:/Users/HP/Desktop/DERSLER/Derin Öğrenme Lab/MNIST", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="C:/Users/HP/Desktop/DERSLER/Derin Öğrenme Lab/MNIST", train=False, download=True, transform=transform)

# Veri yükleyicileri
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)  # 1 kanal (gri), 6 filtre
        self.pool = nn.AvgPool2d(2, 2)               # Havuzlama (2x2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5) # 6 giriş kanalı, 16 filtre
        self.fc1 = nn.Linear(16*4*4, 120)             # Flatten sonrası tam bağlantı
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)                  # 10 sınıf (0-9 rakamlar)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16*4*4)  # Flatten
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.dropout = nn.Dropout(0.5)  # %50 dropout
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16*4*4)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Dropout ekledik
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Eğitim fonksiyonu
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()         # Gradyanları sıfırla
        output = model(data)           # Model tahmini
        loss = criterion(output, target)  # Kayıp hesapla
        loss.backward()                # Geri yayılım
        optimizer.step()               # Optimizasyon adımı

        running_loss += loss.item()
        if batch_idx % 100 == 99:       # Her 100 batch'te bir ekrana yaz
            print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.4f}')

# Test fonksiyonu
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # Toplam kaybı topla
            pred = output.argmax(dim=1, keepdim=True)      # En yüksek olasılıklı sınıfı seç
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    
    return accuracy



# Hiperparametreler
learning_rate = 0.01
num_epochs = 5  

# Model, optimizer ve loss function
model1 = SimpleCNN().to(device)
optimizer1 = optim.SGD(model1.parameters(), lr=learning_rate, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# Eğitim döngüsü
for epoch in range(1, num_epochs + 1):
    train(model1, device, train_loader, optimizer1, criterion, epoch)
    test(model1, device, test_loader, criterion)


# Model, optimizer ve loss function
model2 = ImprovedCNN().to(device)
optimizer2 = optim.SGD(model2.parameters(), lr=learning_rate, momentum=0.9)

# Eğitim döngüsü
for epoch in range(1, num_epochs + 1):
    train(model2, device, train_loader, optimizer2, criterion, epoch)
    test(model2, device, test_loader, criterion)
