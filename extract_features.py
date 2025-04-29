import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np


from MNIST_CNN import SimpleCNN  # 1.modelden import

# Cihaz seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MNIST için dönüşümler
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Verisetini yükleme
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Modeli yükleme
model = SimpleCNN()
torch.save(model.state_dict(), 'simplecnn.pth')
model.load_state_dict(torch.load('simplecnn.pth'))  # Eğitilmiş modeli yüklüyoruz
model = model.to(device)
model.eval()

# Özellikleri çıkartacağımız ara katmanı değiştireceğiz
# Biz fc1 katmanından önceki vektörü almak istiyoruz

def extract_features(model, loader):
    features = []
    labels = []
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            
            x = torch.relu(model.conv1(data))
            x = model.pool(x)
            x = torch.relu(model.conv2(x))
            x = model.pool(x)
            x = x.view(-1, 16*4*4)  # Flatten
            
            features.append(x.cpu().numpy())
            labels.append(target.cpu().numpy())
    
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels

# Özellik çıkarımı
train_features, train_labels = extract_features(model, train_loader)
test_features, test_labels = extract_features(model, test_loader)

# Kaydetme
np.save('train_features.npy', train_features)
np.save('train_labels.npy', train_labels)
np.save('test_features.npy', test_features)
np.save('test_labels.npy', test_labels)

print("Özellikler başarıyla kaydedildi!")
