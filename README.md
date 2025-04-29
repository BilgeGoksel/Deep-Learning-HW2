# Derin Öğrenme Projesi

---

## Giriş

Bu projede, derin öğrenme ve makine öğrenmesi yaklaşımları kullanılarak görüntü sınıflandırması gerçekleştirilmiştir. Evrişimli Sinir Ağları (CNN) mimarileri MNIST ve CIFAR-10 veri setleri üzerinde eğitilmiş, ayrıca CNN tabanlı özellik çıkarımı ile klasik makine öğrenmesi algoritması (SVM) uygulanmıştır. Farklı mimariler ve yöntemler arasındaki başarımlar karşılaştırılmıştır.

---

## Yöntem

### Kullanılan Veri Setleri

- **MNIST**: 28x28 boyutunda gri tonlamalı el yazısı rakamları içeren veri seti (10 sınıf).
- **CIFAR-10**: 32x32 boyutunda RGB renkli görüntülerden oluşan veri seti (10 sınıf).

### Kullanılan Modeller

- **1. Model (SimpleCNN)**: LeNet-5 benzeri klasik bir CNN mimarisi.
- **2. Model (ImprovedCNN)**: SimpleCNN modeline Dropout katmanı eklenerek overfitting azaltılmıştır.
- **3. Model (VGG16)**: Torchvision kütüphanesinden hazır VGG16 mimarisi CIFAR-10 veri seti üzerinde eğitilmiştir.
- **4. Model (Hibrit Yaklaşım)**: SimpleCNN modeli ile çıkarılan özellikler kullanılarak Support Vector Machine (SVM) sınıflandırması yapılmıştır.

### Eğitim Detayları

- Optimizer: SGD (Stochastic Gradient Descent) (momentum=0.9)
- Loss Function: CrossEntropyLoss
- Learning Rate: 0.01
- Epoch Sayısı: 5
- Batch Size: 64

---

## Sonuçlar

### Test Doğrulukları

| Model                           | Kullanılan Veri Seti | Test Doğruluğu (%) |
| ------------------------------- | -------------------- | ------------------ |
| SimpleCNN                       | MNIST                | 98.46%             |
| ImprovedCNN (Dropout)           | MNIST                | 98.83%             |
| VGG16                           | CIFAR-10             | 72.68%             |
| SVM (SimpleCNN Özellikleri ile) | MNIST                | 97.55%             |

---

### Karmaşıklık Matrisi (SVM - MNIST)

```
[[ 973    0    1    0    0    1    3    1    0    1]
 [   0 1127    1    1    0    1    0    1    3    1]
 [   5    2 1005    5    1    0    2    6    3    3]
 [   0    0    8  984    0    4    0    8    3    3]
 [   1    0    0    0  963    0    5    1    2   10]
 [   3    1    0    6    0  868    4    1    6    3]
 [   6    2    3    3    1    4  936    1    2    0]
 [   0    7   16    2    4    0    0  991    1    7]
 [   5    0    4    3    5    3    1    2  948    3]
 [   5    6    1    6   12    2    0    7   10  960]]
```

---

### Eğitim ve Test Kayıpları (Loss) [Özet]

- **SimpleCNN:** Başlangıçta %96.61 doğruluk, 5. epoch sonunda %98.46 doğruluk.
- **ImprovedCNN (Dropout):** Başlangıçta %96.40 doğruluk, 5. epoch sonunda %98.83 doğruluk.
- **VGG16 CIFAR-10:** 1. epoch %40.29 doğruluk, 5. epoch %72.68 doğruluk.

### Eğitim ve Test Doğruluğu / Loss Grafikleri

Karşılaştırmalı Test Doğruluk Grafiği (Tüm Modeller)

Grafik: SimpleCNN, ImprovedCNN (MNIST) ve VGG16 (CIFAR-10) modellerinin epoch bazlı test doğruluklarının karşılaştırılması
---

## Tartışma

- **SimpleCNN ve ImprovedCNN** modelleri MNIST veri seti üzerinde oldukça yüksek başarı sağlamıştır. Özellikle Dropout uygulaması, overfitting'i azaltarak doğruluk oranını artırmıştır.
- **VGG16 modeli** CIFAR-10 gibi daha kompleks bir veri setinde çalıştırılmıştır. Başlangıçta düşük doğruluklar gözlemlenmiş fakat epoch ilerledikçe performans artmıştır. Ancak daha iyi sonuçlar için daha fazla eğitim süresi (daha çok epoch) ve veri artırımı (augmentation) teknikleri uygulanabilir.
- **SVM ile yapılan hibrit yaklaşım**, doğrudan CNN çıktılarından elde edilen özelliklerle oldukça başarılı sonuçlar vermiştir. Bu yöntem özellikle küçük veri setlerinde hesaplama maliyetini azaltarak hızlı ve başarılı bir çözüm sağlamaktadır.
- **Sonuç olarak**, hem klasik CNN yaklaşımları hem de hibrit yöntemler MNIST gibi basit veri setlerinde yüksek performans göstermektedir. Daha karmaşık veri setleri için ise daha derin ve kompleks modeller gerekmektedir.

---

## Referanslar

1. PyTorch Documentation: [https://pytorch.org/](https://pytorch.org/)
2. Torchvision Models: [https://pytorch.org/vision/stable/models.html](https://pytorch.org/vision/stable/models.html)
3. MNIST Dataset: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
4. CIFAR-10 Dataset: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)

---

# 🎯 Notlar:

- Modellerin tam kodları GitHub deposunda paylaşılmıştır.
- Proje Python 3.8+, PyTorch 1.9+, torchvision 0.10+ sürümleriyle test edilmiştir.

