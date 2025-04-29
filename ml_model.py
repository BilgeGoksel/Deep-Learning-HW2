import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Özellikleri yükle
train_features = np.load('train_features.npy')
train_labels = np.load('train_labels.npy')
test_features = np.load('test_features.npy')
test_labels = np.load('test_labels.npy')

# SVM Modeli
svm = SVC(kernel='linear')  # Linear kernel kullanıyoruz
svm.fit(train_features, train_labels)

# Test
predictions = svm.predict(test_features)

# Sonuçlar
accuracy = accuracy_score(test_labels, predictions)
print(f"SVM Test Accuracy: {accuracy * 100:.2f}%")

conf_matrix = confusion_matrix(test_labels, predictions)
print("Confusion Matrix:")
print(conf_matrix)
