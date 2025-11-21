import torch #permet de manipuler des tenseurs
from torch import nn #module pour construire des réseaux de neurones
from torch.utils.data import DataLoader #module pour charger des données
from torchvision import datasets #module pour les datasets d'images
from torchvision.transforms import ToTensor #module pour transformer les images en tenseurs
import matplotlib.pyplot as plt


# Download training data from open datasets.
# FashionMNIST est un dataset d'images 28x28 en noir et blanc (tee-shirts, pantalons, chaussures, etc.)
training_data = datasets.FashionMNIST(
    root="data", #répertoire où les données seront stockées
    train=True, #indique que ce sont des données d'entraînement
    download=True, #télécharge les données si elles ne sont pas déjà présentes
    transform=ToTensor(), #transforme les images en tenseurs PyTorch (entre 0 et 1)
)

#training_data est un dataset PyTorch contenant 60 000 images + labels.

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False, #indique que ce sont des données de test
    download=True, #télécharge les données si elles ne sont pas déjà présentes
    transform=ToTensor(), #transforme les images en tenseurs PyTorch (entre 0 et 1)
)

'''
img, label = training_data[12]

print("Shape du tenseur :", img.shape)
print("Label :", label)

img est un tenseur [1, 28, 28] → on enlève la dimension du canal
plt.imshow(img.squeeze(), cmap="gray")
plt.title(f"Label : {label}")
plt.show()
'''

batch_size = 64

# DataLoader = le moteur qui prépare les données pour l’entraînement.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Donc un batch = tenseur de 64 images 1×28×28.
# Un batch = un petit groupe d’exemples que tu donnes au modèle à chaque étape d’entraînement.
# Au lieu de donner 1 image à la fois, ou 60 000 images d’un coup, tu donnes par exemple 64 images d’un coup.
# Les labels sont un simple vecteur de 64 entiers