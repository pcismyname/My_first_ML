import matplotlib.pyplot as plt
from sklearn import datasets

digit_dataset = datasets.load_digits()
for i in range(0,10):
    plt.imshow(digit_dataset.images[i],cmap=plt.get_cmap('gray'))
    plt.show()