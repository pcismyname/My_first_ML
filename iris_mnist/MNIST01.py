from scipy.io import loadmat
import matplotlib.pyplot as plt

mnist_raw = loadmat("mnist-original.mat")

mnist = {

    "data" : mnist_raw["data"].T, #.T is Transapose of matrix (original (784,70000)
    "target" : mnist_raw["label"][0]

}

#data store all picture
data = mnist["data"]

#label store all target
label = mnist["target"]

print(data.shape)
print(mnist["data"].shape) # (70000 data ,28*28 pixels)

number = data[100]
number_image = number.reshape(28,28)

print(label[100]) #print number

plt.imshow(
    number_image,
    cmap=plt.cm.binary,
    interpolation="nearest"
)
plt.show()