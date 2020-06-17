from sklearn import datasets

iris_dataset = datasets.load_iris()

print(iris_dataset.keys())
print(iris_dataset['feature_names'])
print(iris_dataset['target_names'])
print(iris_dataset['data'][0:10])

print('\n\n\n\n\n\n\n\n')

digit_dataset = datasets.load_digits()

print(digit_dataset.keys())
print(digit_dataset.target_names)
print(digit_dataset.images[0])
print(digit_dataset.images[0].shape)


