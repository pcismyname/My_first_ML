import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

rng = np.random

#simulate data
x = rng.rand(30)*10
y = 2*x+rng.randn(30) # randn is random negative

#linear_regression_model
model = LinearRegression()

x_new = x.reshape(-1,1)


#train
model.fit(x_new,y)


#test model
xfit = np.linspace(-1,11)
xfit_new = xfit.reshape(-1,1)
print(xfit_new)

yfit = model.predict(xfit_new)

#analysis model
plt.scatter(x,y)
plt.plot(xfit,yfit) #linear regression
plt.show()





