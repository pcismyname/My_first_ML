import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5,5,100)
y=2*x+1

#plt.plot(x,y,'-r',label = 'y=2*x+1') # -r is red color
plt.scatter(x,y)
plt.xlabel('x')
plt.ylabel('y')
#plt.legend(loc="upper left") #show label
plt.title("Graph y=2x+1")
plt.grid()
plt.show()

