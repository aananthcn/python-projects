import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 360, 0.1)

plt.figure(1)

# First order polynomial
y = 1 + x
plt.subplot(331)
# plt.plot(x,y, 'bo')
plt.plot(x, y)
plt.title("1st Order")


# Second order polynomial
y = 1 + x + x*x
plt.subplot(332)
plt.plot(x, y)
plt.title("2nd Order")


# 3rd order polynomial
y = 1 + x + x*x + x*x*x
plt.subplot(333)
plt.plot(x, y)
plt.title("3rd Order")


# 4th order polynomial
y = 1 + x + x*x + x*x*x + x*x*x*x
plt.subplot(334)
plt.plot(x, y)
plt.title("4th Order")


# exponential
y = np.exp(x)
plt.subplot(335)
plt.plot(x, y)
plt.title("Exponential")


# (1 + 1/x)^x
y = np.power((1+1/x), x)
plt.subplot(336)
plt.plot(x, y)
plt.title("e = (1 + 1/x)^x")


# Sinusoidal
t = np.arange(0, 720, 5)
y = np.sin(np.deg2rad(x))
plt.subplot(337)
plt.plot(x, y)
plt.title("Sine")


# Cosine
y = np.cos(np.deg2rad(x))
plt.subplot(338)
plt.plot(x, y)
plt.title("Cosine")

plt.tight_layout()
plt.show()
