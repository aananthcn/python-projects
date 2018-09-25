import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps

x_list = [2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2, 1, 1.5, 1.1]
y_list = [2.4, 0.7, 2.9, 2.2, 3.0, 2.7, 1.6, 1.1, 1.6, 0.9]

x_mean = np.mean(x_list)
y_mean = np.mean(y_list)

xm_list = []
ym_list = []

print("X mean:")
for x in x_list:
    xm_list.append(x - x_mean)
    print("{}" .format(x - x_mean))


print("\nY mean:")
for y in y_list:
    ym_list.append(y - y_mean)
    print("{}" .format(y - y_mean))

print("\n")
print("Variance x: {}".format(np.var(x_list)))

# Lindsay I Smith's approach
xvari = 0
i = 0
for x in xm_list:
    xvari += x * x
    i += 1


print("Variance x (Linxsay I Smith's approach): {}".format(xvari / (i - 1)))
print("Std Deviation x: {}".format(np.std(x_list)))
print("Variance y: {}".format(np.var(y_list)))
print("Std Deviation y: {}".format(np.std(y_list)))


plt.figure(1)
plt.plot(xm_list, ym_list, '+')
plt.title("Covariance X mean vs Y mean")

# Gaussian distribution
plt.figure(2)
plt.subplot(211)
xvari = np.var(x_list)
sigma = np.std(x_list)
xaxis = np.linspace((x_mean - 3*sigma), (x_mean + 3*sigma), 100)
plt.plot(xaxis, sps.norm.pdf(xaxis, x_mean, sigma))
plt.title("scipy's Gaussian / Normal distribution")
plt.xlabel("x mean: -3 sigma to +3 sigma")
print("x_mean = {}\n" .format(x_mean))

xgauss = []
xgauss_fixed = np.sqrt(1 / (2 * np.pi * xvari))
for x in x_list:
    xgauss.append(xgauss_fixed * np.exp((-1 / (2 * xvari)) * np.square((x - x_mean))))
#plt.figure(3)
plt.subplot(212)
plt.title("Gaussian Distribution Data")
#xgaxis = np.linspace(min(xgauss), max(xgauss), len(xgauss))
#plt.plot(xgaxis, xgauss, "*")
plt.plot(xgauss, "*")
plt.tight_layout()

plt.show()
