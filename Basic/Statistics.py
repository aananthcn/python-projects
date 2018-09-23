import numpy as np
import matplotlib.pyplot as plt

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
xvar = 0
i = 0
for x in xm_list:
    xvar += x * x
    i += 1


print("Variance x (Linxsay I Smith's approach): {}".format(xvar/(i-1)))
print("Std Deviation x: {}".format(np.std(x_list)))
print("Variance y: {}".format(np.var(y_list)))
print("Std Deviation y: {}".format(np.std(y_list)))


#plt.figure(1)
#plt.plot(xm_list, ym_list, '+')
#plt.show()
