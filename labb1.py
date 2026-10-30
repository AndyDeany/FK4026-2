import matplotlib.pyplot as plt
import numpy as np
import scipy.stats


plt.style.use("dark_background")


array = scipy.stats.norm.rvs(size=10)
print(array)
print("2. The numbers appear to come from a N(0, 1) distribution")


array = scipy.stats.norm.rvs(size=1000)
plt.plot(array)
plt.show()

plt.hist(array)
plt.show()

plt.hist(array, bins=30)
plt.show()

plt.hist(array, bins=100)
plt.show()

plt.hist(array, bins=1000)
plt.show()

print("3. There are more bars in the histogram, which makes the bell shape less 'smooth'.")
print("I think that 30 bins looks pretty good!")

plt.hist(array, bins=30, density=True)
plt.show()

print("With density=True we get the probably density of observations in a given bin,"
      " instead of the quantity of observations in a given bin.")


array_2d = scipy.stats.norm.rvs(size=(3, 1000))
sums = np.sum(array_2d, axis=0)
print(len(sums))

print("4. The `axis` argument specifies which axis to sum over. 1 is rowss, 2 columns, etc.")






