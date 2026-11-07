import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps


plt.style.use("dark_background")


np.random.seed(67)


# Uppgift 1
# 1.1
dist1 = np.loadtxt("dist1.txt")
plt.hist(dist1, bins=30)
plt.show()
print("1.1.3. dist1.txt ser ut som en likformigfördelning.")

print("1.1.4. Jag gissar att μ ≈ 0.5, σ ≈ 1/2*sqrt(3) ≈ 0.289.")
x = np.linspace(0, 1, 100)
y = sps.uniform.pdf(x)
plt.plot(x, y, color="red")
plt.hist(dist1, bins=30, density=True)
plt.show()

mean = np.mean(dist1)
std = np.std(dist1)
print(f"1.1.5. Jag beräknar μ ≈ {mean:.3f}, σ ≈ {std:.3f}.")
# Grafen skulle vara densamma som förut (det beror inte på μ och σ).

# 1.2
dist2 = np.loadtxt("dist2.txt")
plt.hist(dist2, bins=30)
plt.show()
print(f"\n1.2.3 dist1.txt ser ut som en normalfördelning.")

print("1.2.4. Jag gissar at μ ≈ 0, σ ≈ 1.")
x = np.linspace(-4, 4, 100)
y = sps.norm.pdf(x, 0, 1)
plt.plot(x, y, color="red")
plt.hist(dist2, bins=30, density=True)
plt.show()

mean = np.mean(dist2)
std = np.std(dist2)
print(f"1.2.5. Jag beräknar μ ≈ {mean:.3f}, σ ≈ {std:.3f}.")
y = sps.norm.pdf(x, mean, std)
plt.plot(x, y, color="red")
plt.hist(dist2, bins=30, density=True)
plt.show()

# 1.3
dist3 = np.loadtxt("dist3.txt")
plt.hist(dist3, bins=np.arange(-0.5, 26.5))
plt.show()
print("\n1.3.3. dist3.txt ser ut som en Poissonfördelning (alla värden är heltal).")

print("1.3.4. Jag gissar att μ ≈ 11, σ ≈ sqrt(11) ≈ 3.32.")
x = np.arange(0, 26)
y = sps.poisson.pmf(x, 11)
plt.plot(x, y, color="red")
plt.hist(dist3, bins=np.arange(-0.5, 26.5), density=True)
plt.show()

mean = np.mean(dist3)
std = np.std(dist3)
print(f"1.3.5. Jag beräknar μ ≈ {mean:.3f}, σ ≈ {std:.3f}.")
y = sps.poisson.pmf(x, mean)
plt.plot(x, y, color="red")
plt.hist(dist3, bins=np.arange(-0.5, 26.5), density=True)
plt.show()


# Uppgift 2
N = 1000
hastighet_vektorer = sps.norm.rvs(size=(1000, N))
farter = np.linalg.norm(hastighet_vektorer, axis=1)
plt.hist(farter, bins=30)
plt.show()
print("\n2.1. Fördelningen ser ut som en Maxwell-Boltzmann fördelning.")

N = 1000
hastighet_vektorer = sps.norm.rvs(size=(1000, N), scale=10)
farter = np.linalg.norm(hastighet_vektorer, axis=1)
plt.hist(farter, bins=30)
plt.show()
print("2.2. Om standardavvikelsen av hastighetskomponenterna blir 10 gånger större, "
      "så blir medelfarten samt standardavvikelsen för farten 10 gånger större.")
