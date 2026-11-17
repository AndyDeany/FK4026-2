import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps


plt.style.use("dark_background")


np.random.seed(67)


# Uppgift 1
# 1.1
dist1 = np.loadtxt("dist1.txt")
plt.hist(dist1, bins=30, color="#1c549e")
plt.title("Histogram över data från data1.txt")
plt.xlabel("Värde")
plt.ylabel("Antalet observationer")
plt.show()
print("1.1.3. dist1.txt ser ut som en likformigfördelning U(a, b).")

a = 0
b = 1
print(f"1.1.4. Jag gissar att a ≈ {a},b ≈ {b}.")
x = np.linspace(0, 1, 100)
y = sps.uniform.pdf(x, a, b)
plt.plot(x, y, color="#f700ff", label=f"U[{a}, {b}]")
plt.hist(dist1, bins=30, color="#1c549e", density=True)
plt.title("Histogram över data från data1.txt")
plt.xlabel("Värde")
plt.ylabel("Sannolikhetstäthet")
plt.legend()
plt.show()

a = float(np.min(dist1))
b = float(np.max(dist1))
print(f"1.1.5. Jag beräknar a ≈ {a:.3f}, b ≈ {b:.3f}.")
x = np.linspace(a, b, 100)
y = sps.uniform.pdf(x, a, b)
plt.plot(x, y, color="#f700ff", label=f"U[{a:.3f}, {b:.3f}]")
plt.hist(dist1, bins=30, color="#1c549e", density=True)
plt.title("Histogram över data från data1.txt")
plt.xlabel("Värde")
plt.ylabel("Sannolikhetstäthet")
plt.legend()
plt.show()

# 1.2
dist2 = np.loadtxt("dist2.txt")
plt.hist(dist2, bins=30, color="#1c549e")
plt.title("Histogram över data från data2.txt")
plt.xlabel("Värde")
plt.ylabel("Antalet observationer")
plt.show()
print(f"\n1.2.3. dist2.txt ser ut som en normalfördelning N(μ, σ²).")

mu = 0
sigma = 1
print(f"1.2.4. Jag gissar at μ ≈ {mu}, σ ≈ {sigma}.")
x = np.linspace(-4, 4, 100)
y = sps.norm.pdf(x, mu, sigma)
plt.plot(x, y, color="#f700ff", label=f"N({mu}, {sigma}²)")
plt.hist(dist2, bins=30, color="#1c549e", density=True)
plt.title("Histogram över data från data2.txt")
plt.xlabel("Värde")
plt.ylabel("Sannolikhetstäthet")
plt.legend()
plt.show()

mu = np.mean(dist2)
sigma = np.std(dist2)
print(f"1.2.5. Jag beräknar μ ≈ {mu:.3f}, σ ≈ {sigma:.3f}.")
y = sps.norm.pdf(x, mu, sigma)
plt.plot(x, y, color="#f700ff", label=f"N({mu:.3f}, {sigma:.3f}²)")
plt.hist(dist2, bins=30, color="#1c549e", density=True)
plt.title("Histogram över data från data2.txt")
plt.xlabel("Värde")
plt.ylabel("Sannolikhetstäthet")
plt.legend()
plt.show()

# 1.3
dist3 = np.loadtxt("dist3.txt")
plt.hist(dist3, bins=np.arange(-0.5, 26.5), color="#1c549e")
plt.title("Histogram över data från data3.txt")
plt.xlabel("Värde")
plt.ylabel("Antalet observationer")
plt.show()
print("\n1.3.3. dist3.txt ser ut som en Poissonfördelning Po(λ) (alla värden är heltal). "
      "Det kan ju också vara en Binomial fördelning, men en Binomial fördelning "
      "approximeras väldigt bra av en Poisson fördelning då n är stort och p är litet, "
      "som är fallet här.")

l = 11      # λ
print(f"1.3.4. Jag gissar att λ ≈ {l} (λ ≈ μ).")
x = np.arange(0, 26)
y = sps.poisson.pmf(x, l)
plt.plot(x, y, color="#f700ff", label=f"Po({l})")
plt.hist(dist3, bins=np.arange(-0.5, 26.5), color="#1c549e", density=True)
plt.title("Histogram över data från data3.txt")
plt.xlabel("Värde")
plt.ylabel("Sannolikhetstäthet")
plt.legend()
plt.show()

l = np.mean(dist3)
print(f"1.3.5. Jag beräknar λ ≈ {mu:.3f}.")
y = sps.poisson.pmf(x, mu)
plt.plot(x, y, color="#f700ff", label=f"Po({l:.3f})")
plt.hist(dist3, bins=np.arange(-0.5, 26.5), color="#1c549e", density=True)
plt.title("Histogram över data från data3.txt")
plt.xlabel("Värde")
plt.ylabel("Sannolikhetstäthet")
plt.legend()
plt.show()


# Uppgift 2
N = 1000
hastighet_vektorer = sps.norm.rvs(size=(1000, N))
farter = np.linalg.norm(hastighet_vektorer, axis=1)
plt.hist(farter, bins=30, color="#1c549e")
plt.xlabel("Fart")
plt.ylabel("Antalet observationer")
plt.title("Histogram över fartfördelning för partiklar med\nN(0, 1²)-fördelade hastighetskomponenter.")
plt.show()
print("\n2.1. Fördelningen ser ut som en Maxwell-Boltzmann fördelning.")

N = 1000
hastighet_vektorer = sps.norm.rvs(size=(1000, N), scale=10)
farter = np.linalg.norm(hastighet_vektorer, axis=1)
plt.hist(farter, bins=30, color="#1c549e")
plt.xlabel("Fart")
plt.ylabel("Antalet observationer")
plt.title("Histogram över fartfördelning för partiklar med\nN(0, 10²)-fördelade hastighetskomponenter.")
plt.show()
print("2.2. Om standardavvikelsen av hastighetskomponenterna blir 10 gånger större, "
      "så blir medelfarten samt standardavvikelsen för farten 10 gånger större.")
