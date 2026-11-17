import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps


plt.style.use("dark_background")


np.random.seed(67)


print("2.1.1. Det finns många olika oberoende faktorer som bidrar till mätfel, och då "
      "de summeras får vi en fördelning som är ungefär normal, enligt centrala gränsvärdesatsen.")


N = 10**5
ns = [1, 3, 5, 20, 100, 500]

def plot(distribution):
      """Plot a histogram of N sums of n independent observations from the given distribution,
      along with the probability density function of the approximating normal distrubtion.
      """
      axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")[1].flatten()

      for i, n in enumerate(ns):
            stickprov = distribution.rvs(size=(N, n))
            summor = np.sum(stickprov, axis=1)

            axes[i].hist(summor, bins=50, density=True, color="#1c549e")
            x = np.linspace(np.min(summor), np.max(summor), 1000)
            mu = np.mean(summor)
            sigma = np.std(summor)
            y = sps.norm.pdf(x, mu, sigma)
            axes[i].plot(x, y, color="#f700ff", label=f"N({mu:.2f}, {sigma:.2f}²)")
            axes[i].legend()
            axes[i].set_title(f"n={n}")

      plt.suptitle(f"{distribution.name.capitalize()} fördelning: summor av n observationer.")
      plt.show()


plot(sps.uniform)
print("2.1.5. Det ser ganska normalfördelade ut vid n>=5, tycker jag.")


print("2.1.6.")

plot(sps.norm)
print("  * Norm funkar bra även med n=1 (såklart).")

plot(sps.laplace)
print("  * Laplace behöver lite större n för att gör spetsen i mitten mindre. "
      "Det funkar bra med n>=20.")

plot(sps.expon)
print("  * Exponential funkar, men den behöver större n, typ n>=100, eftersom den är skev.")

plot(sps.cauchy)
print("  * Cauchy funkar inte, även för stora n. Den har alltid tjocka svansar som inte "
      "kan representeras av normalapproximationen.")
