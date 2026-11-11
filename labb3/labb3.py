import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps


plt.style.use("dark_background")


np.random.seed(67)


print("2.1.1. Det finns många olika oberoende faktorer som bidrar till mätfel, och då "
      "de summeras får vi en fördelning som är ungefär normal, enligt centrala gränsvärdesatsen.")


def plot(N, n, distribution):
      """Plot a histogram of N sums of n independent observations from the given distribution,
      along with the probability density function of the approximating normal distrubtion.
      """
      stickprov = distribution.rvs(size=(N, n))
      summor = np.sum(stickprov, axis=1)

      plt.hist(summor, bins=50, density=True)
      x = np.linspace(np.min(summor), np.max(summor), 1000)
      y = sps.norm.pdf(x, np.mean(summor), np.std(summor))
      plt.plot(x, y, color="red")
      plt.show()


plot(10**5, 5, sps.uniform)
print("2.1.5. Det ser ganska normalfördelade ut vid n>=5, tycker jag.")


print("2.1.6.")
plot(10**5, 1, sps.norm)
print("  4* Norm funkar bra även med n=1 (såklart).")
plot(10**5, 5, sps.laplace)
print("  * Laplace funkar bra med n>=5.")
plot(10**5, 100, sps.expon)
print("  * Exponential funkar, men den behöver större n, typ n>=100, eftersom den är skev.")
plot(10**5, 1000, sps.cauchy)
print("  * Cauchy funkar inte, även för stora n.")
