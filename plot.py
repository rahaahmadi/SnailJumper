import matplotlib.pyplot as plt

with open('learning_curve.txt', 'r') as file:
    lines = file.readlines()
bests = []
means = []
worsts = []
for line in lines:
    best, worst, mean = line.split()
    bests.append(float(best))
    worsts.append(float(worst))
    means.append(float(mean))

plt.plot(bests, label='bests')
plt.plot(means, label='means')
plt.plot(worsts, label='worsts')

plt.xlabel('Number of Generation')
plt.ylabel('Fitness Value')
plt.legend()
plt.show()