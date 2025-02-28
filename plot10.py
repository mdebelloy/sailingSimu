import matplotlib.pyplot as plt
import numpy as np

# Data
iterations = list(range(10))
baseline_no_penalties = [490, 440, 680, 550, 510, 520, 570, 600, 650, 700]
baseline_with_penalties = [500, 450, 700, 565, 520, 530, 585, 620, 670, 720]
best_training_no_penalties = [450, 375, 430, 410, 430, 370, 380, 390, 400, 420]
best_training_with_penalties = [480, 415, 470, 480, 460, 400, 410, 440, 460, 480]

delta_no_penalties = [baseline - best for baseline, best in zip(baseline_no_penalties, best_training_no_penalties)]
delta_with_penalties = [baseline - best for baseline, best in zip(baseline_with_penalties, best_training_with_penalties)]

percentage_above_no_penalties = [(delta / baseline) * 100 for delta, baseline in zip(delta_no_penalties, baseline_no_penalties)]
percentage_above_with_penalties = [(delta / baseline) * 100 for delta, baseline in zip(delta_with_penalties, baseline_with_penalties)]

# Plot 1: Baseline vs Best Training Times
plt.figure(figsize=(12, 6))
plt.plot(iterations, baseline_no_penalties, label="Baseline (No Penalties)", marker="o")
plt.plot(iterations, best_training_no_penalties, label="Best Training (No Penalties)", marker="o")
plt.plot(iterations, baseline_with_penalties, label="Baseline (With Penalties)", marker="o")
plt.plot(iterations, best_training_with_penalties, label="Best Training (With Penalties)", marker="o")
plt.xlabel("Iteration")
plt.ylabel("Time (seconds)")
plt.title("Baseline vs Best Training Times")
plt.legend()
plt.grid(True)
plt.show()

# Plot 2: Deltas
x = np.arange(len(iterations))
width = 0.35  # Bar width

plt.figure(figsize=(12, 6))
plt.bar(x - width / 2, delta_no_penalties, width, label="Delta (No Penalties)")
plt.bar(x + width / 2, delta_with_penalties, width, label="Delta (With Penalties)")
plt.xlabel("Iteration")
plt.ylabel("Delta (seconds)")
plt.title("Deltas Between Baseline and Best Training Times")
plt.xticks(x, iterations)
plt.legend()
plt.grid(axis="y")
plt.show()

# Plot 3: Percentage Improvement
plt.figure(figsize=(12, 6))
plt.plot(iterations, percentage_above_no_penalties, label="Percentage Improvement (No Penalties)", marker="o")
plt.plot(iterations, percentage_above_with_penalties, label="Percentage Improvement (With Penalties)", marker="o")
plt.xlabel("Iteration")
plt.ylabel("Percentage Improvement (%)")
plt.title("Percentage Improvement Over Baseline")
plt.legend()
plt.grid(True)
plt.show()
