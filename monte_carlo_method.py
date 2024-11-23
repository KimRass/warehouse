import random
import math


def estimate_pi(num_points):
    inside_circle = 0

    for _ in range(num_points):
        if random.uniform(-1, 1)**2 + random.uniform(-1, 1)**2 < 1:
            inside_circle += 1

    # π 근사: (원의 면적 / 정사각형 면적) = π / 4
    estim_pi = 4 * (inside_circle / num_points)
    return estim_pi


if __name__ == "__main__":
    for num_points in [10_000, 100_000, 1_000_000, 10_000_000]:
        print(abs(math.pi - estimate_pi(num_points)))
