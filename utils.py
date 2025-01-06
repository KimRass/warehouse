from collections import defaultdict


def print_defaultdict(d, num_samples, sample_lim, indent=0):
    for key, value in d.items():
        print(" " * indent + f"• {key}: ", end="")
        if isinstance(value, defaultdict):
            print()
            print_defaultdict(
                value, num_samples=num_samples, sample_lim=sample_lim, indent=indent + 4,
            )
        elif isinstance(value, int):
            print(f"{value * num_samples // sample_lim:,}.")
        else:
            print(value)
