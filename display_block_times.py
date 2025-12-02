
import numpy as np
import matplotlib.pyplot as plt
import csv


def main():
    filename: str = ""

       

    with open(filename, 'r', newline='') as file:
        header_lines = 4
        for _ in range(header_lines):
            next(file)

        reader = csv.reader(file)
        for row in reader:
            blocks = row[0]
            threads = row[1]
            total_threads = row[2]
            avg_time_ms = row[3]
            min_time_ms = row[4]
            max_time_ms = row[5]
            
    

    # Example: your array of times in milliseconds
    times = np.array([12, 18, 25, 40, 42, 45, 58, 60, 61, 75])

    # Bin width (in ms)
    bin_width = 10  # X ms

    # Create the bin edges
    bins = np.arange(times.min(), times.max() + bin_width, bin_width)

    plt.hist(times, bins=bins, edgecolor='black')

    plt.xlabel("Time (ms)")
    plt.ylabel("Count")
    plt.title(f"Block times")
    plt.show()


if __name__ == "__main__":
    main()