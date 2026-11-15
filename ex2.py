from common import IMAGES, load_gray, show_images_in_subplots
from ex1 import gradient_magnitude
import numpy as np

# (a)
def findedges(image, sigma, theta):
    Imag, _ = gradient_magnitude(image, sigma)
    height = len(Imag)
    width = len(Imag[0])
    edges = np.zeros((height, width)) # Create a black image.

    for row in range(height): # check every image row
        for col in range(width): # check every pixel in the row
            if Imag[row, col] >= theta:
                edges[row, col] = Imag[row, col]

    return edges

# (b)
def nonmax_suppression(magnitude, direction):
    height = len(magnitude)
    width = len(magnitude[0])
    thinned = np.zeros((height, width)) # Create a black image.

    # (row, col) step along each gradient direction
    steps = [
        (0, 1),   # 0 degrees: right
        (1, 1),   # 45 degrees: down-right
        (1, 0),   # 90 degrees: down
        (1, -1),  # 135 degrees: down-left
    ]

    for row in range(1, height - 1): # skip the image border
        for col in range(1, width - 1):
            angle = direction[row, col]
            direction_bin = round(angle / (np.pi / 4)) % 4 # nearest direction
            row_step, col_step = steps[direction_bin]

            current = magnitude[row, col]
            before = magnitude[row - row_step, col - col_step]
            after = magnitude[row + row_step, col + col_step]
            # Use a strict comparison on one side to thin equal peaks.
            if current > before and current >= after:
                thinned[row, col] = current

    return thinned

# (c)
def hysteresis(thinned, t_low, t_high):
    height = len(thinned)
    width = len(thinned[0])
    edges = np.zeros((height, width), dtype=bool)
    pending = [] # strong pixels waiting to be checked

    for row in range(height): # find starting strong edges
        for col in range(width):
            if thinned[row, col] >= t_high:
                edges[row, col] = True
                pending.append((row, col))

    while pending: # continue through connected weak edges
        row, col = pending.pop()
        for y in range(max(0, row - 1), min(height, row + 2)): # 3x3 neighborhood
            for x in range(max(0, col - 1), min(width, col + 2)):
                if edges[y, x]:
                    continue
                value = thinned[y, x]
                if value > 0 and value >= t_low:
                    edges[y, x] = True
                    pending.append((y, x))

    return edges

def show_results():
    image = load_gray(IMAGES / "museum.jpg")
    sigma = 1
    low = 0.04
    high = 0.16

    thresholded = findedges(image, sigma, high)
    Imag, Idir = gradient_magnitude(image, sigma)
    thinned = nonmax_suppression(Imag, Idir)
    height = len(thinned)
    width = len(thinned[0])
    thinned_thresholded = np.zeros((height, width))
    for row in range(height): # threshold the thinned result for display
        for col in range(width):
            if thinned[row, col] >= high:
                thinned_thresholded[row, col] = thinned[row, col]

    edges = hysteresis(thinned, low, high)

    images = {
        "Original": image,
        f"Thresholded (thr = {high})": thresholded,
        f"Nonmax. supp. (thr = {high})": thinned_thresholded,
        f"Hysteresis (high = {high}, low = {low})": edges,
    }
    show_images_in_subplots(images, 2, 2)

if __name__ == "__main__":
    show_results()
