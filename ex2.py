from common import IMAGES, load_gray, show_images_in_subplots
from ex1 import gradient_magnitude
import numpy as np
import cv2

# (a)
def findedges(image, sigma, theta):
    """Keep gradient magnitudes at or above theta in a grayscale image."""
    magnitude, _ = gradient_magnitude(np.asarray(image, dtype=float), sigma)
    return np.where(magnitude >= theta, magnitude, 0)

def a():
    image = load_gray(IMAGES / "museum.jpg")
    images = {"Original": image}
    for theta in (0.04, 0.08, 0.16):
        images[f"Threshold = {theta}"] = findedges(image, 1, theta)
    show_images_in_subplots(images, 2, 2)

# (b)
def nonmax_suppression(magnitude, direction):
    """Keep local peaks along the nearest of four gradient orientations."""
    center = magnitude[1:-1, 1:-1]
    bins = np.floor(direction[1:-1, 1:-1] / (np.pi / 4) + 0.5) % 4
    # Positive y points down the image, so 45 degrees runs NW to SE.
    neighbors = (
        (magnitude[1:-1, :-2], magnitude[1:-1, 2:]),
        (magnitude[:-2, :-2], magnitude[2:, 2:]),
        (magnitude[:-2, 1:-1], magnitude[2:, 1:-1]),
        (magnitude[:-2, 2:], magnitude[2:, :-2]),
    )
    keep = np.zeros(center.shape, dtype=bool)
    for orientation, (before, after) in enumerate(neighbors):
        # Break ties on one side to thin equal-height ridges.
        keep |= (bins == orientation) & (center > before) & (center >= after)
    result = np.zeros_like(magnitude)
    result[1:-1, 1:-1] = np.where(keep, center, 0)
    return result

def b():
    image = load_gray(IMAGES / "museum.jpg")
    magnitude, direction = gradient_magnitude(image, 1)
    thinned = nonmax_suppression(magnitude, direction)
    theta = 0.16
    show_images_in_subplots({
        "Original": image,
        f"Threshold = {theta}": np.where(magnitude >= theta, magnitude, 0),
        f"Nonmax, threshold = {theta}": np.where(thinned >= theta, thinned, 0),
    }, 1, 3)

# (c)
def hysteresis(thinned, t_low, t_high):
    """Return the 8-connected edge components containing a strong pixel."""
    if not 0 <= t_low < t_high:
        raise ValueError("Expected 0 <= t_low < t_high")
    candidates = (thinned > 0) & (thinned >= t_low)
    count, labels = cv2.connectedComponents(candidates.astype(np.uint8), connectivity=8)
    keep = np.zeros(count, dtype=bool)
    keep[labels[thinned >= t_high]] = True
    return keep[labels]

def c():
    image = load_gray(IMAGES / "museum.jpg")
    magnitude, direction = gradient_magnitude(image, 1)
    thinned = nonmax_suppression(magnitude, direction)
    images = {"Original": image}
    for low, high in ((0.04, 0.16), (0.08, 0.16), (0.04, 0.24)):
        images[f"Hysteresis, low={low}, high={high}"] = hysteresis(thinned, low, high)
    show_images_in_subplots(images, 2, 2)


if __name__ == "__main__":
    c()

