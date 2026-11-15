from common import IMAGES, load_gray, load_rgb_image, show_images, plot_hough_column, plot_hough_lines, show_figure
from a3_utils import draw_line
from ex1 import gradient_magnitude
import cv2
import matplotlib.pyplot as plt
import numpy as np

def edge_map(image, sigma=1):
    gray = np.uint8(np.round(image * 255)) # Canny uses uint8 images
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma) # delete noise
    return cv2.Canny(blurred, 40, 100) # Canny edge map

# (a)
def a(x=50, y=90, bins=360):
    image = np.zeros((100, 100))
    image[y, x] = 1
    height, width = image.shape
    diagonal = np.hypot(height, width) # compute distance
    thetas = np.linspace(-np.pi / 2, np.pi / 2, bins, endpoint=False) # divide theta 0,1,2...359,360
    point_rhos = x * np.cos(thetas) + y * np.sin(thetas) # rho for each theta
    rows = np.rint((point_rhos + diagonal) / (2 * diagonal) * (bins - 1)).astype(int) # rounding
    votes = np.zeros((bins, bins), dtype=int) # Hough accumulator
    votes[rows, np.arange(bins)] = 1

    figure = show_images([image, votes], figsize=(10, 5), interpolation="nearest")
    left = figure.axes[0]
    plt.sca(left)
    for rho, theta in zip(point_rhos, thetas): # draw every line through the point
        draw_line(rho, theta, height, width, clr="green")
    left.plot(x, y, marker="*", color="gold", markersize=9)
    show_figure(figure)

# (b), (f)
def hough_find_lines(image, bins_theta, bins_rho, threshold=0, angles=None, tolerance=5):
    height, width = image.shape
    diagonal = np.hypot(height, width)
    rhos = np.linspace(-diagonal, diagonal, bins_rho) # rho bin values
    thetas = np.linspace(-np.pi / 2, np.pi / 2, bins_theta, endpoint=False) # theta bin values
    votes = np.zeros((bins_rho, bins_theta), dtype=int) # Hough accumulator
    ys, xs = np.nonzero(image > threshold) # voting pixel coordinates
    if angles is not None:
        directions = (angles[ys, xs] + np.pi / 2) % np.pi - np.pi / 2

    for col, theta in enumerate(thetas): # fill one theta column at a time
        if angles is not None:
            difference = (directions - theta + np.pi / 2) % np.pi - np.pi / 2 # angle difference
            nearby = abs(difference) <= tolerance * np.pi / 180 # allowed voters
            rho = xs[nearby] * np.cos(theta) + ys[nearby] * np.sin(theta)
        else:
            rho = xs * np.cos(theta) + ys * np.sin(theta)
        rows = ((rho + diagonal) / (2 * diagonal) * (bins_rho - 1)).astype(int)
        votes[:, col] = np.bincount(rows, minlength=bins_rho) # count votes per rho bin
    return votes, rhos, thetas

def synthetic_images():
    points = np.zeros((100, 100))
    points[10, 10] = points[10, 20] = 1
    images = [("synthetic", points, points)]
    for name in ["oneline.png", "rectangle.png"]: # load provided examples
        image = load_gray(IMAGES / name)
        images.append((name, image, edge_map(image)))
    return images

def b():
    accumulators = []
    for _, _, edges in synthetic_images():
        votes, _, _ = hough_find_lines(edges, 200, 200)
        accumulators.append(votes)
    figure = show_images(accumulators, figsize=(9, 4), cmap="viridis", interpolation="nearest")
    show_figure(figure)

# (c)
def nonmaxima_suppression_box(accumulator, k=9):
    radius = k // 2 # distance from center to box edge
    maximum = cv2.dilate(accumulator.astype(np.float32), np.ones((k, k), np.uint8)) # local maximum
    candidates = (accumulator > 0) & (accumulator == maximum) # peak candidates
    result = np.zeros_like(accumulator)
    height, width = accumulator.shape
    for row, col in np.argwhere(candidates): # keep one peak in each neighborhood
        if candidates[row, col]:
            result[row, col] = accumulator[row, col]
            candidates[max(0, row - radius):min(height, row + radius + 1),
                       max(0, col - radius):min(width, col + radius + 1)] = False
    return result

# (d)
def line_peaks(accumulator, threshold, limit=None, k=9):
    peaks = nonmaxima_suppression_box(accumulator, k)
    rows, cols = np.nonzero(peaks > threshold)
    positions = list(zip(rows, cols))
    positions.sort(key=lambda point: peaks[point], reverse=True) # strongest peaks first
    return positions[:limit]

def d():
    examples = synthetic_images()
    example_images = []
    for _, image, _ in examples:
        example_images.append(image)
    figure = show_images(example_images, figsize=(10, 4))
    for axis, (name, image, edges) in zip(figure.axes, examples): # test each image
        votes, rhos, thetas = hough_find_lines(edges, 360, 601)
        if name == "synthetic":
            threshold = 1
        else:
            threshold = 0.38 * votes.max()
        plt.sca(axis)
        for row, col in line_peaks(votes, threshold): # draw detected lines
            draw_line(rhos[row], thetas[col], image.shape[0], image.shape[1])
    show_figure(figure)


# (e)
def e(sigma=1, bins_theta=400, bins_rho=400):
    figure, axes = plt.subplots(2, 2, figsize=(9, 6))
    for col, name in enumerate(["bricks.jpg", "pier.jpg"]): # one result per column
        gray = load_gray(IMAGES / name)
        image = load_rgb_image(IMAGES / name)
        edges = edge_map(gray, sigma)
        votes, rhos, thetas = hough_find_lines(edges, bins_theta, bins_rho)

        plot_hough_column(axes[:, col], name, image, votes, rhos, thetas,
                          line_peaks(votes, 0, 10), theta_step=200)
    show_figure(figure)

# (f)
def f():
    for name in ["rectangle.png", "pier.jpg"]: # compare on synthetic and real images
        image = load_gray(IMAGES / name)
        edges = edge_map(image)
        _, angles = gradient_magnitude(image, 1)
        figure, axes = plt.subplots(2, 2, figsize=(8, 6))
        voting_modes = [("normal", None), ("orientation", angles)]
        for col, (label, directions) in enumerate(voting_modes): # two voting modes
            votes, rhos, thetas = hough_find_lines(edges, 400, 400, angles=directions)
            if name == "rectangle.png":
                limit = 4
            else:
                limit = 10
            plot_hough_column(axes[:, col], label, image, votes, rhos, thetas, line_peaks(votes, 0, limit))
        show_figure(figure)

# (g)
def g():
    image = load_gray(IMAGES / "rectangle.png")
    edges = edge_map(image)
    votes, rhos, thetas = hough_find_lines(edges, 180, 1201)
    # All pixels voting gives the maximum votes for each line.
    possible, _, _ = hough_find_lines(np.ones_like(edges), len(thetas), len(rhos))
    normalized = np.divide(votes, possible, out=np.zeros_like(votes, dtype=float), where=possible > 0) # vote ratio
    score_sets = [("Raw votes", votes), ("Fraction of possible votes", normalized)]
    for label, scores in score_sets: # compare scores
        peaks = line_peaks(scores, 0, 4)
        figure = show_images([scores, image], label)
        left, right = figure.axes
        left.set_title("Accumulator")
        right.set_title("Lines")
        plot_hough_lines(right, image.shape, rhos, thetas, peaks)
        plt.show()

# (h)
def hough_find_circles(edges, radius):
    # Each edge pixel votes for centers one radius away.
    ring = np.zeros((2 * radius + 1, 2 * radius + 1), np.uint8)
    cv2.circle(ring, (radius, radius), radius, 1, 1)
    return np.round(cv2.filter2D((edges > 0).astype(float), -1, ring, borderType=cv2.BORDER_CONSTANT))

def h():
    image_settings = [("eclipse.jpg", 48, 0.31), ("coins.jpg", 46, 0.60)]
    for name, radius, min_fraction in image_settings: # image settings
        image = load_rgb_image(IMAGES / name)
        edges = edge_map(load_gray(IMAGES / name))
        votes = hough_find_circles(edges, radius)
        diameter = 2 * radius + 1
        centers = line_peaks(votes, min_fraction * votes.max(), k=diameter) # circle centers
        for y, x in centers: # draw detected circles
            cv2.circle(image, (int(x), int(y)), radius, (0, 0.5, 0), 5)
        figure = show_images([votes, image], figsize=(8, 6), cmap="viridis")
        show_figure(figure)
