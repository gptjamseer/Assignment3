from common import IMAGES, load_gray, load_rgb_image, show_images
from common import plot_hough_column, plot_hough_lines, show_figure
from a3_utils import draw_line
from ex1 import gradient_magnitude
import cv2
import matplotlib.pyplot as plt
import numpy as np

def edge_map(image, sigma=1):
    gray = np.uint8(np.round(image * 255))
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    return cv2.Canny(blurred, 40, 100)

# (a)
def a(x=50, y=90):
    image = np.zeros((100, 100))
    image[y, x] = 1
    height, width = image.shape
    diagonal = np.sqrt(height**2 + width**2)
    votes = np.zeros((300, 361), dtype=int)  # 오른쪽 그림의 투표 배열. 행은 거리 구간, 열은 각도(0~360도).
    lines = []  # 왼쪽 그림에 그릴 직선들을 (rho, theta) 쌍으로 저장한다.

    for degree in range(361):
        theta = degree * np.pi / 180  # 직선에 수직인 방향의 각도. cos와 sin에 넣기 위해 라디안으로 바꾼다.
        rho = x * np.cos(theta) + y * np.sin(theta)  # 원점에서 (x, y)를 지나는 이 직선까지의 부호 있는 수직 거리.
        row = int((rho + diagonal) / 2 * diagonal / 300)  # rho가 300개 거리 구간 중 몇 번째에 들어가는지 나타내는 행 번호.
        votes[row, degree] += 1
        lines.append((rho, theta))

    figure = show_images([image, votes], figsize=(10, 5), interpolation="nearest")
    left = figure.axes[0]
    plt.sca(left)
    for rho, theta in lines:
        draw_line(rho, theta, height, width, clr="green")
    left.plot(x, y, marker="*", color="gold", markersize=9)
    show_figure(figure)

# (b), (f)
def hough_find_lines(image, bins_theta, bins_rho, threshold=0, angles=None, tolerance=5):
    height, width = image.shape
    diagonal = np.sqrt(height**2 + width**2)
    rhos = np.array([-diagonal + row * 2 * diagonal / max(1, bins_rho - 1) for row in range(bins_rho)])
    thetas = np.array([-np.pi / 2 + col * (np.pi / bins_theta) for col in range(bins_theta)])
    votes = np.zeros((bins_rho, bins_theta), dtype=int)
    ys, xs = np.nonzero(image > threshold)
    if angles is not None:
        directions = (angles[ys, xs] + np.pi / 2) % np.pi - np.pi / 2

    for col, theta in enumerate(thetas):
        x, y = xs, ys
        if angles is not None:
            # 법선은 반 바퀴마다 같은 방향이므로 각도 차이를 반 바퀴 안으로 맞춘다.
            difference = (directions - theta + np.pi / 2) % np.pi - np.pi / 2
            nearby = abs(difference) <= tolerance * np.pi / 180
            x, y = xs[nearby], ys[nearby]
        rho = x * np.cos(theta) + y * np.sin(theta)
        rows = ((rho + diagonal) / (2 * diagonal) * (bins_rho - 1)).astype(int)
        counts = [0] * bins_rho
        for row in rows.tolist():
            counts[row] += 1
        votes[:, col] = counts
    return votes, rhos, thetas

def synthetic_images():
    points = np.zeros((100, 100))
    points[10, 10] = points[10, 20] = 1
    yield "synthetic", points, points
    for name in ["oneline.png", "rectangle.png"]:
        image = load_gray(IMAGES / name)
        yield name, image, edge_map(image)

def b():
    images = {}
    for name, _, edges in synthetic_images():
        votes, _, _ = hough_find_lines(edges, 200, 200)
        images[name] = votes

    figure = show_images(list(images.values()), figsize=(9, 4), cmap="viridis", interpolation="nearest")
    for axis, name in zip(figure.axes, images):
        axis.set_title(name)
        axis.set_xticks(range(0, 200, 50))
        axis.set_yticks(range(0, 200, 50))
    show_figure(figure, "Figure 6: Accumulator values for various synthetic images.", margin=0.12)

# (c)
def nonmaxima_suppression_box(accumulator, k=9):
    if k < 1 or k % 2 == 0:
        raise ValueError("k must be a positive odd number")
    radius = k // 2
    # theta가 pi만큼 바뀌면 rho의 부호가 바뀐다. 경계 너머의 이웃도 비교한다.
    cols = accumulator.shape[1]
    padded = np.zeros((len(accumulator), cols + 2 * radius))
    for col in range(-radius, cols + radius):
        values = accumulator[:, col % cols]
        if (col // cols) % 2 == 1:
            values = values[::-1]
        padded[:, col + radius] = values
    maximum = cv2.dilate(padded, np.ones((k, k), np.uint8))
    candidates = (accumulator > 0) & (accumulator == maximum[:, radius:radius + cols])
    result = np.zeros_like(accumulator)
    rows, columns = np.nonzero(candidates)
    for row, col in zip(rows, columns):
        if candidates[row, col]:
            result[row, col] = accumulator[row, col]
            for neighbor in range(col - radius, col + radius + 1):
                center = row if (neighbor // cols) % 2 == 0 else len(result) - 1 - row
                candidates[max(0, center - radius):center + radius + 1, neighbor % cols] = False
    return result

# (d)
def line_peaks(accumulator, threshold, limit=None):
    peaks = nonmaxima_suppression_box(accumulator)
    rows, cols = np.nonzero(peaks > threshold)
    positions = list(zip(rows, cols))
    positions.sort(key=lambda point: peaks[point], reverse=True)
    return positions[:limit]

def d():
    examples = list(synthetic_images())
    figure = show_images([image for _, image, _ in examples], figsize=(10, 4))
    for axis, (name, image, edges) in zip(figure.axes, examples):
        votes, rhos, thetas = hough_find_lines(edges, 360, 601)
        threshold = 1 if name == "synthetic" else 0.38 * votes.max()
        axis.set_title(name)
        plt.sca(axis)
        rows, cols = np.nonzero(votes > threshold)
        for row, col in zip(rows, cols):
            draw_line(rhos[row], thetas[col], image.shape[0], image.shape[1])
        if name == "synthetic":
            axis.set_xticks([0, 25, 50, 75])
        else:
            axis.set_xticks([0, 200, 400])
            axis.set_yticks([0, 100, 200])
    show_figure(figure, "Figure 7: Lines resulting from Hough transform for various synthetic images.", margin=0.13)


# (e)
def e(sigma=1, bins_theta=400, bins_rho=400):
    figure, axes = plt.subplots(2, 2, figsize=(9, 6))
    for col, name in enumerate(["bricks.jpg", "pier.jpg"]):
        gray = load_gray(IMAGES / name)
        image = load_rgb_image(IMAGES / name)
        edges = edge_map(gray, sigma)
        votes, rhos, thetas = hough_find_lines(edges, bins_theta, bins_rho)

        plot_hough_column(axes[:, col], name, image, votes, rhos, thetas, line_peaks(votes, 0, 10), theta_step=200)
        axes[1, col].set_xticks(range(0, gray.shape[1], 200))
        axes[1, col].set_yticks(range(0, gray.shape[0], 100))
    show_figure(figure)

# (f)
def f():
    for name in ["rectangle.png", "pier.jpg"]:
        image = load_gray(IMAGES / name)
        edges = edge_map(image)
        _, angles = gradient_magnitude(image, 1)
        figure, axes = plt.subplots(2, 2, figsize=(8, 6))
        for col, (label, directions) in enumerate([("normal", None), ("orientation", angles)]):
            votes, rhos, thetas = hough_find_lines(edges, 400, 400, angles=directions)
            limit = 4 if name == "rectangle.png" else 10
            plot_hough_column(axes[:, col], label, image, votes, rhos, thetas, line_peaks(votes, 0, limit))
        caption = ("Figure 9: Resulting accumulators and lines for the baseline and gradient-filtered\n"
                   "versions of the Hough transform.")
        show_figure(figure, caption, margin=0.09, caption_y=0.02)

# (g)
def g():
    image = load_gray(IMAGES / "rectangle.png")
    edges = edge_map(image)
    votes, rhos, thetas = hough_find_lines(edges, 180, 1201)
    # 모든 픽셀이 투표하면 각 (rho, theta) 칸이 받을 수 있는 최대 표 수가 된다.
    possible, _, _ = hough_find_lines(np.ones_like(edges), len(thetas), len(rhos))
    normalized = np.zeros(votes.shape)
    valid = possible > 0
    normalized[valid] = votes[valid] / possible[valid]
    for label, scores in [("Raw votes", votes), ("Fraction of possible votes", normalized)]:
        peaks = line_peaks(scores, 0, 4)
        print(label)
        for row, col in peaks:
            print(f"  rho={rhos[row]:.1f}, theta={thetas[col] * 180 / np.pi:.0f} deg, score={scores[row, col]:.3f}")
        figure = show_images([scores, image], label)
        left, right = figure.axes
        left.set_title("Accumulator")
        right.set_title("Lines")
        plot_hough_lines(right, image.shape, rhos, thetas, peaks)
        plt.show()

# (h)
def hough_find_circles(edges, radius):
    # 에지에서 radius만큼 떨어진 모든 중심 (x, y)에 한 표씩 준다.
    ring = np.zeros((2 * radius + 1, 2 * radius + 1), np.uint8)
    cv2.circle(ring, (radius, radius), radius, 1, 1)
    return np.round(cv2.filter2D((edges > 0).astype(float), -1, ring, borderType=cv2.BORDER_CONSTANT))

def h():
    for name, radius, min_fraction in [("eclipse.jpg", 48, 0.31), ("coins.jpg", 46, 0.60)]:
        image = load_rgb_image(IMAGES / name)
        edges = edge_map(load_gray(IMAGES / name))
        votes = hough_find_circles(edges, radius)
        # 인접한 중심 후보의 표를 모아 반지름과 픽셀 좌표의 반올림 오차를 완화한다.
        scores = cv2.GaussianBlur(votes, (0, 0), 2)
        diameter = 2 * radius + 1
        maxima = cv2.dilate(scores, np.ones((diameter, diameter), np.uint8))
        centers = (scores == maxima) & (scores > min_fraction * scores.max())
        rows, cols = np.nonzero(centers)
        for y, x in zip(rows, cols):
            cv2.circle(image, (int(x), int(y)), radius, (0, 0.5, 0), 5)
        figure = show_images([votes.T, image], figsize=(8, 6), cmap="viridis")
        right = figure.axes[1]
        right.set_xticks(range(0, image.shape[1], 200))
        show_figure(figure, f"Figure 10: Accumulator and resulting circles for image {name}.",
                    margin=0.10, caption_y=0.04)
