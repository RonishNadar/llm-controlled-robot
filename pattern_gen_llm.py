from image_transform import (
    detect_arena_corners,
    warp_image,
    unwarp_points,
    unwarp_image
)

from llm_process_gemini import (
    configure_gemini,
    generate_points,
    save_points_to_file,
    load_pattern_points,
    visualize_points
)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import json
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

filename = "points.txt"

def read_points(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return [(point['x'], point['y'], point['theta']) for point in data]


def compute_orientations(points):
    """
    Given a list of (x, y) points, return list of (x, y, theta) where
    theta is the orientation angle in degrees of the heading to the next point.
    The last point will copy the angle of the second-last for consistency.
    """
    result = []
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        dx = x2 - x1
        dy = y2 - y1
        # theta = math.degrees(math.atan2(dy, dx))  # Orientation in degrees
        theta = math.atan2(dy, dx)                  # Orientation in radians
        result.append((x1, y1, theta))
    
    # For the last point, reuse the last orientation
    result.append((points[-1][0], points[-1][1], result[-1][2]))
    return result     

def save_pose_points(pose_points, file_path):
    with open(file_path, 'w') as f:
        for x, y, theta in pose_points:
            f.write(f"{x},{y},{theta}\n")  # theta in radians

# def generate_path(filepath, shape, api_key, filename="points.txt"):
#     corners, base_image = detect_arena_corners(filepath)
#     warped, M, side_len, original = warp_image(filepath, corners)

#     # Get real arena size
#     image_h, image_w = warped.shape[0], warped.shape[1]

#     # Build prompt dynamically with real size
#     prompt = (
#         f"Give 20 (x, y, theta) points that trace a {shape} in a "
#         f"{image_h}x{image_w} image. Use radians for theta. "
#         "Give only the list of points in JSON format. "
#         "Don't explain or hallucinate."
#     )

#     model = configure_gemini(api_key)
#     points_json = generate_points(model, prompt)
#     save_points_to_file(points_json, filename)
#     points = read_points(filename)
#     visualize_points(points)

#     # The rest of your drawing, unwarping, and plotting stays the same...


#     xy_points = [(int(x), int(y)) for x, y, theta in points]
#     for i in range(len(xy_points) - 1):
#         cv2.line(warped, xy_points[i], xy_points[i + 1], (0, 255, 0), 3)
#     for pt in xy_points:
#         cv2.circle(warped, pt, 5, (0, 0, 255), -1)

#     M_inv = np.linalg.inv(M)
#     unwarped_pts = unwarp_points(xy_points, M_inv)
#     pose_points = compute_orientations(unwarped_pts)
#     save_pose_points(pose_points, "pattern_target.txt")

#     unwarped_img = unwarp_image(warped, M_inv, (original.shape[1], original.shape[0]))
#     overlay = cv2.addWeighted(original, 0.7, unwarped_img, 0.3, 0)

#     plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
#     for x, y, theta in pose_points:
#         length = 30
#         dx = length * math.cos(theta)
#         dy = length * math.sin(theta)
#         plt.arrow(x, y, dx, dy, head_width=10, head_length=10, fc='blue', ec='blue')
#         plt.plot(x, y, 'ro')
#     plt.axis("off")
#     plt.title("Robot Path with Orientation")
#     plt.show()

def generate_path(filepath, shape, api_key, filename="points.txt"):
    corners, base_image = detect_arena_corners(filepath)
    warped, M, side_len, original = warp_image(filepath, corners)

    image_h, image_w = warped.shape[0], warped.shape[1]

    prompt = (
        f"Give 20 (x, y, theta) points that trace a {shape} in a "
        f"{image_h}x{image_w} image. Use radians for theta. "
        "Give only the list of points in JSON format. "
        "Don't explain or hallucinate."
    )

    model = configure_gemini(api_key)
    points_json = generate_points(model, prompt)
    save_points_to_file(points_json, filename)
    points = read_points(filename)
    pose_fig = visualize_points(points)

    # Draw on warped image
    xy_points = [(int(x), int(y)) for x, y, theta in points]
    for i in range(len(xy_points) - 1):
        cv2.line(warped, xy_points[i], xy_points[i + 1], (0, 255, 0), 3)
    for pt in xy_points:
        cv2.circle(warped, pt, 5, (0, 0, 255), -1)

    # Unwarp and save final path
    M_inv = np.linalg.inv(M)
    unwarped_pts = unwarp_points(xy_points, M_inv)
    pose_points = compute_orientations(unwarped_pts)
    save_pose_points(pose_points, "pattern_target.txt")

    unwarped_img = unwarp_image(warped, M_inv, (original.shape[1], original.shape[0]))
    overlay = cv2.addWeighted(original, 0.7, unwarped_img, 0.3, 0)

    # Create matplotlib Figure (replacing plt.imshow + plt.show)
    fig = Figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    for x, y, theta in pose_points:
        length = 30
        dx = length * math.cos(theta)
        dy = length * math.sin(theta)
        ax.arrow(x, y, dx, dy, head_width=10, head_length=10, fc='blue', ec='blue')
        ax.plot(x, y, 'ro')
    ax.axis("off")
    ax.set_title("Robot Path with Orientation")

    return pose_fig, fig  # Send this back to the GUI