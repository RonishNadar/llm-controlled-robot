import google.generativeai as genai
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import json
from dotenv import load_dotenv
import os

def configure_gemini(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("models/gemini-2.5-pro")

def generate_points(model, prompt):
    response = model.generate_content(prompt)
    raw_text = response.text.strip()

    if raw_text.startswith("```json"):
        lines = raw_text.splitlines()
        clean_text = "\n".join(lines[1:-1])  # Remove ```json and ending ```
    else:
        clean_text = raw_text
    return clean_text

def save_points_to_file(points_json, filename="points.txt"):
    with open(filename, "w") as f:
        f.write(points_json)

def load_pattern_points(filename="points.txt"):
    with open(filename, 'r') as f:
        data = json.load(f)
    return [(point['x'], point['y'], point['theta']) for point in data]

def visualize_points(points, title="Generated Path with Orientations"):
    x_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]
    theta_vals = [p[2] for p in points]

    plt.figure(figsize=(8, 8))
    plt.plot(x_vals, y_vals, 'mo-', label='Generated Path')
    for x, y, theta in zip(x_vals, y_vals, theta_vals):
        dx = 20 * np.cos(theta)
        dy = 20 * np.sin(theta)
        plt.arrow(x, y, dx, dy, head_width=10, head_length=10, fc='blue', ec='blue')

    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.show()

def visualize_points(points, title="Generated Path with Orientations"):
    x_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]
    theta_vals = [p[2] for p in points]

    fig = Figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.plot(x_vals, y_vals, 'mo-', label='Generated Path')

    for x, y, theta in zip(x_vals, y_vals, theta_vals):
        dx = 20 * np.cos(theta)
        dy = 20 * np.sin(theta)
        ax.arrow(x, y, dx, dy, head_width=10, head_length=10, fc='blue', ec='blue')

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    ax.invert_yaxis()
    ax.axis("equal")
    ax.legend()
    fig.tight_layout()

    return fig


if __name__ == "__main__":
    # Minimal example
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    prompt = "Give 30 (x, y, theta) points that trace a rectangle of length 200px and width 100px in a 673x673 image. Use radians for theta. Give only the list of points in JSON format. Don't explain and don't hallucinate."
    filename = "points.txt"

    model = configure_gemini(api_key)
    points_json = generate_points(model, prompt)
    save_points_to_file(points_json, filename)
    points = load_pattern_points(filename)
    visualize_points(points)