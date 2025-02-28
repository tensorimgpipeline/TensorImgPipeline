import colorsys

import numpy as np
from matplotlib.colors import hex2color, rgb2hex


# Function to convert hex to normalized RGB (0 to 1)
def hex_to_rgb(hexcolor):
    hexcolor = hexcolor.lstrip("#")
    return np.array([int(hexcolor[i : i + 2], 16) for i in (0, 2, 4)]) / 255.0


# Function to convert normalized RGB (0 to 1) to hex
def rgb_to_hex(rgb):
    # Convert back to 0-255 range for hex conversion
    rgb = (rgb * 255).astype(int)
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def select_color(rgb_colors):
    vec_rgb_to_hsv = np.vectorize(colorsys.rgb_to_hsv)
    r, g, b = rgb_colors.T
    h, s, v = vec_rgb_to_hsv(r, g, b)

    # To create a new color we place them in the middle between two colors with the biggest gap
    # based on hue.
    hsv = np.stack([h, s, v], axis=-1)
    hsv.sort(axis=0)
    distance_between = np.diff(hsv, axis=0)
    selected_distance = np.argmax(distance_between[:, 0])
    selected_color = hsv[selected_distance]
    selected_color[0] = selected_color[0] + (distance_between[selected_distance] / 2)[0]

    return rgb2hex(colorsys.hsv_to_rgb(*selected_color))


colors = ["#FFFF55", "#FF55FF", "#55FFFF", "#FF5555", "#55FF55", "#5555FF"]
"#aa55ff"
rgb_colors = [hex2color(color) for color in colors]

# Convert to numpy array
rgb_colors = np.array(rgb_colors)

vec_rgb_to_hsv = np.vectorize(colorsys.rgb_to_hsv)

r, g, b = rgb_colors.T

best = select_color(rgb_colors)

print(rgb_colors, best)
