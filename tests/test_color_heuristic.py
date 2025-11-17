from pathlib import Path
from PIL import Image, ImageDraw
from src.main import check_for_aurora_colors


def create_image(colors, size=(200, 120)):
    img = Image.new('RGB', size, (0, 0, 0))
    draw = ImageDraw.Draw(img)
    w, h = size
    step = w // len(colors)
    for i, c in enumerate(colors):
        draw.rectangle([i*step, 0, (i+1)*step - 1, h-1], fill=c)
    return img


def test_color_heuristic_positive_green():
    img = create_image([(10,120,20), (0,0,0)])  # strong green area
    path = Path('temp_green.jpg')
    img.save(path)
    try:
        assert check_for_aurora_colors(path, min_green_pixels=5) is True
    finally:
        path.unlink(missing_ok=True)


def test_color_heuristic_positive_purple():
    img = create_image([(130,30,140), (0,0,0)])  # purple-ish
    path = Path('temp_purple.jpg')
    img.save(path)
    try:
        assert check_for_aurora_colors(path, min_green_pixels=5) is True  # purple counts
    finally:
        path.unlink(missing_ok=True)


def test_color_heuristic_negative_dark():
    img = create_image([(0,0,0),(5,5,5),(10,10,10)])
    path = Path('temp_dark.jpg')
    img.save(path)
    try:
        assert check_for_aurora_colors(path, min_green_pixels=10) is False
    finally:
        path.unlink(missing_ok=True)
