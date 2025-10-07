"""
Компактная учебная реализация задач по компьютерной графике (Python, NumPy, matplotlib).

Содержимое:
1а) Рекурсивная заливка «по строкам» (scanline flood fill) сплошным цветом
1б) Та же заливка, но шаблоном (паттерном) из файла; режимы tile и clip (без масштабирования)
1в) Обход границы связной области (Moore-neighbor tracing) и прорисовка поверх исходника
2)  Рисование отрезка: Брезенхем (целочисленный) и Ву (антиалиасинг)
3)  Градиентная заливка треугольника (барицентрическая интерполяция)

Зависимости: numpy, matplotlib (для загрузки/сохранения/демо).
"""

from dataclasses import dataclass
from math import floor
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys

# -------------------------
# Общие утилиты «пикселей»
# -------------------------
def make_canvas(w, h, color=(255, 255, 255)):
    """Создать RGB-холст w×h, залитый цветом color (uint8)."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :] = color
    return arr

def in_bounds(img, x, y):
    """Проверка выхода за границы массива."""
    h, w = img.shape[:2]
    return 0 <= x < w and 0 <= y < h

def get_pixel(img, x, y):
    """Получить копию пикселя (RGB)."""
    return img[y, x].copy()

def set_pixel(img, x, y, color):
    """Присвоить пикселю цвет (если точка внутри)."""
    if in_bounds(img, x, y):
        img[y, x] = np.array(color, dtype=np.uint8)

def blend_pixel(img, x, y, color, alpha):
    """Альфа-наложение цвета на пиксель (для антиалиасинга)."""
    if not in_bounds(img, x, y):
        return
    bg = img[y, x].astype(np.float32)
    fg = np.array(color, dtype=np.float32)
    out = (1 - alpha) * bg + alpha * fg
    img[y, x] = np.clip(out, 0, 255).astype(np.uint8)

def draw_rect_outline(img, x0, y0, x1, y1, color=(0,0,0)):
    """Нарисовать прямоугольник по контуру."""
    for x in range(x0, x1+1):
        set_pixel(img, x, y0, color)
        set_pixel(img, x, y1, color)
    for y in range(y0, y1+1):
        set_pixel(img, x0, y, color)
        set_pixel(img, x1, y, color)

def load_pattern_from_file(path):
    """
    Загрузить картинку (любого поддерживаемого matplotlib формата) как паттерн RGB uint8.
    Если изображение с альфой или в float — приводится к 0..255 uint8 (каналы RGB).
    """
    arr = mpimg.imread(path)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)  # серое -> RGB
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]  # отбрасываем альфу
    return arr

# -----------------------------------------------------
# 1а) Рекурсивная заливка scanline (цветом, без паттерна)
# -----------------------------------------------------
def scanline_fill_color(img, sx, sy, fill_color):
    """
    Заливка связной области, ограниченной цветом-границей.
    Связность по 4-соседям. Рекурсивная стратегия «по строкам» (сериями пикселей).
    """
    h, w = img.shape[:2]
    target = get_pixel(img, sx, sy)
    target_tuple = tuple(int(c) for c in target)
    fill_tuple   = tuple(int(c) for c in fill_color)
    if target_tuple == fill_tuple:
        return

    sys.setrecursionlimit(1_000_000)

    def colors_equal(a, b):
        return a[0]==b[0] and a[1]==b[1] and a[2]==b[2]

    def fill_from(x, y):
        if not in_bounds(img, x, y):
            return
        if not colors_equal(tuple(int(v) for v in img[y, x]), target_tuple):
            return

        xl = x
        while xl - 1 >= 0 and colors_equal(tuple(int(v) for v in img[y, xl-1]), target_tuple):
            xl -= 1
        xr = x
        while xr + 1 < w and colors_equal(tuple(int(v) for v in img[y, xr+1]), target_tuple):
            xr += 1

        img[y, xl:xr+1] = fill_tuple

        for ny in (y - 1, y + 1):
            if 0 <= ny < h:
                xi = xl
                while xi <= xr:
                    if colors_equal(tuple(int(v) for v in img[ny, xi]), target_tuple):
                        fill_from(xi, ny)
                        while xi <= xr and colors_equal(tuple(int(v) for v in img[ny, xi]), target_tuple):
                            xi += 1
                    else:
                        xi += 1

    fill_from(sx, sy)

# -------------------------------------------------------------------
# 1б) Заливка scanline паттерном (tile/clip)
# -------------------------------------------------------------------
def scanline_fill_pattern(img, sx, sy, pattern, mode="auto", anchor=(0,0)):
    """
    Заливка области, где вместо fill_color пиксели берутся из pattern.

    mode='tile' — паттерн повторяется по модулю размера (циклически);
    mode='clip' — вырезается «кусок» без повторов; вне паттерна пиксели пропускаются;
    mode='auto' — автоматически: если pattern меньше холста по одному из измерений — tile, иначе clip.

    anchor=(x0,y0) — выравнивание паттерна относительно координат холста.
    """
    h, w = img.shape[:2]
    ph, pw = pattern.shape[:2]
    if mode == "auto":
        mode = "tile" if (pw < w or ph < h) else "clip"

    target = tuple(int(v) for v in img[sy, sx])
    sys.setrecursionlimit(1_000_000)

    def pattern_at(x, y):
        px = x - anchor[0]
        py = y - anchor[1]
        if mode == "tile":
            return pattern[py % ph, px % pw]
        else:
            if 0 <= px < pw and 0 <= py < ph:
                return pattern[py, px]
            else:
                return None

    def fill_from(x, y):
        if not in_bounds(img, x, y):
            return
        if tuple(int(v) for v in img[y, x]) != target:
            return

        xl = x
        while xl - 1 >= 0 and tuple(int(v) for v in img[y, xl-1]) == target:
            xl -= 1
        xr = x
        while xr + 1 < w and tuple(int(v) for v in img[y, xr+1]) == target:
            xr += 1

        for xi in range(xl, xr+1):
            pcol = pattern_at(xi, y)
            if pcol is not None:
                img[y, xi] = pcol

        for ny in (y - 1, y + 1):
            if 0 <= ny < h:
                xi = xl
                while xi <= xr:
                    if tuple(int(v) for v in img[ny, xi]) == target:
                        fill_from(xi, ny)
                        while xi <= xr and tuple(int(v) for v in img[ny, xi]) == target:
                            xi += 1
                    else:
                        xi += 1

    fill_from(sx, sy)

# ------------------------------------------------------------
# 1в) Обход границы связной области (Moore-neighbor tracing)
# ------------------------------------------------------------
NEIGH_8 = [(-1,-1),(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0)]

def moore_neighbor_trace(mask, start):
    """
    mask: бинарная маска (True на границе), start: (x,y) — точка на границе.
    Возвращает список точек контура в порядке обхода по часовой стрелке.
    Защищено от зацикливания с помощью множества посещённых пар (curr, prev).
    """
    h, w = mask.shape
    sx, sy = start
    if not (0 <= sx < w and 0 <= sy < h and mask[sy, sx]):
        raise ValueError("Начальная точка не попадает на границу.")

    prev = (sx - 1, sy)  # «точка позади» слева от старта
    curr = (sx, sy)
    contour = []
    first_move_done = False
    first_prev_at_start = None
    visited_pairs = set()

    def neigh_index(center, point):
        cx, cy = center
        dx, dy = point[0]-cx, point[1]-cy
        for i, (nx, ny) in enumerate(NEIGH_8):
            if (dx, dy) == (nx, ny):
                return i
        return None

    while True:
        contour.append(curr)
        key = (curr, prev)
        if key in visited_pairs:
            break
        visited_pairs.add(key)

        cidx = neigh_index(curr, prev)
        if cidx is None:
            cidx = 7  # запад
        i = (cidx + 1) % 8
        found = False
        for _ in range(8):
            nx = curr[0] + NEIGH_8[i][0]
            ny = curr[1] + NEIGH_8[i][1]
            if 0 <= nx < w and 0 <= ny < h and mask[ny, nx]:
                prev, curr = curr, (nx, ny)
                found = True
                if not first_move_done and curr != (sx, sy):
                    first_move_done = True
                    first_prev_at_start = prev
                break
            i = (i + 1) % 8

        if not found:
            break
        if curr == (sx, sy) and first_move_done and prev == first_prev_at_start:
            break

    return contour

def overlay_contour(img, points, color=(255,0,0)):
    """Наложить список точек контура заданным цветом поверх изображения."""
    out = img.copy()
    for (x,y) in points:
        if in_bounds(out, x, y):
            out[y, x] = color
    return out

# --------------------------------------
# 2) Отрезки: Брезенхем и алгоритм Ву
# --------------------------------------
def draw_line_bresenham(img, x0, y0, x1, y1, color=(0,0,0)):
    """Целочисленный Брезенхем (все восьмёрки октантов)."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        set_pixel(img, x0, y0, color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

def ipart(x): return int(floor(x))
def roundi(x): return int(round(x))
def fpart(x): return x - floor(x)
def rfpart(x): return 1 - fpart(x)

def draw_line_wu(img, x0, y0, x1, y1, color=(0,0,0)):
    """Алгоритм Ву — сглаженная линия антиалиасингом."""
    steep = abs(y1 - y0) > abs(x1 - x0)
    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    dx = x1 - x0
    dy = y1 - y0
    grad = dy / dx if dx != 0 else 0

    # первая точка
    xend = roundi(x0)
    yend = y0 + grad * (xend - x0)
    xgap = rfpart(x0 + 0.5)
    xpxl1 = xend
    ypxl1 = ipart(yend)
    if steep:
        blend_pixel(img, ypxl1,   xpxl1, color, rfpart(yend) * xgap)
        blend_pixel(img, ypxl1+1, xpxl1, color, fpart(yend)  * xgap)
    else:
        blend_pixel(img, xpxl1, ypxl1,   color, rfpart(yend) * xgap)
        blend_pixel(img, xpxl1, ypxl1+1, color, fpart(yend)  * xgap)
    intery = yend + grad

    # последняя точка
    xend = roundi(x1)
    yend = y1 + grad * (xend - x1)
    xgap = fpart(x1 + 0.5)
    xpxl2 = xend
    ypxl2 = ipart(yend)

    # основной цикл
    if steep:
        for x in range(xpxl1 + 1, xpxl2):
            blend_pixel(img, ipart(intery),   x, color, rfpart(intery))
            blend_pixel(img, ipart(intery)+1, x, color, fpart(intery))
            intery += grad
        blend_pixel(img, ypxl2,   xpxl2, color, rfpart(yend) * xgap)
        blend_pixel(img, ypxl2+1, xpxl2, color, fpart(yend)  * xgap)
    else:
        for x in range(xpxl1 + 1, xpxl2):
            blend_pixel(img, x, ipart(intery),   color, rfpart(intery))
            blend_pixel(img, x, ipart(intery)+1, color, fpart(intery))
            intery += grad
        blend_pixel(img, xpxl2, ypxl2,   color, rfpart(yend) * xgap)
        blend_pixel(img, xpxl2, ypxl2+1, color, fpart(yend)  * xgap)

# --------------------------------------------------------------
# 3) Растеризация треугольника с градиентной интерполяцией цвета
# --------------------------------------------------------------
@dataclass
class Vertex:
    x: int
    y: int
    color: tuple  # (r,g,b) в 0..255

def draw_triangle_gradient(img, v0: Vertex, v1: Vertex, v2: Vertex):
    """Заливка треугольника, линейная интерполяция цвета по барицентрическим весам."""
    p0 = np.array([v0.x, v0.y], dtype=np.float32)
    p1 = np.array([v1.x, v1.y], dtype=np.float32)
    p2 = np.array([v2.x, v2.y], dtype=np.float32)
    c0 = np.array(v0.color, dtype=np.float32)
    c1 = np.array(v1.color, dtype=np.float32)
    c2 = np.array(v2.color, dtype=np.float32)

    minx = max(0, min(v0.x, v1.x, v2.x))
    maxx = min(img.shape[1]-1, max(v0.x, v1.x, v2.x))
    miny = max(0, min(v0.y, v1.y, v2.y))
    maxy = min(img.shape[0]-1, max(v0.y, v1.y, v2.y))

    def edge(a, b, p):
        return (p[0]-a[0])*(b[1]-a[1]) - (p[1]-a[1])*(b[0]-a[0])

    area = edge(p0, p1, p2)
    if area == 0:
        return  # вырожденный

    for y in range(miny, maxy+1):
        for x in range(minx, maxx+1):
            p = np.array([x + 0.5, y + 0.5], dtype=np.float32)
            w0 = edge(p1, p2, p)
            w1 = edge(p2, p0, p)
            w2 = edge(p0, p1, p)
            if (w0 >= 0 and w1 >= 0 and w2 >= 0) or (w0 <= 0 and w1 <= 0 and w2 <= 0):
                w0 /= area; w1 /= area; w2 /= area
                col = w0*c0 + w1*c1 + w2*c2
                img[y, x] = np.clip(col, 0, 255).astype(np.uint8)

# ----------------------
# Демонстрация (main)
# ----------------------
def demo_and_save():
    # 1а) Холст с границами и заливка цветом
    canvas1 = make_canvas(320, 220, (255,255,255))
    draw_rect_outline(canvas1, 20, 20, 300, 200, (0,0,0))
    draw_rect_outline(canvas1, 100, 80, 220, 160, (0,0,0))
    scanline_fill_color(canvas1, 50, 50, (0,170,255))
    plt.figure(figsize=(5,3.8)); plt.title("1а) Заливка цветом (scanline)"); plt.imshow(canvas1); plt.axis('off')
    plt.savefig("task1a_fill_color.png", dpi=160, bbox_inches='tight')

    # 1б) Паттерны: маленький (tile) и большой (clip)
    pat_small = np.zeros((16,16,3), dtype=np.uint8)
    pat_small[:,:,:] = (230,230,230)
    pat_small[::2, ::2] = (80,80,80)
    pat_small[1::2, 1::2] = (80,80,80)

    ph, pw = 800, 800
    grad_x = np.linspace(0, 255, pw, dtype=np.float32)
    grad_y = np.linspace(0, 255, ph, dtype=np.float32)
    pat_big = np.zeros((ph,pw,3), dtype=np.uint8)
    for c in range(3):
        pat_big[:,:,c] = np.clip((0.6*grad_x + 0.4*grad_y[:,None]) % 256, 0, 255).astype(np.uint8)

    canvas1b_tile = make_canvas(320, 220, (255,255,255))
    draw_rect_outline(canvas1b_tile, 20, 20, 300, 200, (0,0,0))
    draw_rect_outline(canvas1b_tile, 100, 80, 220, 160, (0,0,0))
    scanline_fill_pattern(canvas1b_tile, 50, 50, pat_small, mode="tile", anchor=(0,0))
    plt.figure(figsize=(5,3.8)); plt.title("1б) Маленький паттерн (tile)"); plt.imshow(canvas1b_tile); plt.axis('off')
    plt.savefig("task1b_pattern_tile.png", dpi=160, bbox_inches='tight')

    canvas1b_clip = make_canvas(320, 220, (255,255,255))
    draw_rect_outline(canvas1b_clip, 20, 20, 300, 200, (0,0,0))
    draw_rect_outline(canvas1b_clip, 100, 80, 220, 160, (0,0,0))
    scanline_fill_pattern(canvas1b_clip, 50, 50, pat_big, mode="clip", anchor=(0,0))
    plt.figure(figsize=(5,3.8)); plt.title("1б) Большое изображение (clip)"); plt.imshow(canvas1b_clip); plt.axis('off')
    plt.savefig("task1b_pattern_clip.png", dpi=160, bbox_inches='tight')

    # 1в) Обход границы и наложение
    base_for_contour = make_canvas(320, 220, (255,255,255))
    draw_rect_outline(base_for_contour, 20, 20, 300, 200, (0,0,0))
    draw_rect_outline(base_for_contour, 100, 80, 220, 160, (0,0,0))
    mask = np.all(base_for_contour == np.array([0,0,0], dtype=np.uint8), axis=2)
    start_pt = (20, 20)
    contour_pts = moore_neighbor_trace(mask, start_pt)
    overlay = overlay_contour(base_for_contour, contour_pts, (255,0,0))
    plt.figure(figsize=(5,3.8)); plt.title("1в) Граница (красным)"); plt.imshow(overlay); plt.axis('off')
    plt.savefig("task1c_contour_overlay.png", dpi=160, bbox_inches='tight')

    # 2) Отрезки: Брезенхем и Ву
    lines = [((10,10),(340,30)), ((10,210),(340,50)), ((20,20),(200,200)), ((180,20),(40,200))]

    canvas2_b = make_canvas(360, 220, (255,255,255))
    for (x0,y0),(x1,y1) in lines:
        draw_line_bresenham(canvas2_b, x0,y0,x1,y1, (0,0,0))
    plt.figure(figsize=(5.6,3.8)); plt.title("2) Брезенхем (целочисленный)"); plt.imshow(canvas2_b); plt.axis('off')
    plt.savefig("task2_bresenham.png", dpi=160, bbox_inches='tight')

    canvas2_w = make_canvas(360, 220, (255,255,255))
    for (x0,y0),(x1,y1) in lines:
        draw_line_wu(canvas2_w, x0,y0,x1,y1, (0,0,0))
    plt.figure(figsize=(5.6,3.8)); plt.title("2) Алгоритм Ву (с сглаживанием)"); plt.imshow(canvas2_w); plt.axis('off')
    plt.savefig("task2_wu.png", dpi=160, bbox_inches='tight')

    # 3) Градиентный треугольник
    canvas3 = make_canvas(320, 240, (255,255,255))
    v0 = Vertex(60, 40,  (255, 30, 30))
    v1 = Vertex(260, 60, (30, 255, 30))
    v2 = Vertex(140, 200,(30, 30, 255))
    draw_triangle_gradient(canvas3, v0, v1, v2)
    plt.figure(figsize=(5.6,4.0)); plt.title("3) Градиентный треугольник"); plt.imshow(canvas3); plt.axis('off')
    plt.savefig("task3_triangle_gradient.png", dpi=160, bbox_inches='tight')

if __name__ == "__main__":
    demo_and_save()
    print("Готово. См. PNG-файлы в текущей папке.")
