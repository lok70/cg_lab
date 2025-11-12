import math
import sys
from typing import List, Tuple, Iterable, Optional
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except Exception:  # pragma: no cover
    print("Этому приложению требуется Tkinter.", file=sys.stderr)
    raise

Point = Tuple[float, float]


EPS = 1e-9

def area2(poly: List[Point]) -> float:
    """Вдвойне удвоенная ориентированная площадь (удобно для знака ориентации)."""
    s = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return s

def ensure_ccw(poly: List[Point]) -> List[Point]:
    """Делает порядок вершин против часовой стрелки (CCW)."""
    if len(poly) >= 3 and area2(poly) < 0:
        return list(reversed(poly))
    return poly

def cross(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * by - ay * bx

def cross3(o: Point, a: Point, b: Point) -> float:
    """Векторное произведение OA x OB (знак поворота)."""
    return cross(a[0] - o[0], a[1] - o[1], b[0] - o[0], b[1] - o[1])

def is_convex(poly: List[Point]) -> bool:
    """Проверка выпуклости по знаку поворота (допускается коллинеарность)."""
    n = len(poly)
    if n < 3:
        return False
    sign = 0
    for i in range(n):
        c = cross3(poly[i], poly[(i + 1) % n], poly[(i + 2) % n])
        if abs(c) <= EPS:
            continue
        s = 1 if c > 0 else -1
        if sign == 0:
            sign = s
        elif sign != s:
            return False
    return True

def convex_hull(points: Iterable[Point]) -> List[Point]:
    """Выпуклая оболочка методом монотонной цепи Эндрю. Возвращает CCW-полигон без повторов крайних точек."""
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts

    def build_half(seq):
        half = []
        for p in seq:
            while len(half) >= 2 and cross3(half[-2], half[-1], p) <= EPS:
                half.pop()
            half.append(p)
        return half

    lower = build_half(pts)
    upper = build_half(reversed(pts))
    hull = lower[:-1] + upper[:-1]
    return hull

def line_intersection(p1: Point, p2: Point, q1: Point, q2: Point) -> Point:
    """
    Пересечение прямых p1p2 и q1q2. Возвращает точку на p1p2.
    Если прямые почти параллельны, возвращает середину p1p2 (числовая страховка).
    """
    r = (p2[0] - p1[0], p2[1] - p1[1])
    s = (q2[0] - q1[0], q2[1] - q1[1])
    denom = cross(r[0], r[1], s[0], s[1])
    if abs(denom) <= EPS:
        return ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)
    qp = (q1[0] - p1[0], q1[1] - p1[1])
    t = cross(qp[0], qp[1], s[0], s[1]) / denom
    return (p1[0] + t * r[0], p1[1] + t * r[1])

def is_inside_halfplane(p: Point, a: Point, b: Point) -> bool:
    """Точка p слева от ориентированного ребра a->b (или на линии). Предполагаем CCW-обход клипа."""
    return cross3(a, b, p) >= -EPS

def sutherland_hodgman(subject: List[Point], clip: List[Point]) -> List[Point]:
    """
    Пересечение subject ∩ clip, где clip — выпуклый многоугольник (CCW).
    Работает и для выпуклого subject. Возвращает список точек результата (CCW или пустой).
    """
    if len(subject) < 3 or len(clip) < 3:
        return []

    subj = ensure_ccw(subject)
    clp = ensure_ccw(clip)

    output = subj[:]
    for i in range(len(clp)):
        a = clp[i]
        b = clp[(i + 1) % len(clp)]
        input_list = output
        output = []
        if not input_list:
            break

        S = input_list[-1]
        for E in input_list:
            Ein = is_inside_halfplane(E, a, b)
            Sin = is_inside_halfplane(S, a, b)

            if Ein:
                if not Sin:
                    inter = line_intersection(S, E, a, b)
                    output.append(inter)
                output.append(E)
            elif Sin:
                inter = line_intersection(S, E, a, b)
                output.append(inter)
            S = E
    # Удалим почти совпадающие соседние точки
    result = []
    for p in output:
        if not result or (abs(result[-1][0] - p[0]) > 1e-8 or abs(result[-1][1] - p[1]) > 1e-8):
            result.append(p)
    if len(result) >= 3 and (abs(result[0][0] - result[-1][0]) <= 1e-8 and abs(result[0][1] - result[-1][1]) <= 1e-8):
        result.pop()  # без повторения первой точки в конце
    return result


# ---------------------- Интерфейс ----------------------

class PolygonUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Пересечение выпуклых полигонов — Sutherland–Hodgman")

        self.canvas = tk.Canvas(root, width=900, height=600, bg="white")
        self.canvas.grid(row=0, column=0, rowspan=10, padx=8, pady=8)

        ctrl = ttk.Frame(root)
        ctrl.grid(row=0, column=1, sticky="n", padx=8, pady=8)

        # Состояние
        self.current = tk.StringVar(value="A")
        self.use_hull = tk.BooleanVar(value=True)
        self.polyA: List[Point] = []
        self.polyB: List[Point] = []
        self.closedA = False
        self.closedB = False
        self.drawn_shapes = {"A": [], "B": [], "I": []}  # ids на Canvas
        self.vertex_radius = 3

        ttk.Label(ctrl, text="Текущий полигон:").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(ctrl, text="A", variable=self.current, value="A", command=self._switch_current).grid(row=1, column=0, sticky="w")
        ttk.Radiobutton(ctrl, text="B", variable=self.current, value="B", command=self._switch_current).grid(row=1, column=1, sticky="w")

        ttk.Checkbutton(ctrl, text="Использовать выпуклую оболочку", variable=self.use_hull, command=self.redraw_all).grid(row=2, column=0, columnspan=2, sticky="w", pady=(4,8))

        ttk.Button(ctrl, text="Закрыть полигон", command=self.close_current).grid(row=3, column=0, columnspan=2, sticky="we")
        ttk.Button(ctrl, text="Пересечение", command=self.compute_intersection).grid(row=4, column=0, columnspan=2, sticky="we", pady=(6,0))
        ttk.Button(ctrl, text="Очистить", command=self.reset_all).grid(row=5, column=0, columnspan=2, sticky="we")

        sep = ttk.Separator(ctrl, orient="horizontal")
        sep.grid(row=6, column=0, columnspan=2, sticky="we", pady=8)

        desc = ("ЛКМ — добавить вершину текущего полигона.\n"
                "«Закрыть полигон» — соединяет последнюю и первую точки.\n"
                "При включённой опции оболочка точки упорядочиваются автоматически.\n"
                "Для корректного пересечения оба полигона должны быть выпуклы.")
        ttk.Label(ctrl, text=desc, wraplength=260, justify="left").grid(row=7, column=0, columnspan=2, sticky="w")

        # Привязки событий
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # Красивая сетка
        self._draw_grid()

    # ----------- Рисование -----------

    def _draw_grid(self):
        w = int(self.canvas["width"])
        h = int(self.canvas["height"])
        step = 50
        for x in range(0, w, step):
            self.canvas.create_line(x, 0, x, h, fill="#f0f0f0")
        for y in range(0, h, step):
            self.canvas.create_line(0, y, w, y, fill="#f0f0f0")

    def _switch_current(self):
        self.root.title(f"Пересечение выпуклых полигонов — редактируется {self.current.get()}")

    def on_canvas_click(self, event):
        p = (float(event.x), float(event.y))
        if self.current.get() == "A":
            if self.closedA:
                messagebox.showinfo("Полигон A", "Полигон уже закрыт. Очистите или переключитесь на B.")
                return
            self.polyA.append(p)
        else:
            if self.closedB:
                messagebox.showinfo("Полигон B", "Полигон уже закрыт. Очистите или переключитесь на A.")
                return
            self.polyB.append(p)
        self.redraw_all()

    def _effective_poly(self, pts: List[Point]) -> List[Point]:
        if not pts:
            return []
        if self.use_hull.get():
            hull = convex_hull(pts)
            return ensure_ccw(hull)
        else:
            # используем как есть, но стараемся сделать CCW
            return ensure_ccw(pts[:])

    def redraw_all(self):
        for key in ("A", "B", "I"):
            for item in self.drawn_shapes[key]:
                self.canvas.delete(item)
            self.drawn_shapes[key] = []

        effA = self._effective_poly(self.polyA)
        effB = self._effective_poly(self.polyB)

        self._draw_poly(effA, outline="#1f77b4", fill="#cfe2ff", tag="A")  # синий
        self._draw_poly(effB, outline="#d62728", fill="#ffc9c9", tag="B")  # красный

        # точки
        for p in self.polyA:
            self.drawn_shapes["A"].append(self._draw_point(p, color="#1f77b4"))
        for p in self.polyB:
            self.drawn_shapes["B"].append(self._draw_point(p, color="#d62728"))

    def _draw_point(self, p: Point, color: str):
        r = self.vertex_radius
        return self.canvas.create_oval(p[0]-r, p[1]-r, p[0]+r, p[1]+r, fill=color, outline=color)

    def _draw_poly(self, poly: List[Point], outline: str, fill: Optional[str], tag: str):
        if len(poly) >= 3:
            pid = self.canvas.create_polygon(*sum(([x, y] for x, y in poly), []),
                                             outline=outline, fill=fill, width=2)
            self.drawn_shapes[tag].append(pid)
        elif len(poly) == 2:
            lid = self.canvas.create_line(poly[0][0], poly[0][1], poly[1][0], poly[1][1], fill=outline, width=2)
            self.drawn_shapes[tag].append(lid)

    # ----------- Команды -----------

    def close_current(self):
        if self.current.get() == "A":
            if len(self.polyA) < 3:
                messagebox.showwarning("Полигон A", "Нужно минимум 3 точки.")
                return
            self.closedA = True
        else:
            if len(self.polyB) < 3:
                messagebox.showwarning("Полигон B", "Нужно минимум 3 точки.")
                return
            self.closedB = True
        self.redraw_all()

    def compute_intersection(self):
        effA = self._effective_poly(self.polyA)
        effB = self._effective_poly(self.polyB)
        if len(effA) < 3 or len(effB) < 3:
            messagebox.showwarning("Пересечение", "Оба полигона должны иметь не менее 3 вершин.")
            return

        if not is_convex(effA) or not is_convex(effB):
            messagebox.showwarning("Пересечение", "Полигоны должны быть выпуклые. Включите опцию оболочки или скорректируйте точки.")
            return

        inter = sutherland_hodgman(effA, effB)
        # Рисуем поверх
        for item in self.drawn_shapes["I"]:
            self.canvas.delete(item)
        self.drawn_shapes["I"] = []
        if len(inter) >= 3:
            pid = self.canvas.create_polygon(*sum(([x, y] for x, y in inter), []),
                                             outline="#2ca02c", fill="#d4f8d4", width=2)
            self.drawn_shapes["I"].append(pid)
        else:
            messagebox.showinfo("Пересечение", "Пересечение отсутствует.")

    def reset_all(self):
        self.polyA.clear()
        self.polyB.clear()
        self.closedA = False
        self.closedB = False
        self.redraw_all()


def _demo_cli():
    """Небольшая проверка алгоритма без UI."""
    tri = [(0.0, 0.0), (4.0, 0.0), (2.0, 3.0)]
    sqr = [(1.0, -1.0), (3.0, -1.0), (3.0, 2.0), (1.0, 2.0)]
    res = sutherland_hodgman(tri, sqr)
    print("Пересечение треугольник ∩ квадрат:", res)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        _demo_cli()
        sys.exit(0)
    root = tk.Tk()
    app = PolygonUI(root)
    root.mainloop()
