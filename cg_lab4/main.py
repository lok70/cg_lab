#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

# ----------------------------- МАТРИЦЫ -----------------------------

Matrix = List[List[float]]
Vec2 = Tuple[float, float]

def mat_mul(A: Matrix, B: Matrix) -> Matrix:
    return [[A[i][0]*B[0][j] + A[i][1]*B[1][j] + A[i][2]*B[2][j] for j in range(3)] for i in range(3)]

def mat_apply(M: Matrix, p: Vec2) -> Vec2:
    x, y = p
    X = M[0][0]*x + M[0][1]*y + M[0][2]
    Y = M[1][0]*x + M[1][1]*y + M[1][2]
    W = M[2][0]*x + M[2][1]*y + M[2][2]
    if abs(W) < 1e-12: return (X, Y)
    return (X/W, Y/W)

def T(dx: float, dy: float) -> Matrix:
    return [[1.0, 0.0, dx],[0.0, 1.0, dy],[0.0, 0.0, 1.0]]

def R(angle_deg: float) -> Matrix:
    a = math.radians(angle_deg); c, s = math.cos(a), math.sin(a)
    return [[c,-s,0.0],[s,c,0.0],[0.0,0.0,1.0]]

def S(sx: float, sy: float) -> Matrix:
    return [[sx,0.0,0.0],[0.0,sy,0.0],[0.0,0.0,1.0]]

def around_point(M: Matrix, px: float, py: float) -> Matrix:
    return mat_mul(mat_mul(T(px, py), M), T(-px, -py))

# ----------------------------- ВСПОМОГАТЕЛЬНОЕ -----------------------------

def dist(a: Vec2, b: Vec2) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])

def dist2(a: Vec2, b: Vec2) -> float:
    dx = a[0]-b[0]; dy = a[1]-b[1]; return dx*dx + dy*dy

def orient(a: Vec2, b: Vec2, c: Vec2) -> float:
    """Ориентированная площадь *2 (экранные координаты, Y вниз)."""
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

def on_segment(a: Vec2, b: Vec2, p: Vec2, eps: float=1e-9) -> bool:
    if abs(orient(a,b,p)) > eps: return False
    return (min(a[0],b[0]) - eps <= p[0] <= max(a[0],b[0]) + eps and
            min(a[1],b[1]) - eps <= p[1] <= max(a[1],b[1]) + eps)

def segments_intersect_strict(a: Vec2, b: Vec2, c: Vec2, d: Vec2) -> bool:
    """Пересекаются ли отрезки (включая касания и коллинеарные попадания)."""
    o1 = orient(a,b,c); o2 = orient(a,b,d); o3 = orient(c,d,a); o4 = orient(c,d,b)
    if (o1*o2 < 0) and (o3*o4 < 0): return True
    if abs(o1) < 1e-9 and on_segment(a,b,c): return True
    if abs(o2) < 1e-9 and on_segment(a,b,d): return True
    if abs(o3) < 1e-9 and on_segment(c,d,a): return True
    if abs(o4) < 1e-9 and on_segment(c,d,b): return True
    return False

def point_on_segment(p: Vec2, a: Vec2, b: Vec2, eps: float = 2.0) -> bool:
    ax, ay = a; bx, by = b; px, py = p
    vx, vy = bx-ax, by-ay; wx, wy = px-ax, py-ay
    seg_len2 = vx*vx + vy*vy
    if seg_len2 < 1e-12: return dist2(p, a) <= eps*eps
    t = max(0.0, min(1.0, (wx*vx + wy*vy)/seg_len2))
    proj = (ax + t*vx, ay + t*vy)
    return dist2(p, proj) <= eps*eps

def point_in_polygon(pt: Vec2, vertices: List[Vec2]) -> int:
    x, y = pt; n = len(vertices)
    if n == 0: return -1
    if n == 1: return 0 if dist2(pt, vertices[0]) <= 4.0 else -1
    if n == 2: return 0 if point_on_segment(pt, vertices[0], vertices[1]) else -1
    for i in range(n):
        a = vertices[i]; b = vertices[(i+1) % n]
        if point_on_segment(pt, a, b): return 0
    inside = False
    for i in range(n):
        x1,y1 = vertices[i]; x2,y2 = vertices[(i+1) % n]
        if ((y1 > y) != (y2 > y)):
            t = (y - y1) / (y2 - y1)
            x_at_y = x1 + t*(x2 - x1)
            if x_at_y > x: inside = not inside
    return 1 if inside else -1

def poly_edges(verts: List[Vec2]) -> List[Tuple[Vec2, Vec2]]:
    return [(verts[i], verts[(i+1) % len(verts)]) for i in range(len(verts))]

def line_intersection_point(a1: Vec2, a2: Vec2, b1: Vec2, b2: Vec2) -> Optional[Vec2]:
    x1,y1=a1; x2,y2=a2; x3,y3=b1; x4,y4=b2
    den = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(den) < 1e-12: return None
    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / den
    return (x1 + t*(x2-x1), y1 + t*(y2-y1))

def segments_intersect(a1: Vec2, a2: Vec2, b1: Vec2, b2: Vec2) -> Tuple[bool, Optional[Vec2]]:
    """Единственная точка пересечения отрезков (без «отрезок поверх отрезка»)."""
    x1,y1=a1; x2,y2=a2; x3,y3=b1; x4,y4=b2
    den = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(den) < 1e-12:
        # Параллельные/коллинеарные — единственной точки может не быть
        return (False, None)
    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / den
    u = ((x1-x3)*(y1-y2) - (y1-y3)*(x1-x2)) / den
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        ix = x1 + t*(x2-x1); iy = y1 + t*(y2-y1)
        return (True, (ix, iy))
    return (False, None)

def point_seg_distance(p: Vec2, a: Vec2, b: Vec2) -> float:
    ax, ay = a; bx, by = b; px, py = p
    vx, vy = bx-ax, by-ay; wx, wy = px-ax, py-ay
    L2 = vx*vx + vy*vy
    if L2 == 0: return math.hypot(px-ax, py-ay)
    t = max(0.0, min(1.0, (wx*vx + wy*vy)/L2))
    proj = (ax + t*vx, ay + t*vy)
    return math.hypot(px-proj[0], py-proj[1])

def valid_insertion(poly: List[Vec2], idx: int, p: Vec2) -> bool:
    m = len(poly)
    a = poly[idx]; b = poly[(idx+1) % m]
    new_edges = [(a, p), (p, b)]
    for j in range(m):
        c = poly[j]; d = poly[(j+1) % m]
        if c in (a,b) or d in (a,b):
            continue
        for e in new_edges:
            if segments_intersect_strict(e[0], e[1], c, d):
                return False
    return True

def insert_point_locally(poly: List[Vec2], p: Vec2, w_dist: float = 1.0, w_delta: float = 0.05) -> List[Vec2]:
    n = len(poly)
    if n <= 1:
        return poly + [p]
    if n == 2:
        return [poly[0], p, poly[1]]

    best_idx = None; best_cost = float("inf")
    for i in range(n):
        a = poly[i]; b = poly[(i+1) % n]
        if not valid_insertion(poly, i, p):
            continue
        delta = dist(a,p) + dist(p,b) - dist(a,b)
        cost = w_dist * point_seg_distance(p, a, b) + w_delta * delta
        if cost < best_cost:
            best_cost = cost; best_idx = i

    if best_idx is None:
        k = min(range(n), key=lambda i: dist(poly[i], p))
        best_idx = k

    new_poly = poly[:best_idx+1] + [p] + poly[best_idx+1:]
    return new_poly

# ----------------------------- МОДЕЛЬ ПОЛИГОНА -----------------------------

@dataclass
class PolyShape:
    vertices: List[Vec2] = field(default_factory=list)
    items: List[int] = field(default_factory=list)
    selected: bool = False

    def draw(self, canvas: tk.Canvas) -> None:
        self.erase(canvas)
        n = len(self.vertices)
        if n == 0: return
        if n == 1:
            x,y = self.vertices[0]; r = 4
            self.items = [canvas.create_oval(x-r, y-r, x+r, y+r, fill="#1a73e8", outline="")]
        elif n == 2:
            x1,y1 = self.vertices[0]; x2,y2 = self.vertices[1]
            self.items = [canvas.create_line(x1, y1, x2, y2, fill="#1a73e8", width=2)]
        else:
            flat = [c for p in self.vertices for c in p]
            poly_id = canvas.create_polygon(*flat, outline="#1a73e8", fill="", width=2)
            self.items = [poly_id]
            for (x,y) in self.vertices:
                r = 3
                self.items.append(canvas.create_oval(x-r, y-r, x+r, y+r, outline="#1a73e8"))
        if self.selected:
            self.highlight(canvas, True)

    def erase(self, canvas: tk.Canvas) -> None:
        for it in self.items:
            try: canvas.delete(it)
            except Exception: pass
        self.items.clear()

    def highlight(self, canvas: tk.Canvas, state: bool) -> None:
        self.selected = state
        for it in self.items:
            try:
                canvas.itemconfigure(it,
                    width=3 if state else 2,
                    outline="#d32f2f" if state else "#1a73e8")
            except Exception: pass

    def apply_matrix(self, M: Matrix) -> None:
        self.vertices = [mat_apply(M, p) for p in self.vertices]

# ----------------------------- ПРИЛОЖЕНИЕ -----------------------------

class PolyEditor(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Полигональный редактор — v7")

        # HiDPI
        self.ui_scale = self._auto_scale()
        try: self.tk.call("tk", "scaling", self.ui_scale)
        except Exception: pass

        self._apply_style()
        self._build_layout()

        # Состояние
        self.polys: List[PolyShape] = []
        self.current_poly: Optional[PolyShape] = None
        self.selected_poly: Optional[PolyShape] = None
        self.mode: str = "select"
        self.status_var = tk.StringVar(value="Готово")
        self.status_label.configure(textvariable=self.status_var)

        # Превью/пересечение
        self.preview_items: List[int] = []
        self.inter_state: Dict[str, Optional[object]] = {"a1": None, "a2": None, "b1": None, "b2": None, "line_b_id": None}
        self.pvr_state: Dict[str, Optional[Vec2]] = {"e1": None, "e2": None}
        self._temp_inter_markers: List[int] = []
        self.bind_all("<Escape>", self.on_escape)

        self.update_status("Режим: выделение.")

    # ---------- UI ----------
    def _auto_scale(self) -> float:
        sh = self.winfo_screenheight(); ratio = sh/1080.0
        presets = [1.0,1.25,1.5,1.75,2.0]
        target = max(1.0, min(2.0, ratio))
        return min(presets, key=lambda p: abs(p-target))

    def _apply_style(self) -> None:
        self.configure(bg="#ffffff")
        style = ttk.Style(self)
        try: style.theme_use("clam")
        except Exception: pass
        style.configure("Panel.TFrame", background="#ffffff")
        style.configure("Big.TLabel", background="#ffffff", foreground="#5f6368",
                        font=("TkDefaultFont", max(10, int(10*self.ui_scale))))
        base_font = ("TkDefaultFont", max(11, int(11*self.ui_scale)))
        btn_font  = ("TkDefaultFont", max(11, int(11*self.ui_scale)), "bold")
        style.configure(".", font=base_font, foreground="#111111")
        pad_h = int(14*self.ui_scale); pad_v = int(9*self.ui_scale)
        style.configure("MD.Filled.TButton", font=btn_font, padding=(pad_h, pad_v),
                        background="#1a73e8", foreground="#ffffff", relief="flat", borderwidth=0)
        style.map("MD.Filled.TButton", background=[("active","#185abc")])
        style.configure("MD.Outlined.TButton", font=btn_font, padding=(pad_h, pad_v),
                        background="#ffffff", foreground="#111111", relief="flat", borderwidth=1)
        style.map("MD.Outlined.TButton", background=[("active","#f6f9fe")])
        style.configure("MD.Text.TButton", font=btn_font, padding=(pad_h, pad_v),
                        background="#ffffff", foreground="#1a73e8", relief="flat", borderwidth=0)
        style.map("MD.Text.TButton", background=[("active","#f6f9fe")])

    def _build_layout(self) -> None:
        self.top_bar = ttk.Frame(self, style="Panel.TFrame", padding=(12,12)); self.top_bar.pack(side=tk.TOP, fill=tk.X)
        self.bottom_bar = ttk.Frame(self, style="Panel.TFrame", padding=(12,12)); self.bottom_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_label = ttk.Label(self, style="Big.TLabel", anchor="w"); self.status_label.pack(side=tk.BOTTOM, fill=tk.X, padx=12, pady=(6,10))
        self.canvas = tk.Canvas(self, bg="#ffffff", highlightthickness=0); self.canvas.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)
        self._fill_top_bar(); self._fill_bottom_bar()
        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<Double-Button-1>", self.on_double_left)
        self.canvas.bind("<Motion>", self.on_motion)

    def _btn(self, parent, text, cmd, kind="outlined"):
        style = {"filled":"MD.Filled.TButton","outlined":"MD.Outlined.TButton","text":"MD.Text.TButton"}[kind]
        b = ttk.Button(parent, text=text, style=style, command=cmd); b.pack(side=tk.LEFT, padx=6, pady=2); return b
    def _sep_v(self, parent): ttk.Separator(parent, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=12)

    def _fill_top_bar(self) -> None:
        left = ttk.Frame(self.top_bar, style="Panel.TFrame"); left.pack(side=tk.LEFT)
        self._btn(left, "Выделение", self.set_mode_select, "text")
        self._btn(left, "Рисовать", self.set_mode_draw, "text")
        self._btn(left, "Завершить", self.finish_polygon, "text")
        self._btn(left, "Отменить", self.cancel_polygon, "text")
        self._sep_v(self.top_bar)
        mid = ttk.Frame(self.top_bar, style="Panel.TFrame"); mid.pack(side=tk.LEFT)
        self._btn(mid, "Смещение", self.transform_translate, "filled")
        self._btn(mid, "Поворот (точка)", self.transform_rotate_point, "filled")
        self._btn(mid, "Поворот (центр)", self.transform_rotate_center, "filled")
        self._btn(mid, "Масштаб (точка)", self.transform_scale_point, "filled")
        self._btn(mid, "Масштаб (центр)", self.transform_scale_center, "filled")
        self._sep_v(self.top_bar)
        right = ttk.Frame(self.top_bar, style="Panel.TFrame"); right.pack(side=tk.LEFT)
        self._btn(right, "Очистить сцену", self.clear_scene, "outlined")

    def _fill_bottom_bar(self) -> None:
        left = ttk.Frame(self.bottom_bar, style="Panel.TFrame"); left.pack(side=tk.LEFT)
        self._btn(left, "Пересечение рёбер", self.set_mode_intersections, "outlined")
        self._btn(left, "Точка ∈ полигону", self.set_mode_point_in_poly, "outlined")
        self._btn(left, "Точка vs ребро", self.set_mode_point_vs_edge, "outlined")
        self._btn(left, "Очистить метки", self.clear_markers, "text")

    # ---------- ЛОГИКА ----------
    def update_status(self, msg: str) -> None: self.status_var.set(msg)

    def clear_markers(self) -> None:
        for it in getattr(self, "preview_items", []):
            try: self.canvas.delete(it)
            except Exception: pass
        self.preview_items = []
        # также чистим временные маркеры пересечения
        for it in self._temp_inter_markers:
            try: self.canvas.delete(it)
            except Exception: pass
        self._temp_inter_markers.clear()
        self.inter_state.update({"a1": None, "a2": None, "b1": None, "b2": None, "line_b_id": None})
        self.update_status("Метки очищены.")

    def clear_scene(self) -> None:
        self.canvas.delete("all")
        self.polys.clear(); self.current_poly = None; self.selected_poly = None
        self.preview_items = []
        self._temp_inter_markers.clear()
        self.inter_state.update({"a1": None, "a2": None, "b1": None, "b2": None, "line_b_id": None})
        self.pvr_state = {"e1": None, "e2": None}
        self.update_status("Сцена очищена.")

    # --- Режимы ---
    def set_mode_select(self) -> None:
        self.mode = "select"; self.update_status("Режим: выделение.")

    def set_mode_draw(self) -> None:
        self.mode = "draw"
        self.current_poly = PolyShape()
        self.polys.append(self.current_poly)
        self.update_status("Режим: рисование. ЛКМ — добавить вершину; двойной щелчок — завершить; Esc — отмена.")

    def set_mode_intersections(self) -> None:
        self.mode = "inter"
        self.inter_state.update({"a1": None, "a2": None, "b1": None, "b2": None, "line_b_id": None})
        self.update_status("Пересечение рёбер: A1, A2; затем B1, ведите мышь и клик для B2.")

    def set_mode_point_in_poly(self) -> None:
        self.mode = "pip"; self.update_status("Точка ∈ полигону: кликайте по сцене.")

    def set_mode_point_vs_edge(self) -> None:
        self.mode = "pvr"; self.pvr_state = {"e1": None, "e2": None}
        self.update_status("Точка vs ребро: кликните E1, E2; затем кликайте точку.")

    # --- Мышь ---
    def on_left_click(self, ev) -> None:
        p = (ev.x, ev.y)
        if self.mode == "draw":
            assert self.current_poly is not None
            verts = self.current_poly.vertices
            if len(verts) < 2:
                self.current_poly.vertices = verts + [p]
            else:
                self.current_poly.vertices = insert_point_locally(verts, p)
            self.current_poly.draw(self.canvas)
        elif self.mode == "select":
            self.select_by_point(p)
        elif self.mode == "inter":
            self.intersection_click(p)
        elif self.mode == "pip":
            self.point_in_poly_click(p)
        elif self.mode == "pvr":
            self.point_vs_edge_click(p)

    def on_double_left(self, ev) -> None:
        if self.mode == "draw": self.finish_polygon()

    def on_motion(self, ev) -> None:
        if self.mode == "inter": self.intersection_motion((ev.x, ev.y))

    def on_escape(self, *_args) -> None:
        if self.mode == "draw": self.cancel_polygon()

    # --- Выбор ---
    def select_by_point(self, p: Vec2) -> None:
        best: Optional[PolyShape] = None
        for poly in self.polys:
            verts = poly.vertices
            if len(verts) == 1 and dist2(p, verts[0]) <= 6*6: best = poly
            elif len(verts) == 2 and point_on_segment(p, verts[0], verts[1], eps=4.0): best = poly
            elif len(verts) >= 3 and point_in_polygon(p, verts) >= 0: best = poly
        if best is None:
            if self.selected_poly: self.selected_poly.highlight(self.canvas, False); self.selected_poly = None
            self.update_status("Ничего не выбрано."); return
        if self.selected_poly and self.selected_poly is not best: self.selected_poly.highlight(self.canvas, False)
        self.selected_poly = best; best.highlight(self.canvas, True)
        self.update_status("Полигон выбран.")

    # --- Рисование ---
    def finish_polygon(self) -> None:
        if self.mode != "draw": return
        if self.current_poly is None or not self.current_poly.vertices:
            self.update_status("Нечего завершать."); return
        self.current_poly = None; self.mode = "select"
        self.update_status("Полигон завершён. Режим: выделение.")

    def cancel_polygon(self) -> None:
        if self.mode != "draw": return
        if self.current_poly is not None:
            self.current_poly.erase(self.canvas)
            try: self.polys.remove(self.current_poly)
            except ValueError: pass
            self.current_poly = None
        self.mode = "select"; self.update_status("Рисование отменено. Режим: выделение.")

    # --- Маркеры/превью ---
    def add_marker(self, p: Vec2, color: str, size: int = 4) -> None:
        x,y = p
        it = self.canvas.create_oval(x-size, y-size, x+size, y+size, outline=color, width=2)
        self.preview_items.append(it)

    def add_line_preview(self, a: Vec2, b: Vec2, color: str = "#9aa0a6", dash=(4,2)) -> int:
        it = self.canvas.create_line(a[0], a[1], b[0], b[1], fill=color, dash=dash, width=1)
        self.preview_items.append(it); return it

    def update_line_item(self, item_id: int, a: Vec2, b: Vec2) -> None:
        """Вернул функцию: безопасное обновление координат линии по id."""
        try:
            self.canvas.coords(item_id, a[0], a[1], b[0], b[1])
        except Exception:
            pass

    # --- Пересечения рёбер ---
    def intersection_click(self, p: Vec2) -> None:
        st = self.inter_state
        if st["a1"] is None:
            st["a1"] = p; self.update_status("A1 зафиксирована. Выберите A2.")
        elif st["a2"] is None:
            st["a2"] = p
            self.add_line_preview(st["a1"], st["a2"], color="#9aa0a6", dash=(2,2))
            self.update_status("A готово. Выберите B1.")
        elif st["b1"] is None:
            st["b1"] = p; st["b2"] = p
            line_id = self.add_line_preview(p, p, color="#d32f2f", dash=())
            st["line_b_id"] = line_id
            self.update_status("Ведите мышь — задаётся B2. Кликните для фиксации.")
        else:
            st["b2"] = p
            self.finalize_intersection_case()
            self.inter_state.update({"a1": None, "a2": None, "b1": None, "b2": None, "line_b_id": None})
            self.update_status("Готово. Повторите: A1, A2, затем B1 и B2.")

    def _clear_temp_inter_markers(self):
        for it in self._temp_inter_markers:
            try: self.canvas.delete(it)
            except Exception: pass
        self._temp_inter_markers.clear()

    def intersection_motion(self, p: Vec2) -> None:
        st = self.inter_state
        if st["a1"] and st["a2"] and st["b1"] and st["line_b_id"]:
            # Обновляем линию B по старой утилите
            self.update_line_item(st["line_b_id"], st["b1"], p)
            # Пересчёт точки пересечения
            self._clear_temp_inter_markers()
            ok, ip = segments_intersect(st["a1"], st["a2"], st["b1"], p)
            if ok and ip:
                self._temp_inter_markers.append(
                    self.canvas.create_oval(ip[0]-3, ip[1]-3, ip[0]+3, ip[1]+3, outline="#188038", width=2)
                )

    def finalize_intersection_case(self) -> None:
        st = self.inter_state; a1,a2,b1,b2 = st["a1"],st["a2"],st["b1"],st["b2"]
        if None in (a1,a2,b1,b2): return
        ok, ip = segments_intersect(a1,a2,b1,b2)
        if ok and ip:
            self.add_marker(ip, "#188038", 5); self.update_status("Пересечение.")
        else:
            self.update_status("Отрезки не пересекаются.")

    # --- Точка ∈ полигону ---
    def point_in_poly_click(self, p: Vec2) -> None:
        hit_any = False
        for poly in self.polys:
            res = point_in_polygon(p, poly.vertices)
            if res == 1: self.add_marker(p, "#188038", 5); hit_any = True
            elif res == 0: self.add_marker(p, "#f29900", 5); hit_any = True
        if not hit_any: self.add_marker(p, "#d32f2f", 3)

    # --- Точка vs ребро ---
    def point_vs_edge_click(self, p: Vec2) -> None:
        st = self.pvr_state
        if st["e1"] is None:
            st["e1"] = p; self.update_status("Укажите E2 — направленное ребро.")
        elif st["e2"] is None:
            st["e2"] = p; self._draw_oriented_edge(st["e1"], st["e2"])
            self.update_status("Теперь кликайте точку — классифицируем (слева/справа/на линии).")
        else:
            e1, e2 = st["e1"], st["e2"]
            val = orient(e1, e2, p)
            if abs(val) < 1e-9:
                color = "#f29900"; txt = "НА ЛИНИИ"
            elif val > 0:
                color = "#d32f2f"; txt = "СПРАВА"  # Y вниз
            else:
                color = "#188038"; txt = "СЛЕВА"
            self.add_marker(p, color, 5); self._label_at(p, txt, color)

    def _draw_oriented_edge(self, a: Vec2, b: Vec2) -> None:
        it = self.add_line_preview(a, b, color="#1a73e8", dash=(3,1))
        vx, vy = (b[0]-a[0], b[1]-a[1]); L = max(1e-9, (vx*vx + vy*vy)**0.5)
        ux, uy = vx/L, vy/L; s = 12.0
        left = (b[0] - s*(ux + 0.5*uy), b[1] - s*(uy - 0.5*ux))
        right = (b[0] - s*(ux - 0.5*uy), b[1] - s*(uy + 0.5*ux))
        self.preview_items.append(self.canvas.create_polygon(
            b[0], b[1], left[0], left[1], right[0], right[1],
            outline="#1a73e8", fill=""))

    def _label_at(self, p: Vec2, text: str, color: str) -> None:
        it = self.canvas.create_text(p[0]+10, p[1]-12, text=text, fill=color,
                                     font=("TkDefaultFont", max(11, int(11*self.ui_scale)), "bold"))
        self.preview_items.append(it)

    # --- Аффинные преобразования ---
    def need_selected(self) -> Optional[PolyShape]:
        if self.selected_poly is None:
            messagebox.showinfo("Нет выделения", "Сначала выделите полигон кликом."); return None
        return self.selected_poly

    def transform_translate(self) -> None:
        poly = self.need_selected()
        if not poly: return
        try:
            dx = float(simpledialog.askstring("Смещение", "dx:", parent=self) or "0")
            dy = float(simpledialog.askstring("Смещение", "dy:", parent=self) or "0")
        except Exception:
            messagebox.showerror("Ошибка", "Введите корректные числа."); return
        M = T(dx, dy); poly.apply_matrix(M); poly.draw(self.canvas)
        self.update_status(f"Смещение на ({dx}, {dy}) выполнено.")

    def transform_rotate_point(self) -> None:
        poly = self.need_selected()
        if not poly: return
        messagebox.showinfo("Поворот вокруг точки", "Кликните точку поворота на холсте.")
        self.mode, old_mode = "rotate_point_pick", self.mode
        self._rotate_prev_mode = old_mode
        def on_click(ev):
            px, py = ev.x, ev.y
            try:
                angle = float(simpledialog.askstring("Угол (градусы)", "Против часовой стрелки:", parent=self) or "0")
            except Exception:
                messagebox.showerror("Ошибка", "Введите корректный угол.")
                self.mode = self._rotate_prev_mode; self.canvas.unbind("<Button-1>", bid); return
            M = around_point(R(angle), px, py)
            poly.apply_matrix(M); poly.draw(self.canvas)
            self.update_status(f"Поворот на {angle}° вокруг точки ({px:.1f},{py:.1f}).")
            self.mode = self._rotate_prev_mode; self.canvas.unbind("<Button-1>", bid)
        bid = self.canvas.bind("<Button-1>", on_click, add="+")

    def transform_rotate_center(self) -> None:
        poly = self.need_selected()
        if not poly: return
        try:
            angle = float(simpledialog.askstring("Угол (градусы)", "Против часовой стрелки:", parent=self) or "0")
        except Exception:
            messagebox.showerror("Ошибка", "Введите корректный угол."); return
        cx, cy = self._centroid_safe(poly.vertices)
        M = around_point(R(angle), cx, cy)
        poly.apply_matrix(M); poly.draw(self.canvas)
        self.update_status(f"Поворот на {angle}° вокруг центра ({cx:.1f},{cy:.1f}).")

    def transform_scale_point(self) -> None:
        poly = self.need_selected()
        if not poly: return
        messagebox.showinfo("Масштаб относительно точки", "Кликните опорную точку.")
        self.mode, old_mode = "scale_point_pick", self.mode; self._scale_prev_mode = old_mode
        def on_click(ev):
            px, py = ev.x, ev.y
            try:
                sx = float(simpledialog.askstring("Масштаб по X", "sx:", parent=self) or "1")
                sy = float(simpledialog.askstring("Масштаб по Y", "sy:", parent=self) or "1")
            except Exception:
                messagebox.showerror("Ошибка", "Введите корректные коэффициенты.")
                self.mode = self._scale_prev_mode; self.canvas.unbind("<Button-1>", bid); return
            M = around_point(S(sx, sy), px, py)
            poly.apply_matrix(M); poly.draw(self.canvas)
            self.update_status(f"Масштаб ({sx}, {sy}) относительно точки ({px:.1f},{py:.1f}).")
            self.mode = self._scale_prev_mode; self.canvas.unbind("<Button-1>", bid)
        bid = self.canvas.bind("<Button-1>", on_click, add="+")

    def transform_scale_center(self) -> None:
        poly = self.need_selected()
        if not poly: return
        try:
            sx = float(simpledialog.askstring("Масштаб по X", "sx:", parent=self) or "1")
            sy = float(simpledialog.askstring("Масштаб по Y", "sy:", parent=self) or "1")
        except Exception:
            messagebox.showerror("Ошибка", "Введите корректные коэффициенты."); return
        cx, cy = self._centroid_safe(poly.vertices)
        M = around_point(S(sx, sy), cx, cy)
        poly.apply_matrix(M); poly.draw(self.canvas)
        self.update_status(f"Масштаб ({sx}, {sy}) относительно центра ({cx:.1f},{cy:.1f}).")

    def _centroid_safe(self, verts: List[Vec2]) -> Vec2:
        n = len(verts)
        if n == 0: return (0.0, 0.0)
        if n == 1: return verts[0]
        if n == 2: return ((verts[0][0]+verts[1][0])/2.0, (verts[0][1]+verts[1][1])/2.0)
        A = 0.0; cx = 0.0; cy = 0.0
        for i in range(n):
            x1,y1 = verts[i]; x2,y2 = verts[(i+1)%n]
            cross = x1*y2 - x2*y1
            A += cross
            cx += (x1+x2) * cross; cy += (y1+y2) * cross
        if abs(A) < 1e-9:
            return (sum(x for x,_ in verts)/n, sum(y for _,y in verts)/n)
        A *= 0.5; cx /= (6*A); cy /= (6*A)
        return (cx, cy)

# ----------------------------- MAIN -----------------------------

def main():
    app = PolyEditor()
    app.mainloop()

if __name__ == "__main__":
    main()
