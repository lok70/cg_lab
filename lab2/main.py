import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageTk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def ensure_rgb(img: Image.Image) -> Image.Image:
    if img.mode != "RGB":
        return img.convert("RGB")
    return img

def np_from_image(img: Image.Image) -> np.ndarray:
    return np.asarray(img, dtype=np.uint8)

def image_from_np(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    elif arr.ndim == 3 and arr.shape[2] == 3:
        return Image.fromarray(arr, mode="RGB")
    else:
        raise ValueError("Ожидался массив формы (H,W) или (H,W,3).")

def histogram_uint8(arr: np.ndarray, bins: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    counts, bin_edges = np.histogram(arr.flatten(), bins=bins, range=(0, 255))
    return counts, bin_edges

def to_gray_bt601(rgb: np.ndarray) -> np.ndarray:
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(y, 0, 255).astype(np.uint8)

def to_gray_bt709(rgb: np.ndarray) -> np.ndarray:
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return np.clip(y, 0, 255).astype(np.uint8)

def rgb_to_hsv_pil(rgb_img: Image.Image) -> Image.Image:
    return rgb_img.convert("HSV")

def hsv_to_rgb_pil(hsv_img: Image.Image) -> Image.Image:
    return hsv_img.convert("RGB")

def adjust_hsv(hsv_arr: np.ndarray, hue_shift_deg: float, s_scale: float, v_scale: float) -> np.ndarray:
    hsv = hsv_arr.astype(np.int32).copy()
    shift_units = int(round((hue_shift_deg % 360) * 255 / 360.0))
    hsv[:, :, 0] = (hsv[:, :, 0] + shift_units) % 256
    hsv[:, :, 1] = np.clip(np.round(hsv[:, :, 1] * s_scale), 0, 255)
    hsv[:, :, 2] = np.clip(np.round(hsv[:, :, 2] * v_scale), 0, 255)
    return hsv.astype(np.uint8)


class AutoImageLabel(ttk.Label):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self._src_img: Optional[Image.Image] = None
        self._tk_img: Optional[ImageTk.PhotoImage] = None
        self._resize_job: Optional[str] = None
        self.bind("<Configure>", self._on_configure)

    def set_image(self, img: Image.Image):
        self._src_img = img.copy()
        self._render()

    def _on_configure(self, _event=None):
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(40, self._render)

    def _render(self):
        self._resize_job = None
        if self._src_img is None:
            return
        w = max(self.winfo_width(), 1)
        h = max(self.winfo_height(), 1)
        if w <= 2 or h <= 2:
            return
        src_w, src_h = self._src_img.size
        scale = min(w / src_w, h / src_h)
        new_w = max(1, int(src_w * scale))
        new_h = max(1, int(src_h * scale))
        resized = self._src_img.resize((new_w, new_h), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(resized)
        self.configure(image=self._tk_img)


class ImageLabApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Лаборатория изображений (RGB/Gray/HSV) — Tkinter + Pillow")
        self.geometry("1300x820")
        self.minsize(1100, 720)
        self.original_img: Optional[Image.Image] = None
        self.current_rgb_img: Optional[Image.Image] = None
        self.current_hsv_img: Optional[Image.Image] = None
        self._build_menu()
        self._build_tabs()

    def _build_menu(self):
        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Открыть изображение…", command=self.open_image)
        filemenu.add_command(label="Сохранить результат (активная вкладка)…", command=self.save_active_result)
        filemenu.add_separator()
        filemenu.add_command(label="Выход", command=self.quit)
        menubar.add_cascade(label="Файл", menu=filemenu)
        self.config(menu=menubar)

    def _build_tabs(self):
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True)
        self.tab_gray = ttk.Frame(self.nb)
        self.tab_rgb  = ttk.Frame(self.nb)
        self.tab_hsv  = ttk.Frame(self.nb)
        self.nb.add(self.tab_gray, text="Оттенки серого + гистограммы")
        self.nb.add(self.tab_rgb,  text="Каналы R/G/B + гистограммы")
        self.nb.add(self.tab_hsv,  text="HSV: ползунки и сохранение")
        self._build_tab_gray(self.tab_gray)
        self._build_tab_rgb(self.tab_rgb)
        self._build_tab_hsv(self.tab_hsv)

    def open_image(self):
        path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[
                ("Изображения", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"),
                ("Все файлы", "*.*")
            ]
        )
        if not path:
            return
        try:
            img = Image.open(path)
            img.load()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть файл:\n{e}")
            return
        img = ensure_rgb(img)
        self.original_img = img
        self.current_rgb_img = img.copy()
        self.current_hsv_img = rgb_to_hsv_pil(self.current_rgb_img)
        self.refresh_gray_tab()
        self.refresh_rgb_tab()
        self.refresh_hsv_tab()

    def save_active_result(self):
        idx = self.nb.index(self.nb.select())
        if idx == 0:
            self._save_gray_collage()
        elif idx == 1:
            self._save_rgb_collage()
        elif idx == 2:
            self._save_hsv_result()
        else:
            messagebox.showinfo("Сохранение", "Нет активного результата для сохранения.")

    def _build_tab_gray(self, root: ttk.Frame):
        frm_top = ttk.Frame(root)
        frm_top.pack(side="top", fill="both", expand=True, padx=6, pady=6)
        for c in range(4):
            frm_top.grid_columnconfigure(c, weight=1, uniform="cols")
        frm_top.grid_rowconfigure(0, weight=0)
        frm_top.grid_rowconfigure(1, weight=1)
        self.lbl_orig = ttk.Label(frm_top, text="Исходное изображение")
        self.lbl_g1   = ttk.Label(frm_top, text="Серый (BT.601)")
        self.lbl_g2   = ttk.Label(frm_top, text="Серый (BT.709)")
        self.lbl_diff = ttk.Label(frm_top, text="Разность |BT.601 - BT.709|")
        self.lbl_orig.grid(row=0, column=0, padx=5, pady=(0, 2), sticky="w")
        self.lbl_g1.grid  (row=0, column=1, padx=5, pady=(0, 2), sticky="w")
        self.lbl_g2.grid  (row=0, column=2, padx=5, pady=(0, 2), sticky="w")
        self.lbl_diff.grid(row=0, column=3, padx=5, pady=(0, 2), sticky="w")
        self.canvas_orig = AutoImageLabel(frm_top)
        self.canvas_g1   = AutoImageLabel(frm_top)
        self.canvas_g2   = AutoImageLabel(frm_top)
        self.canvas_diff = AutoImageLabel(frm_top)
        self.canvas_orig.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.canvas_g1.grid  (row=1, column=1, padx=5, pady=5, sticky="nsew")
        self.canvas_g2.grid  (row=1, column=2, padx=5, pady=5, sticky="nsew")
        self.canvas_diff.grid(row=1, column=3, padx=5, pady=5, sticky="nsew")
        frm_bottom = ttk.Frame(root)
        frm_bottom.pack(side="top", fill="both", expand=True, padx=6, pady=(0, 6))
        self.fig_hist1 = Figure(figsize=(5.0, 2.8), dpi=100)
        self.ax_hist1 = self.fig_hist1.add_subplot(111)
        self.ax_hist1.set_title("Гистограмма (BT.601)")
        self.ax_hist1.set_xlabel("Интенсивность (0..255)")
        self.ax_hist1.set_ylabel("Частота")
        self.fig_hist2 = Figure(figsize=(5.0, 2.8), dpi=100)
        self.ax_hist2 = self.fig_hist2.add_subplot(111)
        self.ax_hist2.set_title("Гистограмма (BT.709)")
        self.ax_hist2.set_xlabel("Интенсивность (0..255)")
        self.ax_hist2.set_ylabel("Частота")
        self.canvas_hist1 = FigureCanvasTkAgg(self.fig_hist1, master=frm_bottom)
        self.canvas_hist2 = FigureCanvasTkAgg(self.fig_hist2, master=frm_bottom)
        w1 = self.canvas_hist1.get_tk_widget()
        w2 = self.canvas_hist2.get_tk_widget()
        w1.pack(side="left", fill="both", expand=True, padx=5)
        w2.pack(side="left", fill="both", expand=True, padx=5)

    def refresh_gray_tab(self):
        if self.original_img is None:
            return
        rgb = np_from_image(self.original_img)
        g1 = to_gray_bt601(rgb)
        g2 = to_gray_bt709(rgb)
        diff = np.abs(g1.astype(np.int16) - g2.astype(np.int16)).astype(np.uint8)
        self.canvas_orig.set_image(self.original_img)
        self.canvas_g1.set_image(image_from_np(g1))
        self.canvas_g2.set_image(image_from_np(g2))
        self.canvas_diff.set_image(image_from_np(diff))
        self.ax_hist1.clear()
        self.ax_hist1.set_title("Гистограмма (BT.601)")
        c1, x1 = histogram_uint8(g1)
        self.ax_hist1.bar(x1[:-1], c1, width=1)
        self.ax_hist1.set_xlim(0, 255)
        self.ax_hist1.set_xlabel("Интенсивность (0..255)")
        self.ax_hist1.set_ylabel("Частота")
        self.canvas_hist1.draw_idle()
        self.ax_hist2.clear()
        self.ax_hist2.set_title("Гистограмма (BT.709)")
        c2, x2 = histogram_uint8(g2)
        self.ax_hist2.bar(x2[:-1], c2, width=1)
        self.ax_hist2.set_xlim(0, 255)
        self.ax_hist2.set_xlabel("Интенсивность (0..255)")
        self.ax_hist2.set_ylabel("Частота")
        self.canvas_hist2.draw_idle()
        self._gray_cache = {"g1": g1, "g2": g2, "diff": diff}

    def _save_gray_collage(self):
        if not hasattr(self, "_gray_cache"):
            messagebox.showinfo("Сохранение", "Нет результата для сохранения.")
            return
        g1 = image_from_np(self._gray_cache["g1"])
        g2 = image_from_np(self._gray_cache["g2"])
        diff = image_from_np(self._gray_cache["diff"])
        imgs = [g1, g2, diff]
        h = max(im.height for im in imgs)
        w_sum = sum(im.width for im in imgs)
        collage = Image.new("L", (w_sum, h), color=0)
        x = 0
        for im in imgs:
            collage.paste(im, (x, 0))
            x += im.width
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg"), ("Все файлы", "*.*")],
            title="Сохранить коллаж (BT.601 | BT.709 | Разность)"
        )
        if not path:
            return
        try:
            if path.lower().endswith((".jpg", ".jpeg")):
                collage.convert("RGB").save(path, quality=95)
            else:
                collage.save(path)
            messagebox.showinfo("Готово", f"Коллаж сохранён:\n{path}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить:\n{e}")

    def _build_tab_rgb(self, root: ttk.Frame):
        frm_top = ttk.Frame(root)
        frm_top.pack(side="top", fill="both", expand=True, padx=6, pady=6)
        for c in range(4):
            frm_top.grid_columnconfigure(c, weight=1, uniform="cols")
        frm_top.grid_rowconfigure(0, weight=0)
        frm_top.grid_rowconfigure(1, weight=1)
        self.lbl_rgb_orig = ttk.Label(frm_top, text="Исходное (RGB)")
        self.lbl_r = ttk.Label(frm_top, text="Канал R")
        self.lbl_g = ttk.Label(frm_top, text="Канал G")
        self.lbl_b = ttk.Label(frm_top, text="Канал B")
        self.lbl_rgb_orig.grid(row=0, column=0, padx=5, pady=(0, 2), sticky="w")
        self.lbl_r.grid(row=0, column=1, padx=5, pady=(0, 2), sticky="w")
        self.lbl_g.grid(row=0, column=2, padx=5, pady=(0, 2), sticky="w")
        self.lbl_b.grid(row=0, column=3, padx=5, pady=(0, 2), sticky="w")
        self.canvas_rgb_orig = AutoImageLabel(frm_top)
        self.canvas_r = AutoImageLabel(frm_top)
        self.canvas_g = AutoImageLabel(frm_top)
        self.canvas_b = AutoImageLabel(frm_top)
        self.canvas_rgb_orig.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.canvas_r.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        self.canvas_g.grid(row=1, column=2, padx=5, pady=5, sticky="nsew")
        self.canvas_b.grid(row=1, column=3, padx=5, pady=5, sticky="nsew")
        frm_bottom = ttk.Frame(root)
        frm_bottom.pack(side="top", fill="both", expand=True, padx=6, pady=(0, 6))
        self.fig_hist_rgb = Figure(figsize=(6.8, 3.2), dpi=100)
        self.ax_hist_rgb = self.fig_hist_rgb.add_subplot(111)
        self.ax_hist_rgb.set_title("Гистограммы R/G/B")
        self.ax_hist_rgb.set_xlabel("Интенсивность (0..255)")
        self.ax_hist_rgb.set_ylabel("Частота")
        self.canvas_hist_rgb = FigureCanvasTkAgg(self.fig_hist_rgb, master=frm_bottom)
        self.canvas_hist_rgb.get_tk_widget().pack(side="left", fill="both", expand=True, padx=5)

    def refresh_rgb_tab(self):
        if self.current_rgb_img is None:
            return
        img = self.current_rgb_img
        arr = np_from_image(img)
        r = arr[:, :, 0]
        g = arr[:, :, 1]
        b = arr[:, :, 2]
        r_rgb = np.zeros_like(arr)
        r_rgb[:, :, 0] = r
        g_rgb = np.zeros_like(arr)
        g_rgb[:, :, 1] = g
        b_rgb = np.zeros_like(arr)
        b_rgb[:, :, 2] = b
        self.canvas_rgb_orig.set_image(img)
        self.canvas_r.set_image(image_from_np(r_rgb))
        self.canvas_g.set_image(image_from_np(g_rgb))
        self.canvas_b.set_image(image_from_np(b_rgb))
        self.ax_hist_rgb.clear()
        self.ax_hist_rgb.set_title("Гистограммы R/G/B")
        cr, xr = histogram_uint8(r)
        cg, xg = histogram_uint8(g)
        cb, xb = histogram_uint8(b)
        self.ax_hist_rgb.plot(xr[:-1], cr, label="R")
        self.ax_hist_rgb.plot(xg[:-1], cg, label="G")
        self.ax_hist_rgb.plot(xb[:-1], cb, label="B")
        self.ax_hist_rgb.set_xlim(0, 255)
        self.ax_hist_rgb.set_xlabel("Интенсивность (0..255)")
        self.ax_hist_rgb.set_ylabel("Частота")
        self.ax_hist_rgb.legend()
        self.canvas_hist_rgb.draw_idle()
        self._rgb_cache = {"r_rgb": r_rgb, "g_rgb": g_rgb, "b_rgb": b_rgb}

    def _save_rgb_collage(self):
        if not hasattr(self, "_rgb_cache"):
            messagebox.showinfo("Сохранение", "Нет результата для сохранения.")
            return
        r_img = image_from_np(self._rgb_cache["r_rgb"])
        g_img = image_from_np(self._rgb_cache["g_rgb"])
        b_img = image_from_np(self._rgb_cache["b_rgb"])
        imgs = [r_img, g_img, b_img]
        h = max(im.height for im in imgs)
        w_sum = sum(im.width for im in imgs)
        collage = Image.new("RGB", (w_sum, h), color=(0, 0, 0))
        x = 0
        for im in imgs:
            collage.paste(im, (x, 0))
            x += im.width
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg"), ("Все файлы", "*.*")],
            title="Сохранить коллаж каналов (R | G | B)"
        )
        if not path:
            return
        try:
            if path.lower().endswith((".jpg", ".jpeg")):
                collage.save(path, quality=95)
            else:
                collage.save(path)
            messagebox.showinfo("Готово", f"Коллаж сохранён:\n{path}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить:\n{e}")

    def _build_tab_hsv(self, root: ttk.Frame):
        frm_left = ttk.Frame(root)
        frm_left.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        frm_left.grid_columnconfigure(0, weight=1)
        frm_left.grid_rowconfigure(0, weight=0)
        frm_left.grid_rowconfigure(1, weight=1)
        frm_left.grid_rowconfigure(2, weight=0)
        frm_left.grid_rowconfigure(3, weight=1)
        self.lbl_hsv_orig = ttk.Label(frm_left, text="Исходное (RGB)")
        self.lbl_hsv_orig.grid(row=0, column=0, sticky="w")
        self.canvas_hsv_orig = AutoImageLabel(frm_left)
        self.canvas_hsv_orig.grid(row=1, column=0, sticky="nsew", pady=(0, 8))
        self.lbl_hsv_res = ttk.Label(frm_left, text="Результат (после HSV-коррекции)")
        self.lbl_hsv_res.grid(row=2, column=0, sticky="w")
        self.canvas_hsv_res = AutoImageLabel(frm_left)
        self.canvas_hsv_res.grid(row=3, column=0, sticky="nsew")
        frm_right = ttk.Frame(root)
        frm_right.pack(side="left", fill="y", padx=6, pady=6)
        ttk.Label(frm_right, text="Сдвиг оттенка (Hue), градусы [-180..180]").pack(anchor="w")
        self.scale_hue = ttk.Scale(frm_right, from_=-180.0, to=180.0, orient="horizontal", command=self._on_hsv_change)
        self.scale_hue.set(0.0)
        self.scale_hue.pack(fill="x", pady=5)
        ttk.Label(frm_right, text="Насыщенность (S), множитель [0..2]").pack(anchor="w")
        self.scale_sat = ttk.Scale(frm_right, from_=0.0, to=2.0, orient="horizontal", command=self._on_hsv_change)
        self.scale_sat.set(1.0)
        self.scale_sat.pack(fill="x", pady=5)
        ttk.Label(frm_right, text="Яркость (V), множитель [0..2]").pack(anchor="w")
        self.scale_val = ttk.Scale(frm_right, from_=0.0, to=2.0, orient="horizontal", command=self._on_hsv_change)
        self.scale_val.set(1.0)
        self.scale_val.pack(fill="x", pady=5)
        btn_frame = ttk.Frame(frm_right)
        btn_frame.pack(fill="x", pady=(12, 0))
        ttk.Button(btn_frame, text="Сбросить (0°, S=1, V=1)", command=self.reset_hsv).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Сохранить результат…", command=self._save_hsv_result).pack(side="left", padx=4)

    def refresh_hsv_tab(self):
        if self.current_rgb_img is None:
            return
        self.canvas_hsv_orig.set_image(self.current_rgb_img)
        self._recompute_hsv_result()

    def _on_hsv_change(self, _=None):
        self._recompute_hsv_result()

    def _recompute_hsv_result(self):
        if self.current_rgb_img is None:
            return
        hue_shift = float(self.scale_hue.get())
        s_scale   = float(self.scale_sat.get())
        v_scale   = float(self.scale_val.get())
        hsv_img = rgb_to_hsv_pil(self.current_rgb_img)
        hsv_arr = np.asarray(hsv_img, dtype=np.uint8)
        hsv_adj = adjust_hsv(hsv_arr, hue_shift, s_scale, v_scale)
        hsv_img_adj = Image.fromarray(hsv_adj, mode="HSV")
        rgb_res = hsv_to_rgb_pil(hsv_img_adj)
        self._hsv_result_rgb = rgb_res
        self.canvas_hsv_res.set_image(rgb_res)

    def reset_hsv(self):
        self.scale_hue.set(0.0)
        self.scale_sat.set(1.0)
        self.scale_val.set(1.0)
        self._recompute_hsv_result()

    def _save_hsv_result(self):
        if not hasattr(self, "_hsv_result_rgb"):
            messagebox.showinfo("Сохранение", "Нет результата для сохранения.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg"), ("Все файлы", "*.*")],
            title="Сохранить результат HSV-коррекции (RGB)"
        )
        if not path:
            return
        try:
            img = self._hsv_result_rgb
            if path.lower().endswith((".jpg", ".jpeg")):
                img.save(path, quality=95)
            else:
                img.save(path)
            messagebox.showinfo("Готово", f"Изображение сохранено:\n{path}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить:\n{e}")


def main():
    try:
        app = ImageLabApp()
        app.mainloop()
    except Exception as ex:
        try:
            messagebox.showerror("Критическая ошибка", str(ex))
        except Exception:
            print("Критическая ошибка:", ex, file=sys.stderr)

if __name__ == "__main__":
    main()
