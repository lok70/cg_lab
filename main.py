import io
import math
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageTk

# Встроим matplotlib в Tkinter
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def pil_to_tk(img: Image.Image, max_side: int = 520) -> ImageTk.PhotoImage:
    """Аккуратно уменьшаем изображение под предпросмотр в GUI."""
    if img is None:
        return None
    im = img.copy()
    w, h = im.size
    scale = min(1.0, max_side / max(w, h)) if max(w, h) else 1.0
    if scale < 1.0:
        im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return ImageTk.PhotoImage(im)


def ensure_rgb(img: Image.Image) -> Image.Image:
    """Переводим любое входное изображение в RGB для единообразной обработки."""
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def np_from_image(img: Image.Image) -> np.ndarray:
    """Image -> ndarray uint8 (H, W, 3)."""
    return np.asarray(img, dtype=np.uint8)


def image_from_np(arr: np.ndarray) -> Image.Image:
    """ndarray (H, W) или (H, W, 3) -> Image (L или RGB)."""
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    elif arr.ndim == 3 and arr.shape[2] == 3:
        return Image.fromarray(arr, mode="RGB")
    else:
        raise ValueError("Ожидался массив формы (H,W) или (H,W,3).")


def histogram_uint8(arr: np.ndarray, bins: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """Гистограмма массива uint8. Возвращает (counts, bin_edges)."""
    counts, bin_edges = np.histogram(arr.flatten(), bins=bins, range=(0, 255))
    return counts, bin_edges



# Преобразование серого

def to_gray_bt601(rgb: np.ndarray) -> np.ndarray:
    """Яркость по ITU-R BT.601: Y = 0.299 R + 0.587 G + 0.114 B"""
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(y, 0, 255).astype(np.uint8)


def to_gray_bt709(rgb: np.ndarray) -> np.ndarray:
    """Яркость по ITU-R BT.709: Y = 0.2126 R + 0.7152 G + 0.0722 B"""
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return np.clip(y, 0, 255).astype(np.uint8)




class ImageLabApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Лаборатория изображений (RGB/Gray/HSV) — Tkinter + Pillow")
        self.geometry("1300x820")
        self.minsize(1100, 720)

        self.original_img: Optional[Image.Image] = None  # исходник в RGB
        self.current_rgb_img: Optional[Image.Image] = None  # для вкладок, где нужен RGB
        self.current_hsv_img: Optional[Image.Image] = None  # для вкладки HSV (PIL HSV)

        self._build_menu()
        self._build_tabs()

    # ----------------------- Верхнее меню ----------------------------
    def _build_menu(self):
        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Открыть изображение…", command=self.open_image)
        filemenu.add_command(label="Сохранить результат (активная вкладка)…", command=self.save_active_result)
        filemenu.add_separator()
        filemenu.add_command(label="Выход", command=self.quit)
        menubar.add_cascade(label="Файл", menu=filemenu)
        self.config(menu=menubar)

    # ----------------------- Вкладки Notebook ------------------------
    def _build_tabs(self):
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True)

        self.tab_gray = ttk.Frame(self.nb)
        self.tab_rgb = ttk.Frame(self.nb)
        self.tab_hsv = ttk.Frame(self.nb)

        self.nb.add(self.tab_gray, text="Оттенки серого + гистограммы")
        self.nb.add(self.tab_rgb, text="Каналы R/G/B + гистограммы")
        self.nb.add(self.tab_hsv, text="HSV: ползунки и сохранение")

        self._build_tab_gray(self.tab_gray)
        self._build_tab_rgb(self.tab_rgb)
        self._build_tab_hsv(self.tab_hsv)

    # -------------------- Общие действия -----------------------------
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

        # Обновляем все вкладки
        self.refresh_gray_tab()
        self.refresh_rgb_tab()
        self.refresh_hsv_tab()

    def save_active_result(self):
        idx = self.nb.index(self.nb.select())
        if idx == 0:
            # вкладка серого: сохраняем коллаж результатов (серый1, серый2, разность)
            self._save_gray_collage()
        elif idx == 1:
            # вкладка RGB: сохраняем коллаж каналов
            self._save_rgb_collage()
        elif idx == 2:
            # вкладка HSV: сохраняем текущий RGB после коррекции
            self._save_hsv_result()
        else:
            messagebox.showinfo("Сохранение", "Нет активного результата для сохранения.")

    # ------------------------ Вкладка 1: Gray -------------------------
    def _build_tab_gray(self, root):
        # Верхняя панель с картинками
        frm_top = ttk.Frame(root)
        frm_top.pack(side="top", fill="x", padx=6, pady=6)

        self.lbl_orig = ttk.Label(frm_top, text="Исходное изображение")
        self.lbl_g1 = ttk.Label(frm_top, text="Серый (BT.601)")
        self.lbl_g2 = ttk.Label(frm_top, text="Серый (BT.709)")
        self.lbl_diff = ttk.Label(frm_top, text="Разность |BT.601 - BT.709|")

        self.canvas_orig = ttk.Label(frm_top)
        self.canvas_g1 = ttk.Label(frm_top)
        self.canvas_g2 = ttk.Label(frm_top)
        self.canvas_diff = ttk.Label(frm_top)

        # Сетка 2x4 (подписи + картинки)
        self.lbl_orig.grid(row=0, column=0, padx=5, pady=(0,2))
        self.lbl_g1.grid(row=0, column=1, padx=5, pady=(0,2))
        self.lbl_g2.grid(row=0, column=2, padx=5, pady=(0,2))
        self.lbl_diff.grid(row=0, column=3, padx=5, pady=(0,2))

        self.canvas_orig.grid(row=1, column=0, padx=5, pady=5)
        self.canvas_g1.grid(row=1, column=1, padx=5, pady=5)
        self.canvas_g2.grid(row=1, column=2, padx=5, pady=5)
        self.canvas_diff.grid(row=1, column=3, padx=5, pady=5)

        # Нижняя панель с гистограммами
        frm_bottom = ttk.Frame(root)
        frm_bottom.pack(side="top", fill="both", expand=True, padx=6, pady=(0,6))

        # Две отдельные фигуры: для BT.601 и BT.709
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

        self.canvas_hist1.get_tk_widget().pack(side="left", fill="both", expand=True, padx=5)
        self.canvas_hist2.get_tk_widget().pack(side="left", fill="both", expand=True, padx=5)

    def refresh_gray_tab(self):
        if self.original_img is None:
            return
        rgb = np_from_image(self.original_img)
        g1 = to_gray_bt601(rgb)
        g2 = to_gray_bt709(rgb)
        diff = np.abs(g1.astype(np.int16) - g2.astype(np.int16)).astype(np.uint8)

        # Показываем изображения
        self.tk_orig = pil_to_tk(self.original_img)
        self.canvas_orig.configure(image=self.tk_orig)

        self.tk_g1 = pil_to_tk(image_from_np(g1))
        self.canvas_g1.configure(image=self.tk_g1)

        self.tk_g2 = pil_to_tk(image_from_np(g2))
        self.canvas_g2.configure(image=self.tk_g2)

        self.tk_diff = pil_to_tk(image_from_np(diff))
        self.canvas_diff.configure(image=self.tk_diff)

        # Гистограммы
        self.ax_hist1.clear()
        self.ax_hist1.set_title("Гистограмма (BT.601)")
        c1, x1 = histogram_uint8(g1)
        self.ax_hist1.bar(x1[:-1], c1, width=1)
        self.ax_hist1.set_xlim(0, 255)
        self.ax_hist1.set_xlabel("Интенсивность (0..255)")
        self.ax_hist1.set_ylabel("Частота")
        self.canvas_hist1.draw()

        self.ax_hist2.clear()
        self.ax_hist2.set_title("Гистограмма (BT.709)")
        c2, x2 = histogram_uint8(g2)
        self.ax_hist2.bar(x2[:-1], c2, width=1)
        self.ax_hist2.set_xlim(0, 255)
        self.ax_hist2.set_xlabel("Интенсивность (0..255)")
        self.ax_hist2.set_ylabel("Частота")
        self.canvas_hist2.draw()

        # Сохраняем для функции "сохранить коллаж"
        self._gray_cache = {
            "g1": g1, "g2": g2, "diff": diff
        }

    def _save_gray_collage(self):
        if not hasattr(self, "_gray_cache"):
            messagebox.showinfo("Сохранение", "Нет результата для сохранения.")
            return
        g1 = image_from_np(self._gray_cache["g1"])
        g2 = image_from_np(self._gray_cache["g2"])
        diff = image_from_np(self._gray_cache["diff"])

        # Делаем горизонтальный коллаж
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




def main():
    try:
        app = ImageLabApp()
        app.mainloop()
    except Exception as ex:
        messagebox.showerror("Критическая ошибка", str(ex))


if __name__ == "__main__":
    main()