import sys, math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

try:
    import tkinter as tk
    from tkinter import simpledialog, messagebox
except Exception:
    tk = None
    simpledialog = None
    messagebox = None

Vec2 = Tuple[float, float]

def to_np(points: List[Vec2]) -> np.ndarray:
    return np.array(points, dtype=np.float64)

def as_hom(coords: np.ndarray) -> np.ndarray:
    return np.vstack([coords.T, np.ones((1, coords.shape[0]))])

def from_hom(h: np.ndarray) -> np.ndarray:
    w = h[2, :]
    return (h[:2, :] / w).T

def T(dx: float, dy: float) -> np.ndarray:
    M = np.eye(3); M[0,2]=dx; M[1,2]=dy; return M

def R(theta_deg: float) -> np.ndarray:
    a = math.radians(theta_deg); c,s = math.cos(a), math.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float64)

def S(sx: float, sy: Optional[float]=None) -> np.ndarray:
    if sy is None: sy=sx
    return np.array([[sx,0,0],[0,sy,0],[0,0,1]], dtype=np.float64)

def centroid(poly: np.ndarray) -> Vec2:
    n = poly.shape[0]
    if n == 1:
        return (poly[0,0], poly[0,1])
    if n == 2:
        return (poly[:,0].mean(), poly[:,1].mean())
    x=poly[:,0]; y=poly[:,1]; x1=np.roll(x,-1); y1=np.roll(y,-1)
    a = x*y1 - x1*y; A=a.sum()/2.0
    if abs(A)<1e-12:
        return (poly[:,0].mean(), poly[:,1].mean())
    cx=((x+x1)*a).sum()/(6*A); cy=((y+y1)*a).sum()/(6*A); return (cx,cy)

def apply_affine(poly: np.ndarray, M: np.ndarray) -> np.ndarray:
    return (M @ as_hom(poly))[:2,:].T

def point_segment_distance(p: Vec2, a: Vec2, b: Vec2) -> float:
    ax,ay=a; bx,by=b; px,py=p
    abx,aby=bx-ax,by-ay; apx,apy=px-ax,py-ay
    ab2=abx*abx+aby*aby
    t=0.0 if ab2==0 else max(0.0, min(1.0, (apx*abx+apy*aby)/ab2))
    cx,cy=ax+t*abx, ay+t*aby; return math.hypot(px-cx, py-cy)

def segment_intersection(a: Vec2,b: Vec2,c: Vec2,d: Vec2):
    x1,y1=a; x2,y2=b; x3,y3=c; x4,y4=d
    denom=(x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
    if abs(denom)<1e-12: return (False,None)
    px=((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/denom
    py=((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/denom
    def between(a,b,c): return min(a,b)-1e-9<=c<=max(a,b)+1e-9
    if between(x1,x2,px) and between(y1,y2,py) and between(x3,x4,px) and between(y3,y4,py):
        return (True,(px,py))
    return (False,None)

def is_point_on_segment(p: Vec2, a: Vec2, b: Vec2, eps=1e-9)->bool:
    ax,ay=a; bx,by=b; px,py=p
    cross=(bx-ax)*(py-ay)-(by-ay)*(px-ax)
    if abs(cross)>eps: return False
    dot=(px-ax)*(px-bx)+(py-ay)*(py-by)
    return dot<=eps

def point_in_polygon(p: Vec2, poly: np.ndarray)->int:
    n=poly.shape[0]
    if n==0: return -1
    if n==1: return 0 if math.hypot(p[0]-poly[0,0], p[1]-poly[0,1])<1e-9 else -1
    for i in range(n):
        a=poly[i]; b=poly[(i+1)%n]
        if is_point_on_segment(p, tuple(a), tuple(b)): return 0
    cnt=0; x,y=p
    for i in range(n):
        x1,y1=poly[i]; x2,y2=poly[(i+1)%n]
        if ((y1>y)!=(y2>y)):
            xin=(x2-x1)*(y-y1)/(y2-y1+1e-300)+x1
            if xin> x: cnt+=1
    return 1 if (cnt%2)==1 else -1

def point_side_of_edge(p: Vec2, a: Vec2, b: Vec2, eps=1e-9)->int:
    val=(b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0])
    if val>eps: return 1
    if val<-eps: return -1
    return 0

@dataclass
class Poly:
    verts: np.ndarray
    color: Tuple[float,float,float]

@dataclass
class App:
    polys: List[Poly] = field(default_factory=list)
    selected_idx: Optional[int] = None
    mode: str = "select"
    draw_current: List[Vec2] = field(default_factory=list)

    edge_side_edge: Optional[Tuple[Vec2, Vec2]] = None
    intersect_fixed_edge: Optional[Tuple[Vec2, Vec2]] = None
    intersect_second_a: Optional[Vec2] = None
    intersect_second_b_live: Optional[Vec2] = None

    overlay_points_in: List[Vec2] = field(default_factory=list)
    overlay_points_out: List[Vec2] = field(default_factory=list)
    overlay_points_on: List[Vec2] = field(default_factory=list)
    overlay_edge_side_pts: List[Tuple[Vec2,int]] = field(default_factory=list)
    overlay_intersections: List[Vec2] = field(default_factory=list)
    overlay_free_segments: List[Tuple[Vec2,Vec2]] = field(default_factory=list)

    def current_poly(self)->Optional[Poly]:
        if self.selected_idx is None: return None
        if 0<=self.selected_idx<len(self.polys): return self.polys[self.selected_idx]
        return None

    def add_poly(self, pts: List[Vec2]):
        if not pts: return
        arr = to_np(pts)
        color = (0.15,0.35,0.75) if len(self.polys)%2==0 else (0.75,0.3,0.15)
        self.polys.append(Poly(arr, color))
        self.selected_idx = len(self.polys)-1

    def cycle_selection(self):
        if not self.polys:
            self.selected_idx=None; return
        self.selected_idx = 0 if self.selected_idx is None else (self.selected_idx+1)%len(self.polys)

    def select_by_point(self, p: Vec2, tol=6.0):
        if not self.polys: return
        best=None; bestd=1e9
        for i,poly in enumerate(self.polys):
            d=1e9
            if poly.verts.shape[0]==1:
                d=math.hypot(p[0]-poly.verts[0,0], p[1]-poly.verts[0,1])
            else:
                n=poly.verts.shape[0]
                for k in range(n if n>=3 else n-1):
                    a=tuple(poly.verts[k]); b=tuple(poly.verts[(k+1)%n] if n>=3 else poly.verts[k+1])
                    d=min(d, point_segment_distance(p,a,b))
            if poly.verts.shape[0]>=3 and point_in_polygon(p, poly.verts)>=0:
                d*=0.25
            if d<bestd: bestd=d; best=i
        if best is not None and bestd<=tol: self.selected_idx=best

    def fixed_edge_from_click(self, p: Vec2, tol=6.0):
        best=None; bestd=1e9
        for poly in self.polys:
            n=poly.verts.shape[0]
            if n<2: continue
            for k in range(n if n>=3 else n-1):
                a=tuple(poly.verts[k])
                b=tuple(poly.verts[(k+1)%n] if n>=3 else poly.verts[k+1])
                d=point_segment_distance(p,a,b)
                if d<bestd: bestd=d; best=(a,b)
        return best if (best is not None and bestd<=tol) else None

class View:
    def __init__(self, app: App):
        self.app=app
        self.fig,self.ax=plt.subplots(figsize=(9,7))
        self.ax.set_aspect('equal','box')
        self.ax.set_xlim(0,800); self.ax.set_ylim(0,600); self.ax.invert_yaxis(); self.ax.grid(True,alpha=0.2)
        self.help = ("N:New  V:Select  X:Clear  T:Translate  r/R:Rotate  k/K:Scale  "
                     "G:Point∈Poly  E:Point vs Edge  I:Edge Intersection  TAB:Next  ESC:Cancel  H:Help")
        self.title_ann=self.ax.set_title(self.help, fontsize=10)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.draw_all(); plt.show()

    def set_mode(self, m):
        self.app.mode=m; self.update_title()
    def update_title(self, extra=""):
        t=f"[{self.app.mode}] {self.help}"
        if extra: t+="  —  "+extra
        self.title_ann.set_text(t); self.fig.canvas.draw_idle()

    def draw_all(self):
        self.ax.cla()
        self.ax.set_aspect('equal','box')
        self.ax.set_xlim(0,800); self.ax.set_ylim(0,600); self.ax.invert_yaxis(); self.ax.grid(True,alpha=0.2)
        for i,poly in enumerate(self.app.polys):
            col=poly.color
            if poly.verts.shape[0]==1:
                x,y=poly.verts[0]; self.ax.plot([x],[y],'o',color=col,markersize=6)
            elif poly.verts.shape[0]==2:
                a,b=poly.verts[0],poly.verts[1]
                self.ax.plot([a[0],b[0]],[a[1],b[1]],color=col,linewidth=2)
                self.ax.plot([a[0],b[0]],[a[1],b[1]],'o',color=col,markersize=4)
            else:
                pg=MplPolygon(poly.verts,closed=True,fill=False,edgecolor=col,linewidth=2)
                self.ax.add_patch(pg)
                self.ax.plot(poly.verts[:,0],poly.verts[:,1],'o',color=col,markersize=4)
            if self.app.selected_idx==i:
                bx=[poly.verts[:,0].min(), poly.verts[:,0].max()]
                by=[poly.verts[:,1].min(), poly.verts[:,1].max()]
                self.ax.add_patch(matplotlib.patches.Rectangle((bx[0]-4,by[0]-4),
                    (bx[1]-bx[0])+8,(by[1]-by[0])+8, fill=False, edgecolor='orange', linewidth=1.5, linestyle='--'))
                c=centroid(poly.verts); self.ax.plot([c[0]],[c[1]],'s',color='orange',markersize=6)

        if self.app.mode=='draw' and self.app.draw_current:
            xs=[p[0] for p in self.app.draw_current]; ys=[p[1] for p in self.app.draw_current]
            self.ax.plot(xs,ys,'-o',color='gray')

        if self.app.overlay_points_in:
            xs,ys=zip(*self.app.overlay_points_in); self.ax.scatter(xs,ys,c='green',s=36,marker='o')
        if self.app.overlay_points_out:
            xs,ys=zip(*self.app.overlay_points_out); self.ax.scatter(xs,ys,c='red',s=36,marker='x')
        if self.app.overlay_points_on:
            xs,ys=zip(*self.app.overlay_points_on); self.ax.scatter(xs,ys,c='gold',s=60,marker='o',facecolors='none',edgecolors='gold')
        for pt,side in self.app.overlay_edge_side_pts:
            color='blue' if side==1 else ('purple' if side==-1 else 'black')
            self.ax.plot([pt[0]],[pt[1]],'o',color=color,markersize=5)
            self.ax.text(pt[0]+6,pt[1]-6,'L' if side==1 else ('R' if side==-1 else 'ON'),fontsize=9,color=color)

        if self.app.intersect_fixed_edge:
            a,b=self.app.intersect_fixed_edge
            self.ax.plot([a[0],b[0]],[a[1],b[1]],color='black',linewidth=2,linestyle='--')
        for a,b in self.app.overlay_free_segments:
            self.ax.plot([a[0],b[0]],[a[1],b[1]],color='gray',linewidth=1.5)
        if self.app.mode=='intersect' and self.app.intersect_second_a and self.app.intersect_second_b_live:
            a=self.app.intersect_second_a; b=self.app.intersect_second_b_live
            self.ax.plot([a[0],b[0]],[a[1],b[1]],color='gray',linewidth=1.5,linestyle=':')
        if self.app.overlay_intersections:
            xs,ys=zip(*self.app.overlay_intersections); self.ax.scatter(xs,ys,c='magenta',s=50,marker='+')

        self.update_title()

    def on_key(self,event):
        if event.key is None: return
        key=event.key
        if key=='h':
            self.update_title("Help refreshed.")
        elif key=='n':
            self.app.draw_current=[]; self.set_mode('draw')
            self.update_title("ЛКМ: вершина, ПКМ: завершить.")
        elif key=='v':
            self.set_mode('select'); self.update_title("Клик — выбрать полигон.")
        elif key=='tab':
            self.app.cycle_selection(); self.draw_all()
        elif key=='x':
            self.app.polys.clear(); self.app.selected_idx=None
            self.app.overlay_points_in.clear(); self.app.overlay_points_out.clear(); self.app.overlay_points_on.clear()
            self.app.overlay_edge_side_pts.clear(); self.app.overlay_intersections.clear(); self.app.overlay_free_segments.clear()
            self.app.intersect_fixed_edge=None; self.app.intersect_second_a=None; self.app.intersect_second_b_live=None
            self.draw_all()
        elif key=='t':
            self.affine_translate()
        elif key=='R':  # заглавная
            self.affine_rotate_center()
        elif key=='r':
            self.set_mode('rotate_pivot'); self.update_title("Клик опорной точки, затем угол.")
        elif key=='K':
            self.affine_scale_center()
        elif key=='k':
            self.set_mode('scale_pivot'); self.update_title("Клик опорной точки, затем s или sx,sy.")
        elif key=='g':
            self.set_mode('test_point'); self.update_title("Клики — проверка точки ∈ выбранному полигону.")
        elif key=='e':
            self.set_mode('edge_side'); self.app.edge_side_edge=None; self.update_title("Два клика — ребро; далее клики — L/R/ON.")
        elif key=='i':
            self.set_mode('intersect'); self.update_title("ПКМ — выбрать фиксированное ребро; ЛКМ×2 — второе ребро.")
        elif key=='escape':
            self.app.draw_current.clear(); self.app.edge_side_edge=None
            self.app.intersect_second_a=None; self.app.intersect_second_b_live=None
            if self.app.mode in ('rotate_pivot','scale_pivot','edge_side','edge_side_wait2','edge_side_ready','intersect'):
                self.set_mode('select')
            self.draw_all()

    def on_click(self,event):
        if event.inaxes!=self.ax: return
        p=(float(event.xdata), float(event.ydata))

        if self.app.mode=='draw':
            if event.button==1:
                self.app.draw_current.append(p); self.draw_all()
            elif event.button==3:
                if self.app.draw_current:
                    self.app.add_poly(self.app.draw_current.copy())
                    self.app.draw_current.clear(); self.set_mode('select'); self.draw_all()
            return

        if self.app.mode=='select':
            if event.button==1:
                self.app.select_by_point(p); self.draw_all()
            return

        if self.app.mode=='rotate_pivot' and event.button==1:
            ang=self.prompt_float("Угол поворота (градусы):","30")
            if ang is None:
                return
            poly=self.app.current_poly()
            if poly is None:
                return
            M=T(p[0],p[1])@R(ang)@T(-p[0],-p[1])
            poly.verts=apply_affine(poly.verts,M)
            self.set_mode('select'); self.draw_all()
            return

        if self.app.mode=='scale_pivot' and event.button==1:
            s_txt=self.prompt_text("Масштаб s или sx,sy:","1.2")
            if s_txt is None:
                return
            try:
                parts=[float(z.strip()) for z in s_txt.split(',')]
                if len(parts)==1: sx=sy=parts[0]
                else: sx,sy=parts[0],parts[1]
            except Exception:
                self.alert("Нужно s или sx,sy")
                return
            poly=self.app.current_poly()
            if poly is None:
                return
            M=T(p[0],p[1])@S(sx,sy)@T(-p[0],-p[1])
            poly.verts=apply_affine(poly.verts,M)
            self.set_mode('select'); self.draw_all()
            return

        if self.app.mode=='test_point' and event.button==1:
            poly=self.app.current_poly()
            if poly is None:
                return
            res=point_in_polygon(p, poly.verts)
            if res==1: self.app.overlay_points_in.append(p)
            elif res==-1: self.app.overlay_points_out.append(p)
            else: self.app.overlay_points_on.append(p)
            self.draw_all(); return

        if self.app.mode=='edge_side' and event.button==1:
            self.app.edge_side_edge=(p,p); self.set_mode('edge_side_wait2'); self.update_title("Ещё клик — вторая точка ребра."); return

        if self.app.mode=='edge_side_wait2' and event.button==1:
            a,_=self.app.edge_side_edge; b=p
            self.app.edge_side_edge=(a,b); self.set_mode('edge_side_ready'); self.update_title("Ребро зафиксировано. Клики — классификация."); self.draw_all(); return

        if self.app.mode=='edge_side_ready' and event.button==1:
            a,b=self.app.edge_side_edge; side=point_side_of_edge(p,a,b)
            self.app.overlay_edge_side_pts.append((p,side)); self.draw_all(); return

        if self.app.mode=='intersect':
            if event.button==3:
                edge=self.app.fixed_edge_from_click(p)
                if edge is not None:
                    self.app.intersect_fixed_edge=edge; self.app.intersect_second_a=None; self.app.intersect_second_b_live=None; self.draw_all()
                else:
                    self.update_title("Нет близкого ребра — кликните ближе.")
                return
            if event.button==1:
                if self.app.intersect_fixed_edge is None:
                    self.update_title("Сначала ПКМ выберите фиксированное ребро.")
                    return
                if self.app.intersect_second_a is None:
                    self.app.intersect_second_a=p; self.app.intersect_second_b_live=p; self.draw_all()
                else:
                    a=self.app.intersect_second_a; b=p
                    self.app.overlay_free_segments.append((a,b))
                    ok,pt=segment_intersection(a,b,*self.app.intersect_fixed_edge)
                    if ok and pt is not None: self.app.overlay_intersections.append(pt)
                    self.app.intersect_second_a=None; self.app.intersect_second_b_live=None; self.draw_all()
                return

    def on_move(self,event):
        if event.inaxes!=self.ax: return
        p=(float(event.xdata), float(event.ydata))
        if self.app.mode=='intersect' and self.app.intersect_second_a is not None:
            self.app.intersect_second_b_live=p; self.draw_all()
            if self.app.intersect_fixed_edge:
                ok,pt=segment_intersection(self.app.intersect_second_a, p, *self.app.intersect_fixed_edge)
                if ok and pt is not None:
                    self.ax.scatter([pt[0]],[pt[1]],c='magenta',s=40,marker='+')
                    self.fig.canvas.draw_idle()
            return
        if self.app.mode=='edge_side_wait2' and self.app.edge_side_edge is not None:
            a,_=self.app.edge_side_edge; self.draw_all()
            self.ax.plot([a[0],p[0]],[a[1],p[1]],color='black',linestyle=':',linewidth=1.2); self.fig.canvas.draw_idle()

    # диалоги
    _root=None
    def get_tk_root(self):
        if self._root is None and tk is not None:
            self._root=tk.Tk(); self._root.withdraw()
        return self._root

    def prompt_text(self, title, default=""):
        if simpledialog is None:
            try:
                return input(title+" ")
            except EOFError:
                return None
        else:
            root=self.get_tk_root()
            return simpledialog.askstring("Ввод", title, initialvalue=default, parent=root)

    def prompt_float(self, title, default=""):
        s=self.prompt_text(title, default)
        if s is None: return None
        try: return float(s.strip())
        except: 
            self.alert("Нужно число."); return None

    def alert(self, msg):
        if messagebox is None: print("! ", msg)
        else:
            root=self.get_tk_root(); messagebox.showinfo("Инфо", msg, parent=root)

def main():
    app=App(); View(app)

if __name__=="__main__":
    main()
