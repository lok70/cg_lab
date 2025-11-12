import pygame
import numpy as np
import math
import os

class Vector3D:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
    
    def as_vector(self):
        return np.array([self.a, self.b, self.c, 1])
    
    @classmethod
    def from_vector(cls, vec):
        return cls(vec[0], vec[1], vec[2])
    
    def subtract(self, vec):
        return Vector3D(self.a - vec.a, self.b - vec.b, self.c - vec.c)
    
    def inner_product(self, vec):
        return self.a * vec.a + self.b * vec.b + self.c * vec.c
    
    def add(self, vec):
        return Vector3D(self.a + vec.a, self.b + vec.b, self.c + vec.c)

    def outer_product(self, vec):
        return Vector3D(
            self.b * vec.c - self.c * vec.b,
            self.c * vec.a - self.a * vec.c,
            self.a * vec.b - self.b * vec.a
        )

    def magnitude(self):
        return math.sqrt(self.a**2 + self.b**2 + self.c**2)

    def unit_vector(self):
        mag = self.magnitude()
        if mag > 0:
            return Vector3D(self.a/mag, self.b/mag, self.c/mag)
        return self

    def __str__(self):
        return f"({self.a:.3f}, {self.b:.3f}, {self.c:.3f})"

class Surface:
    def __init__(self, vertices, shade=(200, 200, 200)):
        self.vertices = vertices
        self.shade = shade
    
    def transform(self, matrix):
        new_vertices = []
        for vert in self.vertices:
            vert_vec = vert.as_vector()
            new_vec = np.dot(matrix, vert_vec)
            if new_vec[3] != 0:
                new_vec = new_vec / new_vec[3]
            new_vertices.append(Vector3D.from_vector(new_vec))
        return Surface(new_vertices, self.shade)
    
    def centroid(self):
        avg_a = sum(v.a for v in self.vertices) / len(self.vertices)
        avg_b = sum(v.b for v in self.vertices) / len(self.vertices)
        avg_c = sum(v.c for v in self.vertices) / len(self.vertices)
        return Vector3D(avg_a, avg_b, avg_c)
    
    def surface_normal(self):
        if len(self.vertices) < 3:
            return Vector3D(0, 0, 1)
        dir1 = self.vertices[1].subtract(self.vertices[0])
        dir2 = self.vertices[2].subtract(self.vertices[0])
        na = dir1.b * dir2.c - dir1.c * dir2.b
        nb = dir1.c * dir2.a - dir1.a * dir2.c
        nc = dir1.a * dir2.b - dir1.b * dir2.a
        mag = math.sqrt(na*na + nb*nb + nc*nc)
        if mag > 0:
            na /= mag
            nb /= mag
            nc /= mag
        return Vector3D(na, nb, nc)

class Solid:
    def __init__(self, surfaces):
        self.surfaces = surfaces
        self.transformation = np.identity(4)
        self.name = "Unnamed"
    
    def update_transform(self, matrix):
        self.transformation = np.dot(matrix, self.transformation)
    
    def clear_transform(self):
        self.transformation = np.identity(4)
    
    def transformed_surfaces(self):
        new_surfaces = []
        for surf in self.surfaces:
            new_surf = surf.transform(self.transformation)
            new_surfaces.append(new_surf)
        return new_surfaces
    
    def midpoint(self):
        total_vertices = []
        for surf in self.surfaces:
            total_vertices.extend(surf.vertices)
        if not total_vertices:
            return Vector3D(0, 0, 0)
        avg_a = sum(v.a for v in total_vertices) / len(total_vertices)
        avg_b = sum(v.b for v in total_vertices) / len(total_vertices)
        avg_c = sum(v.c for v in total_vertices) / len(total_vertices)
        return Vector3D(avg_a, avg_b, avg_c)
    
    def resize_around_midpoint(self, scale_factor):
        mid = self.midpoint()
        shift_to_zero = MatrixOps.shift(-mid.a, -mid.b, -mid.c)
        resize_mat = MatrixOps.resize(scale_factor, scale_factor, scale_factor)
        shift_back = MatrixOps.shift(mid.a, mid.b, mid.c)
        combined = np.dot(shift_back, np.dot(resize_mat, shift_to_zero))
        self.update_transform(combined)

    def save_to_obj(self, filename):
        with open(filename, 'w') as f:
            f.write(f"# Exported from Lab7\n")
            f.write(f"o {self.name}\n\n")
            vertex_map = {}
            idx = 1
            for surf in self.surfaces:
                for v in surf.vertices:
                    key = (round(v.a, 6), round(v.b, 6), round(v.c, 6))
                    if key not in vertex_map:
                        vertex_map[key] = idx
                        f.write(f"v {v.a:.6f} {v.b:.6f} {v.c:.6f}\n")
                        idx += 1
            f.write("\n")
            for surf in self.surfaces:
                indices = []
                for v in surf.vertices:
                    key = (round(v.a, 6), round(v.b, 6), round(v.c, 6))
                    indices.append(str(vertex_map[key]))
                f.write(f"f {' '.join(indices)}\n")

class DualPyramid(Solid):
    def __init__(self, scale=1):
        s = scale
        verts = [
            Vector3D(0, s, 0), Vector3D(0, -s, 0), Vector3D(s, 0, 0),
            Vector3D(-s, 0, 0), Vector3D(0, 0, s), Vector3D(0, 0, -s)
        ]
        surfaces = [
            Surface([verts[0], verts[4], verts[2]], (255, 100, 100)),
            Surface([verts[0], verts[2], verts[5]], (100, 255, 100)),
            Surface([verts[0], verts[5], verts[3]], (100, 100, 255)),
            Surface([verts[0], verts[3], verts[4]], (255, 255, 100)),
            Surface([verts[1], verts[2], verts[4]], (255, 100, 255)),
            Surface([verts[1], verts[5], verts[2]], (100, 255, 255)),
            Surface([verts[1], verts[3], verts[5]], (150, 150, 255)),
            Surface([verts[1], verts[4], verts[3]], (255, 150, 100))
        ]
        super().__init__(surfaces)
        self.name = "DualPyramid"

class TwentySided(Solid):
    def __init__(self, scale=1):
        golden = (1 + math.sqrt(5)) / 2
        verts = [Vector3D(v[0]*scale, v[1]*scale, v[2]*scale) for v in [
            (-1, golden, 0), (1, golden, 0), (-1, -golden, 0), (1, -golden, 0),
            (0, -1, golden), (0, 1, golden), (0, -1, -golden), (0, 1, -golden),
            (golden, 0, -1), (golden, 0, 1), (-golden, 0, -1), (-golden, 0, 1)
        ]]
        tris = [
            [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
            [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
            [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
            [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1]
        ]
        shades = [(255,50,50),(255,150,50),(255,255,50),(150,255,50),
                  (50,255,50),(50,255,150),(50,255,255),(50,150,255),
                  (50,50,255),(150,50,255),(255,50,255),(255,50,150),
                  (200,100,100),(100,200,100),(100,100,200),(255,200,100),
                  (255,100,200),(100,255,200),(180,180,180),(120,120,120)]
        surfaces = [Surface([verts[i] for i in tri], shades[idx]) for idx, tri in enumerate(tris)]
        super().__init__(surfaces)
        self.name = "TwentySided"

class MatrixOps:
    @staticmethod
    def shift(da, db, dc):
        return np.array([[1,0,0,da],[0,1,0,db],[0,0,1,dc],[0,0,0,1]])
    
    @staticmethod
    def rotate_a(theta):
        c, s = math.cos(theta), math.sin(theta)
        return np.array([[1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]])
    
    @staticmethod
    def rotate_b(theta):
        c, s = math.cos(theta), math.sin(theta)
        return np.array([[c,0,s,0],[0,1,0,0],[-s,0,c,0],[0,0,0,1]])
    
    @staticmethod
    def rotate_c(theta):
        c, s = math.cos(theta), math.sin(theta)
        return np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]])
    
    @staticmethod
    def resize(sa, sb, sc):
        return np.array([[sa,0,0,0],[0,sb,0,0],[0,0,sc,0],[0,0,0,1]])

    @staticmethod
    def flip_ab(): return np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
    @staticmethod
    def flip_ac(): return np.array([[1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]])
    @staticmethod
    def flip_bc(): return np.array([[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

    @staticmethod
    def rotate_around_central_line(shape, axis, theta):
        mid = shape.midpoint()
        S1 = MatrixOps.shift(-mid.a, -mid.b, -mid.c)
        R = {'a': MatrixOps.rotate_a, 'b': MatrixOps.rotate_b, 'c': MatrixOps.rotate_c}[axis](theta)
        S2 = MatrixOps.shift(mid.a, mid.b, mid.c)
        return np.dot(S2, np.dot(R, S1))

    @staticmethod
    def rotate_around_custom_line(p1, p2, theta):
        dir_vec = Vector3D(p2.a - p1.a, p2.b - p1.b, p2.c - p1.c).unit_vector()
        p, q, r = dir_vec.a, dir_vec.b, dir_vec.c
        S1 = MatrixOps.shift(-p1.a, -p1.b, -p1.c)
        dist = math.sqrt(q*q + r*r)
        if dist > 1e-6:
            Ra = np.array([[1,0,0,0],[0,r/dist,-q/dist,0],[0,q/dist,r/dist,0],[0,0,0,1]])
            Rb = np.array([[dist,0,-p,0],[0,1,0,0],[p,0,dist,0],[0,0,0,1]])
        else:
            Ra = np.identity(4)
            Rb = MatrixOps.rotate_b(math.pi) if p < 0 else np.identity(4)
        Rc = MatrixOps.rotate_c(theta)
        if dist > 1e-6:
            Rb_inv = np.linalg.inv(Rb)
            Ra_inv = np.linalg.inv(Ra)
        else:
            Rb_inv = MatrixOps.rotate_b(-math.pi) if p < 0 else np.identity(4)
            Ra_inv = np.identity(4)
        S2 = MatrixOps.shift(p1.a, p1.b, p1.c)
        return np.dot(S2, np.dot(Ra_inv, np.dot(Rb_inv, np.dot(Rc, np.dot(Rb, np.dot(Ra, S1))))))

def load_obj(filename):
    vertices = []
    faces = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                vertices.append(Vector3D(float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith('f '):
                parts = line.split()
                face_indices = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                face_verts = [vertices[i] for i in face_indices]
                faces.append(Surface(face_verts, (200, 200, 200)))
    model = Solid(faces)
    model.name = os.path.basename(filename).split('.')[0]
    return model

def generate_rotation_surface(profile_points, axis='b', segments=20):
    if len(profile_points) < 2:
        return None
    angle_step = 2 * math.pi / segments
    all_verts = []
    all_faces = []
    for i in range(segments):
        theta = i * angle_step
        rot_mat = {'a': MatrixOps.rotate_a, 'b': MatrixOps.rotate_b, 'c': MatrixOps.rotate_c}[axis](theta)
        rotated = []
        for p in profile_points:
            vec = p.as_vector()
            new_vec = np.dot(rot_mat, vec)
            rotated.append(Vector3D.from_vector(new_vec))
        all_verts.extend(rotated)
    base_offset = 0
    for i in range(segments):
        next_i = (i + 1) % segments
        for j in range(len(profile_points) - 1):
            v1 = all_verts[base_offset + i * len(profile_points) + j]
            v2 = all_verts[base_offset + i * len(profile_points) + j + 1]
            v3 = all_verts[base_offset + next_i * len(profile_points) + j + 1]
            v4 = all_verts[base_offset + next_i * len(profile_points) + j]
            all_faces.append(Surface([v1, v2, v3, v4], (150, 200, 255)))
    model = Solid(all_faces)
    model.name = f"Rotation_{axis.upper()}_{segments}"
    return model

def generate_function_surface(func, x_range, y_range, steps_x=20, steps_y=20):
    x0, x1 = x_range
    y0, y1 = y_range
    dx = (x1 - x0) / steps_x
    dy = (y1 - y0) / steps_y
    vertices = []
    for i in range(steps_x + 1):
        x = x0 + i * dx
        row = []
        for j in range(steps_y + 1):
            y = y0 + j * dy
            z = func(x, y)
            row.append(Vector3D(x, y, z))
        vertices.append(row)
    faces = []
    for i in range(steps_x):
        for j in range(steps_y):
            v1 = vertices[i][j]
            v2 = vertices[i+1][j]
            v3 = vertices[i+1][j+1]
            v4 = vertices[i][j+1]
            faces.append(Surface([v1, v2, v3, v4], (100, 200, 150)))
    model = Solid(faces)
    model.name = "Function_Surface"
    return model

class ShapeVisualizer:
    def __init__(self, w=900, h=700):
        pygame.init()
        self.w, self.h = w, h
        self.display = pygame.display.set_mode((w, h))
        pygame.display.set_caption("Lab 7 - 3D Models")
        self.timer = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 22)
        self.view_dist = 6
        self.view_rot_a = 0.3
        self.view_rot_b = 0.5
        self.active_shape = DualPyramid()
        self.active_shape_type = "builtin"
        self.view_mode = "perspective"
        self.custom_line_start = Vector3D(-2, -2, -2)
        self.custom_line_end = Vector3D(2, 2, 2)
        self.display_custom_line = False
        self.models = {
            "dual_pyramid": DualPyramid(),
            "twenty_sided": TwentySided(0.8)
        }
        self.input_text = ""
        self.input_mode = None
        self.profile_points = [Vector3D(0,0,0), Vector3D(1,1,0), Vector3D(2,0,0)]
        self.rotation_axis = 'b'
        self.rotation_segments = 16
        self.func_surface = None
        self.x0 = self.y0 = -2
        self.x1 = self.y1 = 2
        self.steps = 15

    def render_custom_line(self):
        if not self.display_custom_line: return
        s2d = self.map_3d_to_2d(self.custom_line_start)
        e2d = self.map_3d_to_2d(self.custom_line_end)
        pygame.draw.line(self.display, (255, 0, 255), s2d, e2d, 2)
        pygame.draw.circle(self.display, (255, 0, 0), (int(s2d[0]), int(s2d[1])), 5)
        pygame.draw.circle(self.display, (0, 255, 0), (int(e2d[0]), int(e2d[1])), 5)

    def change_shape(self, kind):
        if kind in self.models:
            self.active_shape = self.models[kind]
            self.active_shape_type = kind
        elif kind == "rotation":
            model = generate_rotation_surface(self.profile_points, self.rotation_axis, self.rotation_segments)
            if model:
                self.active_shape = model
                self.active_shape_type = "rotation"
        elif kind == "function":
            def f(x, y): return math.sin(x) * math.cos(y)
            model = generate_function_surface(f, (self.x0, self.x1), (self.y0, self.y1), self.steps, self.steps)
            self.active_shape = model
            self.active_shape_type = "function"
        self.active_shape.clear_transform()
        self.view_rot_a = 0.3
        self.view_rot_b = 0.5

    def map_3d_to_2d(self, vec):
        ma = MatrixOps.rotate_a(self.view_rot_a)
        mb = MatrixOps.rotate_b(self.view_rot_b)
        comb = np.dot(mb, ma)
        arr = vec.as_vector()
        res = np.dot(comb, arr)
        if self.view_mode == "perspective":
            depth = res[2] + self.view_dist
            if depth == 0: depth = 0.001
            scale = 250 / depth
        else:
            scale = 120
        xa = res[0] * scale + self.w / 2
        yb = res[1] * scale + self.h / 2
        return (xa, yb)

    def render_shape(self):
        self.display.fill((20, 20, 40))
        new_surfs = self.active_shape.transformed_surfaces()
        depth_list = []
        for surf in new_surfs:
            mid = surf.centroid()
            ma = MatrixOps.rotate_a(self.view_rot_a)
            mb = MatrixOps.rotate_b(self.view_rot_b)
            comb = np.dot(mb, ma)
            mid_arr = mid.as_vector()
            new_mid = np.dot(comb, mid_arr)
            depth = new_mid[2] + self.view_dist
            depth_list.append((depth, surf))
        depth_list.sort(reverse=True, key=lambda x: x[0])
        for d, surf in depth_list:
            pts2d = [self.map_3d_to_2d(v) for v in surf.vertices]
            if len(pts2d) >= 3:
                try:
                    pygame.draw.polygon(self.display, surf.shade, pts2d)
                    pygame.draw.polygon(self.display, (255, 255, 255), pts2d, 1)
                except:
                    pygame.draw.lines(self.display, surf.shade, True, pts2d, 1)
        self.render_custom_line()
        self.draw_ui()
        pygame.display.flip()

    def draw_ui(self):
        lines = [
            f"Model: {self.active_shape.name} | View: {self.view_mode}",
            "1-DualPyramid  2-TwentySided  3-Load OBJ  4-Rotation  5-Function",
            "R-Reset  T-Shift  S-Scale  X/Y/Z-Rotate  9/0-ScaleCenter",
            "M/N/B-Flip  C-CenterTurn  L-CustomTurn  P-ViewMode  A-ShowLine",
            "Arrows-View  ENTER-Confirm input"
        ]
        for i, line in enumerate(lines):
            txt = self.small_font.render(line, True, (220, 220, 220))
            self.display.blit(txt, (10, 10 + i * 22))
        if self.input_mode:
            prompt = ""
            if self.input_mode == "load":
                prompt = "Enter OBJ filename: "
            elif self.input_mode == "save":
                prompt = "Save as: "
            elif self.input_mode == "profile":
                prompt = f"Profile point {len(self.profile_points)+1} (x y z): "
            elif self.input_mode == "rotation":
                prompt = f"Axis (a/b/c) [{self.rotation_axis}]: "
            elif self.input_mode == "segments":
                prompt = f"Segments [{self.rotation_segments}]: "
            elif self.input_mode == "range":
                prompt = f"Range x0,x1 y0,y1 [{self.x0},{self.x1} {self.y0},{self.y1}]: "
            elif self.input_mode == "steps":
                prompt = f"Steps [{self.steps}]: "
            txt = self.font.render(prompt + self.input_text, True, (255, 255, 100))
            self.display.blit(txt, (10, self.h - 40))

    def process_input(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            if ev.type == pygame.KEYDOWN:
                if self.input_mode:
                    if ev.key == pygame.K_RETURN:
                        self.handle_input_submit()
                    elif ev.key == pygame.K_BACKSPACE:
                        self.input_text = self.input_text[:-1]
                    elif ev.key == pygame.K_ESCAPE:
                        self.input_mode = None
                        self.input_text = ""
                    else:
                        self.input_text += ev.unicode
                else:
                    if ev.key in [pygame.K_1, pygame.K_KP1]:
                        self.change_shape("dual_pyramid")
                    elif ev.key in [pygame.K_2, pygame.K_KP2]:
                        self.change_shape("twenty_sided")
                    elif ev.key == pygame.K_3:
                        self.input_mode = "load"
                        self.input_text = ""
                    elif ev.key == pygame.K_4:
                        self.profile_points = [Vector3D(0,0,0), Vector3D(1,1,0)]
                        self.input_mode = "profile"
                        self.input_text = ""
                    elif ev.key == pygame.K_5:
                        self.input_mode = "range"
                        self.input_text = ""
                    elif ev.key == pygame.K_r:
                        self.active_shape.clear_transform()
                        self.view_rot_a = 0.3
                        self.view_rot_b = 0.5
                    elif ev.key == pygame.K_t:
                        self.active_shape.update_transform(MatrixOps.shift(0.5, 0, 0))
                    elif ev.key == pygame.K_s:
                        self.active_shape.update_transform(MatrixOps.resize(1.2, 1.2, 1.2))
                    elif ev.key == pygame.K_x:
                        self.active_shape.update_transform(MatrixOps.rotate_a(math.pi / 8))
                    elif ev.key == pygame.K_y:
                        self.active_shape.update_transform(MatrixOps.rotate_b(math.pi / 8))
                    elif ev.key == pygame.K_z:
                        self.active_shape.update_transform(MatrixOps.rotate_c(math.pi / 8))
                    elif ev.key == pygame.K_9:
                        self.active_shape.resize_around_midpoint(1.2)
                    elif ev.key == pygame.K_0:
                        self.active_shape.resize_around_midpoint(0.8)
                    elif ev.key == pygame.K_m:
                        self.active_shape.update_transform(MatrixOps.flip_bc())
                    elif ev.key == pygame.K_n:
                        self.active_shape.update_transform(MatrixOps.flip_ab())
                    elif ev.key == pygame.K_b:
                        self.active_shape.update_transform(MatrixOps.flip_ac())
                    elif ev.key == pygame.K_c:
                        mat = MatrixOps.rotate_around_central_line(self.active_shape, 'a', math.pi / 6)
                        self.active_shape.update_transform(mat)
                    elif ev.key == pygame.K_l:
                        mat = MatrixOps.rotate_around_custom_line(self.custom_line_start, self.custom_line_end, math.pi / 6)
                        self.active_shape.update_transform(mat)
                    elif ev.key == pygame.K_p:
                        self.view_mode = "parallel" if self.view_mode == "perspective" else "perspective"
                    elif ev.key == pygame.K_a:
                        self.display_custom_line = not self.display_custom_line
                    elif ev.key == pygame.K_s and pygame.key.get_mods() & pygame.KMOD_CTRL:
                        self.input_mode = "save"
                        self.input_text = ""
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]: self.view_rot_b -= 0.02
        if keys[pygame.K_RIGHT]: self.view_rot_b += 0.02
        if keys[pygame.K_UP]: self.view_rot_a -= 0.02
        if keys[pygame.K_DOWN]: self.view_rot_a += 0.02
        return True

    def handle_input_submit(self):
        text = self.input_text.strip()
        if not text:
            self.input_mode = None
            self.input_text = ""
            return
        if self.input_mode == "load":
            if os.path.exists(text):
                try:
                    model = load_obj(text)
                    self.models["loaded"] = model
                    self.change_shape("loaded")
                except Exception as e:
                    print(f"Load error: {e}")
            else:
                print(f"File not found: {text}")
        elif self.input_mode == "save":
            filename = text if text.endswith(".obj") else text + ".obj"
            try:
                self.active_shape.save_to_obj(filename)
                print(f"Saved to {filename}")
            except Exception as e:
                print(f"Save error: {e}")
        elif self.input_mode == "profile":
            try:
                parts = text.split()
                if len(parts) != 3: raise ValueError()
                x, y, z = map(float, parts)
                self.profile_points.append(Vector3D(x, y, z))
                if len(self.profile_points) >= 2:
                    self.input_mode = "rotation"
                    self.input_text = self.rotation_axis
            except:
                print("Invalid point")
        elif self.input_mode == "rotation":
            if text.lower() in ['a', 'b', 'c']:
                self.rotation_axis = text.lower()
                self.input_mode = "segments"
                self.input_text = str(self.rotation_segments)
        elif self.input_mode == "segments":
            try:
                val = int(text)
                if val >= 3:
                    self.rotation_segments = val
                    model = generate_rotation_surface(self.profile_points, self.rotation_axis, val)
                    if model:
                        self.active_shape = model
                        self.active_shape_type = "rotation"
            except:
                print("Invalid segments")
        elif self.input_mode == "range":
            try:
                parts = text.replace(',', ' ').split()
                if len(parts) == 4:
                    self.x0, self.x1, self.y0, self.y1 = map(float, parts)
                    self.input_mode = "steps"
                    self.input_text = str(self.steps)
            except:
                print("Format: x0 x1 y0 y1")
        elif self.input_mode == "steps":
            try:
                val = int(text)
                if val >= 3:
                    self.steps = val
                    def f(x, y): return math.sin(math.sqrt(x*x + y*y))
                    model = generate_function_surface(f, (self.x0, self.x1), (self.y0, self.y1), val, val)
                    self.active_shape = model
                    self.active_shape_type = "function"
            except:
                print("Invalid steps")
        self.input_mode = None
        self.input_text = ""

    def start(self):
        running = True
        while running:
            running = self.process_input()
            self.render_shape()
            self.timer.tick(60)
        pygame.quit()

if __name__ == "__main__":
    viz = ShapeVisualizer()
    viz.start()
