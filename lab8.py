import pygame
import numpy as np
import math
import os

# ==================== ВСПОМОГАТЕЛЬНЫЕ КЛАССЫ ====================

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
    
    def add(self, vec):
        return Vector3D(self.a + vec.a, self.b + vec.b, self.c + vec.c)

    def outer_product(self, vec):
        return Vector3D(
            self.b * vec.c - self.c * vec.b,
            self.c * vec.a - self.a * vec.c,
            self.a * vec.b - self.b * vec.a
        )

    def inner_product(self, vec):
        return self.a * vec.a + self.b * vec.b + self.c * vec.c

    def magnitude(self):
        return math.sqrt(self.a**2 + self.b**2 + self.c**2)

    def unit_vector(self):
        mag = self.magnitude()
        if mag > 1e-6:
            return Vector3D(self.a/mag, self.b/mag, self.c/mag)
        return Vector3D(0, 0, 0)

    def scale(self, s):
        return Vector3D(self.a * s, self.b * s, self.c * s)

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
        n = len(self.vertices)
        return Vector3D(
            sum(v.a for v in self.vertices) / n,
            sum(v.b for v in self.vertices) / n,
            sum(v.c for v in self.vertices) / n
        )
    
    def normal(self):
        if len(self.vertices) < 3:
            return Vector3D(0, 0, 1)
        v1 = self.vertices[1].subtract(self.vertices[0])
        v2 = self.vertices[2].subtract(self.vertices[0])
        n = v1.outer_product(v2)
        mag = n.magnitude()
        if mag > 1e-6:
            return Vector3D(n.a/mag, n.b/mag, n.c/mag)
        return Vector3D(0, 0, 1)

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
        return [surf.transform(self.transformation) for surf in self.surfaces]
    
    def midpoint(self):
        total = [v for surf in self.surfaces for v in surf.vertices]
        if not total:
            return Vector3D(0, 0, 0)
        n = len(total)
        return Vector3D(
            sum(v.a for v in total) / n,
            sum(v.b for v in total) / n,
            sum(v.c for v in total) / n
        )
    
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
                indices = [str(vertex_map[(round(v.a,6), round(v.b,6), round(v.c,6))]) for v in surf.vertices]
                f.write(f"f {' '.join(indices)}\n")

# ==================== ГЕНЕРАТОРЫ ОБЪЕКТОВ ====================

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

# ==================== КАМЕРА ====================

class Camera:
    def __init__(self, pos, target, fov=60, near=0.1, far=100):
        self.pos = pos
        self.target = target
        self.fov = math.radians(fov)
        self.near = near
        self.far = far
        self.yaw = 0
        self.pitch = 0
        self.distance = (pos.subtract(target)).magnitude()
        self.update_from_angles()

    def update_from_angles(self):
        self.pos.a = self.target.a + self.distance * math.cos(self.pitch) * math.cos(self.yaw)
        self.pos.b = self.target.b + self.distance * math.sin(self.pitch)
        self.pos.c = self.target.c + self.distance * math.cos(self.pitch) * math.sin(self.yaw)

    def look_at_matrix(self):
        forward = self.target.subtract(self.pos).unit_vector()
        right = forward.outer_product(Vector3D(0, 1, 0)).unit_vector()
        up = right.outer_product(forward).unit_vector()
        view = np.array([
            [right.a, right.b, right.c, -self.pos.inner_product(right)],
            [up.a, up.b, up.c, -self.pos.inner_product(up)],
            [-forward.a, -forward.b, -forward.c, self.pos.inner_product(forward)],
            [0, 0, 0, 1]
        ])
        return view

    def projection_matrix(self, w, h, perspective=True):
        aspect = w / h
        if perspective:
            f = 1.0 / math.tan(self.fov / 2)
            nf = 1.0 / (self.near - self.far)
            return np.array([
                [f/aspect, 0, 0, 0],
                [0, f, 0, 0],
                [0, 0, (self.far + self.near) * nf, 2 * self.far * self.near * nf],
                [0, 0, -1, 0]
            ])
        else:
            scale = 200
            return np.array([
                [scale, 0, 0, 0],
                [0, scale, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

# ==================== Z-БУФЕР ====================

def barycentric_coords(p, a, b, c):
    v0 = b.subtract(a)
    v1 = c.subtract(a)
    v2 = p.subtract(a)
    d00 = v0.inner_product(v0)
    d01 = v0.inner_product(v1)
    d11 = v1.inner_product(v1)
    d20 = v2.inner_product(v0)
    d21 = v2.inner_product(v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-6:
        return None
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    if u >= 0 and v >= 0 and w >= 0:
        return u, v, w
    return None

def render_triangle_zbuffer(screen, zbuffer, v0, v1, v2, color, proj_view_model):
    pts = []
    depths = []
    for v in [v0, v1, v2]:
        vec = np.dot(proj_view_model, v.as_vector())
        if vec[3] != 0:
            vec = vec / vec[3]
        x = int((vec[0] + 1) * screen.get_width() / 2)
        y = int((1 - vec[1]) * screen.get_height() / 2)
        z = vec[2]
        pts.append((x, y))
        depths.append(z)
    
    min_x = max(0, min(pts[0][0], pts[1][0], pts[2][0]))
    max_x = min(screen.get_width() - 1, max(pts[0][0], pts[1][0], pts[2][0]))
    min_y = max(0, min(pts[0][1], pts[1][1], pts[2][1]))
    max_y = min(screen.get_height() - 1, max(pts[0][1], pts[1][1], pts[2][1]))

    a, b, c = [Vector3D(pts[i][0], pts[i][1], 0) for i in range(3)]
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            p = Vector3D(x, y, 0)
            bc = barycentric_coords(p, a, b, c)
            if bc:
                u, v, w = bc
                depth = u * depths[0] + v * depths[1] + w * depths[2]
                if depth < zbuffer[y, x]:
                    zbuffer[y, x] = depth
                    screen.set_at((x, y), color)

# ==================== ЗАГРУЗКА OBJ ====================

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

# ==================== ВИЗУАЛИЗАТОР ====================

class ShapeVisualizer:
    def __init__(self, w=900, h=700):
        pygame.init()
        self.w, self.h = w, h
        self.display = pygame.display.set_mode((w, h))
        pygame.display.set_caption("Lab 7 - 3D Models + Backface Culling + Z-Buffer + Camera")
        self.timer = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 22)

        # Камера
        self.camera = Camera(Vector3D(0, 0, 5), Vector3D(0, 0, 0))
        self.camera_mode = "orbit"  # orbit / free
        self.camera_speed = 0.05

        # Режимы
        self.projection_mode = "perspective"  # perspective / parallel
        self.backface_culling = True
        self.zbuffer_enabled = True

        # Объекты
        self.objects = [
            DualPyramid(),
            TwentySided(0.7),
            load_obj("sphere.obj") if os.path.exists("sphere.obj") else DualPyramid(),
            load_obj("cube.obj") if os.path.exists("cube.obj") else DualPyramid(),
        ]
        for i, obj in enumerate(self.objects):
            shift = MatrixOps.shift(i * 2 - 1.5, 0, 0)
            obj.update_transform(shift)

        # UI
        self.input_text = ""
        self.input_mode = None

    def world_to_screen(self, point, view_proj):
        vec = np.dot(view_proj, point.as_vector())
        if vec[3] == 0:
            return None
        vec /= vec[3]
        x = int((vec[0] + 1) * self.w / 2)
        y = int((1 - vec[1]) * self.h / 2)
        return (x, y), vec[2]

    def render_with_zbuffer(self):
        screen = self.display
        zbuffer = np.full((self.h, self.w), float('inf'))

        view = self.camera.look_at_matrix()
        proj = self.camera.projection_matrix(self.w, self.h, self.projection_mode == "perspective")
        view_proj = np.dot(proj, view)

        triangles = []
        for obj in self.objects:
            model = obj.transformation
            view_proj_model = np.dot(view_proj, model)
            for surf in obj.surfaces:
                if len(surf.vertices) < 3:
                    continue
                trans_surf = surf.transform(model)
                normal = trans_surf.normal()
                view_dir = self.camera.pos.subtract(trans_surf.centroid())
                if self.backface_culling and normal.inner_product(view_dir) < 0:
                    continue
                for i in range(0, len(trans_surf.vertices), 3):
                    v0 = trans_surf.vertices[i % len(trans_surf.vertices)]
                    v1 = trans_surf.vertices[(i+1) % len(trans_surf.vertices)]
                    v2 = trans_surf.vertices[(i+2) % len(trans_surf.vertices)]
                    triangles.append((v0, v1, v2, surf.shade, view_proj_model))

        # Сортировка по глубине центра треугольника
        triangles.sort(key=lambda t: (t[0].add(t[1]).add(t[2])).scale(1/3).as_vector()[2], reverse=True)

        for v0, v1, v2, color, mat in triangles:
            render_triangle_zbuffer(screen, zbuffer, v0, v1, v2, color, mat)

    def render_wireframe(self):
        view = self.camera.look_at_matrix()
        proj = self.camera.projection_matrix(self.w, self.h, self.projection_mode == "perspective")
        view_proj = np.dot(proj, view)

        for obj in self.objects:
            model = obj.transformation
            full_mat = np.dot(view_proj, model)
            for surf in obj.surfaces:
                trans_surf = surf.transform(model)
                normal = trans_surf.normal()
                view_dir = self.camera.pos.subtract(trans_surf.centroid())
                if self.backface_culling and normal.inner_product(view_dir) < 0:
                    continue
                pts = []
                for v in trans_surf.vertices:
                    scr, _ = self.world_to_screen(v, full_mat)
                    if scr:
                        pts.append(scr)
                if len(pts) >= 3:
                    pygame.draw.polygon(self.display, surf.shade, pts)
                    pygame.draw.polygon(self.display, (255,255,255), pts, 1)

    def draw_ui(self):
        lines = [
            f"Camera: {'Orbit' if self.camera_mode=='orbit' else 'Free'} | Proj: {self.projection_mode}",
            f"Backface: {'ON' if self.backface_culling else 'OFF'} | Z-Buffer: {'ON' if self.zbuffer_enabled else 'OFF'}",
            "1/2/3/4 - Select Object | C - Camera Mode | P - Projection | B - Backface | Z - Z-Buffer",
            "Arrows - Move Camera | Mouse - Orbit (hold LMB)"
        ]
        for i, line in enumerate(lines):
            txt = self.small_font.render(line, True, (220, 220, 220))
            self.display.blit(txt, (10, 10 + i * 22))

    def process_input(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_1: self.objects[0].update_transform(MatrixOps.shift(0,0,0))
                elif ev.key == pygame.K_2: pass
                elif ev.key == pygame.K_c: self.camera_mode = "free" if self.camera_mode == "orbit" else "orbit"
                elif ev.key == pygame.K_p: self.projection_mode = "parallel" if self.projection_mode == "perspective" else "perspective"
                elif ev.key == pygame.K_b: self.backface_culling = not self.backface_culling
                elif ev.key == pygame.K_z: self.zbuffer_enabled = not self.zbuffer_enabled

        keys = pygame.key.get_pressed()
        speed = 0.1
        if self.camera_mode == "free":
            if keys[pygame.K_w]: self.camera.pos.a += speed
            if keys[pygame.K_s]: self.camera.pos.a -= speed
            if keys[pygame.K_a]: self.camera.pos.c += speed
            if keys[pygame.K_d]: self.camera.pos.c -= speed
            if keys[pygame.K_SPACE]: self.camera.pos.b += speed
            if keys[pygame.K_LCTRL]: self.camera.pos.b -= speed
        else:
            if keys[pygame.K_LEFT]: self.camera.yaw -= self.camera_speed
            if keys[pygame.K_RIGHT]: self.camera.yaw += self.camera_speed
            if keys[pygame.K_UP]: self.camera.pitch = max(-1.5, self.camera.pitch - self.camera_speed)
            if keys[pygame.K_DOWN]: self.camera.pitch = min(1.5, self.camera.pitch + self.camera_speed)
            self.camera.update_from_angles()

        mouse_buttons = pygame.mouse.get_pressed()
        if mouse_buttons[0] and self.camera_mode == "orbit":
            dx, dy = pygame.mouse.get_rel()
            self.camera.yaw -= dx * 0.005
            self.camera.pitch -= dy * 0.005
            self.camera.pitch = np.clip(self.camera.pitch, -1.5, 1.5)
            self.camera.update_from_angles()

        return True

    def start(self):
        running = True
        auto_rotate = 0
        while running:
            running = self.process_input()
            self.display.fill((20, 20, 40))

            if self.zbuffer_enabled:
                self.render_with_zbuffer()
            else:
                self.render_wireframe()

            # Автовращение объектов
            auto_rotate += 0.01
            for i, obj in enumerate(self.objects):
                rot = MatrixOps.rotate_b(auto_rotate + i)
                shift_back = MatrixOps.shift(i * 2 - 1.5, 0, 0)
                shift_to = MatrixOps.shift(-i * 2 + 1.5, 0, 0)
                obj.clear_transform()
                obj.update_transform(np.dot(shift_back, np.dot(rot, shift_to)))

            self.draw_ui()
            pygame.display.flip()
            self.timer.tick(60)
        pygame.quit()

if __name__ == "__main__":
    viz = ShapeVisualizer()
    viz.start()
