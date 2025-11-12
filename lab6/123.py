import pygame
import numpy as np
import math

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

class Surface:
    def __init__(self, vertices, shade=(255, 255, 255)):
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

class DualPyramid(Solid):
    def __init__(self, scale=1):
        s = scale
        verts = [
            Vector3D(0, s, 0),
            Vector3D(0, -s, 0),
            Vector3D(s, 0, 0),
            Vector3D(-s, 0, 0),
            Vector3D(0, 0, s),
            Vector3D(0, 0, -s)
        ]
        surfaces = [
            Surface([verts[0], verts[4], verts[2]], (255, 0, 0)),
            Surface([verts[0], verts[2], verts[5]], (0, 255, 0)),
            Surface([verts[0], verts[5], verts[3]], (0, 0, 255)),
            Surface([verts[0], verts[3], verts[4]], (255, 255, 0)),
            Surface([verts[1], verts[2], verts[4]], (255, 0, 255)),
            Surface([verts[1], verts[5], verts[2]], (0, 255, 255)),
            Surface([verts[1], verts[3], verts[5]], (128, 128, 255)),
            Surface([verts[1], verts[4], verts[3]], (255, 128, 0))
        ]
        super().__init__(surfaces)

class TwentySided(Solid):
    def __init__(self, scale=1):
        golden = (1 + math.sqrt(5)) / 2
        verts = [
            Vector3D(-1, golden, 0), Vector3D(1, golden, 0), Vector3D(-1, -golden, 0), Vector3D(1, -golden, 0),
            Vector3D(0, -1, golden), Vector3D(0, 1, golden), Vector3D(0, -1, -golden), Vector3D(0, 1, -golden),
            Vector3D(golden, 0, -1), Vector3D(golden, 0, 1), Vector3D(-golden, 0, -1), Vector3D(-golden, 0, 1)
        ]
        verts = [Vector3D(v.a * scale, v.b * scale, v.c * scale) for v in verts]
        surfaces = []
        shades = [
            (255, 0, 0), (255, 128, 0), (255, 255, 0), (128, 255, 0),
            (0, 255, 0), (0, 255, 128), (0, 255, 255), (0, 128, 255),
            (0, 0, 255), (128, 0, 255), (255, 0, 255), (255, 0, 128),
            (255, 128, 128), (128, 255, 128), (128, 128, 255),
            (255, 255, 128), (255, 128, 255), (128, 255, 255),
            (192, 192, 192), (128, 128, 128)
        ]
        tris = [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ]
        for idx, tri in enumerate(tris):
            surf_verts = [verts[tri[0]], verts[tri[1]], verts[tri[2]]]
            surfaces.append(Surface(surf_verts, shades[idx % len(shades)]))
        super().__init__(surfaces)

class TwelveSided(Solid):
    def __init__(self, scale=1):
        base = TwentySided(scale * 1.5)
        dodeca_verts = []
        for surf in base.surfaces:
            mid = surf.centroid()
            mag = math.sqrt(mid.a**2 + mid.b**2 + mid.c**2)
            if mag > 0:
                mid = Vector3D(mid.a / mag * scale * 0.7, mid.b / mag * scale * 0.7, mid.c / mag * scale * 0.7)
            dodeca_verts.append(mid)
        unique_verts = []
        for surf in base.surfaces:
            for vert in surf.vertices:
                unique = True
                for u in unique_verts:
                    if (abs(u.a - vert.a) < 0.001 and abs(u.b - vert.b) < 0.001 and abs(u.c - vert.c) < 0.001):
                        unique = False
                        break
                if unique:
                    unique_verts.append(vert)
        surfaces = []
        shades = [
            (255, 0, 0), (255, 128, 0), (255, 255, 0), (128, 255, 0),
            (0, 255, 0), (0, 255, 128), (0, 255, 255), (0, 128, 255),
            (0, 0, 255), (128, 0, 255), (255, 0, 255), (255, 0, 128)
        ]
        for base_vert in unique_verts:
            dists = []
            for j, d_vert in enumerate(dodeca_verts):
                da = d_vert.a - base_vert.a
                db = d_vert.b - base_vert.b
                dc = d_vert.c - base_vert.c
                distance = math.sqrt(da*da + db*db + dc*dc)
                dists.append((distance, j, d_vert))
            dists.sort(key=lambda x: x[0])
            nearest = [v for d, j, v in dists[:5]]
            if len(nearest) != 5:
                continue
            mid_a = sum(v.a for v in nearest) / 5
            mid_b = sum(v.b for v in nearest) / 5
            mid_c = sum(v.c for v in nearest) / 5
            mid = Vector3D(mid_a, mid_b, mid_c)
            vec1 = nearest[1].subtract(nearest[0])
            vec2 = nearest[2].subtract(nearest[0])
            norm = Vector3D(
                vec1.b * vec2.c - vec1.c * vec2.b,
                vec1.c * vec2.a - vec1.a * vec2.c,
                vec1.a * vec2.b - vec1.b * vec2.a
            )
            mag_n = math.sqrt(norm.a*norm.a + norm.b*norm.b + norm.c*norm.c)
            if mag_n > 0:
                norm = Vector3D(norm.a/mag_n, norm.b/mag_n, norm.c/mag_n)
            if abs(norm.a) > 0.1 or abs(norm.b) > 0.1:
                tang = Vector3D(-norm.b, norm.a, 0)
            else:
                tang = Vector3D(0, -norm.c, norm.b)
            mag_t = math.sqrt(tang.a*tang.a + tang.b*tang.b + tang.c*tang.c)
            if mag_t > 0:
                tang = Vector3D(tang.a/mag_t, tang.b/mag_t, tang.c/mag_t)
            binorm = Vector3D(
                norm.b * tang.c - norm.c * tang.b,
                norm.c * tang.a - norm.a * tang.c,
                norm.a * tang.b - norm.b * tang.a
            )
            def angle_key(vert):
                vec = vert.subtract(mid)
                proj_x = vec.inner_product(tang)
                proj_y = vec.inner_product(binorm)
                return math.atan2(proj_y, proj_x)
            nearest.sort(key=angle_key)
            surfaces.append(Surface(nearest, shades[len(surfaces) % len(shades)]))
            if len(surfaces) >= 12:
                break
        super().__init__(surfaces)

class MatrixOps:
    @staticmethod
    def shift(da, db, dc):
        return np.array([
            [1, 0, 0, da],
            [0, 1, 0, db],
            [0, 0, 1, dc],
            [0, 0, 0, 1]
        ])
    
    @staticmethod
    def rotate_a(theta):
        c = math.cos(theta)
        s = math.sin(theta)
        return np.array([
            [1, 0, 0, 0],
            [0, c, -s, 0],
            [0, s, c, 0],
            [0, 0, 0, 1]
        ])
    
    @staticmethod
    def rotate_b(theta):
        c = math.cos(theta)
        s = math.sin(theta)
        return np.array([
            [c, 0, s, 0],
            [0, 1, 0, 0],
            [-s, 0, c, 0],
            [0, 0, 0, 1]
        ])
    
    @staticmethod
    def rotate_c(theta):
        c = math.cos(theta)
        s = math.sin(theta)
        return np.array([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    
    @staticmethod
    def resize(sa, sb, sc):
        return np.array([
            [sa, 0, 0, 0],
            [0, sb, 0, 0],
            [0, 0, sc, 0],
            [0, 0, 0, 1]
        ])

    @staticmethod
    def flip_ab():
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])

    @staticmethod
    def flip_ac():
        return np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    @staticmethod
    def flip_bc():
        return np.array([
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    @staticmethod
    def rotate_around_central_line(shape, axis, theta):
        mid = shape.midpoint()
        S1 = MatrixOps.shift(-mid.a, -mid.b, -mid.c)
        if axis == 'a':
            R = MatrixOps.rotate_a(theta)
        elif axis == 'b':
            R = MatrixOps.rotate_b(theta)
        elif axis == 'c':
            R = MatrixOps.rotate_c(theta)
        else:
            raise ValueError("axis must be a, b or c")
        S2 = MatrixOps.shift(mid.a, mid.b, mid.c)
        return np.dot(S2, np.dot(R, S1))

    @staticmethod
    def rotate_around_custom_line(p1, p2, theta):
        dir_vec = Vector3D(p2.a - p1.a, p2.b - p1.b, p2.c - p1.c).unit_vector()
        p, q, r = dir_vec.a, dir_vec.b, dir_vec.c
        S1 = MatrixOps.shift(-p1.a, -p1.b, -p1.c)
        dist = math.sqrt(q*q + r*r)
        if dist != 0:
            Ra = np.array([
                [1, 0, 0, 0],
                [0, r/dist, -q/dist, 0],
                [0, q/dist, r/dist, 0],
                [0, 0, 0, 1]
            ])
            Rb = np.array([
                [dist, 0, -p, 0],
                [0, 1, 0, 0],
                [p, 0, dist, 0],
                [0, 0, 0, 1]
            ])
        else:
            Ra = np.identity(4)
            Rb = MatrixOps.rotate_b(math.pi) if p < 0 else np.identity(4)
        Rc = MatrixOps.rotate_c(theta)
        if dist != 0:
            Rb_inv = np.linalg.inv(Rb)
            Ra_inv = np.linalg.inv(Ra)
        else:
            Rb_inv = MatrixOps.rotate_b(-math.pi) if p < 0 else np.identity(4)
            Ra_inv = np.identity(4)
        S2 = MatrixOps.shift(p1.a, p1.b, p1.c)
        if dist != 0:
            return np.dot(S2, np.dot(Ra_inv, np.dot(Rb_inv, np.dot(Rc, np.dot(Rb, np.dot(Ra, S1))))))
        else:
            return np.dot(S2, np.dot(Rb_inv, np.dot(Rc, np.dot(Rb, S1))))

class ShapeVisualizer:
    def __init__(self, w=800, h=600):
        pygame.init()
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((w, h))
        pygame.display.set_caption("3D Solids Display")
        self.timer = pygame.time.Clock()
        self.text_font = pygame.font.Font(None, 36)
        self.view_dist = 5
        self.view_rot_a = 0
        self.view_rot_b = 0
        self.dual_pyramid = DualPyramid()
        self.twenty_sided = TwentySided(0.8)
        self.twelve_sided = TwelveSided(0.6)
        self.active_shape = self.dual_pyramid
        self.active_shape_type = "dual_pyramid"
        self.view_mode = "perspective"
        self.custom_line_start = Vector3D(-2, -2, -2)
        self.custom_line_end = Vector3D(2, 2, 2)
        self.display_custom_line = False

    def render_custom_line(self):
        if not self.display_custom_line:
            return
        s2d = self.map_3d_to_2d(self.custom_line_start)
        e2d = self.map_3d_to_2d(self.custom_line_end)
        pygame.draw.line(self.display, (255, 0, 255), s2d, e2d, 2)
        pygame.draw.circle(self.display, (255, 0, 0), (int(s2d[0]), int(s2d[1])), 5)
        pygame.draw.circle(self.display, (0, 255, 0), (int(e2d[0]), int(e2d[1])), 5)

    def change_shape(self, kind):
        if kind == "dual_pyramid":
            self.active_shape = self.dual_pyramid
            self.active_shape_type = "dual_pyramid"
        elif kind == "twenty_sided":
            self.active_shape = self.twenty_sided
            self.active_shape_type = "twenty_sided"
        elif kind == "twelve_sided":
            self.active_shape = self.twelve_sided
            self.active_shape_type = "twelve_sided"
        else:
            return
        self.active_shape.clear_transform()
        self.view_rot_a = 0
        self.view_rot_b = 0

    def map_3d_to_2d(self, vec):
        if self.view_mode == "perspective":
            ma = MatrixOps.rotate_a(self.view_rot_a)
            mb = MatrixOps.rotate_b(self.view_rot_b)
            comb = np.dot(mb, ma)
            arr = vec.as_vector()
            res = np.dot(comb, arr)
            depth = res[2] + self.view_dist
            if depth == 0:
                depth = 0.001
            scale = 200 / depth
            xa = res[0] * scale + self.w / 2
            yb = res[1] * scale + self.h / 2
            return (xa, yb)
        else:
            ma = MatrixOps.rotate_a(self.view_rot_a)
            mb = MatrixOps.rotate_b(self.view_rot_b)
            comb = np.dot(mb, ma)
            arr = vec.as_vector()
            res = np.dot(comb, arr)
            scale = 100
            xa = res[0] * scale + self.w / 2
            yb = res[1] * scale + self.h / 2
            return (xa, yb)

    def render_shape(self):
        self.display.fill((0, 0, 0))
        new_surfs = self.active_shape.transformed_surfaces()
        shown = new_surfs[:]
        depth_list = []
        for surf in shown:
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
            if len(pts2d) > 2:
                try:
                    pygame.draw.polygon(self.display, surf.shade, pts2d)
                    pygame.draw.polygon(self.display, (255, 255, 255), pts2d, 1)
                except:
                    if len(pts2d) >= 2:
                        pygame.draw.lines(self.display, surf.shade, True, pts2d, 1)
        status = f"Solid: {self.active_shape_type}"
        txt = self.text_font.render(status, True, (255, 255, 255))
        self.display.blit(txt, (10, 10))
        hints = [
            "1-DualPyramid 2-TwentySided 3-TwelveSided",
            "R-Clear T-Shift S-Resize",
            "X/Y/Z-Turn M-Flip C-MidTurn",
            "L-CustomTurn P-ViewMode A-DisplayLine",
            "Arrows-View"
        ]
        small = pygame.font.Font(None, 24)
        for i, line in enumerate(hints):
            line_img = small.render(line, True, (255, 255, 255))
            self.display.blit(line_img, (10, self.h - 150 + i * 25))
        pygame.display.flip()

    def process_input(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            if ev.type == pygame.KEYDOWN:
                if ev.key in [pygame.K_1, pygame.K_KP1]:
                    self.change_shape("dual_pyramid")
                elif ev.key in [pygame.K_2, pygame.K_KP2]:
                    self.change_shape("twenty_sided")
                elif ev.key in [pygame.K_3, pygame.K_KP3]:
                    self.change_shape("twelve_sided")
                elif ev.key == pygame.K_r:
                    self.active_shape.clear_transform()
                    self.view_rot_a = self.view_rot_b = 0
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
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.view_rot_b -= 0.02
        if keys[pygame.K_RIGHT]:
            self.view_rot_b += 0.02
        if keys[pygame.K_UP]:
            self.view_rot_a -= 0.02
        if keys[pygame.K_DOWN]:
            self.view_rot_a += 0.02
        return True

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
