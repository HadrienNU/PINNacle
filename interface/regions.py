import numpy as np
from shapely.geometry import Polygon, LineString, Point
from scipy.spatial import ConvexHull
import trimesh
import csv
from pathlib import Path


class Regions:

    def __init__(self, map_region: dict[int, list[list[float]]], resolution: float, dim: int = 3):
        self.map_region = map_region
        self.resolution = resolution
        self.dim = dim

    def export(self, filename: str):
        """Exporte toutes les régions en triangles (x, y, z, class)."""
        regions_triangles = self._compute_regions_triangles()

        Path("runs").mkdir(exist_ok=True)
        filename_csv = Path(f"runs/{filename}.csv")

        with open(filename_csv, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['x', 'y', 'z', 'class'])

            for region_id, triangles in regions_triangles.items():
                for tri in triangles: 
                    for vertex in tri:
                        writer.writerow([vertex[0], vertex[1], vertex[2], region_id])


    def _compute_regions_triangles(self):
        """Choisit la méthode 2D ou 3D selon la dimension."""
        if self.dim == 2:
            return self._compute_regions_triangles_2d()
        elif self.dim == 3:
            return self._compute_regions_triangles_3d()
        else:
            raise ValueError(f"Dimension {self.dim} non supportée")

    def _compute_regions_triangles_2d(self):
        regions = {}
        pixel_size = 1.0 / self.resolution

        for region_id, pts in self.map_region.items():
            points = np.array(pts)
            n = len(points)
            triangles = []

            if n >= 3:
                try:
                    hull = ConvexHull(points, qhull_options='QJ')
                    polygon = Polygon(points[hull.vertices])
                except Exception:
                    polygon = Polygon(points)

                coords = np.array(polygon.exterior.coords)
                center = np.mean(coords, axis=0)

                for i in range(len(coords) - 2):
                    tri = np.array([
                        [coords[0, 0], coords[0, 1], 0.0],
                        [coords[i + 1, 0], coords[i + 1, 1], 0.0],
                        [coords[i + 2, 0], coords[i + 2, 1], 0.0]
                    ])
                    triangles.append(tri)

            elif n == 2:
                # Créer un rectangle non arrondi entre deux points
                p1, p2 = points[0], points[1]
                direction = p2 - p1
                length = np.linalg.norm(direction)
                direction = direction / length
                
                # Vecteur perpendiculaire
                perpendicular = np.array([-direction[1], direction[0]])
                half_width = pixel_size * 0.5
                
                # Les 4 coins du rectangle
                rect_points = np.array([
                    p1 - perpendicular * half_width,
                    p1 + perpendicular * half_width,
                    p2 + perpendicular * half_width,
                    p2 - perpendicular * half_width
                ])
                
                # Créer deux triangles pour le rectangle
                triangles.append(np.array([
                    [rect_points[0, 0], rect_points[0, 1], 0.0],
                    [rect_points[1, 0], rect_points[1, 1], 0.0],
                    [rect_points[2, 0], rect_points[2, 1], 0.0]
                ]))
                triangles.append(np.array([
                    [rect_points[0, 0], rect_points[0, 1], 0.0],
                    [rect_points[2, 0], rect_points[2, 1], 0.0],
                    [rect_points[3, 0], rect_points[3, 1], 0.0]
                ]))

            else:
                x, y = points[0]
                half_size = pixel_size * 0.5
                square = Polygon([
                    [x - half_size, y - half_size],
                    [x + half_size, y - half_size],
                    [x + half_size, y + half_size],
                    [x - half_size, y + half_size]
                ])
                coords = np.array(square.exterior.coords)
                for i in range(len(coords) - 2):
                    tri = np.array([
                        [coords[0, 0], coords[0, 1], 0.0],
                        [coords[i + 1, 0], coords[i + 1, 1], 0.0],
                        [coords[i + 2, 0], coords[i + 2, 1], 0.0]
                    ])
                    triangles.append(tri)

            regions[region_id] = triangles

        return regions

    def _compute_regions_triangles_3d(self):
        regions = {}
        pixel_size = 1.0 / self.resolution

        for region_id, pts in self.map_region.items():
            points = np.array(pts)
            n = len(points)
            triangles = []

            # --- Cas 1 : Convex Hull 3D
            if n >= 4:
                try:
                    hull = ConvexHull(points, qhull_options='QJ')
                    mesh = trimesh.Trimesh(vertices=points, faces=hull.simplices)
                except Exception:
                    continue

            # --- Cas 2 : 3 points -> triangle
            elif n == 3:
                mesh = trimesh.Trimesh(vertices=points, faces=[[0, 1, 2]])

            # --- Cas 3 : 2 points -> pavé droit
            elif n == 2:
                p1, p2 = points[0], points[1]
                length = np.linalg.norm(p2 - p1)
                center = (p1 + p2) / 2
                
                # Créer un pavé droit orienté entre les deux points
                mesh = trimesh.creation.box(extents=[pixel_size, pixel_size, length])
                
                # Calculer la rotation pour orienter le pavé
                direction = (p2 - p1) / length
                z_axis = np.array([0, 0, 1])
                
                if not np.allclose(direction, z_axis):
                    rotation_axis = np.cross(z_axis, direction)
                    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                    angle = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
                    mesh = mesh.apply_transform(trimesh.transformations.rotation_matrix(angle, rotation_axis))
                
                mesh.apply_translation(center)

            # --- Cas 4 : 1 point -> cube
            else:
                x, y, z = points[0]
                mesh = trimesh.creation.box(extents=[pixel_size, pixel_size, pixel_size])
                mesh.apply_translation([x, y, z])

            if hasattr(mesh, 'triangles') and len(mesh.triangles) > 0:
                for tri in mesh.triangles:
                    triangles.append(tri)

            regions[region_id] = triangles

        return regions
