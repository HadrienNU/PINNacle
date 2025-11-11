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
                line = LineString(points)
                buffer_poly = line.buffer(pixel_size * 0.5, cap_style=1, join_style=2)
                coords = np.array(buffer_poly.exterior.coords)
                for i in range(len(coords) - 2):
                    tri = np.array([
                        [coords[0, 0], coords[0, 1], 0.0],
                        [coords[i + 1, 0], coords[i + 1, 1], 0.0],
                        [coords[i + 2, 0], coords[i + 2, 1], 0.0]
                    ])
                    triangles.append(tri)

            else:
                x, y = points[0]
                circle = Point(x, y).buffer(pixel_size * 0.5, resolution=8)
                coords = np.array(circle.exterior.coords)
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

            # --- Cas 3 : 2 points -> cylinder
            elif n == 2:
                mesh = trimesh.creation.cylinder(
                    radius=pixel_size * 0.5,
                    segment=points,
                    sections=8
                )

            # --- Cas 4 : 1 point -> icosphere
            else:
                x, y, z = points[0]
                mesh = trimesh.creation.icosphere(subdivisions=1, radius=pixel_size * 0.5)
                mesh.apply_translation([x, y, z])

            if hasattr(mesh, 'triangles') and len(mesh.triangles) > 0:
                for tri in mesh.triangles:
                    triangles.append(tri)

            regions[region_id] = triangles

        return regions
