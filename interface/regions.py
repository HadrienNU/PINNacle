import numpy as np
from shapely.geometry import Polygon, LineString, Point
from scipy.spatial import ConvexHull
import trimesh
import csv


class Regions:
    def __init__(self, map_region, circle_center, circle_radius, resolution, dim=2):
        self.map_region = map_region
        self.circle_center = circle_center
        self.circle_radius = circle_radius
        self.resolution = resolution
        self.dim = dim

    def export(self, filename):
        regions = self._compute_regions_hull()
        filename_csv = f"runs/{filename}.csv"
        data = [['x', 'y', 'z', 'class']]
        for region in regions:
            for vertex in regions[region]:
                x = vertex[0]
                y = vertex[1]
                if len(vertex) > 2:
                    z = vertex[2]
                else:
                    z = 0.0
                data.append([x, y, z, region])
                
        with open(filename_csv, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(data)
    
    def _compute_regions_hull(self):
        if self.dim == 2:
            return self._compute_regions_hull_2d()
        elif self.dim == 3:
            return self._compute_regions_hull_3d()
        else:
            raise ValueError(f"Dimension {self.dim} not supported")
    
    def _compute_regions_hull_2d(self):
        regions = {}
        pixel_size = 1.0 / self.resolution  
        cx, cy = self.circle_center
        circle = Point(cx, cy).buffer(self.circle_radius, resolution=256)

        for region_id, pts in self.map_region.items():
            points = np.array(pts)
            n = len(points)

            if n >= 3:
                try:
                    hull = ConvexHull(points, qhull_options='QJ')
                    poly = Polygon(points[hull.vertices])
                except Exception:
                    poly = Polygon(points)
            
            elif n == 2:
                line = LineString(points)
                poly = line.buffer(pixel_size * 0.5, cap_style=1, join_style=2)

            else:
                x, y = points[0]
                poly = Point(x, y).buffer(pixel_size * 0.5, resolution=16)

            clipped = poly.intersection(circle)
            if not clipped.is_empty and isinstance(clipped, (Polygon, LineString)):
                # On n’exporte que les coordonnées de la frontière
                if isinstance(clipped, Polygon):
                    coords = np.array(clipped.exterior.coords)
                else:  # LineString
                    coords = np.array(clipped.coords)
                regions[region_id] = coords

        return regions
    
    def _compute_regions_hull_3d(self):
        regions = {}
        pixel_size = 1.0 / self.resolution  
        cx, cy, cz = self.circle_center
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=self.circle_radius)
        sphere.apply_translation([cx, cy, cz])
        
        for region_id, pts in self.map_region.items():
            points = np.array(pts)
            n = len(points)

            if n >= 4:
                try:
                    hull = ConvexHull(points, qhull_options='QJ')
                    mesh = trimesh.Trimesh(vertices=points, faces=hull.simplices)
                except Exception:
                    continue

            elif n == 3:
                try:
                    mesh = trimesh.Trimesh(vertices=points, 
                                          faces=[[0, 1, 2]])
                except Exception:
                    continue

            elif n == 2:
                line = trimesh.creation.cylinder(radius=pixel_size * 0.5, 
                                                  segment=points,
                                                  sections=8)
                mesh = line

            else:
                x, y, z = points[0]
                mesh = trimesh.creation.icosphere(subdivisions=1, radius=pixel_size * 0.5)
                mesh.apply_translation([x, y, z])

            try:
                clipped = mesh.intersection(sphere)
                if not clipped.is_empty and hasattr(clipped, 'vertices'):
                    coords = np.array(clipped.vertices)
                    if len(coords) > 0:
                        regions[region_id] = coords
            except Exception:
                distances = np.linalg.norm(mesh.vertices - np.array([cx, cy, cz]), axis=1)
                mask = distances <= self.circle_radius
                coords = mesh.vertices[mask]
                if len(coords) > 0:
                    regions[region_id] = np.unique(coords, axis=0)

        return regions

