import numpy as np
from shapely.geometry import Polygon, LineString, Point
from scipy.spatial import ConvexHull
import csv


class Regions:
    def __init__(self, map_region, circle_center, circle_radius, resolution):
        self.map_region = map_region
        self.circle_center = circle_center
        self.circle_radius = circle_radius
        self.resolution = resolution

    def export(self, filename):
        regions = self._compute_regions_hull()
        filename_csv = f"runs/{filename}.csv"
        data = [['x', 'y', 'class']]
        for region in regions:
            for vertex in regions[region]:
                data.append([
                    vertex[0], vertex[1], region
                ])
                
        with open(filename_csv, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(data)
    
    def _compute_regions_hull(self):
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
