import numpy as np
from scipy.spatial import Voronoi, ConvexHull
from shapely.geometry import Polygon, Point
import csv


class Regions:
    def __init__(self, map_region, circle_center=(0, 0), circle_radius=1.0):
        self.map_region = map_region
        self.circle_center = circle_center
        self.circle_radius = circle_radius

    def export(self, filename):
        regions = self._compute_regions()
        filename_csv = f"runs/{filename}_voronoi.csv"
        data = [['x', 'y', 'class']]
        for region_id, polygon in regions.items():
            if polygon.is_empty:
                continue
            for x, y in np.array(polygon.exterior.coords):
                data.append([x, y, region_id])

        with open(filename_csv, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(data)

    def export_hull(self, filename):
        regions = self._compute_regions_hull()
        filename_csv = f"runs/{filename}_hull.csv"
        data = [['x', 'y', 'class']]
        for region in regions:
            for vertex in regions[region]:
                data.append([
                    vertex[0], vertex[1], region
                ])
                
        with open(filename_csv, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(data)

    def _compute_regions(self):
        if not self.map_region:
            return {}

        centers = []
        region_ids = []
        for rid, points in self.map_region.items():
            pts = np.array(points)
            if len(pts) == 0:
                continue
            center = np.mean(pts, axis=0)
            centers.append(center)
            region_ids.append(rid)
        centers = np.array(centers)

        # Calcul du Voronoï
        vor = Voronoi(centers)

        # Domaine circulaire (comme shapely Polygon)
        cx, cy = self.circle_center
        circle = Point(cx, cy).buffer(self.circle_radius, resolution=512)  # précision fine

        regions = {}
        for i, region_idx in enumerate(vor.point_region):
            region = vor.regions[region_idx]
            if not region or -1 in region:
                # Région infinie : on saute
                continue
            polygon = Polygon(vor.vertices[region])
            # Clip dans le cercle
            clipped = polygon.intersection(circle)
            regions[region_ids[i]] = clipped

        return regions
    
    def _compute_regions_hull(self):
        regions = {}
        for region in self.map_region:
            points = np.array(self.map_region[region])
            if (len(points)) < 3:
                continue
            hull = ConvexHull(points, qhull_options='QJ')
            regions[region] = points[hull.vertices]
        return regions
