import numpy as np
from scipy.spatial import ConvexHull

import csv


class Regions:

    def __init__(self, map_region):
        self.map_region = map_region

    def export(self, filename):
        regions = self._compute_regions()
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

    def _compute_regions(self):
        regions = {}
        for region in self.map_region:
            points = np.array(self.map_region[region])
            if (len(points)) < 3:
                continue
            hull = ConvexHull(points, qhull_options='QJ')
            regions[region] = points[hull.vertices]
        return regions
    