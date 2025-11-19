from shapely.geometry import Polygon, LineString, Point
from scipy.spatial import ConvexHull
from pathlib import Path

import trimesh
import csv
import numpy as np
import struct


class Regions:

    HEADER_STRUCT = struct.Struct('i')
    BODY_STRUCT = struct.Struct('fff i')

    def __init__(self, map_region: dict[int, list[list[float]]], resolution: float, dim: int = 3):
        self.map_region = map_region
        self.resolution = resolution
        self.dim = dim

    def export(self, filename: str):
        """Export all regions as triangles (x, y, z, class)."""
        regions_triangles = self._compute_regions_triangles()

        Path("runs").mkdir(exist_ok=True)
        filename_bin = Path(f"runs/{filename}.bin")

        with open(filename_bin, mode='wb') as file:
            header_data = self.HEADER_STRUCT.pack(self.dim)
            file.write(header_data)

            for region_id, triangles in regions_triangles.items():
                for tri in triangles:
                    for vertex in tri:
                        x, y, z = vertex
                        data = self.BODY_STRUCT.pack(x, y, z, region_id)
                        file.write(data)


    def _compute_regions_triangles(self):
        """Select the 2D or 3D method according to the dimension."""
        if self.dim == 2:
            return self._compute_regions_triangles_2d()
        elif self.dim == 3:
            return self._compute_regions_triangles_3d()
        else:
            raise ValueError(f"Dimension {self.dim} not supported")

    def _compute_regions_triangles_2d(self):
        regions = {}
        pixel_size = 1.0 / self.resolution

        for region_id, pts in self.map_region.items():
            points = np.array(pts)
            n = len(points)
            triangles = []

            if n >= 3:
                triangles = self._handle_2d_case_three_or_more_points(points)
            elif n == 2:
                triangles = self._handle_2d_case_two_points(points, pixel_size)
            else:
                triangles = self._handle_2d_case_one_point(points, pixel_size)

            regions[region_id] = triangles

        return regions
    
    def _compute_regions_triangles_3d(self):
        regions = {}
        pixel_size = 1.0 / self.resolution

        for region_id, pts in self.map_region.items():
            points = np.array(pts)
            n = len(points)
            mesh = None

            if n >= 4:
                mesh = self._handle_3d_case_four_or_more_points(points)
            elif n == 3:
                mesh = self._handle_3d_case_three_points(points)
            elif n == 2:
                mesh = self._handle_3d_case_two_points(points, pixel_size)
            else:
                mesh = self._handle_3d_case_one_point(points, pixel_size)

            triangles = []
            if mesh is not None and hasattr(mesh, 'triangles') and len(mesh.triangles) > 0:
                for tri in mesh.triangles:
                    triangles.append(tri)

            regions[region_id] = triangles

        return regions

    def _handle_2d_case_three_or_more_points(self, points):
        """Handle 2D case with 3 or more points (polygon)."""
        triangles = []
        try:
            hull = ConvexHull(points, qhull_options='QJ')
            polygon = Polygon(points[hull.vertices])
        except Exception:
            polygon = Polygon(points)

        coords = np.array(polygon.exterior.coords)
        for i in range(len(coords) - 2):
            tri = np.array([
                [coords[0, 0], coords[0, 1], 0.0],
                [coords[i + 1, 0], coords[i + 1, 1], 0.0],
                [coords[i + 2, 0], coords[i + 2, 1], 0.0]
            ])
            triangles.append(tri)
        return triangles

    def _handle_2d_case_two_points(self, points, pixel_size):
        """Handle 2D case with 2 points (rectangle)."""
        triangles = []
        p1, p2 = points[0], points[1]
        direction = p2 - p1
        length = np.linalg.norm(direction)
        direction = direction / length
        
        # Perpendicular vector
        perpendicular = np.array([-direction[1], direction[0]])
        half_width = pixel_size * 0.5
        
        # Rectangle corners
        rect_points = np.array([
            p1 - perpendicular * half_width,
            p1 + perpendicular * half_width,
            p2 + perpendicular * half_width,
            p2 - perpendicular * half_width
        ])
        
        # Create two triangles for the rectangle
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
        return triangles

    def _handle_2d_case_one_point(self, points, pixel_size):
        """Handle 2D case with 1 point (square)."""
        triangles = []
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
        return triangles

    def _handle_3d_case_four_or_more_points(self, points):
        """Handle 3D case with 4 or more points (convex hull)."""
        try:
            hull = ConvexHull(points, qhull_options='QJ')
            mesh = trimesh.Trimesh(vertices=points, faces=hull.simplices)
            return mesh
        except Exception:
            return None

    def _handle_3d_case_three_points(self, points):
        """Handle 3D case with 3 points (triangle)."""
        mesh = trimesh.Trimesh(vertices=points, faces=[[0, 1, 2]])
        return mesh

    def _handle_3d_case_two_points(self, points, pixel_size):
        """Handle 3D case with 2 points (rectangular box)."""
        p1, p2 = points[0], points[1]
        length = np.linalg.norm(p2 - p1)
        center = (p1 + p2) / 2
        
        # Create a rectangular box oriented between the two points
        mesh = trimesh.creation.box(extents=[pixel_size, pixel_size, length])
        
        # Calculate rotation to orient the box
        direction = (p2 - p1) / length
        z_axis = np.array([0, 0, 1])
        
        if not np.allclose(direction, z_axis):
            rotation_axis = np.cross(z_axis, direction)
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            angle = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
            mesh = mesh.apply_transform(trimesh.transformations.rotation_matrix(angle, rotation_axis))
        
        mesh.apply_translation(center)
        return mesh

    def _handle_3d_case_one_point(self, points, pixel_size):
        """Handle 3D case with 1 point (cube)."""
        x, y, z = points[0]
        mesh = trimesh.creation.box(extents=[pixel_size, pixel_size, pixel_size])
        mesh.apply_translation([x, y, z])
        return mesh
