import numpy as np
from typing import Dict, List
from scipy.spatial import KDTree


def compute_all_statistics(regions_triangles: dict, stats_params: dict) -> dict:
    stats = {}

    region_areas = compute_region_areas(regions_triangles)
    stats['areas'] = region_areas
    
    region_centroids = None
    
    n_neighbors = stats_params.get('n_neighbors', None)
    if n_neighbors is not None and n_neighbors > 0:
        if region_centroids is None:
            region_centroids = compute_region_centroids(regions_triangles)
        area_per_neighbors = compute_area_per_neighbors(region_areas, region_centroids, n_neighbors)
        stats['area_per_neighbors'] = area_per_neighbors
    
    radius = stats_params.get('radius', None)
    if radius is not None and radius > 0:
        if region_centroids is None:
            region_centroids = compute_region_centroids(regions_triangles)
        domain_radius = stats_params.get('domain_radius', float('inf'))
        normalized_area_radius = compute_normalized_area_radius(region_centroids, radius, domain_radius)
        stats['normalized_area_radius'] = normalized_area_radius
    
    return stats


def compute_region_centroids(regions_triangles: dict) -> Dict[int, np.ndarray]:
    region_centroids = {}
    
    for region_id, triangles in regions_triangles.items():
        if len(triangles) == 0:
            continue
        
        all_vertices = []
        for tri in triangles:
            all_vertices.extend(tri)
        
        centroid = np.mean(all_vertices, axis=0)
        region_centroids[region_id] = centroid
    
    return region_centroids


def compute_region_areas(regions_triangles: dict) -> Dict[int, float]:
    region_areas = {}
    
    for region_id, triangles in regions_triangles.items():
        total_area = 0.0
        for tri in triangles:
            total_area += _compute_triangle_area(tri)
        region_areas[region_id] = total_area
    
    return region_areas


def compute_area_per_neighbors(region_areas: Dict[int, float], region_centroids: Dict[int, np.ndarray], n_neighbors: int) -> Dict[int, float]:
    area_per_neighbors = {}
    region_ids = list(region_areas.keys())
    
    if len(region_ids) == 0:
        return area_per_neighbors
    
    centroids_list = []
    valid_region_ids = []
    
    for region_id in region_ids:
        if region_id in region_centroids:
            centroids_list.append(region_centroids[region_id])
            valid_region_ids.append(region_id)
    
    if len(centroids_list) < 2:
        for region_id in valid_region_ids:
            area_per_neighbors[region_id] = region_areas[region_id]
        return area_per_neighbors
    
    centroids_array = np.array(centroids_list)
    tree = KDTree(centroids_array)
    
    k = min(n_neighbors + 1, len(valid_region_ids))
    distances, indices = tree.query(centroids_array, k=k)
    
    for i, region_id in enumerate(valid_region_ids):
        current_area = region_areas[region_id]
        neighbor_indices = indices[i][1:]  # Exclude itself
        neighbors_area_sum = sum(region_areas[valid_region_ids[idx]] for idx in neighbor_indices)
        
        if neighbors_area_sum > 0:
            area_per_neighbors[region_id] = current_area / neighbors_area_sum
        else:
            area_per_neighbors[region_id] = 0.0
    
    return area_per_neighbors


def compute_normalized_area_radius(region_centroids: Dict[int, np.ndarray], radius: float, domain_radius: float) -> Dict[int, float]:
    normalized_areas = {}
    
    valid_region_ids = list(region_centroids.keys())
    if len(valid_region_ids) == 0:
        return normalized_areas
    
    centroids_list = [region_centroids[rid] for rid in valid_region_ids]
    centroids_array = np.array(centroids_list)
    
    tree = KDTree(centroids_array)
    neighbors_list = tree.query_ball_point(centroids_array, r=radius)
    
    ref_area = np.pi * (radius ** 2)
    
    for i, region_id in enumerate(valid_region_ids):
        centroid_distance = np.linalg.norm(centroids_array[i])
        
        if (centroid_distance + radius) > domain_radius:
            normalized_areas[region_id] = -1.0
        else:
            neighbor_count = len(neighbors_list[i])
            if neighbor_count > 0:
                normalized_areas[region_id] = ref_area / neighbor_count
            else:
                normalized_areas[region_id] = 0.0
    
    return normalized_areas


def _compute_triangle_area(triangle: np.ndarray) -> float:
    v1 = triangle[1] - triangle[0]
    v2 = triangle[2] - triangle[0]
    cross = np.cross(v1, v2)
    area = 0.5 * np.linalg.norm(cross)
    return area


def pack_statistics(stats: dict) -> bytes:
    import struct
    
    data = b''
    
    num_stat_types = len(stats)
    data += struct.pack('i', num_stat_types)
    
    for stat_name, stat_values in stats.items():
        name_bytes = stat_name.encode('utf-8')
        data += struct.pack('i', len(name_bytes))
        data += name_bytes
        
        data += struct.pack('i', len(stat_values))
        
        for region_id in sorted(stat_values.keys()):
            value = stat_values[region_id]
            data += struct.pack('i f', region_id, value)
    
    return data