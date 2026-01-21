import numpy as np
from typing import Dict, List


def compute_all_statistics(regions_triangles: dict, stats_params: dict) -> dict:
    stats = {}
    
    region_areas = compute_region_areas(regions_triangles)
    stats['areas'] = region_areas
    
    n_neighbors = stats_params.get('n_neighbors', None)
    if n_neighbors is not None and n_neighbors > 0:
        region_centroids = compute_region_centroids(regions_triangles)
        area_per_neighbors = compute_area_per_neighbors(region_areas, region_centroids, n_neighbors)
        stats['area_per_neighbors'] = area_per_neighbors
    
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
    
    for region_id in region_ids:
        if region_id not in region_centroids:
            continue
            
        current_centroid = region_centroids[region_id]
        current_area = region_areas[region_id]
        
        distances = []
        for other_id in region_ids:
            if other_id == region_id or other_id not in region_centroids:
                continue
            
            other_centroid = region_centroids[other_id]
            dist = np.linalg.norm(current_centroid - other_centroid)
            distances.append((dist, other_id))
        
        distances.sort(key=lambda x: x[0])
        closest_neighbors = distances[:min(n_neighbors, len(distances))]
        
        neighbors_area_sum = sum(region_areas[neighbor_id] for _, neighbor_id in closest_neighbors)
        
        if neighbors_area_sum > 0:
            area_per_neighbors[region_id] = current_area / neighbors_area_sum
        else:
            area_per_neighbors[region_id] = 0.0
    
    return area_per_neighbors


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
