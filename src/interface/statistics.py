import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.spatial import KDTree


def compute_all_statistics(regions_triangles: dict, stats_params: dict) -> dict:
    stats = {}

    region_areas = compute_region_areas(regions_triangles)
    stats['areas'] = region_areas
    
    region_centroids = None
    kdtree_data = None
    
    n_neighbors = stats_params.get('n_neighbors', None)
    radius = stats_params.get('radius', None)
    needs_kdtree = (n_neighbors is not None and n_neighbors > 0) or (radius is not None and radius > 0)
    
    if needs_kdtree:
        region_centroids = compute_region_centroids(regions_triangles)
        kdtree_data = _build_kdtree(region_centroids)
    
    if n_neighbors is not None and n_neighbors > 0:
        average_neighborhood_area = compute_average_neighborhood_area(region_areas, kdtree_data, n_neighbors)
        stats['average_neighborhood_area'] = average_neighborhood_area
    
    if radius is not None and radius > 0:
        domain_radius = stats_params.get('domain_radius', float('inf'))
        normalized_area_radius = compute_normalized_area_radius(kdtree_data, radius, domain_radius)
        stats['normalized_area_radius'] = normalized_area_radius
    
    return stats


def _build_kdtree(region_centroids: Dict[int, np.ndarray]) -> Optional[Tuple[KDTree, List[int], np.ndarray]]:
    if not region_centroids or len(region_centroids) < 1:
        return None
    
    region_ids_list = list(region_centroids.keys())
    centroids_array = np.array([region_centroids[rid] for rid in region_ids_list])
    tree = KDTree(centroids_array)
    
    return tree, region_ids_list, centroids_array


def compute_region_centroids(regions_triangles: dict) -> Dict[int, np.ndarray]:
    region_centroids = {}
    
    for region_id, triangles in regions_triangles.items():
        if len(triangles) == 0:
            continue
        
        triangles_array = np.array(triangles)
        
        # Compute triangle areas
        v1 = triangles_array[:, 1] - triangles_array[:, 0]
        v2 = triangles_array[:, 2] - triangles_array[:, 0]
        cross = np.cross(v1, v2)
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        
        # Compute triangle centroids (mean of 3 vertices)
        triangle_centroids = np.mean(triangles_array, axis=1)
        
        # Area-weighted centroid
        total_area = np.sum(areas)
        if total_area > 0:
            centroid = np.sum(triangle_centroids * areas[:, np.newaxis], axis=0) / total_area
        else:
            centroid = np.mean(triangle_centroids, axis=0)
        
        region_centroids[region_id] = centroid
    
    return region_centroids


def compute_region_areas(regions_triangles: dict) -> Dict[int, float]:
    region_areas = {}
    
    for region_id, triangles in regions_triangles.items():
        if len(triangles) == 0:
            region_areas[region_id] = 0.0
            continue
        
        triangles_array = np.array(triangles)
        v1 = triangles_array[:, 1] - triangles_array[:, 0]
        v2 = triangles_array[:, 2] - triangles_array[:, 0]
        cross = np.cross(v1, v2)
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        region_areas[region_id] = float(np.sum(areas))
    
    return region_areas


def compute_average_neighborhood_area(region_areas: Dict[int, float], kdtree_data: Optional[Tuple], n_neighbors: int) -> Dict[int, float]:
    average_neighborhood_area = {}
    
    if kdtree_data is None:
        return average_neighborhood_area
    
    tree, region_ids_list, centroids_array = kdtree_data
    
    if len(region_ids_list) < 1:
        return average_neighborhood_area
    
    areas_array = np.array([region_areas[rid] for rid in region_ids_list])
    
    k = min(n_neighbors + 1, len(region_ids_list))
    _, indices = tree.query(centroids_array, k=k)
    
    for i, region_id in enumerate(region_ids_list):
        neighbor_indices = indices[i][:k]  # Include self in the average
        total_area = np.sum(areas_array[neighbor_indices])
        average_neighborhood_area[region_id] = total_area / k
    
    return average_neighborhood_area


def compute_normalized_area_radius(kdtree_data: Optional[Tuple], radius: float, domain_radius: float) -> Dict[int, float]:
    normalized_areas = {}
    
    if kdtree_data is None:
        return normalized_areas
    
    tree, region_ids_list, centroids_array = kdtree_data
    
    if len(region_ids_list) == 0:
        return normalized_areas
    
    neighbors_list = tree.query_ball_point(centroids_array, r=radius)
    
    centroid_distances = np.linalg.norm(centroids_array, axis=1)
    ref_area = np.pi * (radius ** 2)
    
    for i, region_id in enumerate(region_ids_list):
        if (centroid_distances[i] + radius) > domain_radius:
            normalized_areas[region_id] = -1.0
        else:
            neighbor_count = len(neighbors_list[i])
            normalized_areas[region_id] = ref_area / neighbor_count if neighbor_count > 0 else ref_area
    
    return normalized_areas


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
