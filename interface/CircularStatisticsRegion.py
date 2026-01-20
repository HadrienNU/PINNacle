import struct
import os
import numpy as np
from scipy.spatial import cKDTree

DENSITY_RADIUS = 0.01  
DOMAIN_RADIUS = 1.0    

def add_area_to_binary(filename):
    if not os.path.exists(filename):
        print(f"Erreur: '{filename}' introuvable.")
        return -1

    file_size = os.path.getsize(filename)
    header_size = 4
    record_size = 16 
    num_records = (file_size - header_size) // record_size
    valid_data_end = header_size + (num_records * record_size)

    try:
        with open(filename, 'rb+') as f:
            f.seek(0)
            dim = struct.unpack('<i', f.read(4))[0]
            
            raw_data = f.read(num_records * record_size)
            dt = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('id', '<i4')])
            data = np.frombuffer(raw_data, dtype=dt)

            unique_ids, indices, counts = np.unique(data['id'], return_index=True, return_counts=True)
            sort_order = np.argsort(indices)
            unique_ids = unique_ids[sort_order]
            indices = indices[sort_order]
            counts = counts[sort_order]
            
            num_regions = len(unique_ids)
            print(f"Nombre de régions : {num_regions}")

            centroids_x = np.zeros(num_regions)
            centroids_y = np.zeros(num_regions)
            
            for i in range(num_regions):
                start = indices[i]
                end = start + counts[i]
                centroids_x[i] = np.mean(data['x'][start:end])
                centroids_y[i] = np.mean(data['y'][start:end])

            centroids = np.column_stack((centroids_x, centroids_y))
            tree = cKDTree(centroids)
            
            normalized_areas = []
            ref_circle_area = np.pi * (DENSITY_RADIUS ** 2)
            neighbors_list = tree.query_ball_point(centroids, r=DENSITY_RADIUS)

            for i, neighbors in enumerate(neighbors_list):
                cx, cy = centroids[i]
                dist_to_center = np.sqrt(cx**2 + cy**2)
                if (dist_to_center + DENSITY_RADIUS) > DOMAIN_RADIUS:
                    normalized_areas.append(-1.0)
                else:
                    count = len(neighbors)
                    val = ref_circle_area / float(count)
                    normalized_areas.append(val)

            f.seek(valid_data_end)
            f.write(struct.pack('<i', num_regions))
            areas_array = np.array(normalized_areas, dtype='<f4')
            f.write(areas_array.tobytes())
            f.truncate()
            
            print(f"{num_regions} aires ajoutées.")
            return num_regions

    except Exception as e:
        print(f"Erreur : {e}")
        return -1
