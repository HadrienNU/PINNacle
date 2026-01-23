import struct
import os
import numpy as np
from scipy.spatial import cKDTree

DENSITY_RADIUS = 0.01
DOMAIN_RADIUS = 1.0

def find_true_end_of_points(f, file_size, header_size):
    
    start_scan = max(header_size, file_size - 5_000_000)
    current_pos = file_size - 4 
    
    while current_pos >= start_scan:
        if (current_pos - header_size) % 16 != 0:
            current_pos -= 4 
            continue

        f.seek(current_pos)
        try:
            val_bytes = f.read(4)
            if len(val_bytes) < 4: 
                break
            val = struct.unpack('<i', val_bytes)[0]
        except:
            current_pos -= 4
            continue
            
        expected_eof = current_pos + 4 + (val * 4)
        
        if expected_eof == file_size and val > 0:
            return current_pos

        current_pos -= 4

    remainder = (file_size - header_size) % 16
    if remainder != 0:
        return file_size - remainder
        
    return file_size

def add_area_to_binary(filename):
    if not os.path.exists(filename):
        print(f"Erreur: '{filename}' introuvable.")
        return

    file_size = os.path.getsize(filename)
    header_size = 4
    record_size = 16 


    try:
        with open(filename, 'rb+') as f:
            true_end_pos = find_true_end_of_points(f, file_size, header_size)
            if true_end_pos < file_size:
                f.seek(true_end_pos)
                f.truncate()
            
            num_records = (true_end_pos - header_size) // record_size

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
            print(f"Régions détectées : {num_regions}")

            centroids_x = np.zeros(num_regions)
            centroids_y = np.zeros(num_regions)
            
            for i in range(num_regions):
                start = indices[i]
                end = start + counts[i]
                centroids_x[i] = np.mean(data['x'][start:end])
                centroids_y[i] = np.mean(data['y'][start:end])

            centroids = np.column_stack((centroids_x, centroids_y))
            tree = cKDTree(centroids)
            neighbors_list = tree.query_ball_point(centroids, r=DENSITY_RADIUS)
            
            normalized_areas = np.zeros(num_regions, dtype='<f4')
            ref_circle_area = np.pi * (DENSITY_RADIUS ** 2)
            
            for i, neighbors in enumerate(neighbors_list):
                cx, cy = centroids[i]
                if (np.sqrt(cx**2 + cy**2) + DENSITY_RADIUS) > DOMAIN_RADIUS:
                    normalized_areas[i] = -1.0
                else:
                    normalized_areas[i] = ref_circle_area / float(len(neighbors))

            f.seek(true_end_pos)
            f.write(struct.pack('<i', num_regions))
            f.write(normalized_areas.tobytes())
            final_size = f.tell()
            print(f"Taille finale : {final_size} octets.")

    except Exception as e:
        print(f"Erreur critique : {e}")