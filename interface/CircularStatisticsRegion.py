import struct
import os
import numpy as np

def calculate_signed_area(x, y):
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    y1, y2, y3 = y[:, 0], y[:, 1], y[:, 2]
    return 0.5 * np.abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

def add_area_to_binary(filename, domain_radius=1.0):
    if not os.path.exists(filename):
        print(f"Erreur: Le fichier '{filename}' est introuvable.")
        return -1
    
    file_size = os.path.getsize(filename)
    header_size = 4
    record_size = 16 
    
    num_records = (file_size - header_size) // record_size
    valid_data_end = header_size + (num_records * record_size)
    
    print(f"--- Traitement de : {filename} ---")

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
            
            areas = []
            all_x = data['x']
            all_y = data['y']
            
            for i, start_idx in enumerate(indices):
                count = counts[i]
                end_idx = start_idx + count
                
                region_x = all_x[start_idx:end_idx]
                region_y = all_y[start_idx:end_idx]
                
                cx = np.mean(region_x)
                cy = np.mean(region_y)
                
                if (cx**2 + cy**2) > (domain_radius**2):
                    areas.append(-1.0)
                    continue
                
                n_tris = count // 3
                if n_tris == 0:
                    areas.append(0.0)
                    continue
                    
                tri_x = region_x[:n_tris*3].reshape(n_tris, 3)
                tri_y = region_y[:n_tris*3].reshape(n_tris, 3)
                
                total_area = np.sum(calculate_signed_area(tri_x, tri_y))
                areas.append(float(total_area))

            f.seek(valid_data_end)
            
            areas_array = np.array(areas, dtype='<f4')
            f.write(areas_array.tobytes())
            
            num_regions = len(areas)
            f.write(struct.pack('<i', num_regions))
            
            f.truncate()
            
            print(f"{num_regions} aires ajoutées.")
            return num_regions

    except Exception as e:
        print(f"Une erreur est survenue : {e}")
        return -1