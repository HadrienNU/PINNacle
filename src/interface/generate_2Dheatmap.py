import os
import shutil
import struct
import glob
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
from scipy.spatial import KDTree

# --- Constantes ---
DEFAULT_PARAMS = {
    'n_neighbors': 50,
    'radius': 0.01,
    'domain_radius': 1.0
}

MAGIC_TAG = b'STATS_EOF'
FOOTER_FORMAT = '<q9s'
FOOTER_SIZE = struct.calcsize(FOOTER_FORMAT)

# ==========================================
# CALCUL DES STATISTIQUES
# ==========================================

def _compute_triangle_area(triangle):
    """Calcule l'aire d'un triangle 3D."""
    t64 = triangle.astype(np.float64)
    cross = np.cross(t64[1] - t64[0], t64[2] - t64[0])
    return 0.5 * np.linalg.norm(cross)

def compute_region_areas(regions_triangles: dict) -> Dict[int, float]:
    """Somme les aires de tous les triangles par région."""
    return {rid: sum(_compute_triangle_area(tri) for tri in tris) 
            for rid, tris in regions_triangles.items()}

def compute_region_centroids(regions_triangles: dict) -> Dict[int, np.ndarray]:
    """Calcule le barycentre de chaque région."""
    centroids = {}
    for rid, tris in regions_triangles.items():
        if len(tris) > 0:
            centroid = np.mean(tris.reshape(-1, 3).astype(np.float64), axis=0)
            if np.all(np.isfinite(centroid)):
                centroids[rid] = centroid
    return centroids

def compute_area_per_neighbors(region_areas, region_centroids, n_neighbors) -> Dict[int, float]:
    """Calcule l'aire moyenne locale via la méthode des N-plus-proches voisins."""
    valid_ids = [rid for rid in region_areas.keys() if rid in region_centroids]
    if len(valid_ids) < 2: return {rid: region_areas.get(rid, 0.0) for rid in valid_ids}

    points = np.array([region_centroids[rid] for rid in valid_ids])
    tree = KDTree(points)
    
    k = min(n_neighbors + 1, len(valid_ids))
    _, indices = tree.query(points, k=k)
    
    return {rid: sum(region_areas[valid_ids[idx]] for idx in indices[i]) / len(indices[i])
            for i, rid in enumerate(valid_ids)}

def compute_normalized_area_radius(region_centroids, radius, domain_radius) -> Dict[int, float]:
    """Calcule l'aire normalisée par région via une recherche sphérique (radius)."""
    valid_ids = list(region_centroids.keys())
    if not valid_ids: return {}
    
    points = np.array([region_centroids[rid] for rid in valid_ids])
    tree = KDTree(points)
    neighbors_list = tree.query_ball_point(points, r=radius)
    ref_area = np.pi * (radius ** 2)
    
    areas = {}
    for i, rid in enumerate(valid_ids):
        if (np.linalg.norm(points[i]) + radius) > domain_radius:
            areas[rid] = -1.0
        else:
            count = len(neighbors_list[i])
            areas[rid] = (ref_area / count) if count > 0 else 0.0
    return areas

def compute_all_statistics(regions_triangles: dict, params: dict) -> dict:
    """Orchestre tous les calculs statistiques sur les régions."""
    clean_regions = {rid: tris for rid, tris in regions_triangles.items() if np.all(np.isfinite(tris))}
    stats = {'areas': compute_region_areas(clean_regions)}
    centroids = compute_region_centroids(clean_regions)

    if params.get('n_neighbors', 0) > 0:
        stats['area_per_neighbors'] = compute_area_per_neighbors(stats['areas'], centroids, params['n_neighbors'])
        
    if params.get('radius', 0) > 0:
        stats['normalized_area_radius'] = compute_normalized_area_radius(centroids, params['radius'], params.get('domain_radius', float('inf')))
        
    return stats

# ==========================================
# GESTION FICHIERS ET BINAIRES
# ==========================================

def pack_statistics(stats: dict) -> bytes:
    """Convertit le dictionnaire de stats en flux d'octets."""
    data = struct.pack('i', len(stats))
    for name, values in stats.items():
        name_b = name.encode('utf-8')
        data += struct.pack('i', len(name_b)) + name_b
        data += struct.pack('i', len(values))
        for rid in sorted(values.keys()):
            val = values[rid]
            data += struct.pack('i f', rid, val if np.isfinite(val) else 0.0)
    return data

def get_data_boundaries(filename):
    """Détecte la fin des points et lit l'historique des stats si existant."""
    file_size = os.path.getsize(filename)
    if file_size < FOOTER_SIZE: return file_size, {}

    with open(filename, 'rb') as f:
        f.seek(-FOOTER_SIZE, 2)
        try:
            offset, tag = struct.unpack(FOOTER_FORMAT, f.read(FOOTER_SIZE))
            if tag == MAGIC_TAG:
                return offset, read_stats_at_offset(f, offset, file_size - FOOTER_SIZE)
        except struct.error: pass
    return file_size, {}

def read_stats_at_offset(f, start_offset, end_offset):
    """Désérialise les statistiques depuis le fichier binaire."""
    if start_offset >= end_offset or (end_offset - start_offset) < 4: return {}
    f.seek(start_offset)
    stats = {}
    num_stats = struct.unpack('i', f.read(4))[0]
    for _ in range(num_stats):
        name = f.read(struct.unpack('i', f.read(4))[0]).decode('utf-8')
        raw = f.read(struct.unpack('i', f.read(4))[0] * 8)
        arr = np.frombuffer(raw, dtype=np.dtype([('id', 'i4'), ('val', 'f4')]))
        stats[name] = {int(r['id']): float(r['val']) for r in arr}
    return stats

def load_triangles_exact(filename, limit_offset):
    """Charge les données géométriques brutes jusqu'à la limite spécifiée."""
    if limit_offset <= 4: return {}
    with open(filename, 'rb') as f:
        f.read(4) 
        data = np.frombuffer(f.read(limit_offset - 4), dtype=np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('id', '<i4')]))
    
    n_tris = len(data) // 3
    coords = np.column_stack((data['x'], data['y'], data['z']))[:n_tris*3].reshape(n_tris, 3, 3)
    tri_ids = data['id'][:n_tris*3].reshape(n_tris, 3)[:, 0]
    
    sort_idx = np.argsort(tri_ids)
    u_ids, indices = np.unique(tri_ids[sort_idx], return_index=True)
    groups = np.split(coords[sort_idx], indices[1:])
    return {int(rid): grp for rid, grp in zip(u_ids, groups)}

def read_binary_points_only(filename, limit_offset):
    """Extrait uniquement les points bruts pour le plot."""
    with open(filename, 'rb') as f:
        f.seek(4)
        raw = f.read(limit_offset - 4)
    return np.frombuffer(raw, dtype=np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('id', '<i4')]))

# ==========================================
# PLOT & PIPELINE PRINCIPAL
# ==========================================

def plot_heatmaps(data, stats, output_path, title_name):
    """Génère et sauvegarde la double heatmap des densités."""
    stat_voisins = stats.get('area_per_neighbors', {})
    stat_rayon = stats.get('normalized_area_radius', {})

    mask = (data['x'] >= -1.0) & (data['x'] <= 1.0) & (data['y'] >= -1.0) & (data['y'] <= 1.0)
    data_clean = data[mask]
    
    vals_voisins = np.zeros(len(data_clean), dtype=np.float32)
    vals_rayon = np.zeros(len(data_clean), dtype=np.float32)
    
    u_ids, starts, counts = np.unique(data_clean['id'], return_index=True, return_counts=True)
    for rid, start, count in zip(u_ids, starts, counts):
        vals_voisins[start:start+count] = stat_voisins.get(rid, 0.0)
        vals_rayon[start:start+count] = stat_rayon.get(rid, 0.0)

    density_voisins = np.divide(1.0, vals_voisins, out=np.zeros_like(vals_voisins), where=vals_voisins>0)
    density_rayon = np.divide(1.0, vals_rayon, out=np.zeros_like(vals_rayon), where=vals_rayon>0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 11), dpi=150)
    fig.suptitle(f"Heatmaps des densités surfaciques locales pour {title_name}", fontsize=20, fontweight='bold')
    plt.subplots_adjust(wspace=0.15, top=0.92)
    
    sc_params = {'s': 15.0, 'cmap': 'turbo', 'edgecolors': 'none', 'alpha': 1.0}

    # Plot 1: Disque
    valid_d1 = density_rayon[density_rayon > 0]
    v1_min, v1_max = np.percentile(valid_d1, [1, 99]) if len(valid_d1) > 0 else (0, 1)
    sc1 = ax1.scatter(data_clean['x'], data_clean['y'], c=density_rayon, vmin=v1_min, vmax=v1_max, **sc_params)
    ax1.set_title("Méthode Disque", fontsize=16, fontweight='bold')
    ax1.axis('equal'); ax1.grid(False)
    fig.colorbar(sc1, ax=ax1, label='Densité', fraction=0.046, pad=0.04)

    # Plot 2: N-Voisins
    valid_d2 = density_voisins[density_voisins > 0]
    v2_min, v2_max = np.percentile(valid_d2, [1, 99]) if len(valid_d1) > 0 else (0, 1)
    sc2 = ax2.scatter(data_clean['x'], data_clean['y'], c=density_voisins, vmin=v2_min, vmax=v2_max, **sc_params)
    ax2.set_title("Méthode N-Voisins", fontsize=16, fontweight='bold')
    ax2.axis('equal'); ax2.grid(False)
    fig.colorbar(sc2, ax=ax2, label='Densité', fraction=0.046, pad=0.04)

    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)

def process_and_plot(original_filepath, params=DEFAULT_PARAMS):
    """
    Crée une copie temporaire, calcule et ajoute les stats, 
    génère la heatmap dans PINNacle/heatmap/, puis supprime le fichier temporaire.
    """
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_script_dir)) 
    heatmap_dir = os.path.join(project_root, "heatmap")
    
    os.makedirs(heatmap_dir, exist_ok=True)
    
    orig_name = os.path.basename(original_filepath)
    out_png_name = f"heatmap_{orig_name}.png"
    out_png_path = os.path.join(heatmap_dir, out_png_name)
    
    temp_filepath = f"temp_{orig_name}"

    print(f"--- Démarrage du pipeline pour : {orig_name} ---")
    
    shutil.copy2(original_filepath, temp_filepath)

    try:
        points_end, existing_stats = get_data_boundaries(temp_filepath)
        regions = load_triangles_exact(temp_filepath, points_end)
        new_stats = compute_all_statistics(regions, params)
        
        final_stats = existing_stats.copy()
        final_stats.update(new_stats)

        with open(temp_filepath, 'r+b') as f:
            f.seek(points_end)
            f.write(pack_statistics(final_stats))
            f.write(struct.pack(FOOTER_FORMAT, points_end, MAGIC_TAG))
            f.truncate()

        raw_points = read_binary_points_only(temp_filepath, points_end)
        plot_heatmaps(raw_points, final_stats, out_png_path, orig_name)
        print(f" > Heatmap sauvegardée dans : {out_png_path}")

    finally:
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
            print(" > Fin")

# ==========================================
# RECHERCHE DE FICHIERS
# ==========================================

def resolve_file_path(user_input: str, runs_dir: str) -> Optional[str]:
    """
    Tente de trouver le fichier spécifié par l'utilisateur.
    """

    if os.path.exists(user_input) and os.path.isfile(user_input):
        return user_input
    
    path_in_runs = os.path.join(runs_dir, user_input)
    if os.path.exists(path_in_runs) and os.path.isfile(path_in_runs):
        return path_in_runs
    
    print(f" > Recherche de '{user_input}' dans {runs_dir}...")
    pattern = os.path.join(runs_dir, "**", user_input)
    candidates = glob.glob(pattern, recursive=True)
    
    if candidates:
        return candidates[0]
        
    return None

def get_latest_binary_automatic(runs_dir: str) -> Optional[str]:
    """Trouve le fichier .bin le plus récent automatiquement."""
    if not os.path.isdir(runs_dir): return None
    subdirs = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
    if not subdirs: return None
    latest_dir = max(subdirs, key=os.path.getmtime)
    bin_files = glob.glob(os.path.join(latest_dir, "*.bin"))
    if not bin_files: return None
    return max(bin_files, key=os.path.getmtime)

if __name__ == "__main__":
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_script_dir)) 
    runs_directory = os.path.join(project_root, "runs")
    
    print("=== Générateur de Heatmap PINNacle ===")
    print(f"Dossier de recherche : {runs_directory}")
    print("Vous pouvez entrer :")
    print(" - Rien (Entrée) : Prend le fichier le plus récent automatiquement")
    print(" - Nom du fichier (ex: test.bin) : Cherche partout dans runs")
    print(" - Chemin relatif (ex: 02.03-22.06.25-benchmark/test.bin)")
    print(" - Chemin absolu ")
    print("-" * 40)
    
    user_input = input("Fichier binaire : ").strip()
    
    target_file = None
    
    if user_input:
        target_file = resolve_file_path(user_input, runs_directory)
        if not target_file:
            print(f"ERREUR: Le fichier '{user_input}' est introuvable.")
    else:
        print(" > Aucun fichier spécifié, recherche du plus récent...")
        target_file = get_latest_binary_automatic(runs_directory)
        if target_file:
            print(f" > Fichier trouvé : {os.path.basename(target_file)}")

    if target_file:
        process_and_plot(target_file)
    else:
        print("Aucun fichier binaire valide n'a pu être traité.")