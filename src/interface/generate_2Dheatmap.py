import os
import struct
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from typing import Dict, Optional

# Binary file format constants
HEADER_STRUCT = struct.Struct('i')
BODY_STRUCT = struct.Struct('fff i')
STATS_MARKER = -999999999

def read_header(file):
    """Read the .bin file header (dimension)."""
    header_data = file.read(HEADER_STRUCT.size)
    if not header_data:
        return None
    return HEADER_STRUCT.unpack(header_data)[0]


def read_triangles_and_find_stats(file):
    """Read triangles and find the statistics marker."""
    triangles_by_region = {}  # region_id -> list of triangles
    
    triangle_buffer = []
    
    while True:
        data = file.read(BODY_STRUCT.size)
        if not data:
            return triangles_by_region, False
        
        x, y, z, region_id = BODY_STRUCT.unpack(data)
        
        if region_id == STATS_MARKER:
            return triangles_by_region, True
        
        # Accumulate points to form triangles (3 points = 1 triangle)
        triangle_buffer.append((x, y, z, region_id))
        
        if len(triangle_buffer) == 3:
            # All three points should have the same region_id
            rid = triangle_buffer[0][3]
            if rid not in triangles_by_region:
                triangles_by_region[rid] = []
            
            # Store the triangle as 3 (x, y, z) tuples
            triangles_by_region[rid].append([
                (triangle_buffer[0][0], triangle_buffer[0][1], triangle_buffer[0][2]),
                (triangle_buffer[1][0], triangle_buffer[1][1], triangle_buffer[1][2]),
                (triangle_buffer[2][0], triangle_buffer[2][1], triangle_buffer[2][2])
            ])
            triangle_buffer = []


def read_statistics(file):
    """Deserialize statistics from the binary file."""
    stats = {}
    
    try:
        num_stat_types_data = file.read(struct.calcsize('i'))
        if not num_stat_types_data:
            return None
        
        num_stat_types = struct.unpack('i', num_stat_types_data)[0]
        
        for _ in range(num_stat_types):
            # Statistic name
            name_len = struct.unpack('i', file.read(struct.calcsize('i')))[0]
            stat_name = file.read(name_len).decode('utf-8')
            
            # Number of regions
            num_regions = struct.unpack('i', file.read(struct.calcsize('i')))[0]
            
            # Values per region
            stat_values = {}
            for _ in range(num_regions):
                region_id, value = struct.unpack('i f', file.read(struct.calcsize('i f')))
                stat_values[region_id] = value
            
            stats[stat_name] = stat_values
        
        return stats
    except Exception as e:
        print(f"Error reading statistics: {e}")
        return None


def load_binary_file(filepath):
    """Load a .bin file and extract triangles by region and statistics."""
    with open(filepath, 'rb') as file:
        dim = read_header(file)
        if dim is None:
            raise ValueError("Unable to read file header")
        
        triangles_by_region, has_stats = read_triangles_and_find_stats(file)
        
        if not has_stats:
            raise ValueError("No statistics found in file. Make sure the file was generated with --compute-stats")
        
        stats = read_statistics(file)
        if stats is None:
            raise ValueError("Error reading statistics")
        
        return triangles_by_region, stats, dim


def create_polygon_patches(triangles_by_region, stat_dict):
    valid_patches = []
    colors = []
    invalid_patches = []
    
    for region_id, triangles in triangles_by_region.items():
        has_valid_stat = (region_id in stat_dict and stat_dict[region_id] > 0)
        
        for triangle in triangles:
            poly = Polygon([(triangle[0][0], triangle[0][1]),
                           (triangle[1][0], triangle[1][1]),
                           (triangle[2][0], triangle[2][1])], closed=True)
            
            if has_valid_stat:
                area_value = stat_dict[region_id]
                density = 1.0 / area_value
                valid_patches.append(poly)
                colors.append(density)
            else:
                invalid_patches.append(poly)
    
    return valid_patches, colors, invalid_patches


def plot_density_subplot(ax, fig, triangles_by_region, stat_dict, title, stat_name):
    if not stat_dict or len(stat_dict) == 0:
        ax.text(0.5, 0.5, f'No statistics available\n({stat_name})', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.axis('off')
        return
    
    valid_patches, colors, invalid_patches = create_polygon_patches(triangles_by_region, stat_dict)
    
    # Add invalid patches (gray background)
    if len(invalid_patches) > 0:
        invalid_collection = PatchCollection(invalid_patches, facecolor='lightgray', 
                                            alpha=0.3, edgecolors='gray', linewidths=0.5)
        ax.add_collection(invalid_collection)
    
    # Add valid patches with colors
    if len(valid_patches) > 0:
        colors_array = np.array(colors)
        v_min, v_max = np.percentile(colors_array, [1, 99])
        
        collection = PatchCollection(valid_patches, cmap='turbo', alpha=1.0, 
                                    edgecolors='face', linewidths=0.5)
        collection.set_array(colors_array)
        collection.set_clim(v_min, v_max)
        
        ax.add_collection(collection)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal')
        ax.grid(False)
        fig.colorbar(collection, ax=ax, label='Density (1/area)', fraction=0.046, pad=0.04)
    else:
        ax.text(0.5, 0.5, 'No valid density data', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.axis('off')


def plot_heatmaps(triangles_by_region, stats, output_path, title_name):
    """Generate and save dual density heatmaps using region-based statistics with triangular mesh."""
    # Support both old and new key names for backward compatibility
    stat_voisins = stats.get('average_neighborhood_area', stats.get('area_per_neighbors', {}))
    stat_rayon = stats.get('normalized_area_radius', {})
    
    if len(triangles_by_region) == 0:
        return False
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 11), dpi=150)
    fig.suptitle(f"Local surface density heatmaps for {title_name}", fontsize=20, fontweight='bold')
    plt.subplots_adjust(wspace=0.15, top=0.92)
    
    # Set background color to light gray to see gaps better
    for ax in [ax1, ax2]:
        ax.set_facecolor('#f0f0f0')
    
    # Plot 1: Disk method (radius)
    plot_density_subplot(ax1, fig, triangles_by_region, stat_rayon, "Disk Method (radius)", "normalized_area_radius")
    
    # Plot 2: N-Neighbors method
    plot_density_subplot(ax2, fig, triangles_by_region, stat_voisins, "N-Neighbors Method", "average_neighborhood_area")

    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    return True


def process_and_plot(filepath):
    """
    Read a .bin file, extract existing statistics,
    and generate heatmaps in the same directory as the .bin file.
    """
    bin_dir = os.path.dirname(os.path.abspath(filepath))
    
    filename = os.path.basename(filepath)
    out_png_name = f"heatmap_{filename.replace('.bin', '')}.png"
    out_png_path = os.path.join(bin_dir, out_png_name)
    
    try:
        triangles_by_region, stats, dim = load_binary_file(filepath)
        
        if dim != 2:
            return False
        
        success = plot_heatmaps(triangles_by_region, stats, out_png_path, filename)
        return success
            
    except Exception as e:
        return False

if __name__ == "__main__":
    import sys
    
    # Check if an argument is provided
    if len(sys.argv) < 2:
        print("USAGE:")
        print(f"  python {os.path.basename(__file__)} <path/to/file.bin>")
        print()
        print("Example:")
        print(f"  python {os.path.basename(__file__)} runs/02.06-14.30.00/test-epoch1000.bin")
        print()
        print("NOTE: The file must have been generated with --compute-stats")
        sys.exit(1)
    
    # Get the file path
    bin_path = sys.argv[1]
    
    # Check if the file exists
    if not os.path.exists(bin_path):
        print(f"ERROR: The file '{bin_path}' does not exist.")
        sys.exit(1)
    
    if not os.path.isfile(bin_path):
        print(f"ERROR: '{bin_path}' is not a file.")
        sys.exit(1)
    
    # Generate the heatmap
    success = process_and_plot(bin_path)
    sys.exit(0 if success else 1)