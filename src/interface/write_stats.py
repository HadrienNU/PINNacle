import sys
import struct
from pathlib import Path


HEADER_STRUCT = struct.Struct('i')
BODY_STRUCT = struct.Struct('fff i')
STATS_MARKER = -999999999


def read_header(file):
    header_data = file.read(HEADER_STRUCT.size)
    if not header_data:
        return None
    dim = HEADER_STRUCT.unpack(header_data)[0]
    return dim


def skip_triangles(file):
    triangles_count = 0
    while True:
        data = file.read(BODY_STRUCT.size)
        if not data:
            return False 
        
        x, y, z, region_id = BODY_STRUCT.unpack(data)
        triangles_count += 1
        
        if region_id == STATS_MARKER:
            return True  


def read_statistics(file):
    stats = {}
    
    # Read number of stat types
    num_stat_types_data = file.read(struct.calcsize('i'))
    if not num_stat_types_data:
        return None
    
    num_stat_types = struct.unpack('i', num_stat_types_data)[0]
    
    for _ in range(num_stat_types):
        # Read stat name
        name_len_data = file.read(struct.calcsize('i'))
        if not name_len_data:
            break
        name_len = struct.unpack('i', name_len_data)[0]
        
        name_bytes = file.read(name_len)
        stat_name = name_bytes.decode('utf-8')
        
        # Read number of regions
        num_regions_data = file.read(struct.calcsize('i'))
        num_regions = struct.unpack('i', num_regions_data)[0]
        
        # Read region stats
        stat_values = {}
        for _ in range(num_regions):
            region_data = file.read(struct.calcsize('i f'))
            region_id, value = struct.unpack('i f', region_data)
            stat_values[region_id] = value
        
        stats[stat_name] = stat_values
    
    return stats


def write_statistics(stats, dim, output_file):
    with open(output_file, 'w') as f:
        if not stats:
            f.write("No statistics found in file.\n")
            return
        
        for stat_name, stat_values in stats.items():
            f.write(f"\n{stat_name.upper().replace('_', ' ')}\n")
            f.write("-" * 80 + "\n")
            
            if not stat_values:
                f.write("  No data\n")
                continue
            
            sorted_regions = sorted(stat_values.items())
            values = list(stat_values.values())
            total = sum(values)
            avg = total / len(values) if values else 0
            min_val = min(values) if values else 0
            max_val = max(values) if values else 0
            
            f.write(f"  Total regions: {len(values)}\n")
            f.write(f"  Average: {avg:.6f}\n")
            f.write(f"  Min:     {min_val:.6f}\n")
            f.write(f"  Max:     {max_val:.6f}\n")
            f.write("\n")
            f.write("  Region ID | Value\n")
            f.write("  " + "-" * 30 + "\n")
            for region_id, value in sorted_regions:
                f.write(f"  {region_id:9d} | {value:12.6f}\n")
        
        f.write("\n" + "=" * 80 + "\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python print_stats.py <path_to_bin_file>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    output_file = filepath.replace('.bin', '_stats.txt')
    
    if not Path(filepath).exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    
    try:
        with open(filepath, 'rb') as file:
            dim = read_header(file)
            if dim is None:
                print("Error: Could not read header.")
                sys.exit(1)
            
            has_stats = skip_triangles(file)
            if not has_stats:
                print("Warning: No statistics found in file.")
                sys.exit(0)
            
            stats = read_statistics(file)
            write_statistics(stats, dim, output_file)
            print(f"Statistics written to: {output_file}")
    
    except Exception as e:
        print(f"Error reading file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
