import os
from collections import defaultdict
from pathlib import Path
from multiprocessing import Pool, cpu_count


def obj_info(obj_path):
    v_count = 0
    vt_count = 0
    vn_count = 0
    f_count = 0
    unique_vts = set()

    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                v_count += 1
            elif line.startswith('vt '):
                vt_count += 1
                unique_vts.add(line.strip())
            elif line.startswith('vn '):
                vn_count += 1
            elif line.startswith('f '):
                f_count += 1
            

    print(f"Vertices (v): {v_count}")
    print(f"Texture coords (vt): {vt_count}")
    print(f"Unique texture coords (vt): {len(unique_vts)}")
    print(f"Normals (vn): {vn_count}")
    print(f"Faces (f): {f_count}")

def float_eq(a, b, eps=1e-12):
    return abs(a - b) < eps

def floats_eq_2d(a, b, eps=1e-12):
    # Compare two 2D coords with tolerance
    return float_eq(a[0], b[0], eps) and float_eq(a[1], b[1], eps)

def parse_face_vertex(fv):
    """
    Parse a face vertex specification 'v_idx/vt_idx/vn_idx' or variations.
    Returns a tuple (v_idx, vt_idx, vn_idx), each int or 0 if missing.
    """
    # Example formats:
    #  "12/34/56"
    #  "12//56"   (no vt)
    #  "12/34"    (no vn)
    #  "12"       (no vt, no vn)
    parts = fv.split('/')
    v_i = int(parts[0]) if parts[0] else 0
    
    vt_i = 0
    vn_i = 0
    if len(parts) >= 2 and parts[1] != '':
        vt_i = int(parts[1])
    if len(parts) == 3 and parts[2] != '':
        vn_i = int(parts[2])
    
    return (v_i, vt_i, vn_i)

def write_obj(
    out_path, 
    vertices,        # list of (x, y, z)
    texcoords,       # list of (u, v)
    normals,         # list of (nx, ny, nz)
    faces            # list of [ (v_i, vt_i, vn_i), (v_i, vt_i, vn_i), (v_i, vt_i, vn_i) ] 
):
    """
    Write a new .obj file with 1-based indexing for v, vt, vn.
    If you have no normals or no texcoords, pass an empty list and faces with 0 references.
    """
    with open(out_path, 'w') as f:
        # Write vertices
        for (x, y, z) in vertices:
            f.write(f"v {x} {y} {z}\n")
        
        # Write texcoords
        for (u, v) in texcoords:
            f.write(f"vt {u} {v}\n")
        
        # Write normals
        for (nx, ny, nz) in normals:
            f.write(f"vn {nx} {ny} {nz}\n")
        
        # Write faces
        for face in faces:
            # Each face is a list of (v_i, vt_i, vn_i)
            # We must build something like "f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3"
            parts = []
            for (vi, ti, ni) in face:
                # Only write the needed slashes
                if ti > 0 and ni > 0:
                    parts.append(f"{vi}/{ti}/{ni}")
                elif ti > 0:
                    parts.append(f"{vi}/{ti}")
                elif ni > 0:
                    parts.append(f"{vi}//{ni}")
                else:
                    parts.append(f"{vi}")
            
            f.write("f " + " ".join(parts) + "\n")


def unify_uvs(input_obj_path, output_obj_path, epsilon=1e-12):
    """
    Reads an OBJ with per-face-vertex UVs (like those from Open3D) and merges them
    so each 3D vertex index has exactly one vt index. If multiple vt coords differ
    for the same vertex, it warns.
    """
    vertices = []
    normals = []
    texcoords = []
    faces = []
    
    # We'll store each face as a list of (v_idx, vt_idx, vn_idx).
    # OBJ indexes start at 1. We'll keep them 1-based to avoid confusion,
    # only converting to 0-based when referencing Python lists if needed.
    
    with open(input_obj_path, 'r') as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            prefix = parts[0]
            
            if prefix == 'v':
                # Vertex
                # e.g. v x y z
                coords = list(map(float, parts[1:4]))
                vertices.append(coords)
            
            elif prefix == 'vt':
                # Texture coord
                # e.g. vt u v (optionally w)
                coords = list(map(float, parts[1:3]))  # ignoring w if present
                texcoords.append(coords)
            
            elif prefix == 'vn':
                # Normal
                coords = list(map(float, parts[1:4]))
                normals.append(coords)
            
            elif prefix == 'f':
                face_vertices = parts[1:]
                face_triplets = []
                for fv in face_vertices:
                    v_i, vt_i, vn_i = parse_face_vertex(fv)
                    face_triplets.append((v_i, vt_i, vn_i))
                faces.append(face_triplets)
            
            # ignore other lines (g, o, usemtl, etc.)
    
    # Now we unify the per-face-vertex UV so there's exactly one uv for each v_idx.
    # 1. For each vertex index, gather all its vt indices.
    v_to_vt_indices = defaultdict(set)
    
    for face in faces:
        for (v_i, vt_i, vn_i) in face:
            if vt_i != 0:
                v_to_vt_indices[v_i].add(vt_i)
    
    # 2. For each v_i, see if all vt coords are (within epsilon) the same.
    #    We'll pick the first one if they differ; you can decide how to handle.
    
    # Convert texcoords to a list of tuples for easier comparison
    texcoords_tuples = [tuple(tc) for tc in texcoords]  # 0-based in Python
    
    vertex_to_unified_vt = {}
    
    for v_i, vt_set in v_to_vt_indices.items():
        # If there's only one, trivial.
        if len(vt_set) == 1:
            vertex_to_unified_vt[v_i] = list(vt_set)[0]
        else:
            # We must check if they're all the same within epsilon
            # Compare them to the first one:
            vt_list = sorted(list(vt_set))
            first_uv = texcoords_tuples[vt_list[0] - 1]  # -1 for 0-based
            all_same = True
            for vt_id in vt_list[1:]:
                this_uv = texcoords_tuples[vt_id - 1]
                if not floats_eq_2d(this_uv, first_uv, epsilon):
                    all_same = False
                    break
            if all_same:
                vertex_to_unified_vt[v_i] = vt_list[0]
            else:
                print(f"Warning: vertex {v_i} has multiple distinct UV coords. Using the first one among {vt_list}.")
                vertex_to_unified_vt[v_i] = vt_list[0]
    
    # Some vertices might never appear in faces, so they won't be in v_to_vt_indices.
    # For those, we won't assign a vt. They effectively have no UV. That might be okay.
    
    # 3. "Compress" or rebuild the new list of unique texcoords we actually use.
    #    Grab the chosen ones. We want to build a map old_vt_index -> new_vt_index (1-based).
    used_vt_indices = set(vertex_to_unified_vt.values())  # the old vt indices that we are actually using
    used_vt_indices = sorted(list(used_vt_indices))
    
    # Build the mapping to new 1-based indices
    old_vt_to_new_vt = {}
    new_texcoords_list = []
    
    next_new_idx = 1
    for old_vt_i in used_vt_indices:
        old_vt_to_new_vt[old_vt_i] = next_new_idx
        new_texcoords_list.append(texcoords_tuples[old_vt_i - 1])  # store the actual 2D coords
        next_new_idx += 1
    
    # 4. Update face references. For each (v_i, vt_i, vn_i), we set vt_i => new vt if we have one. 
    #    If the vertex never had a vt, it remains 0.
    updated_faces = []
    for face in faces:
        new_face = []
        for (v_i, vt_i, vn_i) in face:
            # If v_i is in vertex_to_unified_vt, then that's the vt
            if v_i in vertex_to_unified_vt:
                chosen_old_vt = vertex_to_unified_vt[v_i]
                new_vt_i = old_vt_to_new_vt[chosen_old_vt]  # 1-based
                new_face.append((v_i, new_vt_i, vn_i))
            else:
                # no UV
                new_face.append((v_i, 0, vn_i))
        updated_faces.append(new_face)
    
    # 5. Write the new OBJ with:
    #    - the same vertices
    #    - the new, reduced texcoords
    #    - the same normals
    #    - updated faces
    write_obj(
        output_obj_path,
        vertices,
        new_texcoords_list,
        normals,
        updated_faces
    )

def process_obj_file(input_path, output_path, epsilon=1e-12):
    """Process a single OBJ file"""
    print(f"Processing {input_path}")
    unify_uvs(input_path, output_path, epsilon)
    obj_info(output_path)

def process_obj_task(args):
    """Wrapper for process_obj_file to use with Pool.map"""
    input_path, output_path, epsilon = args
    return process_obj_file(input_path, output_path, epsilon)

def process_directory(dir_path, epsilon=1e-12):
    """Process all OBJ files in a directory in parallel"""
    dir_path = Path(dir_path)
    output_dir = dir_path / "optimised"
    output_dir.mkdir(exist_ok=True)
    
    # Create list of (input, output, epsilon) tuples for all .obj files
    tasks = [
        (obj_file, output_dir / f"{obj_file.stem}_optimised.obj", epsilon)
        for obj_file in dir_path.glob("*.obj")
    ]
    
    # Use all available CPUs except one
    num_processes = max(1, cpu_count() - 1)
    
    # Process files in parallel
    with Pool(num_processes) as pool:
        pool.map(process_obj_task, tasks)

if __name__ == "__main__":
    path = "/Users/jamesdarby/Desktop/thaumato_gp_preds"
    path = "/Users/jamesdarby/Desktop/thaumato_gp_preds/mesh_0_window_949831_999831_sm_flatboi.obj"
    if os.path.isdir(path):
        process_directory(path)
    else:
        output_file = path.replace(".obj", "_optimised.obj")
        process_obj_file(path, output_file)