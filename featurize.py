from collections import defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache
import gc
import hashlib
from itertools import product
import math
import os
from typing import List, Tuple, Dict, Set

import ipdb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from tqdm import tqdm

import networkx as nx
import RNA

from collections import Counter

from dna import DNA_ENERGIES
from gm_energy_functions import (
    _pair, _stack, _hairpin, _bulge, _internal_loop,
    gm_multi_branch, Struct, _d_g, _j_s
)
from utils import parse_data_frame, get_libs



# =========
# BOTLZMANN
# =========

def compute_structure_probabilities(sequence, delta=500):
    """
    For a given RNA sequence:
    - Computes the ensemble free energy G
    - Recovers the partition function Z
    - Finds MFE and suboptimal structures within 'delta' kcal/mol of the MFE
    - Computes Boltzmann probability of each structure
    - Returns a list of (structure, energy, probability) tuples
    """
    # Constants
    R = 0.001987  # kcal/mol·K
    T = 310.15    # temperature in K (ViennaRNA default is 37°C)

    # Create fold compound and compute partition function
    fc = RNA.fold_compound(sequence)
    fc.pf()
    structure, G = RNA.pf_fold(sequence)

    # Compute partition function Z = exp(-G / RT)
    Z = math.exp(-G / (R * T))

    # Enumerate all suboptimal structures within delta * 0.01 kcal/mol of MFE
    subopts = fc.subopt(delta, sorted=1, nullfile=None)

    # Compute P(S_i | s) for each structure
    result = []
    for s in subopts:
        if s.structure is None:  # End of list
            continue
        E = s.energy
        weight = math.exp(-E / (R * T))
        prob = weight / Z
        result.append((s.structure, E, prob))

    # Return top 50 by probability
    result.sort(key=lambda x: -x[2])
    return result[:50]


def normalize_probs(probs):
    """ Normalize the probabilities of structures to sum to 1. """
    total = sum(prob for _, _, prob in probs)
    return [(struct, energy, prob / total) for struct, energy, prob in probs]



# ============
# PRIMAL FACES
# ============

class EnergyDict(dict):
    """A dictionary subclass that allows for reverse key lookups."""
    def __init__(self, name, base=None):
        super().__init__()
        self.name = name
        if base:
            self.update(base)

    def __getitem__(self, key):
        if key in self:
            return super().__getitem__(key)
        # Also try reverse key
        try:
            left, right = key.split("/")
            rev_key = f"{right[::-1]}/{left[::-1]}"
            if rev_key in self:
                return super().__getitem__(rev_key)
        except Exception:
            pass
        raise KeyError(f"{key} not found in {self.name} energy table")

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


# ---------------------------
# ENERGY MAP & GLOBALS
# ---------------------------


def parse_dotbracket(dotbracket: str) -> List[Tuple[int, int]]:
    """Parses a dot-bracket notation string into a list of pairs (i, j)."""
    stack = []
    pairs = []
    for i, c in enumerate(dotbracket):
        if c == '(':
            stack.append(i)
        elif c == ')':
            j = stack.pop()
            pairs.append((j, i))
    return sorted(pairs)

def build_pair_map(pairs: List[Tuple[int, int]]) -> Dict[int, int]:
    """Builds a mapping of pairs (i, j) and undirected (j, i)."""
    pair_map = {i: j for i, j in pairs}
    pair_map.update({j: i for i, j in pairs})
    return pair_map

def find_nested(pairs: List[Tuple[int, int]], i: int, j: int) -> List[Tuple[int, int]]:
    """Minimal children strictly inside (i,j) with no intermediate parent between."""
    nested = []
    for a, b in pairs:
        if i < a < b < j:
            # keep only minimal intervals
            if not any(c <= a < d or c < b <= d for (c, d) in nested):
                nested.append((a, b))
    return sorted(nested)

# ---------------------------
# SAFE ENERGY WRAPPERS  (PATCH)
# ---------------------------
def _safe_stack(seq, i, i1, j, j1, temp, emap) -> float:
    """Stack with robust lookups; uses pair-string keys consistently."""
    try:
        pair = _pair(seq, i, i1, j, j1)
        if not pair:
            return 0.0
        if pair not in emap.NN:
            emap.NN[pair] = (0.0, 0.0)
        return _stack(seq, i, i1, j, j1, temp, emap)
    except Exception:
        return 0.0

def _safe_bulge(seq, i, i1, j, j1, temp, emap) -> float:
    """Bulge with robust lookups; uses pair-string keys consistently."""
    try:
        return _bulge(seq, i, i1, j, j1, temp, emap)
    except Exception:
        return 0.0

def _safe_internal_loop(seq, i, i1, j, j1, temp, emap) -> float:
    """Internal loop with robust lookups; uses pair-string keys consistently."""
    try:
        return _internal_loop(seq, i, i1, j, j1, temp, emap)
    except Exception:
        return 0.0

def safe_hairpin(seq: str, i: int, j: int, temp: float, emap) -> float:
    """Hairpin with robust lookups; uses pair-string keys consistently."""
    try:
        if j - i < 4:
            return 0.0

        hairpin = seq[i : j + 1]
        hairpin_len = len(hairpin) - 2
        pair = _pair(seq, i, i + 1, j, j - 1)  # string key for TERMINAL_MM

        d_g = 0.0
        # Tri/tetra bonus
        if getattr(emap, "TRI_TETRA_LOOPS", None) and hairpin in emap.TRI_TETRA_LOOPS:
            d_h, d_s = emap.TRI_TETRA_LOOPS[hairpin]
            d_g += _d_g(d_h, d_s, temp)

        # Size-dependent loop energy
        if hairpin_len in emap.HAIRPIN_LOOPS:
            d_h, d_s = emap.HAIRPIN_LOOPS[hairpin_len]
            d_g += _d_g(d_h, d_s, temp)
        else:
            d_h, d_s = emap.HAIRPIN_LOOPS[30]
            d_g_inc = _d_g(d_h, d_s, temp)
            d_g += _j_s(hairpin_len, 30, d_g_inc, temp)

        # Terminal mismatch term if available; create neutral if missing
        if pair and pair not in emap.TERMINAL_MM:
            emap.TERMINAL_MM[pair] = (0.0, 0.0)
        if hairpin_len > 3 and pair in emap.TERMINAL_MM:
            d_h, d_s = emap.TERMINAL_MM[pair]
            d_g += _d_g(d_h, d_s, temp)

        # Special 3-loop A bonus
        if hairpin_len == 3 and ('A' in (hairpin[0], hairpin[-1])):
            d_g += 0.5

        return d_g
    except Exception:
        return 0.0

def safe_multi_branch(seq, i, j, temp, e_cache, emap, branches: List[Tuple[int,int]]) -> Struct:
    """Robust wrapper for gm_multi_branch; never returns inf."""
    try:
        return gm_multi_branch(seq, i, j, temp, e_cache, emap, branches)
    except Exception as e:
        return Struct(0.0, f"BIFURCATION:FORCED({len(branches)} arms)", branches)

# ---------------------------
# RECONSTRUCTION
# ---------------------------
def reconstruct(i: int, j: int, seq: str, pair_map: Dict[int, int], pairs: List[Tuple[int, int]]) -> List[Struct]:
    """Reconstructs RNA structures recursively from pairs (i, j) in the sequence."""
    nested = find_nested(pairs, i, j)

    # Base case: hairpin
    if not nested:
        e = safe_hairpin(seq, i, j, temp, emap)  # PATCH: safe
        pstr = _pair(seq, i, i+1, j, j-1)
        return [Struct(e, f"HAIRPIN:{pstr}", [(i, j)])]

    # Single nested structure: could be stack, bulge, or internal loop
    elif len(nested) == 1:
        i1, j1 = nested[0]
        try:
            child = reconstruct(i1, j1, seq, pair_map, pairs)
            if not child or child[0].e == math.inf:
                raise ValueError("Invalid child structure")
        except Exception:
            return [Struct(0.0, f"STACK:ERROR_CHILD({i1},{j1})", [(i, j)])]  # PATCH: finite

        # Determine local face type
        if i1 == i + 1 and j1 == j - 1:
            face_type = "STACK"
            e_local = _safe_stack(seq, i, i1, j, j1, temp, emap)  # PATCH: safe
        elif (i1 > i + 1) and (j1 == j - 1):
            face_type = f"BULGE:{i1 - i - 1}"
            e_local = _safe_bulge(seq, i, i1, j, j1, temp, emap)  # PATCH: safe
        elif (i1 == i + 1) and (j1 < j - 1):
            face_type = f"BULGE:{j - j1 - 1}"
            e_local = _safe_bulge(seq, i, i1, j, j1, temp, emap)  # PATCH: safe
        else:
            face_type = f"INTERNAL:{(i1 - i - 1)}+{(j - j1 - 1)}"
            e_local = _safe_internal_loop(seq, i, i1, j, j1, temp, emap)  # PATCH: safe

        return [Struct(e_local, f"{face_type}:{_pair(seq, i, i1, j, j1)}", [(i, j)])] + child

    # Multibranch (two or more children)
    else:
        branches = sorted(nested, key=lambda x: x[0])

        # --- Patch 1: detect pure stack chains (no true bifurcation) ---
        # Accept a long chain of (i+k+1, j-k-1) children as stacked run.
        if all(
            branches[k][0] == i + k + 1 and branches[-(k + 1)][1] == j - k - 1
            for k in range(len(branches))
        ):
            chain_structs = []
            i_curr, j_curr = i, j
            for (i1, j1) in branches:
                e_local = _safe_stack(seq, i_curr, i1, j_curr, j1, temp, emap)
                chain_structs.append(
                    Struct(e_local, f"STACK:{_pair(seq, i_curr, i1, j_curr, j1)}", [(i_curr, j_curr)])
                )
                chain_structs.extend(reconstruct(i1, j1, seq, pair_map, pairs))
                i_curr, j_curr = i1, j1
            return chain_structs

        # --- Normal multibranch case ---
        L = len(seq)
        branch_structs = []
        e_cache = [[STRUCT_NULL for _ in range(L)] for _ in range(L)]

        for (i1, j1) in branches:
            try:
                substructs = reconstruct(i1, j1, seq, pair_map, pairs)
                root = substructs[0] if substructs else STRUCT_NULL
                if root.e < math.inf:
                    e_cache[i1][j1] = root
                    branch_structs.extend(substructs)
                else:
                    e_cache[i1][j1] = Struct(0.0, "FORCED:ZERO", [(i1, j1)])  # PATCH: finite
            except Exception:
                e_cache[i1][j1] = Struct(0.0, "FORCED:ZERO", [(i1, j1)])      # PATCH: finite

        # sanitize cache bounds
        for (a, b) in branches:
            if not (0 <= a < L and 0 <= b < L and a < b):
                return [Struct(0.0, "BIFURCATION:FORCED(BOUNDS)", branches)] + branch_structs

        bifurc = safe_multi_branch(seq, i, j, temp, e_cache, emap, branches)  # PATCH: safe call

        return [bifurc] + branch_structs


def reconstruct_structs_from_dotbracket(seq: str, dotbracket: str) -> List[Struct]:
    """Reconstructs RNA structures from a dot-bracket notation."""
    pairs = parse_dotbracket(dotbracket)
    if not pairs:
        return []

    pair_map = build_pair_map(pairs)

    def get_outermost_pairs(pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        return [
            (i, j)
            for (i, j) in pairs
            if not any(
                (a < i and j < b)
                for (a, b) in pairs
                if (a, b) != (i, j)
            )
        ]

    top_pairs = get_outermost_pairs(pairs)

    structs = []
    for i, j in sorted(top_pairs):
        try:
            sub = reconstruct(i, j, seq, pair_map, pairs)
            structs.extend(sub)
        except Exception as e:
            print(f"Error in reconstruct() for substructure ({i}, {j}) in dotbracket:\n{dotbracket}")
            print(f"Exception: {e}")
            raise  # Optionally remove this to skip over errors instead of halting

    return structs

# ==========

# EXPECTED BAG OF FACES
# ==========

def compute_expected_bag_of_faces(faces_energy_list, boltz_distrib, words0):
    """
    Given an aptamer's list of face-energy lists and its Boltzmann-distributed structures,
    compute the expected bag-of-face (BoF) vector.

    Args:
        faces_energy_list: List of face-energy lists, one for each structure.
            Example: [[('STACK:AC/TG', -1.45), ('STACK:CG/GC', -2.16)], ...]
        boltz_distrib: List of (dotbracket, energy, prob) tuples for each structure.
        words0: A list of all possible face descriptors (length 848).

    Returns:
        A numpy array of shape (848,) representing the expected BoF vector.
    """
    # Create an index map for words0 for fast lookup
    face_to_idx = {face: idx for idx, face in enumerate(words0)}
    vector_len = len(words0)

    expected_vec = np.zeros(vector_len)

    for struct_idx, face_energy in enumerate(faces_energy_list):
        if struct_idx >= len(boltz_distrib):
            continue  # skip if mismatch
        _, _, prob = boltz_distrib[struct_idx]

        # Count the number of times each face appears in the current structure
        counts = Counter(face for face, _ in face_energy)

        # Create the vector for this structure
        struct_vec = np.zeros(vector_len)
        for face, count in counts.items():
            if face in face_to_idx:
                struct_vec[face_to_idx[face]] = count

        # Weight by probability
        expected_vec += prob * struct_vec

    return expected_vec

# ==========

# DUAL GRAPH
# ==========

@dataclass
class FaceNode:
    face_type: str                  # "STACK" / "HAIRPIN" / "BULGE:k" / "INTERNAL:a+b" / "BIFURCATION"
    ij: Tuple[int, int]             # defining (i, j) for this face
    verts: Set[Tuple[str, int]]     # {(base_char, idx), ...}
    struct: Struct                  # original Struct (unchanged)

def head_pair(st: Struct) -> Tuple[int,int]:
    """Extracts the defining (i, j) pair from a Struct object."""
    # each Struct you emit has exactly one defining (i,j) in st.ij
    return tuple(st.ij[0])

def all_pairs(structs: List[Struct]) -> List[Tuple[int,int]]:
    """Extracts all (i, j) pairs from a list of Struct objects."""
    return sorted(head_pair(s) for s in structs)

def outermost_pairs(pairs: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
    """Finds outermost pairs (i, j) that are not nested within any other pair."""
    return [(i,j) for (i,j) in pairs if not any(a < i and j < b for (a,b) in pairs if (a,b) != (i,j))]

def direct_children_of(pairs: List[Tuple[int,int]], i:int, j:int) -> List[Tuple[int,int]]:
    """Finds direct children of the pair (i, j) that are strictly inside it,"""
    # children strictly inside (i,j) with none strictly between parent and child
    cand = [(a,b) for (a,b) in pairs if i < a < b < j]
    children = []
    for (a,b) in cand:
        if not any(i < c < a < b < d < j for (c,d) in cand if (c,d) != (a,b)):
            children.append((a,b))
    return sorted(children)

def make_face_node(seq: str, st: Struct, children: List[Tuple[int,int]]) -> FaceNode:
    """Constructs a FaceNode from a Struct and its children pairs."""
    i, j = head_pair(st)
    # face type from description prefix; keep your richer desc if needed later
    face_type = st.desc.split(":")[0] if ":" in st.desc else st.desc

    if not children:
        # HAIRPIN: unpaired i+1..j-1
        verts = {(seq[k], k) for k in range(i+1, j)}
        return FaceNode("HAIRPIN", (i,j), verts, st)

    if len(children) == 1:
        (i1, j1) = children[0]
        if i1 == i+1 and j1 == j-1:
            # STACK: include all four vertices (A,i), (B,i1), (C,j1), (D,j)
            verts = {(seq[k], k) for k in (i, i1, j1, j)}
            return FaceNode("STACK", (i,j), verts, st)
        # BULGE / INTERNAL: include unpaired flanks only
        left_gap  = range(i+1, i1)   # i+1..i1-1
        right_gap = range(j1+1, j)   # j1+1..j-1
        verts = {(seq[k], k) for k in list(left_gap) + list(right_gap)}
        # keep your informative subtype name if you want
        subtype = face_type if face_type else "INTERNAL"
        return FaceNode(subtype, (i,j), verts, st)

    # MULTIBRANCH: indices in [i..j] not covered by any child; exclude closing pair endpoints
    covered: Set[int] = set()
    for (a,b) in children:
        covered.update(range(a, b+1))
    loop_idxs = [k for k in range(i, j+1) if k not in covered and k not in (i, j)]
    verts = {(seq[k], k) for k in loop_idxs}
    return FaceNode("BIFURCATION", (i,j), verts, st)

def build_dual_from_structs(seq: str, dotbracket: str):
    """
    Uses your reconstruct_* to identify faces; DOES NOT modify Struct.
    Returns:
      faces_meta: List[FaceNode]         # nodes with (base, index) sets
      adj:        Dict[int, List[int]]   # undirected adjacency; includes EXTERIOR=-1
      top_pairs:  List[(i,j)]            # outermost (i,j) touching EXTERIOR
    """
    structs: List[Struct] = reconstruct_structs_from_dotbracket(seq, dotbracket)
    if not structs:
        return [], {EXTERIOR: []}, [] # This is a strand

    # Map faces by (i,j)
    pairs = all_pairs(structs)
    id_by_pair: Dict[Tuple[int,int], int] = {head_pair(s): k for k, s in enumerate(structs)} 

    # Children relationships via pair inclusion
    children_by_pair: Dict[Tuple[int,int], List[Tuple[int,int]]] = {
        (i,j): direct_children_of(pairs, i, j) for (i,j) in pairs
    }

    # Build node payloads
    faces_meta: List[FaceNode] = []
    for st in structs:
        ij = head_pair(st)
        kids = children_by_pair[ij]
        faces_meta.append(make_face_node(seq, st, kids))

    # Dual edges: parent ↔ child; plus EXTERIOR ↔ outermost faces
    adj_sets: Dict[int, Set[int]] = defaultdict(set)

    tops = outermost_pairs(pairs)
    for (i,j) in tops:
        u = id_by_pair[(i,j)]
        adj_sets[EXTERIOR].add(u)
        adj_sets[u].add(EXTERIOR)

    for (i,j), kids in children_by_pair.items():
        u = id_by_pair[(i,j)]
        for (a,b) in kids:
            v = id_by_pair[(a,b)]
            if u != v:                      # guard against self-loop
                adj_sets[u].add(v)
                adj_sets[v].add(u)

    # finalize adjacency lists
    adj: Dict[int, List[int]] = {u: sorted(v for v in nbrs if v != u) for u, nbrs in adj_sets.items()}

    return faces_meta, adj, tops

# ---------
# Balancing
# ---------

def smallest_vertex_index(face) -> int:
    """Return smallest index in the node's verts; large sentinel if empty."""
    return min((idx for (_, idx) in getattr(face, "verts", [])), default=10**9)

def label_from_face(face):
    verts = ",".join(f"{b}{i}" for (b,i) in sorted(face.verts, key=lambda x: x[1]))
    return f"{face.face_type}: {{{verts}}}" if verts else face.face_type

def graph_from_adj(adj, drop_exterior=True):
    G = nx.Graph()
    for u, nbrs in adj.items():
        if drop_exterior and u == EXTERIOR:
            continue
        for v in nbrs:
            if drop_exterior and v == EXTERIOR:
                continue
            if u != v:
                G.add_edge(u, v)
    return G

def bfs_farthest(G, start):
    """Return (farthest_node, parent_map)."""
    parent = {start: None}
    q = deque([start])
    last = start
    while q:
        u = q.popleft()
        last = u
        for v in G.neighbors(u):
            if v not in parent:
                parent[v] = u
                q.append(v)
    return last, parent

def path_from_parent(parent, end):
    path = [end]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    path.reverse()
    return path

def tree_diameter_path(G):
    """Double-BFS to get a diameter path."""
    start = next(iter(G.nodes))
    u, _ = bfs_farthest(G, start)
    v, parent = bfs_farthest(G, u)
    return path_from_parent(parent, v)  # list of node ids along a diameter

def choose_center(path, faces):
    """Pick the center; if two, pick the one with smallest vertex index."""
    D = len(path) - 1
    if D % 2 == 0:
        return path[D // 2]  # single center
    else:
        a = path[D // 2]
        b = path[D // 2 + 1]
        # tie-break by smallest nucleotide index in verts
        ka = smallest_vertex_index(faces[a])
        kb = smallest_vertex_index(faces[b])
        return a if ka <= kb else b

def balance_root(faces, adj):
    G = graph_from_adj(adj, drop_exterior=True)
    path = tree_diameter_path(G)
    center = choose_center(path, faces)
    return center, path

# -------------
# Visualization
# -------------

def label_from_face(node):
    # node is a FaceNode
    verts = ",".join(f"{b}{i}" for (b,i) in sorted(node.verts, key=lambda x: x[1]))
    return f"{node.face_type}: {{{verts}}}" if verts else node.face_type

def visualize_dual_with_labels(faces_meta, adj, figsize=(12, 8), root=None, highlight_path=None):
    # Build graph
    G = nx.Graph()
    for u, nbrs in adj.items():
        for v in nbrs:
            G.add_edge(u, v)

    # Positions (any layout you like)
    pos = nx.spring_layout(G, seed=0)

    # Labels: EXTERIOR plus one per face id
    labels = {EXTERIOR: "EXTERIOR"}
    for i, face in enumerate(faces_meta):
        labels[i] = label_from_face(face)

    plt.figure(figsize=figsize)
    nx.draw(G, pos, node_size=800, with_labels=False)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    plt.axis("off")
    plt.savefig("./data/dual_graph.png", dpi=300)





# ============
# FINGERPRINTS
# ============

@lru_cache(maxsize=None)
def _safe_build_dual_from_structs(seq, db):
    try:
        return _original_build_dual_from_structs(seq, db)
    except Exception as e:
        print(f"⚠️ Skipping structure due to: {e}")
        return ([], {-1: []}, [])  # sentinel empty dual

def smallest_vertex_index(face) -> int:
    return min((idx for (_, idx) in getattr(face, "verts", [])), default=10**9)

def depths_from_exterior(adj):
    depth = {EXTERIOR: 0}
    q = deque([EXTERIOR])
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if v not in depth:
                depth[v] = depth[u] + 1
                q.append(v)
    depth.pop(EXTERIOR, None)  # keep only interior nodes
    return depth

def base_label(face):
    """
    Neighborhood identity at radius 0:
      - face type
      - multiset of base letters in this face (no indices)
    """
    bases = sorted(b for (b, _) in getattr(face, "verts", []))  # drop indices
    return f"{face.face_type}|bases:{''.join(bases)}"

def multiset_hash(strings):
    h = hashlib.sha256()
    for s in sorted(strings):
        h.update(s.encode()); h.update(b"|")
    return h.hexdigest()

def k_hop_directed_colors(faces, adj, k: int = 2):
    """
    WL refinement for exactly k rounds with direction (parent/sibling/child),
    base labels DO NOT use indices or absolute depth.
    Returns: dict node -> color (hash string)
    """
    depth = depths_from_exterior(adj)
    nodes = list(depth.keys())  # interior nodes only

    # round 0: index-free, depth-free node label
    color = {
        u: hashlib.sha256(base_label(faces[u]).encode()).hexdigest()
        for u in nodes
    }

    for _ in range(k):
        new_color = {}
        for u in nodes:
            parents  = [v for v in adj[u] if v != EXTERIOR and depth.get(v, 10**9) == depth[u]-1]
            siblings = [v for v in adj[u] if depth.get(v, -1) == depth[u]]
            children = [v for v in adj[u] if depth.get(v, -1) == depth[u]+1]

            sig = "|".join([
                color[u],
                "P:"+multiset_hash(color[v] for v in parents),
                "S:"+multiset_hash(color[v] for v in siblings),
                "C:"+multiset_hash(color[v] for v in children),
            ])
            new_color[u] = hashlib.sha256(sig.encode()).hexdigest()
        color = new_color
    return color

@lru_cache(maxsize=None)
def _dual(seq, db):
    # one reconstruction per unique (seq, db)
    return build_dual_from_structs(seq, db)  # -> (faces, adj, tops)

def _color2minindex_at_K(faces, adj, K):
    # returns {color: smallest_root_index_plus_1}
    colors = k_hop_directed_colors(faces, adj, k=K)
    m = {}
    for u, c in colors.items():
        idx1 = smallest_vertex_index(faces[u]) + 1
        if c not in m or idx1 < m[c]:
            m[c] = idx1
    return m

def _as_list_of_triples(cell):
    """Normalize whatever is in the cell to: [(idx, val, (1, U)), ...] or []."""
    if cell is None:
        return []
    # If someone stored it as a 2D object array of shape (k,3)
    if isinstance(cell, np.ndarray) and cell.ndim == 2 and cell.shape[1] == 3:
        return [ (cell[i,0], cell[i,1], cell[i,2]) for i in range(cell.shape[0]) ]
    # If it's already a python list
    if isinstance(cell, list):
        return cell
    # Fallback: wrap single dense vector
    return [cell]

def expected_vectors_from_boltz(dataframe, universe_size):
    """
    For each row (aptamer):
      - grab its list of embedding vectors (aligned, length = universe_size)
      - grab probs from Boltz_Distrib (already normalized to sum=1)
      - compute weighted average: E[vec] = sum_j p_j * vec_j
    Writes:
      dataframe["NeighborhoodExpected"] -> 1 vector (np.ndarray, float) per aptamer
      dataframe["NeighborhoodPresenceProb"] -> per-feature presence probability
    """
    exp_vectors = []
    presence_prob_vectors = []

    for i in range(len(dataframe)):
        distrib = dataframe.at[i, "Boltz_Distrib"] or []
        vecs = dataframe.at[i, "NeighborhoodEmbeddings"]
        if vecs is None:
            vecs = []

        triples = _as_list_of_triples(vecs)
        is_sparse = (len(triples) > 0 and isinstance(triples[0], tuple) and len(triples[0]) == 3  and isinstance(triples[0][0], np.ndarray))

        if not vecs or not distrib:
            exp_vectors.append(np.zeros(universe_size, dtype=float))
            presence_prob_vectors.append(np.zeros(universe_size, dtype=float))
            continue

        # Extract probs (already normalized!)
        probs = np.array([prob for (_, _, prob) in distrib], dtype=float)

        if is_sparse:
            # ---- Sparse path: vecs[t] = (idx:int[], val:uint32[], shape=(1,U)) ----
            m = min(len(probs), len(triples))
            E = np.zeros(U, dtype=np.float32)
            P = np.zeros(U, dtype=np.float32)

            for t in range(m):
                p = probs[t]
                if p <= 0:
                    continue
                idx, val, _shape = triples[t]
                if idx.size == 0:
                    continue
                # val is your ordinal; presence is 1 for all active cols
                E[idx] += p * val.astype(np.float32)
                P[idx] += p

            exp_vectors.append(E)
            presence_prob_vectors.append(P)
        else:
            # Defensive length match
            m = min(len(probs), V.shape[0])
            V = V[:m]
            probs = probs[:m]

            # Expected embedding
            E = probs @ V  # shape = (universe_size,)

            # Per-feature presence probability
            presence = (V > 0).astype(float)
            P_present = probs @ presence

            exp_vectors.append(E)
            presence_prob_vectors.append(P_present)

    dataframe["NeighborhoodExpected"] = exp_vectors
    dataframe["NeighborhoodPresenceProb"] = presence_prob_vectors


def build_sparse_neighborhood_embeddings(per_seq_color_maps, pos_global, U=None):
    """
    Produces dataframe['NeighborhoodEmbeddings'] in a RAM-light format:
      For each sequence: a list over structures of tuples
         (idx:int32[nnz], val:uint32[nnz], shape=(1,U))
    """
    if U is None:
        U = len(pos_global)

    embeddings_all = []
    empty_idx = np.empty(0, dtype=np.int32)
    empty_val = np.empty(0, dtype=np.uint32)
    empty_shape = (1, U)

    for maps_for_seq in per_seq_color_maps:
        row_list = []
        for c2i in (maps_for_seq or []):
            if not c2i:
                row_list.append((empty_idx, empty_val, empty_shape))
                continue
            # map feature/color -> global column
            idx = np.fromiter((pos_global[c] for c in c2i.keys()), dtype=np.int32)
            val = np.fromiter((c2i[c] for c in c2i.keys()), dtype=np.uint32)
            # keep indices sorted for painless merging later
            order = np.argsort(idx)
            row_list.append((idx[order], val[order], (1, U)))
        embeddings_all.append(row_list)

    return embeddings_all

# ---------------------------
# Helpers
# ---------------------------

EXTERIOR = -1
emap = DNA_ENERGIES
temp = 310.15  # 37°C in Kelvin
STRUCT_NULL = Struct(math.inf)

# Patch all 4-letter stack dimers with default zero energy if missing
bases = "ACGT"
for a, b, c, d in product(bases, repeat=4):
    pair_str = f"{a+b}/{d+c}"  # left/right stacked, reversed right
    if pair_str not in emap.NN:
        emap.NN[pair_str] = (0.0, 0.0)

# Your explicit patches
emap.NN['AT/AG'] = emap.NN['GA/TA'] = (0.7, 0.7)  # dummy but correct type
emap.NN['GG/GT'] = emap.NN['TG/GG'] = (0.0, 0.0)
emap.NN['TT/GC'] = emap.NN['CG/TT'] = (0.0, 0.0)

emap.INTERNAL_MM['GG/GT'] = emap.INTERNAL_MM['TG/GG'] = (0.0, 0.0)
emap.TERMINAL_MM['TT/GC'] = emap.TERMINAL_MM['CG/TT'] = (0.0, 0.0)

emap.NN['XX/XX'] = emap.INTERNAL_MM['XX/XX'] = emap.TERMINAL_MM['XX/XX'] = emap.DE['XX/XX'] = (0.0, 0.0)

_original_build_dual_from_structs = build_dual_from_structs  # keep a handle

# Monkey-patch so your helper uses the safe, cached version
build_dual_from_structs = _safe_build_dual_from_structs




if __name__=='__main__':
    # Example aptamer sequence (you can replace this with your own)
    sequence = "ACGACGGGGCACATTGTGCTGTTCATCTGTTCCGCAGGAGAGTCGT"


    probs = compute_structure_probabilities(sequence)
    probs = normalize_probs(probs)

    df_9th = pd.read_csv("./data/df_9th.csv")
    df_12th = pd.read_csv("./data/df_12th.csv")
    df_13th = pd.read_csv("./data/df_13th.csv")
    df_16th = pd.read_csv("./data/df_16th.csv")

    lib1_merged, lib2_merged, dataframe, lib1_overlap, lib2_overlap = get_libs(df_9th, df_12th, df_13th, df_16th, clean=False)

    dataframe = parse_data_frame(dataframe)

    # TEST THE DUAL GRAPH
    seq = dataframe.at[0, "Sequence"]
    distrib = compute_structure_probabilities(seq)
    dotbracket = distrib[2][0]  # Get the first structure's dotbracket

    structs = reconstruct_structs_from_dotbracket(seq, dotbracket)
    faces, adj, top_paris = build_dual_from_structs(seq, dotbracket)
    center, diam_path = balance_root(faces, adj)

    for struct in structs:
        print(f"Struct: {struct.desc} | Energy: {struct.e} | Pairs: {struct.ij}")

    print("Dotbracket:", dotbracket)
    print("Faces:", len(faces), "Adjacency size:", (adj))
    print("Diameter path:", diam_path, "| length =", len(diam_path))
    print("Chosen center:", center, "| tie-break key =", smallest_vertex_index(faces[center]))


    visualize_dual_with_labels(faces, adj)
    #visualize_dual_with_labels(faces, adj, root=center, highlight_path=diam_path)

    # ---------------------------
    # PASS A: build feature space
    # ---------------------------
    K = 4
    per_seq_color_maps = []   # list (per sequence) of list (per structure) of dict(color -> min_idx+1)
    universe_set = set()
    _debug = {"total": 0, "empty_faces": 0, "empty_colors": 0, "ok": 0, "exceptions": 0}


    for i in tqdm(range(len(dataframe)), desc=f"Pass A: duals→colors @K={K}"):
        seq = dataframe.at[i, "Sequence"]
        distrib = dataframe.at[i, "Boltz_Distrib"] or []
        maps_for_seq = []
        for (db, *_rest) in distrib:
            if not db:
                maps_for_seq.append({})
                continue
            try:
                faces, adj, _ = _dual(seq, db)
                if not faces:
                    _debug["empty_faces"] += 1
                    maps_for_seq.append({})
                    continue

                c2i = _color2minindex_at_K(faces, adj, K)
                if not c2i:
                    _debug["empty_colors"] += 1
                else:
                    universe_set.update(c2i.keys())
                    _debug["ok"] += 1

                maps_for_seq.append(c2i)
            except Exception as e:
                _debug["exceptions"] += 1
                print(f"[EXC] row {i}, db[:30]={db[:30]} -> {e}")
                maps_for_seq.append({})
            _debug["total"] += 1
        per_seq_color_maps.append(maps_for_seq)

    # ---------------------------
    # Freeze global universe
    # ---------------------------

    UNIVERSE = sorted(universe_set)
    pos_global = {c: j for j, c in enumerate(UNIVERSE)}

    print(f"[OK] Global universe size = {len(UNIVERSE)} at K={K}")
    print("DEBUG counts:", _debug)

    # ---------------------------
    # PASS B: Subgraph fingerprints creation
    # ---------------------------

    out_dir = "./data"
    os.makedirs(out_dir, exist_ok=True)

    # --- New PASS B: streaming expectations (no dense intermediates) ---
    U = len(UNIVERSE)                 # universe size
    dataframe["NeighborhoodEmbeddings"] = build_sparse_neighborhood_embeddings(
        per_seq_color_maps, pos_global, U=U
    )
    # pos_global already built above from UNIVERSE:
    # pos_global = {c: j for j, c in enumerate(UNIVERSE)}

    NeighborhoodExpected = []
    NeighborhoodPresenceProb = []

    for i, maps_for_seq in tqdm(enumerate(per_seq_color_maps),
                                total=len(per_seq_color_maps),
                                desc="Pass B: streaming E/P"):
        distrib = dataframe.at[i, "Boltz_Distrib"] or []

        if not maps_for_seq or not distrib:
            NeighborhoodExpected.append((np.array([], dtype=np.int64),
                                         np.array([], dtype=np.float32),
                                         (1, U)))
            NeighborhoodPresenceProb.append((np.array([], dtype=np.int64),
                                             np.array([], dtype=np.float32),
                                             (1, U)))
            continue

        m = min(len(distrib), len(maps_for_seq))
        E_map = defaultdict(np.float32)  # j -> E[idx]
        P_map = defaultdict(np.float32)  # j -> presence prob

        for t in range(m):
            p_t = float(distrib[t][2])
            if p_t <= 0:
                continue
            c2i = maps_for_seq[t] or {}
            for c, idx1 in c2i.items():
                j = pos_global.get(c)
                if j is None:  # shouldn't happen if UNIVERSE built from maps
                    continue
                # expectation of ordinal (idx1) and presence probability
                E_map[j] += np.float32(p_t * idx1)
                P_map[j] += np.float32(p_t)

        if E_map:
            idx = np.fromiter(E_map.keys(), dtype=np.int64)
            datE = np.fromiter((E_map[j] for j in idx), dtype=np.float32)
            datP = np.fromiter((P_map[j] for j in idx), dtype=np.float32)
            order = np.argsort(idx)
            NeighborhoodExpected.append((idx[order], datE[order], (1, U)))
            NeighborhoodPresenceProb.append((idx[order], datP[order], (1, U)))
        else:
            NeighborhoodExpected.append((np.array([], dtype=np.int64),
                                         np.array([], dtype=np.float32),
                                         (1, U)))
            NeighborhoodPresenceProb.append((np.array([], dtype=np.int64),
                                             np.array([], dtype=np.float32),
                                             (1, U)))

    dataframe["NeighborhoodExpected"] = NeighborhoodExpected
    dataframe["NeighborhoodPresenceProb"] = NeighborhoodPresenceProb

    print("Batched embeddings written to", out_dir)
    ##----------Build expected ordinal vector for each sequence--------
    expected_vectors_from_boltz(dataframe, universe_size=len(UNIVERSE))
    # save this to npz
    # convert to scipy sparse matrix format
    np.savez_compressed(os.path.join(out_dir, f"neighborhood_embeddings_{K}.npz"),
                        NeighborhoodExpected=dataframe["NeighborhoodExpected"].values,
                        NeighborhoodPresenceProb=dataframe["NeighborhoodPresenceProb"].values)
    print("[OK] NeighborhoodExpected and NeighborhoodPresenceProb added to DataFrame")
    print("Example expected vector:", dataframe["NeighborhoodExpected"].iloc[0])
    # get the dim of the expected vectors
    print("Expected vector dimension:", dataframe["NeighborhoodExpected"].iloc[0].shape)
    # get how many non-zeros
    print("Non-zero features in expected vector:", np.count_nonzero(dataframe["NeighborhoodExpected"].iloc[0]))
