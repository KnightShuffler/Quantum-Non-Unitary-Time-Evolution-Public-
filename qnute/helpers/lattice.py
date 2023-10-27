import numpy as np
from . import int_to_base

#----------------------------#
# Manhattan Distance Helpers #
#----------------------------#

def in_lattice(point, d, l):
    '''
    Returns True if the point is within a d-dimensional lattice
    with non-negative coordinates, of side length l
    '''
    if type(point) is not tuple:
        point = (point,)
    for i in range(d):
        if point[i] < 0 or point[i] >= l:
            return False
    return True

def get_center(points):
    '''
    Returns the coordinates of the center of the points
    '''
    n = len(points)
    return np.sum(points,axis=0)/n

def manhattan_dist(a, b):
    '''
    Returns the Manhatan distance between points a and b
    '''
    return np.sum(np.abs(a - b))

def within_radius(center, point, radius):
    '''
    Returns True if the point is within a Manhattan distance of radius 
    away from the center
    '''
    return manhattan_dist(point, center) <= radius

def get_m_sphere(c, R, d, l):
    '''
    Returns the list of points at a Manhattan distance of R from the point c
    in a d-dimensional lattice of side length l
    '''
    sphere = []
    # Cast c to a tuple, important for the d=1 base case
    try:
        c = tuple(c)
    except TypeError as e:
        if type(c) is not tuple:
            c = (c,)

    lb = max(int(np.ceil(c[0] - R)), 0)
    ub = min(int(np.floor(c[0] + R)), l-1)
    
    for i in np.arange(lb, ub+1, 1):
        if d > 1:
            # calculate new radius
            if i <= c[0]:
                r = i - (c[0] - R)
            else:
                r = R - (i - c[0])
            sub_sphere = get_m_sphere(c[1:], r, d-1, l)
            for point in sub_sphere:
                sphere.append( (i,) + point )
        else:
            sphere.append( (i,) )
    return sphere

def min_bounding_sphere(points):
    center = get_center(points)
    max_rad = 0
    for p in points:
        rad = manhattan_dist(p, center)
        if rad > max_rad:
            max_rad = rad
    return center, max_rad

#------------------------#
# k-local Domain Related #
#------------------------#
def get_lattice_paths(start, depth,d,l, illegal_dir=None):
    '''
    returns all paths of depth=depth originating from the point start in the l^d lattice using depth first search
    the paths are sorted coordinate wise
    '''
    if type(start) is not tuple:
        start = (start, )
    assert len(start) == d, 'start must be a d-dimensional tuple'
    
    if depth == 0:
        return [[start]]  if d > 1 else [[start[0]]]
    
    paths = []
    for i in range(d):
        for j in range(2):
            if illegal_dir == (i,(-1)**j):
                continue
            n_node = list(start)
            n_node[i] += (-1)**j
            n_node = tuple(n_node)
            if in_lattice(n_node,d,l):
                n_paths = get_lattice_paths(n_node, depth-1, d,l, illegal_dir=(i,(-1)**(j+1)))
                for path in n_paths:
                    paths.append( [start if d > 1 else start[0]]+path )
                    
    return paths

def get_k_local_domains(k,d,l):
    '''
    returns a list of all possible geometric k-local domains (lattice coordinates lists) for a given lattice.
    '''
    domains = set()
    for i in range(l**d):
        point = tuple(int_to_base(i,l,d))
        for k_ in range(k):
            n_paths = get_lattice_paths(point,k_,d,l)
            for path in n_paths:
                s_path = sorted(path)
                domains.add(tuple(s_path))
    return sorted(list(domains))