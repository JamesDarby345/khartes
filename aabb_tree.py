import numpy as np
from collections import namedtuple

AABB = namedtuple('AABB', ['min', 'max'])

class BVHNode:
    __slots__ = ['bbox', 'left', 'right', 'triangles']
    def __init__(self, bbox, left=None, right=None, triangles=None):
        self.bbox = bbox
        self.left = left
        self.right = right
        self.triangles = triangles if triangles is not None else []

class AABBTree:
    def __init__(self, pts, trgls, max_tris=32, max_depth=32):
        """
        pts: Nx3 array of vertex positions
        trgls: Mx3 array of triangle indices
        """
        self.pts = pts
        self.trgls = trgls
        self.max_tris = max_tris
        self.max_depth = max_depth
        if len(trgls) > 0:
            self.root = self._build_bvh(trgls, 0)
        else:
            self.root = None

    def _compute_aabb_for_triangles(self, trgls):
        p = self.pts[trgls.flatten()]
        pmin = p.min(axis=0)
        pmax = p.max(axis=0)
        return AABB(pmin, pmax)

    def _split_triangles(self, trgls, axis):
        centroids = self.pts[trgls].mean(axis=1)
        median = np.median(centroids[:, axis])
        left_mask = centroids[:, axis] < median
        right_mask = ~left_mask
        # If all on one side, this is degenerate split
        if not np.any(right_mask):
            right_mask = left_mask
        if not np.any(left_mask):
            left_mask = right_mask
        return trgls[left_mask], trgls[right_mask]

    def _build_bvh(self, trgls, depth):
        if len(trgls) <= self.max_tris or depth >= self.max_depth:
            bbox = self._compute_aabb_for_triangles(trgls)
            return BVHNode(bbox, triangles=trgls)

        bbox = self._compute_aabb_for_triangles(trgls)
        axis_lengths = bbox.max - bbox.min
        axis = np.argmax(axis_lengths)

        left_tris, right_tris = self._split_triangles(trgls, axis)
        if len(left_tris) == 0 or len(right_tris) == 0:
            # Degenerate split
            return BVHNode(bbox, triangles=trgls)

        left_child = self._build_bvh(left_tris, depth+1)
        right_child = self._build_bvh(right_tris, depth+1)
        cmin = np.minimum(left_child.bbox.min, right_child.bbox.min)
        cmax = np.maximum(left_child.bbox.max, right_child.bbox.max)
        parent_bbox = AABB(cmin, cmax)
        return BVHNode(parent_bbox, left_child, right_child, None)

    def _aabb_intersect(self, a, b):
        return (a.min[0] <= b.max[0] and a.max[0] >= b.min[0] and
                a.min[1] <= b.max[1] and a.max[1] >= b.min[1] and
                a.min[2] <= b.max[2] and a.max[2] >= b.min[2])

    def _query_bvh(self, node, query_box):
        if node is None:
            return []
        if not self._aabb_intersect(node.bbox, query_box):
            return []
        if node.triangles is not None:
            # Leaf
            return node.triangles
        return self._query_bvh(node.left, query_box) + self._query_bvh(node.right, query_box)

    def query(self, bbox):
        """
        bbox: ((xmin,ymin,zmin), (xmax,ymax,zmax)) query AABB
        returns indices of triangles intersecting bbox
        """
        query_box = AABB(np.array(bbox[0]), np.array(bbox[1]))
        return self._query_bvh(self.root, query_box)

