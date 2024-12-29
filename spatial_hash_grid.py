import numpy as np
from collections import defaultdict
import math

class SpatialHashGrid:
    """
    Maintains a 3-axis dictionary-based "grid" for fast slice-based queries.
    Each axis is stored in a dict: { integer_axis_value -> set_of_point_indices }.
    We define a 'thickness' so that when we query e.g. z=100 slice, we gather points in
    z in [100-thickness .. 100+thickness].
    """

    def __init__(self, points, thickness=10):
        """
        :param points: Nx3 array of point positions
        :param thickness: number of voxels on either side of the slice plane
                        that we consider "in slice." Default=10
        """
        self.thickness = thickness
        
        # For each axis (x=0, y=1, z=2), we keep a dict:
        #    axis_buckets[axis][int_coord] = set_of_point_indices
        # e.g. axis_buckets[0][ix] = {point1, point5, ...} for all points with floor(x) == ix
        self.axis_buckets = [
            defaultdict(set),  # x-buckets
            defaultdict(set),  # y-buckets
            defaultdict(set)   # z-buckets
        ]

        # Store point coordinates for convenience
        self.points = points.copy()
        
        # Build the structure
        self.build(points)

    def build(self, points):
        """
        Builds the 3-axis hash from the point positions.
        Clears old data if any.
        """
        # Clear existing data
        for a in range(3):
            self.axis_buckets[a].clear()

        for point_index, coord in enumerate(points):
            x, y, z = coord
            # Convert to integer voxel indices:
            ix = math.floor(x)
            iy = math.floor(y)
            iz = math.floor(z)

            # Insert into each axis bucket
            self.axis_buckets[0][ix].add(point_index)
            self.axis_buckets[1][iy].add(point_index)
            self.axis_buckets[2][iz].add(point_index)

    def queryPointsOnSlice(self, axis, slice_val):
        """
        Returns a set of point indices that are within 'thickness' of 'slice_val'
        in the specified axis.
        
        :param axis: 0 for x, 1 for y, 2 for z
        :param slice_val: the integer plane coordinate for that axis
        :return: set of point indices that are in [slice_val-thickness .. slice_val+thickness]
        """
        result_points = set()

        # Gather all buckets from slice_val - thickness to slice_val + thickness
        start_bucket = slice_val - self.thickness
        end_bucket   = slice_val + self.thickness

        axis_dict = self.axis_buckets[axis]

        for b in range(start_bucket, end_bucket + 1):
            if b in axis_dict:
                result_points.update(axis_dict[b])

        return result_points

    def updatePointPosition(self, point_index, old_coord, new_coord):
        """
        Updates the position of a single point in the 3-axis hash.
        
        :param point_index: the integer ID of the point
        :param old_coord: (x, y, z) float of the old position
        :param new_coord: (nx, ny, nz) float of the new position
        """
        ox, oy, oz = old_coord
        nx, ny, nz = new_coord

        # Floor the old and new coords:
        iox = math.floor(ox)
        ioy = math.floor(oy)
        ioz = math.floor(oz)

        inx = math.floor(nx)
        iny = math.floor(ny)
        inz = math.floor(nz)

        # For each axis, if the integer coordinate changed, remove from old, add to new
        # Axis=0 => x-buckets
        if iox != inx:
            # remove from old x bucket
            old_set = self.axis_buckets[0].get(iox, None)
            if old_set and point_index in old_set:
                old_set.remove(point_index)
            # add to new x bucket
            self.axis_buckets[0][inx].add(point_index)
        
        # Axis=1 => y-buckets
        if ioy != iny:
            old_set = self.axis_buckets[1].get(ioy, None)
            if old_set and point_index in old_set:
                old_set.remove(point_index)
            self.axis_buckets[1][iny].add(point_index)
        
        # Axis=2 => z-buckets
        if ioz != inz:
            old_set = self.axis_buckets[2].get(ioz, None)
            if old_set and point_index in old_set:
                old_set.remove(point_index)
            self.axis_buckets[2][inz].add(point_index)

        # Update the stored point coordinate
        self.points[point_index] = new_coord

    def getPointCoord(self, point_index):
        """Retrieve the (x,y,z) coordinate from our internal store."""
        return self.points[point_index]

    def removePoint(self, point_index):
        """Remove a point from all axis buckets."""
        if point_index < len(self.points):
            (ox, oy, oz) = self.points[point_index]
            iox = math.floor(ox)
            ioy = math.floor(oy)
            ioz = math.floor(oz)
            if iox in self.axis_buckets[0]:
                self.axis_buckets[0][iox].discard(point_index)
            if ioy in self.axis_buckets[1]:
                self.axis_buckets[1][ioy].discard(point_index)
            if ioz in self.axis_buckets[2]:
                self.axis_buckets[2][ioz].discard(point_index)
            # Mark point as removed in points array
            self.points[point_index] = (None, None, None)
