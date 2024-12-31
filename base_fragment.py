import time
from utils import Utils
import numpy as np
from scipy.spatial import KDTree
from spatial_hash_grid import SpatialHashGrid
from enum import Enum
from PyQt5.QtGui import QColor
import json

class BaseFragment:
    class Type(Enum):
        TRGL_FRAGMENT = "3D"
        FRAGMENT = "2.5D" 
        UMBILICUS = "U"

    def __init__(self, name):
        self.name = name
        self.color = QColor()
        self.cvcolor = (0,0,0,0)
        self.created = Utils.timestamp()
        self.modified = Utils.timestamp()
        self.valid = False
        self.project = None
        self.type = None

    def notifyModified(self, tstamp=""):
        if tstamp == "":
            tstamp = Utils.timestamp()
        self.modified = tstamp
        # self.project can be None if BaseFragment is
        # a working Fragment of a TrglFragment
        if self.project is not None:
            self.project.notifyModified(tstamp)

    def setColor(self, qcolor, no_notify=False):
        self.color = qcolor
        rgba = qcolor.getRgbF()
        self.cvcolor = [int(65535*c) for c in rgba] 
        if not no_notify:
            self.notifyModified()

    def createView(self, project_view):
        print("BaseFragment: need to implement this class!")
        return None

    # class function
    def saveList(frags, path, stem):
        class_lists = {}
        for frag in frags:
            print("bsl", frag.name)
            # print(type(frag))
            t = type(frag)
            # t.asdf()
            l = class_lists.setdefault(t, [])
            l.append(frag)
        for cl, l in class_lists.items():
            cl.saveList(l, path, stem)

    def meshExportNeedsInfill(self):
        return False

    # class function
    def saveListAsObjMesh(fvs, path, infill, ppm):
        class_lists = {}
        for fv in fvs:
            frag = fv.fragment
            print("bsl", frag.name)
            # print(type(frag))
            t = type(frag)
            # t.asdf()
            l = class_lists.setdefault(t, [])
            l.append(fv)
        for cl, l in class_lists.items():
            err = cl.saveListAsObjMesh(l, path, infill, ppm, len(class_lists.items()))
            if err != "":
                return err
        return ""

    # class function
    # returns normals at points
    def pointNormals(pts, trgls):
        v0 = trgls[:,0]
        v1 = trgls[:,1]
        v2 = trgls[:,2]
        d01 = (pts[v1] - pts[v0]).astype(np.float64)
        d02 = (pts[v2] - pts[v0]).astype(np.float64)
        n3d = np.cross(d01, d02)
        ptn = np.zeros((len(pts), 3), np.float32)
        ptn[v0] += n3d
        ptn[v1] += n3d
        ptn[v2] += n3d
        l2 = np.sqrt(np.sum(ptn*ptn, axis=1)).reshape(-1,1)
        l2[l2==0] = 1.
        ptn /= l2
        return ptn

    # class function
    # returns normals of triangles
    def faceNormals(pts, trgls):
        v0 = trgls[:,0]
        v1 = trgls[:,1]
        v2 = trgls[:,2]
        d01 = (pts[v1] - pts[v0]).astype(np.float64)
        d02 = (pts[v2] - pts[v0]).astype(np.float64)
        n3d = np.cross(d01, d02)
        l2 = np.sqrt(np.sum(n3d*n3d, axis=1)).reshape(-1,1)
        l2[l2==0] = 1.
        n3d /= l2
        return n3d

    # class function
    def pointNormal(pt_index, pts, trgls):
        ltrgl_indexes = BaseFragment.trglsAroundPoint(pt_index, trgls)
        ltrgls = trgls[ltrgl_indexes]

        v0 = ltrgls[:,0]
        v1 = ltrgls[:,1]
        v2 = ltrgls[:,2]
        d01 = (pts[v1] - pts[v0]).astype(np.float64)
        d02 = (pts[v2] - pts[v0]).astype(np.float64)
        n3d = np.cross(d01, d02)
        npt = n3d.sum(axis=0)
        l2 = np.sqrt(np.sum(npt*npt))
        if l2 == 0:
            return npt
        return npt/l2

    # class function
    # returns 3 axes: axis along increasing stx, axis along increasing sty,
    # normal.  The 3 axes are orthonormal.
    # TODO: the calculation of stxaxis and styaxis should take into
    # account the local stxy values (uvpts), intead of looking
    # at the z axis as a proxy
    @staticmethod
    def pointThreeAxes(pt_index, xyzpts, uvpts, trgls):
        if uvpts is None or len(xyzpts) != len(uvpts):
            return None
        ltrgl_indexes = BaseFragment.trglsAroundPoint(pt_index, trgls)
        ltrgls = trgls[ltrgl_indexes]

        v0 = ltrgls[:,0]
        v1 = ltrgls[:,1]
        v2 = ltrgls[:,2]
        d01 = (xyzpts[v1] - xyzpts[v0]).astype(np.float64)
        d02 = (xyzpts[v2] - xyzpts[v0]).astype(np.float64)
        n3d = np.cross(d01, d02)
        npt = n3d.sum(axis=0)
        # print("npt", npt)
        l2 = np.sqrt(np.sum(npt*npt))
        if l2 == 0:
            return None
        normal = npt/l2
        # print("normal", normal)
        # In the transposed coordinate system, this represents
        # the axis along the scroll's original z axis.
        # This should be more or less aligned with the sty axis
        zaxis = np.array((0., 1., 0.), dtype=np.float32)
        stxaxis = np.cross(normal, zaxis)
        # stxaxis = np.cross(zaxis, normal)
        # stxaxis *= -1
        # print("stxaxis", stxaxis)
        l2 = np.sqrt(np.sum(stxaxis*stxaxis))
        if l2 == 0:
            return None
        stxaxis /= l2
        styaxis = np.cross(normal, stxaxis)
        # styaxis *= -1
        return np.array((stxaxis, styaxis, normal)).T


    @staticmethod
    def calculateSqCm(pts, trgls, voxel_size_um):
        if trgls is None or len(trgls) == 0:
            return 0.
        v0 = trgls[:,0]
        v1 = trgls[:,1]
        v2 = trgls[:,2]
        d01 = (pts[v1] - pts[v0]).astype(np.float64)
        d02 = (pts[v2] - pts[v0]).astype(np.float64)
        n3d = np.cross(d01, d02)
        l2 = np.sqrt(np.sum(n3d*n3d, axis=1))
        area_sq_mm_trg = np.sum(l2)*voxel_size_um*voxel_size_um/(2*1000000)
        sqcm = area_sq_mm_trg/100.
        return sqcm

    # class function
    def findNeighbors(trgls):
        index = np.indices((len(trgls),1))[0]
        ones = np.ones((len(trgls),1), dtype=np.int32)
        zeros = np.zeros((len(trgls),1), dtype=np.int32)
        twos = 2*ones
        
        e01 = np.concatenate((trgls[:, (0,1)], index, twos, ones), axis=1)
        e12 = np.concatenate((trgls[:, (1,2)], index, zeros, ones), axis=1)
        e20 = np.concatenate((trgls[:, (2,0)], index, ones, ones), axis=1)
        
        edges = np.concatenate((e01,e12,e20), axis=0)
        rev = (edges[:,0] > edges[:,1])
        edges[rev,0:2] = edges[rev,1::-1]
        edges[rev,4] = -1
        edges = edges[edges[:,4].argsort()]
        edges = edges[edges[:,1].argsort(kind='mergesort')]
        edges = edges[edges[:,0].argsort(kind='mergesort')]
        
        ediff = np.diff(edges, axis=0)
        duprows = np.where(((ediff[:,0]==0) & (ediff[:,1]==0)))[0]
        duprows2 = np.sort(np.append(duprows, duprows+1))
        bdup = np.zeros((len(edges)), dtype=np.bool_)
        bdup[duprows2] = True
        
        neighbors = np.full((len(trgls), 3), -1, dtype=np.int32)
        
        eplus = edges[duprows+1,:4]
        eminus = edges[duprows,:4]
        # print(eplus)
        # print(eminus)
        neighbors[eplus[:,2],eplus[:,3]] = eminus[:,2]
        neighbors[eminus[:,2],eminus[:,3]] = eplus[:,2]
        return neighbors

    # returns list of indexes of those trgls that have pt_index as
    # a vertex
    def trglsAroundPoint(pt_index, trgls):
        bvec = (trgls[:,0] == pt_index) | (trgls[:,1] == pt_index) | (trgls[:,2] == pt_index)
        tindexes = np.where(bvec)[0]
        return tindexes.tolist()
    
    def getType(self):
        return self.type.value if self.type else None

class BaseFragmentView:

    def __init__(self, project_view, fragment):
        self.project_view = project_view
        self.fragment = fragment
        self.sqcm = 0.
        self.cur_volume_view = None
        self.visible = True
        self.active = False
        self.mesh_visible = True
        self.map_image = None
        self.map_corners = None
        self.modified = Utils.timestamp()
        self.local_points_modified = Utils.timestamp()
        self.normal_offset = 0.
        self.kd_tree = None  # For spatial queries
        self.adjacency_list = None
        self.spatial_hash_grid = None
        self.k_neighbors = 1   # Default number of neighbors
        self.current_radius = 30.0  # Default radius in global units
        self.selected_nodes = set()  # Store selected node indices

    def allowAutoExtrapolation(self):
        return False

    def allowAutoInterpolation(self):
        return False

    def setVolumeView(self, vol_view):
        if vol_view == self.cur_volume_view:
            return
        self.cur_volume_view = vol_view
        self.clearCaches()
        if vol_view is not None:
            # Don't rebuild adjacency list when just changing volume view
            self.setLocalPoints(False, True, False, False)

    def notifyModified(self, tstamp=""):
        if tstamp == "":
            tstamp = Utils.timestamp()
        self.modified = tstamp
        # print("fragment view", self.fragment.name,"modified", tstamp)
        self.project_view.notifyModified(tstamp)

    def getZsurfPoints(self, axis, axis_pos):
        return None

    def line(self):
        return None

    def getLinesOnSlice(self, axis, axis_pos):
        return None, None

    def triangulate(self):
        return None

    def addPoint(self, tijk, stxy):
        return None

    def deletePointByIndex(self, index):
        return None

    def setLiveZsurfUpdate(self, flag):
        return None

    def setWorkingRegion(self, index, max_angle):
        return None

    def workingZsurf(self):
        return None

    def workingSsurf(self):
        return None

    def workingVpoints(self):
        return np.zeros((0,4), dtype=np.bool_)

    def workingTrgls(self):
        return None

    def hasWorkingNonWorking(self):
        return (False, False)

    def workingLine(self):
        return None

    def workingLineAxis(self):
        return -1

    def workingLineAxisPosition(self):
        return 0

    def activeAndAligned(self):
        if not self.active:
            return False
        return self.aligned()

    def rebuildStPoints(self):
        return

    def reparameterize(self):
        return
    
    def fragFromDict(self):
        return

    # direction is not used here, but this notifies fragment view
    # to recompute things
    def setVolumeViewDirection(self, direction):
        self.clearCaches()
        # Don't rebuild adjacency list when just changing direction
        self.setLocalPoints(False, True, False, False)

    def clearCaches(self):
        return None
    
    def pointNormals(self):
        self.triangulate()
        trgls = self.trgls()
        if trgls is None:
            return
        pts3d = self.fpoints[:,:3]
        return BaseFragment.pointNormals(pts3d, trgls)

    def pointNormal(self, pt_index):
        self.triangulate()
        trgls = self.trgls()
        if trgls is None:
            return
        pts3d = self.vpoints[:,:3]
        return BaseFragment.pointNormal(pt_index, pts3d, trgls)

    def moveAlongNormalsSign(self):
        return 1.

    def moveAlongNormals(self, step):
        ns = self.pointNormals()
        if ns is None:
            print("No normals found")
            return
        # print("man", self.fpoints.shape, ns.shape)
        # fpoints has 4 elements; the 4th is the index
        sgn = self.moveAlongNormalsSign()
        self.fpoints[:, :3] += sgn*step*ns
        self.fragment.gpoints = self.cur_volume_view.volume.transposedIjksToGlobalPositions(self.fpoints, self.fragment.direction)
        self.fragment.notifyModified()
        # Don't rebuild adjacency list when just moving along normals
        self.setLocalPoints(True, True, True, False)

    def moveInK(self, step):
        # if len(self.fpoints) > 0:
        #     print("before", self.fragment.gpoints[0], self.fpoints[0])
        self.fpoints[:,2] += step
        self.fragment.gpoints = self.cur_volume_view.volume.transposedIjksToGlobalPositions(self.fpoints, self.fragment.direction)
        # if len(self.fpoints) > 0:
        #     print("after", self.fragment.gpoints[0], self.fpoints[0])
        self.fragment.notifyModified()
        # Don't rebuild adjacency list when just moving in K
        self.setLocalPoints(True, True, True, False)

    # returns 3 axes: axis along increasing stx, axis along increasing sty,
    # normal.  The 3 axes are orthonormal.
    # TODO: the calculation of stxaxis and styaxis should take into
    # account the local stxy values (uvpts), intead of looking
    # at the z axis as a proxy
    def localStAxes(self, pt_index):
        xyzpts = self.fragment.gpoints
        uvpts = self.stpoints
        trgls = self.trgls()
        if uvpts is None or len(xyzpts) != len(uvpts):
            return None
        ltrgl_indexes = BaseFragment.trglsAroundPoint(pt_index, trgls)
        ltrgls = trgls[ltrgl_indexes]

        v0 = ltrgls[:,0]
        v1 = ltrgls[:,1]
        v2 = ltrgls[:,2]
        d01 = (xyzpts[v1] - xyzpts[v0]).astype(np.float64)
        d02 = (xyzpts[v2] - xyzpts[v0]).astype(np.float64)
        n3d = np.cross(d01, d02)
        npt = n3d.sum(axis=0)
        # print("npt", npt)
        l2 = np.sqrt(np.sum(npt*npt))
        if l2 == 0:
            return None
        normal = npt/l2
        # print("normal", normal)
        # In the global coordinate system, this represents
        # the axis along the scroll's original z axis.
        # This should be more or less aligned with the sty axis
        zaxis = np.array((0., 0., 1.), dtype=np.float32)
        stxaxis = np.cross(normal, zaxis)
        # stxaxis = np.cross(zaxis, normal)
        # stxaxis *= -1
        # print("stxaxis", stxaxis)
        l2 = np.sqrt(np.sum(stxaxis*stxaxis))
        if l2 == 0:
            return None
        stxaxis /= l2
        styaxis = np.cross(normal, stxaxis)
        # styaxis *= -1
        axes = np.array((stxaxis, styaxis, normal)).T
        # print(normal, axes)
        # return np.array((stxaxis, styaxis, normal)).T
        return axes
    
    def localStAxesBatched(self, indices):
        """
        Fully vectorized version of localStAxes that handles multiple indices at once.
        Returns array of shape (n_indices, 3, 3) containing axes for each point.
        """
        n_indices = len(indices)
        axes_list = np.zeros((n_indices, 3, 3), dtype=np.float64)
        
        # Get all triangles
        trgls = self.trgls()
        if len(trgls) == 0:
            return axes_list
            
        # Find all triangles containing any of the query points
        point_mask = np.isin(trgls, indices)
        relevant_trgls = trgls[point_mask.any(axis=1)]
        
        if len(relevant_trgls) == 0:
            return axes_list
            
        # Calculate normals for all relevant triangles at once
        trgl_pts = self.vpoints[relevant_trgls, :3]
        v1 = trgl_pts[:, 1] - trgl_pts[:, 0]
        v2 = trgl_pts[:, 2] - trgl_pts[:, 0]
        normals = np.cross(v1, v2)
        
        # Normalize all normals at once
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        mask = norms > 0
        normals = np.where(mask, normals / norms, 0)
        
        # Create a mapping from point indices to their normals
        point_to_normals = {idx: [] for idx in indices}
        for i, trgl in enumerate(relevant_trgls):
            for vertex in trgl:
                if vertex in point_to_normals:
                    point_to_normals[vertex].append(normals[i])
        
        # Convert lists to arrays and compute average normals
        avg_normals = np.zeros((n_indices, 3))
        for i, idx in enumerate(indices):
            if point_to_normals[idx]:
                normal = np.mean(point_to_normals[idx], axis=0)
                norm = np.linalg.norm(normal)
                if norm > 0:
                    avg_normals[i] = normal / norm
        
        # Compute all axes at once
        z_axis = np.array([0., 0., 1.])
        
        # Calculate x axes (cross product with z_axis)
        x_axes = np.cross(np.tile(z_axis, (n_indices, 1)), avg_normals)
        x_norms = np.linalg.norm(x_axes, axis=1, keepdims=True)
        mask = x_norms > 0
        x_axes = np.where(mask, x_axes / x_norms, np.array([1., 0., 0.]))
        
        # Calculate y axes (cross product of normal and x axis)
        y_axes = np.cross(avg_normals, x_axes)
        
        # Stack all axes
        axes_list[:, 0] = x_axes
        axes_list[:, 1] = y_axes
        axes_list[:, 2] = avg_normals
        
        return axes_list

    def buildKDTrees(self, recursion_ok, build_kd_tree=True, build_adjacency_list=True, build_spatial_hash_grid=True):
        # print("buildKDTrees", recursion_ok, build_kd_tree, build_adjacency_list, build_spatial_hash_grid)
        if not recursion_ok:
            return
        # print("fragment datastructures; adj list, kdtree, spatial hash grid")
        if not hasattr(self, 'vpoints') or self.vpoints is None or len(self.vpoints) == 0:
            self.kd_tree = None
            self.adjacency_list = None
            self.spatial_hash_grid = None
            return
        
        # Build adjacency list from triangles
        trgls = self.trgls()
        if build_adjacency_list:
            print("building adjacency list")
            if trgls is not None and len(trgls) > 0:
                # print("building adjacency list")
                stime = time.time()
                self.adjacency_list = [set() for _ in range(len(self.vpoints))]
                
                # Check if trgls contains triangles (3 vertices each) or just indices
                if len(trgls.shape) > 1 and trgls.shape[1] == 3:
                    # Handle triangulated mesh
                    for tri in trgls:
                        a, b, c = tri
                        self.adjacency_list[a].update([b, c])
                        self.adjacency_list[b].update([a, c])
                        self.adjacency_list[c].update([a, b])
                else:
                    # Handle line segments (like umbilicus)
                    for i in range(len(trgls)-1):
                        self.adjacency_list[i].add(i+1)
                        self.adjacency_list[i+1].add(i)
                        
                # print("adjacency list built in", time.time() - stime)
            else:
                self.adjacency_list = None
        
        # Build KD tree using global xyz coordinates
        if hasattr(self, 'fragment') and hasattr(self.fragment, 'gpoints'):
            if build_kd_tree:   
                # print("building kd tree")
                stime = time.time()
                self.kd_tree = KDTree(self.fragment.gpoints)
                # print("kd tree built in", time.time() - stime)

            # if build_spatial_hash_grid:
            #     # print("building spatial hash grid")
            #     stime = time.time()
            #     self.spatial_hash_grid = SpatialHashGrid(self.fragment.gpoints, thickness=10)
            #     # print("spatial hash grid built in", time.time() - stime)

    def updateSelectedNodes(self, point_index, k=None, radius=None, use_3d=False):
        """
        Select nodes either by k-nearest neighbors or radius.
        Uses either connectivity-based or spatial-based selection.
        
        Args:
            point_index: Index of the center point
            k: Number of neighbors (if None, uses self.k_neighbors)
            radius: Radius to search within (if provided, overrides k)
            use_3d: If True, use spatial distance, otherwise use connectivity
        """
        
        if point_index < 0 or point_index >= len(self.vpoints):
            print("point_index out of range")
            self.selected_nodes = set()
            return

        if use_3d:
            # Use KDTree for spatial queries
            if not hasattr(self, 'kd_tree') or self.kd_tree is None:
                print("KD tree is None")
                self.selected_nodes = set()
                return
            
            points = self.fragment.gpoints
            if radius is not None:
                # Radius-based query
                indices = self.kd_tree.query_ball_point(points[point_index], radius)
                self.selected_nodes = set(indices)
            else:
                # K-nearest neighbors query
                if k is None:
                    k = self.k_neighbors
                k = min(k + 1, len(points))  # +1 to include the point itself
                distances, indices = self.kd_tree.query(points[point_index], k=k)
                self.selected_nodes = set(indices.tolist())
        else:
            # Use adjacency list for connectivity-based queries
            if not hasattr(self, 'adjacency_list') or self.adjacency_list is None:
                print("Adjacency list is None")
                self.selected_nodes = set()
                return
            
            # Start with just the selected vertex
            self.selected_nodes = {point_index}

            # If steps is 0, we're done - just return the single point
            steps = k if k is not None else self.k_neighbors
            if steps <= 0:
                return

            # Otherwise do the BFS traversal
            current_nodes = {point_index}
            all_nodes = current_nodes.copy()

            # Traverse the graph using BFS
            for _ in range(steps):
                next_nodes = set()
                for node in current_nodes:
                    next_nodes.update(self.adjacency_list[node])
                current_nodes = next_nodes - all_nodes
                all_nodes.update(current_nodes)
                if not current_nodes:  # No more nodes to explore
                    break

            self.selected_nodes = all_nodes
        
        # return self.selected_nodes

    def updateSelectedNodesFromPoints(self, points, radius):
        """
        Select nodes within radius of any of the given points.
        Uses KD-tree for efficient spatial queries.
        
        Args:
            points: Array of points in the same coordinate space as the KD-tree
            radius: Search radius around each point
        """
        if self.kd_tree is None:
            print("KD tree is None")
            self.selected_nodes = set()
            return
        
        # Query KD-tree for each point
        self.selected_nodes = set()
        print("kd tree query", points.shape, points[0], radius)
        for point in points:
            indices = self.kd_tree.query_ball_point(point, radius)
            self.selected_nodes.update(indices)

