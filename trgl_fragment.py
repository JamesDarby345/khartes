import math
import json
import numpy as np

from pathlib import Path
from collections import deque
import traceback
from scipy.spatial import Delaunay
import scipy
import cv2
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import os

from utils import Utils
from base_fragment import BaseFragment, BaseFragmentView
from fragment import Fragment, FragmentView
from uv_mapper import UVMapper

from PyQt5.QtGui import QColor

class TrglFragment(BaseFragment):
    def __init__(self, name):
        super(TrglFragment, self).__init__(name)
        self.gpoints = np.zeros((0,3), dtype=np.float32)
        self.gtpoints = np.zeros((0,2), dtype=np.float32)
        self.trgls = np.zeros((0,3), dtype=np.int32)
        self.direction = 0
        self.params = {}
        self.type = BaseFragment.Type.TRGL_FRAGMENT
        self.obj_path = None

    # class function
    # expected to return a list of fragments, but always
    # returns only one
    def load(obj_file):
        print("loading obj file", obj_file)
        pname = Path(obj_file)
        try:
            fd = pname.open("r")
        except:
            return None

        name = pname.stem
        
        vrtl = []
        tvrtl = []
        trgl = []
        
        created = ""
        frag_name = ""
        for line in fd:
            line = line.strip()
            words = line.split()
            if words == []: # prevent crash on empty line
                continue
            if words[0][0] == '#':
                if len(words) > 2: 
                    if words[1] == "Created:":
                        created = words[2]
                    if words[1] == "Name:":
                        frag_name = words[2]
            elif words[0] == 'v':
                # len is 7 if the vrt has color attached
                # (color is ignored)
                if len(words) == 4 or len(words) == 7:
                    vrtl.append([float(w) for w in words[1:4]])
            elif words[0] == 'vt':
                if len(words) == 3:
                    tvrtl.append([float(w) for w in words[1:]])
            elif words[0] == 'f':
                if len(words) == 4:
                    # implicit assumption that v == vt
                    trgl.append([int(w.split('/')[0])-1 for w in words[1:]])
        print("tf obj reader", len(vrtl), len(tvrtl), len(trgl))
        
        if frag_name == "":
        #     frag_name = name.replace("_",":").replace("p",".")
            frag_name = name
        trgl_frag = TrglFragment(frag_name)
        if len(vrtl) > 0:
            trgl_frag.gpoints = np.array(vrtl, dtype=np.float32)
        else:
            trgl_frag.gpoints = np.zeros((0,3), dtype=np.float32)
        if len(tvrtl) > 0:
            trgl_frag.gtpoints = np.array(tvrtl, dtype=np.float32)
        else:
            trgl_frag.gtpoints = np.zeros((0,2), dtype=np.float32)
        if len(trgl) > 0:
            trgl_frag.trgls = np.array(trgl, dtype=np.int32)
        else:
            trgl_frag.trgls = np.zeros((0,3), dtype=np.int32)
        if created == "":
            ts = Utils.vcToTimestamp(name)
            if ts is not None:
                created = ts
        if created != "":
            trgl_frag.created = created
        trgl_frag.params = {}
        
        mname = pname.with_suffix(".mtl")
        fd = None
        color = None
        try:
            fd = mname.open("r")
        except:
            print("failed to open mtl file",mname.name)
            pass

        if fd is not None:
            for line in fd:
                words = line.split()
                # print("words[0]", words[0])
                if len(words) == 4 and words[0] == "Kd":
                    try:
                        # print("words", words)
                        r = float(words[1])
                        g = float(words[2])
                        b = float(words[3])
                    except:
                        continue
                    # print("rgb", r,g,b)
                    color = QColor.fromRgbF(r,g,b)
                    break

        if color is None:
            color = Utils.getNextColor()
        trgl_frag.setColor(color, no_notify=True)
        trgl_frag.valid = True
        trgl_frag.neighbors = BaseFragment.findNeighbors(trgl_frag.trgls)
        trgl_frag.obj_path = obj_file
        print(trgl_frag.name, trgl_frag.color.name(), trgl_frag.gpoints.shape, trgl_frag.gtpoints.shape, trgl_frag.trgls.shape)
        # print("tindexes", BaseFragment.trglsAroundPoint(100, trgl_frag.trgls))
        if len(trgl_frag.gtpoints) > 0:
            tmp_fv = trgl_frag.createView(None)
            tmp_fv.setScaledTexturePoints(similar=False)
            trgl_frag.gtpoints = tmp_fv.stpoints

        return [trgl_frag]

    def createView(self, project_view):
        return TrglFragmentView(project_view, self)

    def createCopy(self, name):
        frag = TrglFragment(name)
        frag.setColor(self.color, no_notify=True)
        frag.gpoints = np.copy(self.gpoints)
        frag.gtpoints = np.copy(self.gtpoints)
        frag.trgls = np.copy(self.trgls)
        frag.neighbors = np.copy(self.neighbors)
        frag.valid = True
        return frag

    # class function
    def saveList(frags, path, stem):
        # cfixed = self.created.replace(':',"_").replace('.',"p")
        for frag in frags:
            cfixed = Utils.timestampToVc(frag.created)
            if cfixed is None:
                print("Could not convert self.created", self.created, "to vc")
                cfixed = frag.created.replace(':',"_").replace('.',"p")
            print("tsl", frag.name)
            fpath = path / cfixed
            frag.save(fpath)

    # class function
    def saveListAsObjMesh(fvs, path, infill, ppm, class_count):
        print("TF slaom", len(fvs), class_count)
        name = path.name
        stem = path.stem
        for fv in fvs:
            frag = fv.fragment
            if class_count > 1 or len(fvs) > 1:
                newname = stem+"_"+frag.name
                opath = path.with_name(newname)
            else:
                opath = path
            print("TF slaom", opath)
            frag.save(opath, ppm, fv)

        return ""

    def save(self, fpath, ppm=None, fv=None):
        """Save both OBJ file and update obj_path"""
        # First save the OBJ file
        obj_path = fpath.with_suffix(".obj")
        self.obj_path = obj_path  # Update the path for JSON saving
        
        name = fpath.name
        stem = fpath.stem
        print("TF save", obj_path)
        of = obj_path.open("w")
        # print("hello", file=of)
        print("# Khartes OBJ File", file=of)
        print("# Created: %s"%self.created, file=of)
        print("# Name: %s"%self.name, file=of)
        print("# Vertices: %d"%len(self.gpoints), file=of)
        ns = BaseFragment.pointNormals(self.gpoints, self.trgls)
        vrts = self.gpoints
        if ppm is not None:
            vrts = ppm.layerIjksToScrollIjks(vrts)
        for i, pt in enumerate(vrts):
            print("v %f %f %f"%(pt[0], pt[1], pt[2]), file=of)
            if ns is not None:
                n = ns[i]
                print("vn %f %f %f"%(n[0], n[1], n[2]), file=of)
        print("# Color and texture information", file=of)
        # print("mtllib %s.mtl"%self.name, file=of)
        print("mtllib %s.mtl"%stem, file=of)
        print("usemtl default", file=of)
        image_file = ""
        rgb = self.color.getRgbF()
        has_texture = (len(self.gtpoints) == len(self.gpoints))
        if has_texture:
            tpts = self.gtpoints
            if fv is not None and fv.stpoints is not None and fv.map_corners is not None:
                image_ext = ".png"
                image_file = "%s%s"%(stem, image_ext)
                image_path = fpath.with_suffix(image_ext)
                map_image = fv.map_image
                # image corners, in uv coordinates
                ic = fv.map_corners
                ms = map_image.shape
                # print("ms", ms)
                # image size, in uv coordinates
                dc = (ic[1][0]-ic[0][0], ic[1][1]-ic[0][1])
                tpts = fv.stpoints.copy()
                # corners of entire surface, in uv coords
                a0 = tpts.min(axis=0)
                a1 = tpts.max(axis=0)
                ac = (a0, a1)
                dac = (a1[0]-a0[0], a1[1]-a0[1])
                if dc[0] > 0 and dc[1] > 0 and ms[0] > 0 and ms[1] > 0 and dac[0] > 0 and dac[1] > 0:
                    # when zoomed in, pxsz is small
                    # pxsz[0] should be almost the same as pxsz[1]
                    # size of image pixel, in uv coordinates
                    pxsz = (dc[0]/ms[1], dc[1]/ms[0])
                    # print("pxsz", pxsz)
                    # print(ic, ms, pxsz)
                    # image corners, in pixel coordinates
                    ipc = [[int((ic[j][i]-ac[0][i])/pxsz[i]) for i in range(2)] for j in range(2)]
                    # corners of entire surface, in pixel coordinates
                    apc = ((0,0), [int((ac[1][i]-ac[0][i])/pxsz[i]) for i in range(2)])
                    # windowed image corners, in pixel coordinates
                    rpc = Utils.rectIntersection(apc, ipc)
                    # windowed image corners, in image-pixel coordinates
                    ric = [[rpc[j][i]-ipc[0][i] for i in range(2)] for j in range(2)]
                    # print("ipc", ipc)
                    # print("apc", apc)
                    # print("rpc", rpc)
                    # print("ric", ric)
                    # map_image is RGBA, so img must be as well
                    # TODO: img is actually BGRA, not RGBA, though
                    # the difference is not visible with gray-scale images
                    img = np.full((apc[1][1], apc[1][0], 4), 65536//2, dtype=map_image.dtype)
                    # reversed because cv2 uses BGRA not RGBA
                    img[:,:,2] = 65535*rgb[0]
                    img[:,:,1] = 65535*rgb[1]
                    img[:,:,0] = 65535*rgb[2]
                    img[rpc[0][1]:rpc[1][1], rpc[0][0]:rpc[1][0]] = map_image[ric[0][1]:ric[1][1], ric[0][0]:ric[1][0]]
                    cv2.imwrite(str(image_path), img)
                    rgb = (1.,1.,1.)

                    # rst = ((st0[0], st0[1]), (st1[0], st1[1]))
                    # print(rst)
                    # ri = Utils.rectIntersection(mc, rst)
                    # print(ri)
                    # p0 = [int((mc[0][i]+ri[0][i])/pxsz[i]) for i in range(2)]
                    # p1 = [int((mc[1][i]+ri[1][i])/pxsz[i]) for i in range(2)]
                    # print(p0, p1)

                # tpts = fv.stpoints.copy()
                if dac[0] != 0. and dac[1] != 0.:
                    tpts = (tpts-ac[0])/dac
                    tpts[:,1] = 1.-tpts[:,1]

                '''
                # c = fv.map_corners
                # full_image = np.zeros((c[1][1], c[1][0]))
                cv2.imwrite(str(image_path), fv.map_image)
                c = fv.map_corners
                tpts = fv.stpoints.copy()
                st0 = np.array(c[0])
                st1 = np.array(c[1])
                # st0 = st0[::-1]
                dst = st1-st0
                # print(st0, st1, dst)
                if dst[0] != 0. and dst[1] != 0.:
                    # print("converting tpts")
                    tpts = (tpts-st0)/dst
                    tpts[:,1] = 1.-tpts[:,1]
                    tpts[tpts>1.] = -1.e+3
                    tpts[tpts<0.] = -1.e+3
                    tpts[(tpts<0).any(axis=1), :] = -1.e+3
                '''

            for i, pt in enumerate(tpts):
                print("vt %f %f"%(pt[0], pt[1]), file=of)
        print("# Faces: %d"%len(self.trgls), file=of)
        for trgl in self.trgls:
            ostr = "f"
            for i in range(3):
                v = trgl[i]+1
                if has_texture:
                    ostr += " %d/%d/%d"%(v,v,v)
                else:
                    ostr += " %d/%d"%(v,v)
            print(ostr, file=of)
        mtl_path = fpath.with_suffix(".mtl")
        try:
            of = mtl_path.open("w")
        except Exception as e:
            print("Could not open %s: %s"%(str(mtl_path), e))
            return
            
        print("newmtl default", file=of)
        print("Ka %f %f %f"%(rgb[0],rgb[1],rgb[2]), file=of)
        print("Kd %f %f %f"%(rgb[0],rgb[1],rgb[2]), file=of)
        print("Ks 0.0 0.0 0.0", file=of)

        # TODO: testing only!
        # print("map_Kd asdf.tif", file=of)
        if image_file != "":
            print("map_Kd", image_file, file=of)

        print("illum 2", file=of)
        print("d 1.0", file=of)
        # TODO: print this only if TIFF file exists
        # if has_texture:
        #     print("map_Kd %s.tif"%stem, file=of)

        if fv is not None:
            jfilename = fpath.with_suffix(".json")
            jdict = {}
            fdict = {}
            fdict["name"] = self.name
            area = fv.sqcm
            fdict["area_sq_cm"] = area
            fdict["n_vrts"] = len(self.gpoints)
            fdict["n_trgls"] = len(self.trgls)
            jdict[self.name] = fdict
            info_txt = json.dumps(jdict, indent=4)
            try:
                ofj = jfilename.open("w")
                print(info_txt, file=ofj)
            except Exception as e:
                print("Could not open %s: %s"%(str(jfilename), e))
                return

    # find the intersections between a plane defined by axis and position,
    # and the triangles.  
    # The first return value is an array
    # with 6 columns (the x,y,z location of each of the two
    # intersections with a given trgl), and as many rows as
    # there are intersected triangles.
    # The second return value is a vector, as long as the first
    # array, with the trgl index of each intersected triangle.
    def findIntersections(pts, trgls, axis, position):
        gpts = pts
        # print("min", np.min(gpts, axis=0))
        # print("max", np.max(gpts, axis=0))
        # trgls = self.trgls
        # print("trgls", trgls.shape)
        # print(axis, position)

        # shift intersection plane slightly so that
        # no vertices lie on the plane
        while len(gpts[gpts[:,axis]==position]) > 0:
            position += .01
        
        # print(axis, position)
        
        # -1 or 1 depending on which side gpt is in relation
        # to the plane defined by axis and position
        gsgns = np.sign(gpts[:,axis] - position)
        # print("gsgns", gsgns.shape)
        # print(gsgns)
        # -1 or 1 for each vertex of each triangle
        trglsgns = gsgns[trgls]
        # sum of the signs for each trgl
        tssum = trglsgns.sum(axis=1)
        # if sum is -3 or 3, the trgl is entirely on one
        # side of the plane, and can be ignored from now on
        esor = (tssum != -3) & (tssum != 3)
        trglsgns = trglsgns[esor]
        trglvs = trgls[esor]
        # print("trglsgns", trglsgns.shape)
        # print(trglsgns)
        # print(trglvs)
        
        # shift the trglsgns by one, to compare each vertex to 
        # its adjacent vertex around the trgl
        trglroll = np.roll(trglsgns, 1, axis=1)
        # assuming vertices of each trgl are labeled 0,1,2,
        # for each trgl set a boolean showing whether each edge
        # of the trgl crosses the plane, in order:
        # 1 to 2, 2 to 0, 0 to 1
        es = np.roll((trglsgns != trglroll), 1, axis=1)
        
        # print(es)
        
        # repeat each column of es
        es2 = np.repeat(es, 2, axis=1)
        
        # for each trgl, assuming its vertices are numbered 0,1,2,
        # create a row of six vertices, corresponding to the
        # edge ordering in es, namely: 1,2,2,0,0,1
        vs0 = np.roll(np.repeat(trglvs, 2, axis=1), 3, axis=1)
        
        # if a given edge does NOT cross the plane, replace its
        # vertex numbers by -1
        vs0[~es2] = -1
        # print(vs0)
        
        # find all "-1" vertices in columns 0 and 1 and
        # roll them to columns 4 and 5
        m = vs0[:,0] == -1
        vs0[m] = np.roll(vs0[m], -2, axis=1)
        
        # find all "-1" vertices in columns 2 and 3 and
        # roll them to columns 4 and 5
        m = vs0[:,2] == -1
        vs0[m] = np.roll(vs0[m], 2, axis=1)
        
        # each row of vs contains two pairs of vertices specifying
        # the two edges of the triangle that cross the plane
        vs = vs0[:,:4]
        # print(vs)
        
        # There shouldn't be any "-1" values in vs at this point,
        # but if there are, filter them out
        vs = vs[vs[:,0] != -1]
        vs = vs[vs[:,2] != -1]
        
        # gpts projected on the axis
        gax = gpts[:,axis]
        
        # vsh has only one edge (vertex pair) per row
        vsh = vs.reshape(-1,2)
        # extract the two vertices of each edge pair
        v0 = vsh[:,0]
        v1 = vsh[:,1]
        
        # calculate the point where each edge intersects the plane.
        # d is always non-zero because v0 and v1 lie on opposite
        # sides of the plane
        d = gax[v1] - gax[v0]
        a = (position - gax[v0])/d
        a = a.reshape(-1,1)
        i = (1-a)*gpts[v0] + a*gpts[v1]
        
        # put the two intersection points, for the two edges of the single
        # triangle, back into a single row
        i01 = i.reshape(-1,6)
        # print(i01)
        # print("i01", i01.shape)

        trglist = np.indices((len(trgls),))[0]
        # print(trgls.shape, trglist.shape, esor.shape)
        trglist = trglist[esor]

        return i01, trglist
    
    def toDict(self):
        info = {}
        info['name'] = self.name
        info['created'] = self.created
        info['modified'] = self.modified
        info['color'] = self.color.name()
        info['type'] = self.type.value if self.type else Fragment.Type.TRGL_FRAGMENT.value
        
        # Convert any NumPy arrays in params to lists
        params = {}
        for key, value in self.params.items():
            if isinstance(value, np.ndarray):
                params[key] = value.tolist()
            else:
                params[key] = value
        info['params'] = params
        
        info['obj_path'] = str(self.obj_path) if self.obj_path else None
        # Don't save gpoints/trgls in JSON as they're in the OBJ file
        return info

    @staticmethod
    def fragFromDict(info):
        """Reconstruct a TrglFragment from a dictionary, loading geometry from OBJ file"""
        if not info.get('obj_path'):
            print("No OBJ path found in fragment info")
            return None
            
        # Load the geometry from OBJ file
        obj_path = Path(info['obj_path'])
        if not obj_path.exists():
            print(f"OBJ file not found: {obj_path}")
            return None
            
        # Use existing load method to get geometry
        fragments = TrglFragment.load(obj_path)
        if not fragments or not fragments[0]:
            print(f"Failed to load OBJ file: {obj_path}")
            return None
            
        # Get the fragment and update its metadata from JSON
        frag = fragments[0]
        frag.name = info.get('name', frag.name)
        frag.created = info.get('created', frag.created)
        frag.modified = info.get('modified', frag.modified)
        if 'color' in info:
            frag.setColor(QColor(info['color']), no_notify=True)
        frag.params = info.get('params', {})
        frag.type = BaseFragment.Type.TRGL_FRAGMENT
        frag.obj_path = obj_path
        frag.valid = True
        
        return frag

    @staticmethod
    def load(obj_file):
        """Load a TrglFragment from an OBJ file"""
        print("loading obj file", obj_file)
        pname = Path(obj_file)
        try:
            fd = pname.open("r")
        except:
            return None

        name = pname.stem
        
        vrtl = []
        tvrtl = []
        trgl = []
        
        created = ""
        frag_name = ""
        for line in fd:
            line = line.strip()
            words = line.split()
            if words == []: # prevent crash on empty line
                continue
            if words[0][0] == '#':
                if len(words) > 2: 
                    if words[1] == "Created:":
                        created = words[2]
                    if words[1] == "Name:":
                        frag_name = words[2]
            elif words[0] == 'v':
                # len is 7 if the vrt has color attached
                # (color is ignored)
                if len(words) == 4 or len(words) == 7:
                    vrtl.append([float(w) for w in words[1:4]])
            elif words[0] == 'vt':
                if len(words) == 3:
                    tvrtl.append([float(w) for w in words[1:]])
            elif words[0] == 'f':
                if len(words) == 4:
                    # implicit assumption that v == vt
                    trgl.append([int(w.split('/')[0])-1 for w in words[1:]])
        print("tf obj reader", len(vrtl), len(tvrtl), len(trgl))
        
        if frag_name == "":
        #     frag_name = name.replace("_",":").replace("p",".")
            frag_name = name
        trgl_frag = TrglFragment(frag_name)
        if len(vrtl) > 0:
            trgl_frag.gpoints = np.array(vrtl, dtype=np.float32)
        else:
            trgl_frag.gpoints = np.zeros((0,3), dtype=np.float32)
        if len(tvrtl) > 0:
            trgl_frag.gtpoints = np.array(tvrtl, dtype=np.float32)
        else:
            trgl_frag.gtpoints = np.zeros((0,2), dtype=np.float32)
        if len(trgl) > 0:
            trgl_frag.trgls = np.array(trgl, dtype=np.int32)
        else:
            trgl_frag.trgls = np.zeros((0,3), dtype=np.int32)
        if created == "":
            ts = Utils.vcToTimestamp(name)
            if ts is not None:
                created = ts
        if created != "":
            trgl_frag.created = created
        trgl_frag.params = {}
        
        mname = pname.with_suffix(".mtl")
        fd = None
        color = None
        try:
            fd = mname.open("r")
        except:
            print("failed to open mtl file",mname.name)
            pass

        if fd is not None:
            for line in fd:
                words = line.split()
                # print("words[0]", words[0])
                if len(words) == 4 and words[0] == "Kd":
                    try:
                        # print("words", words)
                        r = float(words[1])
                        g = float(words[2])
                        b = float(words[3])
                    except:
                        continue
                    # print("rgb", r,g,b)
                    color = QColor.fromRgbF(r,g,b)
                    break

        if color is None:
            color = Utils.getNextColor()
        trgl_frag.setColor(color, no_notify=True)
        trgl_frag.valid = True
        trgl_frag.neighbors = BaseFragment.findNeighbors(trgl_frag.trgls)
        trgl_frag.obj_path = obj_file
        print(trgl_frag.name, trgl_frag.color.name(), trgl_frag.gpoints.shape, trgl_frag.gtpoints.shape, trgl_frag.trgls.shape)
        # print("tindexes", BaseFragment.trglsAroundPoint(100, trgl_frag.trgls))
        if len(trgl_frag.gtpoints) > 0:
            tmp_fv = trgl_frag.createView(None)
            tmp_fv.setScaledTexturePoints(similar=False)
            trgl_frag.gtpoints = tmp_fv.stpoints

        # Store the obj path for future saves
        trgl_frag.obj_path = pname
        
        return [trgl_frag]


class TrglFragmentView(BaseFragmentView):
    def __init__(self, project_view, trgl_fragment):
        super(TrglFragmentView, self).__init__(project_view, trgl_fragment)
        # self.project_view = project_view
        # self.fragment = trgl_fragment
        # TODO fix:
        self.line = None
        self.setWorkingRegion(-1, 0.)
        self.has_working_non_working = (False, False)
        self.prev_pt_count = 0
        self.stpoints = np.zeros((0,2))
        self.all_stpoints = np.zeros((0,2))
        self.setStxyDefaults()
        self.normals = None
        self.normal_offset = 0.
        # self.half_width_multiplier = 10
        self.half_width_multiplier = 5
        self.retriangulate_enabled = True
        self.gpoints_history = []
        if len(trgl_fragment.trgls) == 0:
            self.mesh_visible = False

    def allowAutoExtrapolation(self):
        return False

    def allowAutoInterpolation(self):
        return False

    def setLocalPoint(self, index):
        self.local_points_modified = Utils.timestamp()
        self.vpoints[index, :3] = self.cur_volume_view.globalPositionToTransposedIjk(self.fragment.gpoints[index])
        self.vpoints[index, 3] = index
        self.fpoints[index] = self.vpoints[index, :3]

    def addLocalPoint(self, index):
        self.vpoints = np.insert(self.vpoints, index, [0.]*4, axis=0)
        self.fpoints = np.insert(self.fpoints, index, [0.]*3, axis=0)
        self.setLocalPoint(index)

    # TODO: if cur_volume_view changed, unset working region
    # NOTE that Fragment.setLocalPoints sets stpoints,
    # but TrglFragment.setLocalPoints does not.
    def setLocalPoints(self, recursion_ok=True, always_update_zsurfs=True, build_kd_trees=False, build_adjacency_list=False):
        """
        Update local point coordinates and optionally rebuild spatial data structures.
        
        Args:
            recursion_ok: Whether to allow recursive updates
            always_update_zsurfs: Whether to always update z-surfaces
            build_kd_trees: Whether to rebuild KD trees and spatial hash grid
                          (should be False during dragging operations)
            build_adjacency_list: Whether to rebuild adjacency list
                                (should be True when topology changes: adding/removing nodes)
        """
        self.local_points_modified = Utils.timestamp()
        if self.cur_volume_view is None:
            self.vpoints = np.zeros((0,4), dtype=np.float32)
            self.fpoints = self.vpoints
            self.setWorkingRegion(-1, 0.)
            return
            
        self.vpoints = self.cur_volume_view.globalPositionsToTransposedIjks(self.fragment.gpoints)
        self.fpoints = self.vpoints
        self.fragment.direction = self.cur_volume_view.direction
        npts = self.vpoints.shape[0]
        
        indices = np.reshape(np.arange(npts), (npts,1))
        self.vpoints = np.concatenate((self.vpoints, indices), axis=1)
        
        if recursion_ok:
            if always_update_zsurfs:
                self.setScaledTexturePoints()
            # Pass through both build flags
            self.buildKDTrees(True, 
                            build_kd_tree=build_kd_trees,
                            build_adjacency_list=build_adjacency_list,
                            build_spatial_hash_grid=build_kd_trees)
        self.calculateSqCm()
        vv = self.cur_volume_view
        if vv.stxytf is not None:
            uvxytf = self.stxyToUv(vv.stxytf)
        self.setScaledTexturePoints()
        if vv.stxytf is not None:
            vv.stxytf = self.uvToStxy(uvxytf)
        self.mesh_visible = (len(self.trgls()) > 0)
        '''
        mapper = UVMapper(self.fragment.gtpoints, self.trgls())
        pt0, pt1 = mapper.getTwoAdjacentBoundaryPoints()
        mapper.constraints = np.array([[pt0, 0., 0.], [pt1, 1., 0.]], dtype=np.float64)
        uvs = mapper.computeUvsFromABF()
        self.fragment.gtpoints = uvs
        self.stpoints = None
        self.setScaledTexturePoints()
        '''
        # timer = Utils.Timer()
        # print("computing normals")
        # TODO: compute only modified normals
        self.normals = BaseFragment.pointNormals(self.vpoints[:,:3], self.trgls())
        # timer.time("normals")
        
        # self.createTetras()
        # self.setWorkingRegion(35555, 60.)
        # TODO:
        # update positions of working points
        if self.working_fv is not None:
            self.working_fragment.gpoints = self.fragment.gpoints[self.working_vpoints]
            # recursion_ok=True causes crash due to FragmentView.setLocalPoints
            # looping over project_view.fragments
            # print("before wfv slp")
            self.working_fv.setLocalPoints(False, build_kd_trees=build_kd_trees, build_adjacency_list=build_adjacency_list)
            # print("after wfv slp")

        # super().buildKDTrees(recursion_ok)


    '''
    
    Each vertex in a .obj file that is created by vc carries two
    pieces of information: the xyz location, and the texture (uv)
    coordinate.
    The xyz location is in scroll coordinates, but the uv coordinates
    are stretched or compressed so they both extend over the entire
    range 0.0 to 1.0.  The uv coordinates may also be rotated relative
    to the viewing angle we may prefer.
    The setScaledTexturePoints routine attepts to find a transformation 
    (scale, rotate, shift, aka an affine transformation) 
    that will convert uv coordinates into what I called "st" 
    (scaled texture) coordinates.  I denote the two resulting coordinates 
    as stx and sty.  
    Given the fundamental constraints (all stx and sty values are
    generated from u and v using the same affine transformation),
    the goals are:
        1) sty is, as much as possible, parallel to the original z axis;
        2) each triangle in st coordinates preserves, as much as possible,
           the area and angles of the original triangle in xyz space.
    
    The steps are:
        1) transform each triangle into a flattened xy space, exactly
        preserving areas and angles, and where the original alignment with
        the z axis is preserved;
        2) shift the triangle so that its center is at the origin of the
        flattened xy space;
        3) for each triangle, shift the uv coordinates of its vertices
        so that the center of the triangle in uv space is at the origin
        of uv space;
        4) solve a least-squares equation to find the coefficients
        (I call them a,b,c,d) of a rotation+scale transform (no
        shift), with the objective function being: the sum of the distances
        between the transformed uv points (transformed by the a,b,c,d
        matrix) and the recentered xy points be as small as possible.
    
    '''

    def setScaledTexturePoints(self, similar=True):
        timer = Utils.Timer()
        timer.active = False  # Enable timing
        
        timer.time("Start setScaledTexturePoints")
        f = self.fragment
        
        if self.stpoints is not None and len(f.gpoints) == self.prev_pt_count:
            return
            
        self.prev_pt_count = len(f.gpoints)
        if len(f.gtpoints) != len(f.gpoints):
            print("length mismatch", len(f.gtpoints), len(f.gpoints), "in volume",self.fragment.name)
            print('''*******************************************************
* Warning!  This segment does not have the same number
* of uv points as xyz points.  There is probably
* something wrong with the input obj file.
* Khartes will probably crash soon.
*******************************************************
              ''')
            return

        timer.time("Initial checks")
        
        self.deleteDisconnectedComponents()
        timer.time("Delete disconnected components")

        # original xyzs
        oxyzs = self.fragment.gpoints.astype(np.float64)
        txyzs = oxyzs[self.trgls()].astype(np.float64)
        gtps = self.fragment.gtpoints.astype(np.float64)
        
        timer.time("Array setup")

        # centers of triangles
        cxyzs = txyzs.sum(axis=1)/3
        cxyzs = cxyzs[:,np.newaxis,:]
        txyzs -= cxyzs
        
        timer.time("Triangle centers calculation")

        t01 = txyzs[:,1]-txyzs[:,0]
        t02 = txyzs[:,2]-txyzs[:,0]
        tnorm = np.cross(t01, t02)
        
        timer.time("Normal calculation")

        fxyaxis = np.cross(tnorm, (0.,0.,1))
        weights = np.sqrt((fxyaxis*fxyaxis).sum(axis=1))
        
        timer.time("Weight calculation")

        nw = len(weights)
        if nw > 10:
            wsort = np.argsort(weights)
            median_weight = weights[wsort[nw//2]]
            max_allowed_weight = 4*median_weight
            weights[weights > max_allowed_weight] = max_allowed_weight
        
        timer.time("Weight adjustment")

        weights = weights.reshape(-1,1,1)
        fzaxis = np.cross(tnorm, fxyaxis)

        # normalize axes
        norm = np.linalg.norm(fzaxis, axis=1, keepdims=True)
        norm[norm==0] = 1.
        fzaxis = fzaxis/norm

        norm = np.linalg.norm(fxyaxis, axis=1, keepdims=True)
        norm[norm==0] = 1.
        fxyaxis /= norm
        
        timer.time("Axis normalization")

        fxyaxis = fxyaxis[:,np.newaxis,:]
        tfxy = (txyzs*fxyaxis).sum(axis=2)
        fzaxis = fzaxis[:,np.newaxis,:]
        tfz = (txyzs*fzaxis).sum(axis=2)
        tfxy = np.stack((tfxy, tfz), axis=2)
        
        timer.time("Coordinate transformation")

        tuvs = gtps[self.trgls()].astype(np.float64)
        cuvs = tuvs.sum(axis=1)/3
        cuvs = cuvs[:,np.newaxis,:]
        tfuv = tuvs-cuvs
        
        timer.time("UV coordinate processing")

        tfxynw = tfxy.copy()
        tfxy *= weights
        tfxy = tfxy.reshape(-1,2)
        tfuv *= weights
        tfuv = tfuv.reshape(-1,2)

        u = tfuv[:,0]
        v = tfuv[:,1]
        x = tfxy[:,0]
        y = tfxy[:,1]
        
        timer.time("Coordinate reshaping")

        uu = (u*u).sum()
        uv = (u*v).sum()
        vv = (v*v).sum()
        ux = (u*x).sum()
        uy = (u*y).sum()
        vx = (v*x).sum()
        vy = (v*y).sum()

        mden = uu*vv - uv*uv
        mden2 = uu+vv
        
        timer.time("Matrix calculations")

        if mden == 0 or mden2 == 0:
            print("mden, mden2", mden, mden2)
            return

        gu = gtps[:,0]
        gv = gtps[:,1]
        stp = np.zeros_like(gtps)

        if similar:
            abcds = []

            a = (ux - vy)/mden2
            b = (vx + uy)/mden2
            c = b
            d = -a
            abcds.append((a,b,c,d))

            a = (ux + vy)/mden2
            b = (vx - uy)/mden2
            c = -b
            d = a
            abcds.append((a,b,c,d))

            a = ( vv*ux - uv*vx)/mden
            b = (-uv*ux + uu*vx)/mden
            c = ( vv*uy - uv*vy)/mden
            d = (-uv*uy + uu*vy)/mden
            abcds.append((a,b,c,d))

            errors = []
            for a,b,c,d in abcds:
                stp[:,0] = a*gu + b*gv
                stp[:,1] = c*gu + d*gv
                tuvs = stp[self.trgls()].astype(np.float64)
                cuvs = tuvs.sum(axis=1)/3
                cuvs = cuvs[:,np.newaxis,:]
                tfuv = tuvs-cuvs
                dd = tfxynw.flatten() - tfuv.flatten()
                error = np.sqrt((dd*dd).sum())/len(dd)
                errors.append(error)

            if errors[0] < errors[1]:
                a,b,c,d = abcds[0]
            else:
                a,b,c,d = abcds[1]
        else:
            print("affine")
            a = ( vv*ux - uv*vx)/mden
            b = (-uv*ux + uu*vx)/mden
            c = ( vv*uy - uv*vy)/mden
            d = (-uv*uy + uu*vy)/mden
            
        timer.time("Transformation matrix calculation")

        stp[:,0] = a*gu + b*gv
        stp[:,1] = c*gu + d*gv
        self.st_abcd = (a,b,c,d)

        stmin = stp.min(axis=0)
        stmax = stp.max(axis=0)
        styc = .5*(stmin[1]+stmax[1])
        xyzmin = oxyzs.min(axis=0)
        xyzmax = oxyzs.max(axis=0)
        zc = .5*(xyzmin[1]+xyzmax[1])
        
        timer.time("Final coordinate calculations")

        self.st_shift = -stmin
        self.st_shift[1] = zc-styc
        stp += self.st_shift

        self.stpoints = stp
        self.stmin = stp.min(axis=0)
        self.stmax = stp.max(axis=0)

        lens = TrglPointSet.edgeLengths(self.stpoints, self.trgls())
        self.avg_st_len = 0.
        if len(lens) > 0:
            self.avg_st_len = lens.sum()/len(lens)
        
        timer.time("Edge length calculations")

        self.outside_stpoints = self.outsidePoints(self.avg_st_len)
        self.all_stpoints = np.concatenate((stp, self.outside_stpoints), axis=0)
        self.retriangulateAll()
        
        timer.time("Final triangulation")

    def setStxyDefaults(self):
        self.st_abcd = (1.,0.,0.,1.)
        self.st_shift = np.zeros(2, dtype=np.float64)
        self.stmin = np.zeros(2, dtype=np.float64)
        self.stmax = np.zeros(2, dtype=np.float64)
        self.avg_st_len = 100

    def uvToStxy(self, uv):
        u, v = uv
        a,b,c,d = self.st_abcd
        sh = self.st_shift
        stxy = (a*u+b*v+sh[0], c*u+d*v+sh[1])
        return stxy

    def uvsToStxys(self, uvs):
        a,b,c,d = self.st_abcd
        # A = np.array(((a,b),(c,d)), dtype=np.float64)
        stxys = np.zeros((uvs.shape[0], 2), dtype=np.float64)
        sh = self.st_shift
        stxys[:,0] = a*uvs[:,0]+b*uvs[:,1]+sh[0]
        stxys[:,1] = c*uvs[:,0]+d*uvs[:,1]+sh[1]
        return stxys

    def stxyToUv(self, stxy):
        stxy = stxy - self.st_shift
        a,b,c,d = self.st_abcd
        det = a*d-b*c
        u = (d*stxy[0] - b*stxy[1])/det
        v = (-c*stxy[0] + a*stxy[1])/det
        return (u,v)
        
    def stxysToUvs(self, stxys):
        stxys = stxys.copy() - self.st_shift
        a,b,c,d = self.st_abcd
        # print("stu", a,b,c,d)
        det = a*d-b*c
        uvs = np.zeros((stxys.shape[0], 2), dtype=np.float64)
        uvs[:,0] = (d*stxys[:,0] - b*stxys[:,1])/det
        uvs[:,1] = (-c*stxys[:,0] + a*stxys[:,1])/det
        return uvs
        
    def setVolumeViewDirection(self, direction):
        self.setWorkingRegion(-1, 0.)
        self.prev_pt_count = 0
        super(TrglFragmentView, self).setVolumeViewDirection(direction)
        
    def setVolumeView(self, vol_view):
        self.setWorkingRegion(-1, 0.)
        self.prev_pt_count = 0
        super(TrglFragmentView, self).setVolumeView(vol_view)

    def pushFragmentState(self):
        """Push the current fragment state onto the undo stack."""
        
        # Save triangulation state
        state = {
            'gpoints': np.copy(self.fragment.gpoints),
            'gtpoints': np.copy(self.fragment.gtpoints),
            'stpoints': np.copy(self.stpoints) if self.stpoints is not None else None,
            'all_stpoints': np.copy(self.all_stpoints) if self.all_stpoints is not None else None,
            'trgls': np.copy(self.fragment.trgls) if self.fragment.trgls is not None else None,
            'sqcm': self.sqcm
        }
        print("pushFragmentState", len(self.gpoints_history))
        # Maintain max history of 10 states
        if len(self.gpoints_history) >= 10:
            self.gpoints_history.pop(0)  # Remove oldest state
        self.gpoints_history.append(state)

    def popFragmentState(self):
        """Restore the previous fragment state from the undo stack."""
        hist_size = len(self.gpoints_history)
        if hist_size > 0:
            state = self.gpoints_history.pop()
            self.fragment.gpoints = state['gpoints']
            self.fragment.gtpoints = state['gtpoints']
            self.stpoints = state['stpoints']
            self.all_stpoints = state['all_stpoints']
            self.fragment.trgls = state['trgls']
            self.sqcm = state['sqcm']
            self.fragment.notifyModified()
            self.setLocalPoints(True, False, build_kd_trees=True, build_adjacency_list=True)
            print("popFragmentState", len(self.gpoints_history))

    def setWorkingRegion(self, index, max_angle):
        if index < 0:
            # self.working_trgls = np.zeros((0, 3), dtype=np.int32)
            # self.working_vpoints = np.zeros((0,4), dtype=np.float32)
            self.working_trgls = np.full((len(self.trgls()),), False)
            self.working_vpoints = np.full((len(self.fragment.gpoints),), False)
            self.working_fragment = None
            self.working_fv = None
            self.has_working_non_working = (False, True)
            self.working_fragment = None
            self.working_fv = None
            # print("swr cleared all")
            return
        tbn = self.regionByNormals(index, max_angle)
        # print("tbn", len(tbn))
        wt = np.full((len(self.trgls()),), False)
        wt[tbn] = True
        # self.working_trgls = self.trgls()[tbn]
        self.working_trgls = wt
        vs = np.unique(self.trgls()[tbn].flatten())
        wv = np.full((len(self.vpoints),), False)
        # print(vs.shape, wv.shape)
        # print(vs)
        wv[(vs)] = True
        self.working_vpoints = wv

        lworking = len(vs)
        lnonworking = len(self.vpoints)-lworking
        self.has_working_non_working = (lworking>0, lnonworking>0)
        # print("lwlnw wnw", lworking, lnonworking, self.has_working_non_working)
        # self.working_vpoints = self.vpoints[vs]
        # print("wvp", len(self.working_vpoints))
        # print("trgl_fragment set local points")

        self.working_fragment = Fragment("working", self.fragment.direction)
        self.working_fragment.setColor(self.fragment.color)
        # self.working_fragment.gpoints = np.copy(self.fragment.gpoints)
        self.working_fragment.gpoints = self.fragment.gpoints[self.working_vpoints]
        self.working_fv = FragmentView(None, self.working_fragment)
        self.working_fv.setVolumeView(self.cur_volume_view)
        # print("swr set all")

    def workingZsurf(self):
        if self.working_fv is not None:
            return self.working_fv.workingZsurf()

    def workingSsurf(self):
        if self.working_fv is not None:
            return self.working_fv.workingSsurf()

    def moveAlongNormalsSign(self):
        return -1.

    def workingVpoints(self):
        return self.working_vpoints

    def hasWorkingNonWorking(self):
        return self.has_working_non_working

    def workingTrgls(self):
        return self.working_trgls

    def calculateSqCmOfTrgls(self, trgls):
        pts = self.fragment.gpoints
        voxel_size_um = self.project_view.project.voxel_size_um
        sqcm = BaseFragment.calculateSqCm(pts, trgls, voxel_size_um)
        return sqcm

    def calculateSqCm(self):
        simps = self.fragment.trgls
        sqcm = self.calculateSqCmOfTrgls(simps)
        self.sqcm = sqcm

    def calculateStArea(self):
        pts = self.stpoints
        simps = self.fragment.trgls
        voxel_size_um = 1000000/100 # 1 cm in um
        sqcm = BaseFragment.calculateSqCm(pts, simps, voxel_size_um)

    def getPointsOnSlice(self, axis, i):
        # matches = self.vpoints[(self.vpoints[:, axis] == i)]
        matches = self.vpoints[(self.vpoints[:, axis] >= i-.5001) & (self.vpoints[:, axis] < i+.5001)]
        return matches

    # outputs a list of lines; each line has two vertices
    # NOTE that input axis and position are in local tijk coordinates,
    # and that output vertices are in tijk coordinates
    def getLinesOnSlice(self, axis, axis_pos):
        '''
        tijk = [0,0,0]
        tijk[axis] = axis_pos
        gijk = self.cur_volume_view.transposedIjkToGlobalPosition(tijk)
        gaxis = self.cur_volume_view.globalAxisFromTransposedAxis(axis)
        gpos = gijk[gaxis]
        ints = self.fragment.findIntersections(gaxis, gpos)
        gpts = ints.reshape(-1, 3)
        vpts = self.cur_volume_view.globalPositionsToTransposedIjks(gpts)
        plines = vpts.reshape(-1,2,3)
        return plines
        '''
        ints, trglist = TrglFragment.findIntersections(self.fpoints, self.trgls(), axis, axis_pos)
        plines = ints.reshape(-1,2,3)
        return plines, trglist

    def aligned(self):
        return True

    def trgls(self):
        return self.fragment.trgls

    def outsidePoints(self, ptstep):
        stp = self.stpoints
        stmin = self.stmin
        stmax = self.stmax
        minid = stmin/ptstep - 5
        maxid = stmax/ptstep + 5
        id0 = np.floor(minid).astype(np.int32)
        id1 = np.ceil(maxid).astype(np.int32)
        idn = id1-id0
        # We want to find all the points outside of the obj surface.
        # To do this, create an array, and set all the cells of
        # the array to 255.  Then set all cells that contain a
        # trgl vertex to 0.
        # Look for connected components, and take the connected
        # component that extends to 0,0; that is the outer region.
        # Note that connected components are created from non-zero
        # components, so need to make sure the points in the area 
        # we are interested in is non-zero.
        arr = np.full(idn, 255, dtype=np.uint8)
        istps = np.floor(stp/ptstep - id0).astype(np.int32)
        # print("outsidePoints arr", id0, idn, arr.shape)
        # arr[istps[:,0], istps[:,1]] = 0
        arr[istps[:,0], istps[:,1]] = 0
        # cv2.imwrite("test.png", arr)
        ccoutput = cv2.connectedComponentsWithStats(arr, 4, cv2.CV_32S)
        (nlabels, labels, stats, centroids) = ccoutput
        label0 = labels[0,0]
        # print("nlabels", nlabels, "label0", label0)
        # print("stats", stats[label0])
        arr2 = np.full(idn, 255, dtype=np.uint8)
        # points that are outside are 0, points not in the outside are 255
        arr2[labels == label0] = 0
        # cv2.imwrite("test.png", arr2)
        kernel = np.ones((3,3), np.uint8)
        dilo = cv2.dilate(arr2, kernel, iterations=2)
        dili = cv2.dilate(arr2, kernel, iterations=1)
        diff = dilo-dili
        # cv2.imwrite("test.png", diff)
        pts = np.argwhere(diff)
        # print("pts", pts.shape, pts[0:5])
        pts = (pts+id0)*ptstep
        # print("pts", pts.shape, pts[0:5])
        return pts

    def retriangulateAll(self):
        # st_area = self.calculateStArea()
        # if self.fragment.trgls is None or len(self.fragment.trgls) == 0:
        #     return
        stp = self.stpoints
        if stp is None or len(stp) == 0:
            return
        all_pts = self.all_stpoints
        # print("ra", len(stp), len(all_pts), len(self.trgls()))

        all_trgls = None
        try:
            all_trgls = Delaunay(all_pts).simplices
        except Exception as err:
            err = str(err).splitlines()[0]
            print("retriangulateAll triangulation error: %s"%err)
        # print("allt", len(all_trgls))
        if all_trgls is not None:
            new_trgls = all_trgls[(all_trgls < len(stp)).all(axis=1), :]
            # print("newt", len(new_trgls))
            # new_trgls = all_trgls
            self.fragment.trgls = TrglPointSet.rotateToMin(new_trgls)
            # self.fragment.trgls = new_trgls
        # print("ra2", len(stp), len(all_pts), len(self.trgls()))

    def rebuildStPoints(self):
        self.stpoints = None
        # print("rsp set stpoints to None")
        self.setScaledTexturePoints()
        self.fragment.notifyModified()

    def reparameterize(self):
        self.stpoints = None
        # print("rpm set stpoints to None")
        self.setScaledTexturePoints()
        xyzs = self.vpoints[:,0:3]
        trgls = self.trgls()
        # print("rt before")
        # print(self.trgls()[(self.trgls()==181).any(axis=1)])
        TrglPointSet.findSpikes(xyzs, trgls, "before reparameterize")
        # print("rt after findSpikes")
        # print(self.trgls()[(self.trgls()==181).any(axis=1)])
        self.disconnectColocatedPoints()
        # print("rt after coloc")
        # print(self.trgls()[(self.trgls()==181).any(axis=1)])
        self.deleteDisconnectedComponents()
        # print("rt after disconn")
        # print(self.trgls()[(self.trgls()==181).any(axis=1)])
        self.deleteFreePoints()
        # print("rt after free points")
        # print(self.trgls()[(self.trgls()==181).any(axis=1)])
        xyzs = self.vpoints[:,0:3]
        # txyzs is array[trgl #][trgl pt (0, 1, or 2)][pt xyz]
        trgls = self.trgls()
        # print("rt before mapper")
        # print(self.trgls()[(self.trgls()==181).any(axis=1)])
        mapper = UVMapper(xyzs, trgls)
        # print("rt after mapper")
        # print(self.trgls()[(self.trgls()==181).any(axis=1)])

        # set the floating points, and the window-boundary points,
        # as constraints (the floating points need to be constrained,
        # though they have no effect, otherwise the parameterizer
        # may complain of a singular matrix)
        # constraints = np.zeros((bpts.shape[0], 3), dtype=np.float64)
        # constraints[:, 0] = bpts
        # constraints[:, (1,2)] = pts[bpts]

        # mapper = UVMapper(self.fragment.gtpoints, self.trgls())
        ta = mapper.getTwoAdjacentBoundaryPoints()
        if ta is None:
            print("reparameterize: could not find boundary points!")
            return
        pt0, pt1 = ta
        mapper.constraints = np.array([[pt0, 0., 0.], [pt1, 1., 0.]], dtype=np.float64)
        weight = .000001
        mapper.ip_weights = np.full(self.stpoints.shape[0], weight)
        mapper.initial_points = self.fragment.gtpoints
        adjusted_sts = mapper.computeUvsFromABF()
        if adjusted_sts is None:
            print("reparameterize failed!")
            return
        self.fragment.gtpoints = adjusted_sts
        # print(adjusted_sts)
        self.stpoints = None
        # TrglPointSet.findSpikes(xyzs, trgls, "after reparam")
        # print("reparameterize set stpoints to None")
        self.setScaledTexturePoints()
        TrglPointSet.findSpikes(xyzs, trgls, "after reparameterize")
        # print(self.stpoints)
        self.fragment.notifyModified()


    # This depends on self.fragment.trgls being
    # up to date
    def adjustStPoints(self, index, half_width, stxy=None):
        timer = Utils.Timer()
        timer.active = False
        if stxy is None:
            stxy = self.all_stpoints[index]
        # print("stxy", stxy)
        osts = TrglPointSet(self.all_stpoints, len(self.stpoints), stxy, half_width)
        timer.time(" astpts TrglPointSet")
        # print("osts indexes", osts.indexes)
        retval = osts.adjustSts(self.fragment.gpoints, self.fragment.trgls, index)
        timer.time(" astpts adjustSts")
        if retval is None:
            print("Adjustment failed")
            return None

        adjusted_inds, adjusted_sts, constrained = retval
        # print(len(adjusted_sts))
        self.stpoints[adjusted_inds] = adjusted_sts
        self.all_stpoints[adjusted_inds] = adjusted_sts
        adj_uvs = self.stxysToUvs(adjusted_sts)
        timer.time(" astps stxysToUvs")
        self.fragment.gtpoints[adjusted_inds] = adj_uvs
        # print("Adjustment done")
        return constrained

    def movePoint(self, index, new_vijk, update_xyz, update_st, build_kd_trees=True, build_adjacency_list=False):
        """
        Move a single point to a new position.
        
        Args:
            index: Index of point to move
            new_vijk: New position in volume coordinates
            update_xyz: Whether to update xyz coordinates
            update_st: Whether to update st coordinates
            build_kd_trees: Whether to rebuild KD trees (False during dragging)
            build_adjacency_list: Whether to rebuild adjacency list
        """
        timer = Utils.Timer()
        timer.active = False
        vv = self.cur_volume_view
        new_gijk = vv.transposedIjkToGlobalPosition(new_vijk)
        new_uijk = vv.transposedIjkToIjk(new_vijk)
        old_vijk = self.vpoints[index, :3]
        old_uijk = vv.transposedIjkToIjk(old_vijk)
        duijk = [new_uijk[i]-old_uijk[i] for i in range(3)]
        axes = self.localStAxes(index)
        
        if axes is None:
            print("TrglFragmentView.movePoint: could not compute axes")
            axes = np.zeros((3,3), dtype=np.float64)
            
        rduijk = (axes.T)@duijk
        old_stxy = self.all_stpoints[index]
        new_stxy = old_stxy+rduijk[:2]

        if update_st and (new_stxy != old_stxy).all() and self.pointExists(new_stxy):
            print("move: point already exists")
            return

        timer.time("startup")

        # Only do triangulation-related work if retriangulation is enabled
        if update_st and self.retriangulate_enabled:
            mel = self.maxStEdgeLengthAroundPoint(index)
            half_width = self.half_width_multiplier*self.avg_st_len
            half_width = max(half_width, 2.*mel)
            ops = TrglPointSet(self.all_stpoints, len(self.stpoints), new_stxy, half_width)
            osqcm = self.calculateSqCmOfTrgls(ops.triangulate())

        if update_xyz:
            self.fragment.gpoints[index, :] = new_gijk
            self.setLocalPoint(index)
            timer.time("update xyz")

        if update_st:
            self.stpoints[index, :] = new_stxy
            self.all_stpoints[index, :] = new_stxy
            uv = self.stxyToUv(new_stxy)
            self.fragment.gtpoints[index, :] = uv
            timer.time("set up update_st")

            if self.retriangulate_enabled:
                constrained = self.adjustStPoints(index, half_width)
                timer.time("adjust st points")

                nps = TrglPointSet(self.all_stpoints, len(self.stpoints), new_stxy, half_width)
                nsqcm = self.calculateSqCmOfTrgls(nps.triangulate())
                dsqcm = nsqcm-osqcm
                self.sqcm += dsqcm
                self.applyTrglDiff(ops, nps)
                timer.time("apply diff")
                
                if not constrained:
                    self.rebuildStPoints()
                    timer.time("rebuild st points")
            else:
                # Use the simpler area calculation like in movePoints
                old_sqcm = self.calculateSqCmOfTrgls(self.trgls())
                self.sqcm = old_sqcm

        self.fragment.notifyModified()
        # Only rebuild KD trees if requested (not during dragging)
        if build_kd_trees:
            self.buildKDTrees(True, build_kd_trees, 
                             build_adjacency_list=build_adjacency_list,
                             build_spatial_hash_grid=build_kd_trees)
        return True

    def movePoints(self, indices, new_vijks, update_xyz, update_st, build_kd_trees=True, build_adjacency_list=False):
        """
        Move multiple points to new positions.
        
        Args:
            indices: Array of point indices to move
            new_positions: Array of new positions
            update_xyz: Whether to update xyz coordinates
            update_st: Whether to update st coordinates
            build_kd_trees: Whether to rebuild KD trees (False during dragging)
        """
        timer = Utils.Timer()
        timer.active = False
        vv = self.cur_volume_view
        print("move points called, saving undo state")
        # Save current state for undo
        self.pushFragmentState()
        
        # Convert all positions at once
        timer.time("Start movePoints")
        new_gijks = np.array([vv.transposedIjkToGlobalPosition(vijk) for vijk in new_vijks])
        new_uijks = np.array([vv.transposedIjkToIjk(vijk) for vijk in new_vijks])
        old_vijks = self.vpoints[indices, :3]
        old_uijks = np.array([vv.transposedIjkToIjk(vijk) for vijk in old_vijks])
        duijks = new_uijks - old_uijks
        timer.time("Position conversions")

        # Get axes for all points at once
        axes_list = self.localStAxesBatched(indices)
        timer.time("Get axes")

        # Calculate new positions for all points at once
        rduijks = np.array([axes.T @ duijk for axes, duijk in zip(axes_list, duijks)])
        old_stxys = self.all_stpoints[indices]
        new_stxys = old_stxys + rduijks[:, :2]
        timer.time("Calculate new positions")

        # Update xyz coordinates if requested
        if update_xyz:
            self.fragment.gpoints[indices] = new_gijks
            # Batch update local points
            self.vpoints[indices, :3] = new_vijks
            timer.time("Update xyz")

        # Update st coordinates if requested
        if update_st:
            # Simply update the coordinates without adjusting triangulation
            self.stpoints[indices] = new_stxys
            self.all_stpoints[indices] = new_stxys
            uvs = np.array([self.stxyToUv(stxy) for stxy in new_stxys])
            self.fragment.gtpoints[indices] = uvs
            
            # Update area calculation
            old_sqcm = self.calculateSqCmOfTrgls(self.trgls())
            self.sqcm = old_sqcm
            timer.time("Update st")

        self.fragment.notifyModified()
        if build_kd_trees:
            self.buildKDTrees(True, build_kd_trees, 
                             build_adjacency_list=build_adjacency_list,
                             build_spatial_hash_grid=build_kd_trees)  # Explicitly rebuild KD trees without updating adjacency
        timer.time("Notify modified")
        return True

    def applyTrglDiff(self, ops, nps):
        result = TrglPointSet.trglDiff(ops, nps)
        if result is not None:
            otrgls, ntrgls = result
            # osqcm = self.calculateSqCmOfTrgls(otrgls)
            # nsqcm = self.calculateSqCmOfTrgls(ntrgls)
            # dsqcm = nsqcm-osqcm
            # print("o,n sqcm", osqcm, nsqcm)
            # print("d sqcm", dsqcm)
            # print(" ", self.sqcm+dsqcm)
            # print("otrgls", len(otrgls), self.maxEdgeLengthTrgls(otrgls))
            # print("ntrgls", len(ntrgls), self.maxEdgeLengthTrgls(ntrgls))
            if len(otrgls) > 0 or len(ntrgls) > 0:
                # print("uo", otrgls)
                # print("un", ntrgls)
                # self.replaceTrgls(otrgls, ntrgls)
                # self.fragment.trgls = TrglPointSet.replaceTrgls(self.fragment.trgls, otrgls, ntrgls)
                trgls = TrglPointSet.replaceTrgls(self.fragment.trgls, otrgls, ntrgls)
                if trgls is not None:
                    self.fragment.trgls = trgls
                else:
                    print("applyTrglDiff: retriangulating")
                    self.retriangulateAll()


    def pointExists(self, stxy):
        existing = np.nonzero((self.stpoints == stxy).all(axis=1))[0]
        return len(existing)

    def maxEdgeLengthTrgls(self, trgls):
        return TrglPointSet.maxEdgeLength(self.fragment.gpoints, trgls)

    def maxEdgeLength(self, tps):
        trgls = tps.triangulate()
        return TrglPointSet.maxEdgeLength(self.fragment.gpoints, trgls)

    def maxEdgeLengthAll(self):
        return TrglPointSet.maxEdgeLength(self.fragment.gpoints, self.fragment.trgls)

    def maxStEdgeLengthAroundPoint(self, index):
        sft = self.fragment.trgls
        trgls = sft[(sft==index).any(axis=1)]
        if len(trgls) == 0:
            return 0.
        return TrglPointSet.maxEdgeLength(self.stpoints, trgls)

    def maxStEdgeLengthNearPoint(self, stxy):
        hw = 2*self.avg_st_len
        if len(self.all_stpoints) == 0:
            return 0.
        index = -1
        for i in range(3):
           tps = TrglPointSet(self.stpoints, len(self.stpoints), stxy, hw)
           # print("hw", hw, len(tps.indexes))
           if len(tps.indexes) > 0:
               stpt = self.stpoints[tps.indexes]
               dpt = stpt - stxy
               d = (dpt*dpt).sum(axis=1)
               a = np.argmax(d)
               index = tps.indexes[a]
               # print("a,in", a, index)
               break
           hw *= 2.
        if index >= 0:
            return self.maxStEdgeLengthAroundPoint(index)
        else:
            return 0.

    def addPoint(self, tijk, stxy):
        # print("tf add point", tijk, stxy)
        if stxy is None:
            print("TrglFragment.addPoint failed because stxy not given")
            return
        if tijk is None:
            print("TrglFragment.addPoint failed because tijk not given")
            return
        # print("a before", self.maxEdgeLengthAll())
        timer = Utils.Timer()
        timer.active = False

        vv = self.cur_volume_view
        gijk = vv.transposedIjkToGlobalPosition(tijk)

        # existing = np.nonzero((self.stpoints == stxy).all(axis=1))[0]
        # if len(existing > 0):
        #     print("Point already exists at stxy", stxy)
        #     return
        if self.pointExists(stxy):
            print("Point already exists at stxy", stxy)
            return

        astxy = np.array(stxy)
        # agijk = np.array(gijk)
        # Need to compute this before point is added to gpoints
        mel = self.maxStEdgeLengthNearPoint(astxy)
        # print("mel", mel)
        half_width = self.half_width_multiplier*self.avg_st_len
        half_width = max(half_width, 2*mel)

        self.fragment.gpoints = np.append(self.fragment.gpoints, [gijk], axis=0)
        uv = self.stxyToUv(stxy)
        nstp = len(self.stpoints)
        self.fragment.gtpoints = np.append(self.fragment.gtpoints, [uv], axis=0)
        self.stpoints = np.append(self.stpoints, [stxy], axis=0)
        self.all_stpoints = np.insert(self.all_stpoints, nstp, stxy, axis=0)
        timer.time("setup")

        ops = TrglPointSet(self.all_stpoints, len(self.stpoints), astxy, half_width)
        ops.deletePoint(nstp)
        osqcm = self.calculateSqCmOfTrgls(ops.triangulate())
        # print("a ops", self.maxEdgeLength(ops))
        timer.time("ops")

        # Can't do this here; self.fragment.trgls is not
        # up to date yet (new point hasn't been added)
        # constrained = self.adjustStPoints(nstp, half_width)

        nps = TrglPointSet(self.all_stpoints, len(self.stpoints), astxy, half_width)

        '''
        nsqcm = self.calculateSqCmOfTrgls(nps.triangulate())
        dsqcm = nsqcm-osqcm
        self.sqcm += dsqcm
        print(self.sqcm, self.calculateSqCmOfTrgls(self.trgls()))
        '''

        # print("a nps", self.maxEdgeLength(nps))
        # nps.addPointAtEnd(stxy)
        timer.time("nps")

        self.applyTrglDiff(ops, nps)
        timer.time("diff")
        constrained = self.adjustStPoints(nstp, half_width)
        timer.time("adj")
        nps2 = TrglPointSet(self.all_stpoints, len(self.stpoints), astxy, half_width)
        nsqcm = self.calculateSqCmOfTrgls(nps2.triangulate())
        dsqcm = nsqcm-osqcm
        self.sqcm += dsqcm
        # print(self.sqcm, self.calculateSqCmOfTrgls(self.trgls()))

        # TODO: will this crash if lens are not equal?
        nps2match = len(nps.indexes) == len(nps2.indexes) and (nps.indexes == nps2.indexes).all()
        # if not nps2match:
        #     print("nps, nps2 index mismatch")
        # print("a nps2", self.maxEdgeLength(nps2))
        timer.time("nps2")
        self.applyTrglDiff(nps, nps2)
        timer.time("diff2")

        # tcount will be zero if the new point has no triangles,
        # non-zero otherwise
        tcount = (self.fragment.trgls==nstp).any(axis=1).sum()

        if tcount > 0 and constrained and nps2match:
            self.addLocalPoint(nstp)
        else:
            if not nps2match:
                print("addPoint: set local points", tcount, constrained, nps2match)
            self.setLocalPoints(True, False, build_kd_trees=True, build_adjacency_list=True)
        # print("a after", self.maxEdgeLengthAll())
        self.fragment.notifyModified()

    # If two points are colocated in xyz, disconnect
    # one of them from all the trgls, replacing it by
    # the other.
    # The disconnected point can be deleted in
    # another function
    def disconnectColocatedPoints(self):
        if self.stpoints is None:
            return
        trgls = self.fragment.trgls
        if len(trgls) == 0:
            return
        pts = self.fragment.gpoints
        npt = pts.shape[0]
        # print("npt", npt)
        # print("min edge length", TrglPointSet.minEdgeLength(pts,trgls))
        # This is wrong, and caused annoying pyramids:
        # lind = np.lexsort((pts[:,1], pts[:,0]))
        lind = np.lexsort((pts[:,2], pts[:,1], pts[:,0]))
        rlind = np.zeros(npt, dtype=np.int64)
        rlind[lind] = np.ogrid[:npt]
        sarr = pts[lind]
        value,inds,counts = np.unique(sarr, return_index=True, return_counts=True, axis=0)
        tinds = np.repeat(inds, counts)
        ndup = np.sum(counts-1)
        if ndup > 0:
            print("found",ndup,"colocated point(s)")
        dedup = lind[tinds[rlind]]
        # print("a", len(trgls))
        # print(trgls[(trgls==181).any(axis=1)])
        trgls = dedup[trgls]
        # print("b", len(trgls))
        # print(trgls[(trgls==181).any(axis=1)])
        trgls = trgls[trgls[:,0] != trgls[:,1]]
        # print("c", len(trgls))
        # print(trgls[(trgls==181).any(axis=1)])
        trgls = trgls[trgls[:,1] != trgls[:,2]]
        # print("d", len(trgls))
        # print(trgls[(trgls==181).any(axis=1)])
        trgls = trgls[trgls[:,2] != trgls[:,0]]
        # print("e", len(trgls))
        # print(trgls[(trgls==181).any(axis=1)])
        self.fragment.trgls = trgls

    # This will create free points by deleting the
    # triangles that hold them
    def deleteDisconnectedComponents(self):
        if self.stpoints is None:
            return
        trgls = self.fragment.trgls
        if len(trgls) < 3:
            return
        neighbors = BaseFragment.findNeighbors(trgls)
        '''
        gpoints = self.fragment.gpoints
        # testinds = np.nonzero(gpoints[:,2] == 10536)
        # print("testinds", testinds)
        # print(gpoints[testinds])
        testinds = np.nonzero(trgls==281)[0]
        print(testinds)
        print(trgls[testinds])
        print(neighbors[testinds])
        # print("trgls")
        # print(trgls[1774:1780])
        # print("neighbors")
        # print(neighbors[1774:1780])
        '''
        nt = trgls.shape[0]
        # print("t n", trgls.shape, neighbors.shape)
        # each triangle also has itself as a neighbor
        # (this is so that triangles with no neighbors will still
        # show up in the connectivity graph)
        neighbors = np.append(neighbors, np.ogrid[:nt,:0][0], axis=1)
        tindex = np.ogrid[:4*nt]//4
        nindex = neighbors.flatten()
        is_valid = (nindex > -1)
        tindex = tindex[is_valid]
        nindex = nindex[is_valid]
        ones = np.full(nindex.shape[0], 1)
        connections = scipy.sparse.csr_array((ones, (tindex, nindex)), shape=(nt,nt))
        # print("connections")
        # print(connections)
        nc, labels = scipy.sparse.csgraph.connected_components(connections, directed=False)
        if nc > 1:
            print("number of components", nc)
            hist = np.zeros(nc, dtype=np.int64)
            np.add.at(hist, labels, 1)
            print("hist", hist)
            mainind = np.argmax(hist)
            keep_trgl = (labels==mainind)
            self.fragment.trgls = trgls[keep_trgl]

    def deleteFreePoints(self):
        if self.stpoints is None:
            return
        trgls = self.fragment.trgls
        # Don't delete free points unless there is at
        # least one trgl
        if len(trgls) == 0:
            return
        npt = len(self.stpoints)
        free_flag = np.full(npt, True, dtype=np.bool_)
        free_flag[trgls.flatten()] = False
        nf = free_flag.sum()
        if nf == 0:
            return
        print(nf, "free points")
        nc = npt-nf
        o2n = np.full(npt, -1, dtype=np.int64)
        o2n[~free_flag] = np.ogrid[:nc]
        # fp_index = np.nonzero(free_flag)[0]
        self.fragment.trgls = o2n[trgls]
        old_outside = self.stpoints[npt:].copy()
        self.fragment.gpoints = self.fragment.gpoints[~free_flag]
        self.fragment.gtpoints = self.fragment.gtpoints[~free_flag]
        self.stpoints = self.stpoints[~free_flag]
        self.all_stpoints = np.concatenate((self.stpoints, old_outside))
        self.setLocalPoints(True, False, build_kd_trees=True, build_adjacency_list=True)

    def deletePointByIndex(self, index):
        if index < 0:
            return
        if index >= len(self.fragment.gpoints):
            return
        if self.stpoints is None or index >= len(self.stpoints):
            return

        # If the node is part of selected_nodes, delete all selected nodes
        points_to_delete = {index}
        if hasattr(self, 'selected_nodes') and index in self.selected_nodes:
            points_to_delete = self.selected_nodes.copy()
        
        # Sort indices in descending order to avoid index shifting issues
        sorted_indices = sorted(points_to_delete, reverse=True)
        
        # Get parameters for triangulation update (using the first point as reference)
        mel = self.maxStEdgeLengthAroundPoint(sorted_indices[-1])  # Use first point (lowest index)
        half_width = self.half_width_multiplier*self.avg_st_len
        half_width = max(half_width, 2.*mel)

        # Store old state for triangulation
        old_stxy = self.all_stpoints[sorted_indices[-1]]  # Use first point's position
        ops = TrglPointSet(self.all_stpoints, len(self.stpoints), old_stxy, half_width)
        osqcm = self.calculateSqCmOfTrgls(ops.triangulate())

        # Delete all points at once from arrays
        mask = np.ones(len(self.fragment.gpoints), dtype=bool)
        mask[list(points_to_delete)] = False
        
        self.fragment.gpoints = self.fragment.gpoints[mask]
        self.fragment.gtpoints = self.fragment.gtpoints[mask]
        self.fpoints = self.fpoints[mask]
        self.vpoints = self.vpoints[mask]
        self.vpoints[:,3] = np.arange(len(self.vpoints))
        self.stpoints = self.stpoints[mask]
        self.all_stpoints = np.delete(self.all_stpoints, list(points_to_delete), 0)

        # Update triangle indices - remove triangles that reference deleted points
        old_trgls = self.fragment.trgls.copy()
        valid_trgls = np.ones(len(old_trgls), dtype=bool)
        
        # Mark triangles containing deleted points as invalid
        for idx in sorted_indices:
            valid_trgls &= ~np.any(old_trgls == idx, axis=1)
            old_trgls[old_trgls > idx] -= 1
        
        self.fragment.trgls = old_trgls[valid_trgls]

        # Update triangulation once for all deleted points
        nps = TrglPointSet(self.all_stpoints, len(self.stpoints), old_stxy, half_width)
        self.applyTrglDiff(ops, nps)
        
        # Final triangulation update
        ops2 = TrglPointSet(self.all_stpoints, len(self.stpoints), old_stxy, half_width)
        constrained = self.adjustStPoints(-1, half_width, old_stxy)
        nps2 = TrglPointSet(self.all_stpoints, len(self.stpoints), old_stxy, half_width)
        self.applyTrglDiff(ops2, nps2)
        nsqcm = self.calculateSqCmOfTrgls(nps2.triangulate())
        dsqcm = nsqcm-osqcm
        self.sqcm += dsqcm
        nps2match = len(ops2.indexes) == len(nps2.indexes) and (ops2.indexes == nps2.indexes).all()

        # Update local points if needed
        if not (constrained and nps2match):
            if not nps2match:
                print("deletePointByIndex: set local points", constrained, nps2match)
            self.setLocalPoints(True, False, build_kd_trees=True, build_adjacency_list=True)

        # Clear selected nodes after deletion
        if hasattr(self, 'selected_nodes'):
            self.selected_nodes = set()
            
        self.fragment.notifyModified()

    # returns list of trgl indexes
    def regionByNormals(self, ptind, max_angle):
        pts = self.fpoints
        trgls = self.fragment.trgls
        neighbors = self.fragment.neighbors
        minz = math.cos(math.radians(max_angle))
        # print("minz", minz)
        normals = BaseFragment.faceNormals(pts, trgls)
        trgl_stack = deque()
        tap = BaseFragment.trglsAroundPoint(ptind, trgls)
        zsgn = np.sum(normals[tap,2])
        # print("zsgn", zsgn)
        if zsgn < 0:
            zsgn = -1
        else:
            zsgn = 1
        # print("tap", tap)
        trgl_stack.extend(tap)
        done = set()
        out_trgls = []
        while len(trgl_stack) > 0:
            trgl = trgl_stack.pop()
            # print("1", trgl)
            if trgl in done:
                continue
            done.add(trgl)
            # print("2", trgl)
            n = normals[trgl]
            # print("n", n)
            # if abs(n[2]) < minz:
            if zsgn*n[2] < minz:
                continue
            # print("3", trgl)
            out_trgls.append(trgl)
            for neigh in neighbors[trgl]:
                # print("4", neigh)
                if neigh < 0:
                    continue
                # print("5", neigh)
                if neigh in done:
                    continue
                # print("6", neigh)
                trgl_stack.append(neigh)
            # print("ts", len(trgl_stack))
        return out_trgls
    
    def moveSelectedNodesToBrushArc(self, brush_points, brush_radius, z_val):
        """
        Moves selected nodes to positions interpolated from brush points, maintaining their angles
        but adjusting their radii to match the interpolated brush arc.
        
        Args:
            brush_points (np.ndarray): Array of (x,y) brush stroke points
            brush_radius (float): Radius around brush stroke to consider
            z_val (int): Z-level index
        """
        if not hasattr(self.fragment, 'params') or self.fragment.params is None:
            print("Missing params")
            return None
            
        if 'umbilicus_points' not in self.fragment.params or self.fragment.params['umbilicus_points'] is None:
            print("Missing umbilicus_points")
            return None
            
        if not hasattr(self, 'selected_nodes') or not self.selected_nodes:
            print("No selected_nodes")
            return None

        # Get umbilicus point for this z-level
        umbilicus_point_3d = self.fragment.params['umbilicus_points'][z_val]
        umbilicus_xy = umbilicus_point_3d[:2]

        # Convert brush points to polar coordinates
        brush_points = np.asarray(brush_points)
        brush_vectors = brush_points - umbilicus_xy
        brush_angles = np.degrees(np.arctan2(brush_vectors[:, 1], brush_vectors[:, 0])) % 360
        brush_radii = np.sqrt(np.sum(brush_vectors**2, axis=1))

        # Sort brush points by angle
        sort_idx = np.argsort(brush_angles)
        brush_angles = brush_angles[sort_idx]
        brush_radii = brush_radii[sort_idx]

        # Convert selected nodes to polar coordinates
        selected_indices = np.array(list(self.selected_nodes))
        selected_points = self.fragment.gpoints[selected_indices][:, :2]
        vectors = selected_points - umbilicus_xy
        node_angles = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0])) % 360
        node_radii = np.sqrt(np.sum(vectors**2, axis=1))

        # Handle wrap-around for interpolation
        # If the brush stroke crosses the 0/360 boundary, adjust angles
        if brush_angles[-1] - brush_angles[0] > 180:
            # Some points need to be adjusted by +360 for proper interpolation
            brush_angles = np.where(brush_angles < brush_angles[0], brush_angles + 360, brush_angles)
            node_angles = np.where(node_angles < brush_angles[0], node_angles + 360, node_angles)

        # Interpolate radii for each node based on its angle
        new_radii = np.interp(node_angles, brush_angles, brush_radii)

        # Convert back to cartesian coordinates
        angles_rad = np.radians(node_angles)
        new_x = umbilicus_xy[0] + new_radii * np.cos(angles_rad)
        new_y = umbilicus_xy[1] + new_radii * np.sin(angles_rad)
        
        # Create array of new positions, maintaining Z coordinates
        #move points assumes x,z,y
        new_positions = np.column_stack((
            new_x, 
            self.fragment.gpoints[selected_indices][:, 2],
            new_y
        ))

        # print("new_positions", new_positions.shape, new_positions[0])
        
        # Move all points at once
        self.movePoints(selected_indices, new_positions, True, True)

        # print(f"Moved {len(self.selected_nodes)} nodes to interpolated positions")
        
        
        





    def findNodesInBrushArc(self, brush_points, brush_radius, z_val):
        """
        Finds nodes in self.selected_nodes that lie within the 'pizza slice' arc
        defined by the brush stroke around the umbilicus point.

        Args:
            brush_points (array-like): 2D array of (x, y) points from the brush stroke
            brush_radius (float):      Radius around brush stroke to consider
            z_val (int or float):      The z-index or slice index for the umbilicus_points

        Returns:
            Set[int]: Subset of self.selected_nodes that are inside the pizza slice arc.
        """
        # Check prerequisites
        if not hasattr(self.fragment, 'params') or self.fragment.params is None:
            print("Missing params")
            return None
            
        if 'umbilicus_points' not in self.fragment.params or self.fragment.params['umbilicus_points'] is None:
            print("Missing umbilicus_points")
            return None
            
        if not hasattr(self, 'selected_nodes') or not self.selected_nodes:
            print("No selected_nodes")
            return None

        if not hasattr(self, 'adjacency_list') or self.adjacency_list is None:
            print("No adjacency_list")
            return None

        # Get umbilicus point for this z-level
        umbilicus_point_3d = self.fragment.params['umbilicus_points'][z_val]
        umbilicus_xy = umbilicus_point_3d[:2]

        # Convert brush points to numpy array if not already
        brush_points = np.asarray(brush_points)
    
        if brush_points.shape[0] == 0:
            print("No brush points—nothing to do")
            return None
            
        if brush_points.shape[0] == 1:
            # For single point, use KD tree for efficient radius search
            point = np.array([brush_points[0][0], brush_points[0][1], z_val])
            if not hasattr(self, 'kd_tree') or self.kd_tree is None:
                print("No KD tree")
                return set()
            indices = self.kd_tree.query_ball_point(point, brush_radius)
            # Only return nodes that are in selected_nodes
            return set(indices) & self.selected_nodes

        # Calculate angles for all brush points relative to umbilicus
        angles = []
        for point in brush_points:
            dx = point[0] - umbilicus_xy[0]
            dy = point[1] - umbilicus_xy[1]
            angle = np.arctan2(dy, dx)
            # Convert to degrees and ensure positive angles (0 to 360)
            angle_deg = np.degrees(angle) % 360
            angles.append(angle_deg)

        # Sort angles and find largest gap
        angles = np.array(sorted(angles))
        angle_diffs = np.diff(angles)
        # Add the wrap-around difference
        angle_diffs = np.append(angle_diffs, 360 - (angles[-1] - angles[0]))
        max_gap_idx = np.argmax(angle_diffs)
        max_gap = angle_diffs[max_gap_idx]

        # Determine if we have a complete circle or an arc
        GAP_THRESHOLD = 60  # degrees
        is_complete_circle = max_gap <= GAP_THRESHOLD

        if is_complete_circle:
            print("Complete circle detected")
            arc_start = 0
            arc_end = 360
        else:
            # The arc starts after the largest gap
            if max_gap_idx == len(angles) - 1:
                arc_start = angles[0]
                arc_end = angles[-1]
            else:
                arc_start = angles[max_gap_idx + 1]
                arc_end = angles[max_gap_idx] + 360 if angles[max_gap_idx + 1] < angles[max_gap_idx] else angles[max_gap_idx]
            print(f"Arc detected: {arc_start:.1f}° to {arc_end:.1f}°")

        # Find nodes within the arc
        nodes_in_arc = set()
        for node in self.selected_nodes:
            node_point = self.fragment.gpoints[node][:2]  # Get x,y coordinates
            
            # Calculate angle for this node
            dx = node_point[0] - umbilicus_xy[0]
            dy = node_point[1] - umbilicus_xy[1]
            node_angle = np.degrees(np.arctan2(dy, dx)) % 360

            # Check if node is within the arc
            if is_complete_circle:
                nodes_in_arc.add(node)
            else:
                # Handle wrap-around case
                if arc_start > arc_end:
                    if node_angle >= arc_start or node_angle <= arc_end:
                        nodes_in_arc.add(node)
                else:
                    if arc_start <= node_angle <= arc_end:
                        nodes_in_arc.add(node)

        return nodes_in_arc


    # def moveSelectedNodesToBrushArc(self):
    def findDominantWrap2D(self):
        """
        For each selected node, finds its adjacent nodes at the same z-level.
        Returns the group that contains the most selected nodes, oriented so the wrap's
        discontinuity is in the middle of the largest gap between selected nodes.
        """
        # Initial validation checks remain the same
        if not hasattr(self.fragment, 'params') or self.fragment.params is None:
            print("Missing params")
            return None
        
        if 'pts_per_wrap' not in self.fragment.params or self.fragment.params['pts_per_wrap'] is None:
            print("Missing pts_per_wrap")
            return None
        
        if not hasattr(self, 'selected_nodes') or not self.selected_nodes:
            print("no selected_nodes")
            return None

        if not hasattr(self, 'adjacency_list') or self.adjacency_list is None:
            print("no adjacency_list")
            return None
        
        if 'umbilicus_points' not in self.fragment.params or self.fragment.params['umbilicus_points'] is None:
            print("Missing umbilicus_points")
            return None

        # Get target number of adjacent points
        target_adjacent = int(self.fragment.params['pts_per_wrap'])

        # First find the best group containing most selected nodes
        groups = []
        for start_node in self.selected_nodes:
            start_z = round(self.fragment.gpoints[start_node][2], 2)
            
            visited = {start_node}
            group = {start_node}
            queue = [start_node]
            
            while queue and len(group) < target_adjacent:
                current = queue.pop(0)
                for neighbor in self.adjacency_list[current]:
                    if neighbor >= len(self.fragment.gpoints):  # Add bounds check
                        continue
                    if neighbor not in visited:
                        visited.add(neighbor)
                        neighbor_z = round(self.fragment.gpoints[neighbor][2], 2)
                        if neighbor_z == start_z:
                            group.add(neighbor)
                            queue.append(neighbor)
                            if len(group) >= target_adjacent:
                                break
            
            if len(group) >= target_adjacent:
                groups.append(group)
        
        # Find group with most selected nodes
        best_group = None
        max_selected = 0
        for group in groups:
            selected_count = len(group & self.selected_nodes)
            if selected_count > max_selected:
                max_selected = selected_count
                best_group = group
        
        if best_group is None:
            return None

        # Find the best split point (in largest gap)
        z_val = int(round(self.fragment.gpoints[list(self.selected_nodes)[0]][2], 2))
        umbilicus_point_3d = self.fragment.params['umbilicus_points'][z_val]
        umbilicus_xy = umbilicus_point_3d[:2]
        
        # Calculate angles for selected nodes
        selected_angles = []
        for node in self.selected_nodes:
            node_point = self.fragment.gpoints[node][:2]
            dx = node_point[0] - umbilicus_xy[0]
            dy = node_point[1] - umbilicus_xy[1]
            angle = np.degrees(np.arctan2(dy, dx)) % 360
            selected_angles.append(angle)
        
        # Find largest gap
        angles = np.array(sorted(selected_angles))
        angle_diffs = np.diff(angles)
        wrap_diff = 360 - (angles[-1] - angles[0])
        angle_diffs = np.append(angle_diffs, wrap_diff)
        max_gap_idx = np.argmax(angle_diffs)
        max_gap = angle_diffs[max_gap_idx]
        
        # Calculate target angle in middle of largest gap
        if max_gap_idx == len(angles) - 1:
            start_angle = angles[-1]
            end_angle = angles[0] + 360
        else:
            start_angle = angles[max_gap_idx]
            end_angle = angles[max_gap_idx + 1]
            if end_angle < start_angle:
                end_angle += 360
        
        target_angle = ((start_angle + end_angle) / 2) % 360
        
        # Find node closest to target angle in best_group
        best_split_node = None
        min_angle_diff = float('inf')
        
        for node in best_group:
            node_point = self.fragment.gpoints[node][:2]
            dx = node_point[0] - umbilicus_xy[0]
            dy = node_point[1] - umbilicus_xy[1]
            node_angle = np.degrees(np.arctan2(dy, dx)) % 360
            angle_diff = min((node_angle - target_angle) % 360, 
                            (target_angle - node_angle) % 360)
            
            if angle_diff < min_angle_diff:
                min_angle_diff = angle_diff
                best_split_node = node
        
        if best_split_node is None:
            return None

        # Now build the wrap starting from the opposite side of the split
        opposite_angle = (target_angle + 180) % 360
        start_node = None
        min_angle_diff = float('inf')
        
        # Find the node closest to the opposite angle
        for node in best_group:
            node_point = self.fragment.gpoints[node][:2]
            dx = node_point[0] - umbilicus_xy[0]
            dy = node_point[1] - umbilicus_xy[1]
            node_angle = np.degrees(np.arctan2(dy, dx)) % 360
            angle_diff = min((node_angle - opposite_angle) % 360, 
                            (opposite_angle - node_angle) % 360)
            
            if angle_diff < min_angle_diff:
                min_angle_diff = angle_diff
                start_node = node

        # Build the final wrap group starting from the opposite point
        start_z = round(self.fragment.gpoints[start_node][2], 2)
        final_group = {start_node}
        visited = {start_node}
        queue = [start_node]
        
        while queue and len(final_group) < target_adjacent:
            current = queue.pop(0)
            for neighbor in self.adjacency_list[current]:
                if neighbor >= len(self.fragment.gpoints):  # Add bounds check
                    continue
                if neighbor not in visited and neighbor != best_split_node:  # Avoid crossing the split
                    visited.add(neighbor)
                    neighbor_z = round(self.fragment.gpoints[neighbor][2], 2)
                    if neighbor_z == start_z:
                        final_group.add(neighbor)
                        queue.append(neighbor)
                        if len(final_group) >= target_adjacent:
                            break

        self.selected_nodes = final_group
        return final_group



class TrglPointSet:

    # Create a TrglPointSet that contains all stpoints
    # that lie in a square whose center and half-width are given
    def __init__(self, all_stpoints, nstpoints, pt, half_width):
        # TODO: this isn't valid!
        # if all_stpoints is None or len(all_stpoints) == 0:
        #     return None
        ptsb = ((pt-half_width <= all_stpoints) & 
                (all_stpoints <= pt+half_width)).all(axis=1)
        self.indexes = np.nonzero(ptsb)[0]
        self.reverse_indexes = np.full((len(all_stpoints)), -1, dtype=np.int64)
        self.reverse_indexes[ptsb] = np.ogrid[:ptsb.sum()]
        self.pts = all_stpoints[self.indexes]
        # number of normal (not outside) points in all_stpoints
        self.nstpoints = nstpoints
        # bs = np.nonzero(self.indexes < nstpoints)[0]
        # self.nipoints = bs[-1]+1
        # self.nipoints = bs.shape[0]

        # number of normal (not outside) points in self.indexes
        self.nipoints = (self.indexes < nstpoints).sum()


    # Finds the points on the boundary that was
    # created by windowing the surface.
    # For efficiency, assumes that the input triangles
    # have local indexes instead of global, with -1
    # for indexes outside the window.
    def cutBoundaryPoints(self, ritrgls):
        # ritrgls = self.reverse_indexes[trgls]
        # ctrgls (cut trgls) are trgls that have either 
        # one or two points outside of the window
        ctrgls = ritrgls[((ritrgls<0).sum(axis=1)+1)//2 == 1]
        # print("ctrgls")
        # print(ctrgls)
        # print("cbp", trgls.shape, ritrgls.shape, ctrgls.shape)
        # print(ctrgls)
        bpts = ctrgls[ctrgls>=0].flatten()
        ubpts = np.unique(bpts)
        return ubpts

    def deletePoint(self, index):
        row = (self.indexes == index).nonzero()[0]
        # print("before", self.indexes)
        self.indexes = np.delete(self.indexes, row, 0)
        # print("after", self.indexes)
        self.pts = np.delete(self.pts, row, 0)

    ''' Won't work; points are st points not xyz points
    def calculateSqCm(self, trgls, voxel_size_um):
        if trgls is None or len(trgls) == 0.:
            return 0.
        sqcm = BaseFragment.calculateSqCm(self.pts, trgls, voxel_size_um)
    '''

    @staticmethod
    def trglDiff(oldps, newps):
        # print("doing old trgls")
        old_trgls = oldps.triangulate()
        # print("doing new trgls")
        new_trgls = newps.triangulate()
        # print(-1 if old_trgls is None else len(old_trgls), 
        #       -1 if new_trgls is None else len(new_trgls))

        if old_trgls is not None:
            lo = len(old_trgls)
        # print("None", old_trgls is None, new_trgls is None)

        # if old_trgls is None or new_trgls is None:
        #     return None
        if old_trgls is None:
            old_trgls = np.zeros((0,3), dtype=np.int64)
        if new_trgls is None:
            new_trgls = np.zeros((0,3), dtype=np.int64)

        unique_old_trgls = Utils.setDiff2DIndex(old_trgls, new_trgls)
        unique_new_trgls = Utils.setDiff2DIndex(new_trgls, old_trgls)
        # print("uo", unique_old_trgls)
        # print("un", unique_new_trgls)
        return old_trgls[unique_old_trgls], new_trgls[unique_new_trgls]

    # Input: a trgls array (3 columns, n rows)
    # Output: the same array, but each row has been
    # rotated so that the smallest index of that row is
    # moved to the first column
    @staticmethod
    def rotateToMin(trgls):
        # print("rtm")
        mins = np.argmin(trgls, axis=1)
        otrgls = trgls.copy()
        otrgls[mins==1] = np.roll(trgls[mins==1], 2, axis=1)
        otrgls[mins==2] = np.roll(trgls[mins==2], 1, axis=1)
        return otrgls

    @staticmethod
    def replaceTrgls(trgls, uo, un):
        if len(uo) > 0:
            orows = []
            for o in uo:
                row = (trgls == o).all(axis=1).nonzero()[0]
                # print("row", row, len(row))
                if len(row) != 1:
                    print("replaceTrgls unexpected row len", len(row))
                    return None
                if len(row) == 0:
                    # continue
                    return None
                orows.append(row[0])
            # print("before", len(trgls))
            # print("orows", orows)
            trgls = np.delete(trgls, orows, 0)
            # print("after", len(trgls))
        if len(un) > 0:
            trgls = np.concatenate((trgls, un), axis=0)
        return trgls

    @staticmethod
    def edgeLengths(pts, trgls):
        if trgls is None or len(trgls) == 0:
            return np.zeros((0,3), dtype=np.float64)
        tpts = pts[trgls]
        dtpts = tpts - np.roll(tpts, 1, axis=1)
        dsq = (dtpts*dtpts).sum(axis=2)
        return np.sqrt(dsq)

    # pts should be xyz points, not uv points
    @staticmethod
    def maxEdgeLength(pts, trgls):
        '''
        if trgls is None or len(trgls) == 0:
            return 0.
        tpts = pts[trgls]
        dtpts = tpts - np.roll(tpts, 1, axis=1)
        dsq = (dtpts*dtpts).sum(axis=2)
        maxdsq = np.max(dsq)
        return np.sqrt(maxdsq)
        '''
        lens = TrglPointSet.edgeLengths(pts, trgls)
        if len(lens) == 0:
            return 0.
        return np.max(lens)

    # pts should be xyz points, not uv points
    @staticmethod
    def minEdgeLength(pts, trgls):
        '''
        if trgls is None or len(trgls) == 0:
            return 0.
        tpts = pts[trgls]
        dtpts = tpts - np.roll(tpts, 1, axis=1)
        dsq = (dtpts*dtpts).sum(axis=2)
        mindsq = np.min(dsq)
        return np.sqrt(mindsq)
        '''
        lens = TrglPointSet.edgeLengths(pts, trgls)
        if len(lens) == 0:
            return 0.
        return np.min(lens)

    '''
    # pts should be xyz points, not uv points
    # returns ntrgl*3 array of the 3 angles around each trgl
    @staticmethod
    def computeAngles(self, pts, trgls):
        if pts is None or len(pts) < 3:
            return None
        if trgls is None or len(trgls) == 0:
            return None
        tpts = pts[trgls]
        # d02, d10, d21
        # i.e. vector from pt 0 to pt 2, etc
        tvecs = np.roll(tpts, 1, axis=1) - tpts
        tlens = np.sqrt((tvecs*tvecs).sum(axis=2))
        tlens[tlens[:,:]==0] = 1.

        # normalized vecs
        tnvecs = tvecs/tlens[:,:,np.newaxis]

        # dot products of normalized vectors
        tndps = (-tnvecs*np.roll(tnvecs, -1, axis=1)).sum(axis=2)
        angles = np.arccos(tndps)
        return angles

    @staticmethod
    def boundaryPoints(self, trgls):

    @staticmethod
    def computeAngleDeficits(self, pts, trgls):
        if pts is None or len(pts) < 3:
            return None
        if trgls is None or len(trgls) == 0:
            return None
        angles = self.computeAngles(pts, trgls)
        if angles is None:
            return None
        sums = np.zeros(len(points), dtpye=np.float64)
        is_on_boundary = self.onBoundaryArray()
    '''

    # pts should be xyz points, not uv points
    @staticmethod
    def findSpikes(pts, trgls, txt=""):
        mapper = UVMapper(pts, trgls)
        mapper.createAngles()
        sums = mapper.sumAnglesAroundPoints()
        on_boundary = mapper.onBoundaryArray()
        sums[on_boundary] = 100.
        inds = np.argsort(sums)
        minangle = 3.14
        dinds = inds[sums[inds] < minangle]
        if len(dinds) > 0:
            print("Spikes",txt)
            print(dinds)
            print(sums[dinds])
            print(pts[dinds])


    def triangulate(self):
        trgls = None
        if len(self.pts) < 4:
            print("triangulate: fewer than 4 points")
            return None
        try:
            trgls = Delaunay(self.pts).simplices
        except Exception as err:
            err = str(err).splitlines()[0]
            print("triangulate: triangulation error: %s"%err)
            return None

        nst = self.nstpoints
        # Besides rotating to min pt index, this line
        # replaces the local pt index by the global pt index
        # trgls = self.rotateToMin(np.array(self.indexes)[trgls])
        trgls = self.rotateToMin(self.indexes[trgls])
        # Remove trgls that contain 1 or more outside points
        trgls = trgls[(trgls < nst).all(axis=1)]
        return trgls

    # xyzpts is all the fragment points (gpoints),
    # trgls is all the trgls,
    # ptindex is the index (into the list of all fragment points)
    # of the point that is being moved.
    def adjustSts(self, xyzpts, trgls, ptindex):
        timer = Utils.Timer()
        timer.active = False
        if trgls is None:
            return
        if len(self.pts) == 0:
            return
        # indices (relative to fragment point list) 
        # of windowed non-outside points
        inds = self.indexes[:self.nipoints]
        # stxy locations of windowed non-outside points
        pts = self.pts[:self.nipoints]
        # print("inds", inds)
        # print("xyzpts", xyzpts.shape)
        # xyz locations of windowed non-outside points
        localxyz = xyzpts[inds]
        # Don't triangulate here; the local convex
        # hull would not be desirable!
        # trgls = self.triangulate()
        # print(trgls)
        timer.time("  a")

        # trgls converted to use windowed-point indexing
        ltrgls = self.reverse_indexes[trgls]
        timer.time("  b")
        # trgls that have at least one point inside the window
        in_trgls = ltrgls[(ltrgls >= 0).any(axis=1)]
        timer.time("  b2")

        ## eliminate trgls that have points outside of the
        ## window 
        # ltrgls = ltrgls[(ltrgls >= 0).all(axis=1)]

        # trgls that have no points outside the window
        fully_in_trgls = in_trgls[(in_trgls >= 0).all(axis=1)]
        timer.time("  c")
        """
        bounds_flag = mapper.onBoundaryArray()
        # input point is never used as a constraint
        if ptindex >= 0:
            bounds_flag[self.reverse_indexes[ptindex]] = False

        '''
        ptrgls = trgls.copy()
        ptrgls[bounds_flag[ptrgls]] *= -1
        print(ptrgls)
        '''

        nb = bounds_flag.sum()
        constraints = np.zeros((nb, 3), dtype=np.float64)

        # print(bounds_flag)
        # print(xyzpts.shape, localxyz.shape, self.pts.shape, pts.shape, constraints.shape)
        constraints[:, 0] = np.nonzero(bounds_flag)[0]
        constraints[:, (1,2)] = pts[bounds_flag]
        """

        """
        # floating points (points that are not part of
        # any trgl), using windowed-point indexing:
        # print("ltrgls")
        # print(ltrgls)
        floating = np.full(pts.shape[0], True, dtype=np.bool_)
        floating[ltrgls.flatten()] = False

        # floating = np.logical_not(np.isin(inds, ltrgls.flatten()))
        # print("floating", floating.sum(),"of",len(inds))
        print("floating", floating.sum(), np.nonzero(floating)[0])
        """

        # bpts contains points that are on the boundaries
        # of cut triangles.  The points use windowed-point indexing
        # bpts = self.cutBoundaryPoints(trgls)
        bpts = self.cutBoundaryPoints(in_trgls)
        timer.time("  d")

        """
        # print("bpts floating", bpts.shape, floating.shape, floating.sum())
        # add to "floating" the points on boundaries created
        # by windowing
        bfloating = floating.copy()
        bfloating[bpts] = True
        bpts = np.nonzero(bfloating)[0]
        # print("bpts floating again", bpts.shape, bfloating.shape, bfloating.sum())
        # print(bpts.shape)

        # print("bpts")
        # print(bpts)
        # lbpts = self.indexes[bpts]
        # print("lbpts")
        # print(lbpts)


        # set the floating points, and the window-boundary points,
        # as constraints (the floating points need to be constrained,
        # though they have no effect, otherwise the parameterizer
        # may complain of a singular matrix)
        """

        mapper = UVMapper(localxyz, fully_in_trgls)
        timer.time("  e")

        if ptindex >= 0:
            bpts = bpts[bpts != self.reverse_indexes[ptindex]]
        # print("cut bpts", len(bpts))
        constraints = np.zeros((bpts.shape[0], 3), dtype=np.float64)
        constraints[:, 0] = bpts
        constraints[:, (1,2)] = pts[bpts]

        mapper.constraints = constraints
        weight = .5
        mapper.ip_weights = np.full(pts.shape[0], weight)
        if ptindex >= 0:
            mapper.ip_weights[self.reverse_indexes[ptindex]] = 0.
        mapper.initial_points = pts
        # adjusted_sts = mapper.computeUvsFromABF()
        # print(pts.shape[0], nb)
        adjusted_sts = mapper.computeUvsFromXyzs()
        # Would prefer to this because it may handle flipped triangles
        # better than computeUvsFromXyzs(), but it
        # doesn't seem as stable
        # adjusted_sts = mapper.computeUvsFromAngles()
        if adjusted_sts is None:
            print("adjustSts: parameterization failed, no adjustment made")
            # return inds, pts
            return None
        # print("before", pts[~bounds_flag])
        # print("after", adjusted_sts[~bounds_flag])
        # print("delta", (adjusted_sts-pts)[~bounds_flag])
        # adjusted_sts[floating] = pts[floating]
        '''
        delta = adjusted_sts-pts
        delta = np.sqrt((delta*delta).sum(axis=1))
        delta[bpts] *= -1
        np.set_printoptions(suppress=True)
        # print(delta[np.abs(delta)>1000])
        # print(np.nonzero(np.abs(delta)>1000)[0])
        bigds = np.nonzero(np.abs(delta)>1000)[0]
        print("bigds", bigds)
        print(adjusted_sts[bigds])
        for bigd in bigds.tolist():
            print("  ", bigd)
            print(trgls[(trgls==bigd).any(axis=1)])
        '''
        timer.time("  z")
        return inds, adjusted_sts, len(bpts) > 1
