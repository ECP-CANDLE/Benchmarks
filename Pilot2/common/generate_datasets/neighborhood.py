from scipy import spatial
import numpy

class Neighborhood(object):

    # constructor
    def __init__(self, points, pdist, box):

        self.points = points
        self.pdist = pdist

        self.sidx = numpy.argsort(self.pdist, axis=1)

        print 'Initializing Neighborhood with', self.points.shape, 'points and', pdist.shape, 'distances'
        # compute triangulation
        # self.delaunay = spatial.Delaunay(self.points[:,0:2])
        # self.simplices = self.delaunay.simplices



    def mesh_nbrs(self, p):

        (indices, indptr) = self.delaunay.vertex_neighbor_vertices
        return indptr[ indices[p]:indices[p+1] ]


    # orient the points ccw around the origin
    def ccw(self, points, origin):

        # return polar coordinats (transform the angle to [0, 2pi])
        def cart2pol(p):
            return ( numpy.sqrt(p[0]**2 + p[1]**2),
                    (numpy.arctan2(p[1], p[0]) + 2*numpy.pi) % (2*numpy.pi))

        def pol2cart(p):
            return (p[0] * numpy.cos(p[1]), p[0] * numpy.sin(p[1]) )

        opoints = self.points[points,0:2] - self.points[origin,0:2]
        opoints = numpy.apply_along_axis(cart2pol, 1, opoints)

        # sort on angle
        return points[numpy.argsort(opoints[:,1])]


    # orient points ccw in rings around p
    def orient_rings(self, points, pidx):

        # need to store all points in first k-rings
        knbrs = [pidx]
        kidx = [0,1]

        # points in k-ring that are also in input points
        kpoints = [pidx]
        kpidx = [0,1]

        i = 0
        while len(kpoints) < len(points):

            # find the points in i'th ring
            inbrs = []
            for v in knbrs[ kidx[i]:kidx[i+1] ]:

                vnbrs = self.mesh_nbrs(v)
                vnbrs = [x for x in vnbrs if x not in knbrs and x not in inbrs]
                inbrs.extend( vnbrs )

            knbrs.extend( inbrs )
            kidx.append( len(knbrs) )

            # consider only the points in i-ring that belong to input points
            inbrs = [x for x in inbrs if x in points]
            inbrs = self.ccw( numpy.array(inbrs), pidx)

            kpoints.extend(inbrs)
            kpidx.append(len(kpoints))

            i = i+1

        setdiff = set(points) ^ set(kpoints)
        if len(setdiff) > 0:
            print points
            print kpoints
            raise ValueError('sorting failed!')

        return (numpy.array(kpidx), numpy.array(kpoints))

     # get all neighbors in k-ring (sort them wrt rings)
    def get_nbrs_kring(self, pidx, k):

        knbrs = [pidx]
        kidx = [0,1]

        for i in xrange(k):

            inbrs = []
            for v in knbrs[kidx[i]:kidx[i+1]]:

                vnbrs = self.mesh_nbrs(v)
                vnbrs = [x for x in vnbrs if x not in knbrs and x not in inbrs]
                inbrs.extend( vnbrs )

            inbrs = self.ccw( numpy.array(inbrs), pidx)
            knbrs.extend( inbrs )
            kidx.append( len(knbrs) )

        return (numpy.array(kidx), numpy.array(knbrs))

    # get k nearest neighbors (sort them ccw wrt triangulation)
    def get_nbrs_k(self, pidx, k, ccwsort=True):

        # distance-based k-nearest neighbors
        #dnbrs = numpy.argsort(self.pdist[pidx,:])[0:k+1]
        dnbrs = self.sidx[pidx, 0:k+1]

        if not ccwsort:
            return ([0,1,len(dnbrs)], dnbrs)

        p = self.orient_rings(dnbrs, pidx)

        return p


    # get neighbors < r distance away (sort them ccw wrt triangulation)
    def get_nbrs_r(self, pidx, r, ccwsort=True):

        # distance-based neighbors
        dnbrs = numpy.where(self.pdist[pidx,:] < r)[0]

        if not ccwsort:
            return ([0,1,len(dnbrs)], dnbrs)

        return self.orient_rings(dnbrs, pidx)
