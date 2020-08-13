import vtk
import numpy as np
from vtk.util import numpy_support
from itertools import combinations
from numpy import linalg as LA


# Write polydata to vtk file
def vtk_writevtk(pd, filename):
    writer = vtk.vtkDataSetWriter()
    writer.SetInputData(pd)
    writer.SetFileName(filename)
    writer.Write()


# get angle between p3 and p4 assuming both triangles share p1 and p2
# https://stackoverflow.com/questions/2142552/calculate-the-angle-between-two-triangles-in-cuda
def get_t_angle(p1, p2, p3, p4):
    d1 = np.cross(p3 - p1, p2 - p1)
    d2 = np.cross(p4 - p1, p2 - p1)

    nd1 = d1 / LA.norm(d1)
    nd2 = d2 / LA.norm(d2)

    return np.arccos(nd1.dot(nd2))


# https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
def rotate_around_axis(n, p, angle):
    return p * np.cos(angle) + (np.cross(n, p)) * np.sin(angle) + n * (n.dot(p)) * (1 - np.cos(angle))


def correct(p1, p2, p3, p4):
    old_angle = get_t_angle(p1, p2, p3, p4)
    angle = np.pi - old_angle
    n = p2 - p1
    n = n / LA.norm(n)
    return rotate_around_axis(n, p4, angle)


# https://stackoverflow.com/questions/28910718/give-3-points-and-a-plot-circle
def define_circle(ps):
    p1 = ps[0]
    p2 = ps[1]
    p3 = ps[2]

    """
    Returns the center and radius of the circle passing the given 3 points.
    In case the 3 points form a line, returns (None, infinity).
    """
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

    if abs(det) < 1.0e-6:
        return (None, np.inf)

    # Center of circle
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    return ((cx, cy), radius)


def rotate_into_xz(p):
    x = p[0]
    y = p[1]
    z = p[2]
    d = np.sqrt(y**2 + z**2)

    t = np.array([
        [1, 0, 0],
        [0, y/d, z/d],
        [0, -z/d, y/d]
    ])

    return t.dot(p)

def remove_sublist_from_list(sub, l):
    return [x for x in l if x not in sub]


class Polydata():
    def __init__(self, pd):
        self.polydata = pd
        self.vertices = self.polydata.GetPoints()
        self.n_order = 2
        self.tolerance = 0.05

    # gets all neighbors of vertex_id
    def get_neighbors_id_h(self, vertex_id):
        point_cells = vtk.vtkIdList()
        self.polydata.GetPointCells(vertex_id, point_cells)

        neighbor_ids = []

        # Find all cells containing vertex_id
        for cell in range(point_cells.GetNumberOfIds()):
            cell_points = vtk.vtkIdList()
            neighbor_cell_id = point_cells.GetId(cell)

            # Get point of cell
            self.polydata.GetCellPoints(neighbor_cell_id, cell_points)

            for neighbor in range(cell_points.GetNumberOfIds()):
                neighbor_point_id = cell_points.GetId(neighbor)

                if neighbor_point_id not in neighbor_ids and neighbor_point_id != vertex_id:
                    neighbor_ids.append(neighbor_point_id)

        return neighbor_ids

    # Get neighbors from n_order surrounding
    def get_neighbors_ids(self, vertex_id):
        neighbors = self.get_neighbors_id_h(vertex_id)

        if self.n_order == 1:
            return neighbors

        new = []
        for _ in range(self.n_order - 1):
            for n in neighbors:
                t = self.get_neighbors_id_h(n)
                new = list(np.unique(new + t))
            # remove duplicates
            new = remove_sublist_from_list(neighbors, new)
            neighbors = new

        # remove own vertex_id out of neighbor list
        if vertex_id in new:
            new.remove(vertex_id)

        return new

    def get_minmax_neighbors(self, vertex_id):
        neighbors = self.get_neighbors_ids(vertex_id)
        distances = [self.get_distance(vertex_id, neighbor_id) for neighbor_id in neighbors]
        return neighbors[np.argmin(distances)], neighbors[np.argmax(distances)]

    # Get distance between v1 and v2
    def get_distance(self, v0_id, v1_id):
        # Get 3d vector from id
        p0 = self.get_point(v0_id)
        p1 = self.get_point(v1_id)

        squaredDistance = vtk.vtkMath().Distance2BetweenPoints(p0, p1)

        return np.sqrt(squaredDistance)

    # Add data array to polydata
    def add_data_array(self, data_array, name):
        vtk_da = numpy_support.numpy_to_vtk(data_array)
        vtk_da.SetName(name)
        self.polydata.GetPointData().AddArray(vtk_da)

    # Calculate mean curvature via Jaap's method
    def get_mean_curvature(self):
        mcs = []

        # for all vertices in mesh
        for vertex_id in range(self.vertices.GetNumberOfPoints()):
            m = np.array(self.get_point(vertex_id))
            n = self.get_mean_normal_vector(vertex_id)
            p = n

            # get all neighbors of m
            neighbors = self.get_neighbors_ids(vertex_id)

            # get all neighbor pairs
            cs = np.array(list(combinations(neighbors, 2)))

            # get top 3 neighbors pairs were points are most diametrically
            angles = []

            for a, b in cs:
                l = np.array(self.get_point(a))
                r = np.array(self.get_point(b))
                angles.append(get_t_angle(m, p, l, r))

            top_three = cs[np.argsort(angles)[-3:][::-1]]

            # draw circle around top 3 (neighbor pairs + m)
            radii = []

            for a, b in top_three:
                l = np.array(self.get_point(a))
                r = np.array(self.get_point(b))

                # get current angle between l and r
                angle = get_t_angle(m, p, l, r)

                # if angle error is below tolerance, correct it
                if abs(np.pi - angle) / np.pi > self.tolerance:
                    r = correct(m, p, l, r)

                # rotate l, r and m into xz plane
                la = rotate_into_xz(l)
                ra = rotate_into_xz(r)
                ma = rotate_into_xz(m)

                # draw circle around these points
                _, r = define_circle([la, ra, ma])
                radii.append(r)

            radii = np.array(radii)

            mean_curvature = 1/2 * (1/np.min(radii) + 1/np.max(radii))
            mcs.append(mean_curvature)

        self.add_data_array(mcs, "Mean_Curvature_Steven")

    # Get normal vector of vertex_id
    def get_normal_vector(self, vertex_id):
        return np.array(self.polydata.GetPointData().GetNormals().GetTuple(vertex_id))

    def get_mean_normal_vector(self, vertex_id):
        neighbors = self.get_neighbors_ids(vertex_id)

        # initialize empty matrix
        all_normal_vectors = np.zeros((3, len(neighbors)))

        # derive normal vector for every neighbor
        # add to all_normal_vectors
        for i in range(len(neighbors)):
            normal_vector = self.get_normal_vector(neighbors[i])
            all_normal_vectors[:, i] = normal_vector

        # get mean in y direction of matrix
        return np.mean(all_normal_vectors, axis=1)

    def get_point(self, vertex_id):
        return self.vertices.GetPoint(vertex_id)


if __name__ == "__main__":
    reader = vtk.vtkPLYReader()
    reader.SetFileName("/home/steven/Documents/school/scriptie/INPUTS/meshes/sphere_kutter.ply")
    reader.Update()

    polydata = reader.GetOutput()

    normals = vtk.vtkPolyDataNormals()
    normals.SplittingOff()
    normals.SetInputData(polydata)
    normals.ComputePointNormalsOn()
    normals.Update()

    polydata = normals.GetOutput()

    pd = Polydata(polydata)
    pd.get_mean_curvature()

    vtk_writevtk(pd.polydata, "test4.vtk")