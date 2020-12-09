import sys
sys.path.insert(1, '../functions')

import vtk_functions
import general

import numpy as np
from numpy import linalg as LA
import vtk
import cv2
import os


def is_upside_down(img):
    height, _ = img.shape
    height_middle = round(height / 2)
    top_part = img[0:height_middle, :]
    bottom_part = img[height_middle:height, :]

    top_nzp = cv2.countNonZero(top_part)
    bottom_nzp = cv2.countNonZero(bottom_part)

    return top_nzp < bottom_nzp


def get_transformation_matrix(x, y, z):
    u = x / LA.norm(x)
    v = y / LA.norm(y)
    w = z / LA.norm(z)

    uvw = np.concatenate((np.concatenate((u, v), axis=1), w), axis=1)
    uvw_i = LA.inv(uvw)

    return uvw_i


def bounds_to_extent(bounds, origin, spacing):
    x1, x2, y1, y2, z1, z2 = bounds

    i_min = int(np.round((x1 - origin[0]) / spacing[0]))
    i_max = int(np.round((x2 - origin[0]) / spacing[0]))
    j_min = int(np.round((y1 - origin[1]) / spacing[1]))
    j_max = int(np.round((y2 - origin[1]) / spacing[1]))
    k_min = int(np.round((z1 - origin[2]) / spacing[2]))
    k_max = int(np.round((z2 - origin[2]) / spacing[2]))

    return i_min, i_max, j_min, j_max, k_min, k_max


def get_points_actor(ps):
    points = vtk.vtkPoints()

    for p in ps:
        points.InsertNextPoint(p)

    pointsPolydata = vtk.vtkPolyData()
    pointsPolydata.SetPoints(points)

    vertexFilter = vtk.vtkVertexGlyphFilter()
    vertexFilter.SetInputData(pointsPolydata)
    vertexFilter.Update()

    polydata2 = vtk.vtkPolyData()
    polydata2.ShallowCopy(vertexFilter.GetOutput())

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata2)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(10)
    return actor


def prepare_slices(imgdata, visualize, padding=1):
    dims = imgdata.GetDimensions()
    print("Original dimensions: %s" % str(dims))

    max_z = 2000
    omega = 1

    if dims[2] > max_z:
        omega = general.trunc(max_z / dims[2], 1)
        imgdata = vtk_functions.resize_imgdata(imgdata, omega)
        imgdata.SetSpacing(1, 1, 1)
        print("Resampled imagedata dimensions (%s): %s" %
              (omega, str(imgdata.GetDimensions())))

    # Create contour (input for OBBtree)
    sran = imgdata.GetScalarRange()
    polydata = vtk.vtkContourFilter()
    polydata.SetInputData(imgdata)
    polydata.SetValue(0, sran[0] + (sran[1] - sran[0]) / 2.0)
    polydata.Update()

    print("Created contour of imagedata...")

    # Generate OBB
    obb = vtk.vtkOBBTree()
    obb.SetDataSet(polydata.GetOutput())
    obb.SetMaxLevel(1)
    obb.BuildLocator()

    obb_corner = [0.0, 0.0, 0.0]
    obb_max = [0.0, 0.0, 0.0]
    obb_mid = [0.0, 0.0, 0.0]
    obb_min = [0.0, 0.0, 0.0]
    obb_size = [0.0, 0.0, 0.0]

    obb.ComputeOBB(polydata.GetOutput(), obb_corner, obb_max, obb_mid,
                   obb_min, obb_size)

    print("Derived OBB from imagedata's contour...")

    # Create t using output of OBB

    obb_corner = np.array(obb_corner).reshape((3, 1))
    obb_max = (np.array(obb_max)).reshape((3, 1))
    obb_mid = (np.array(obb_mid)).reshape((3, 1))
    obb_min = (np.array(obb_min)).reshape((3, 1))

    uvw_i = get_transformation_matrix(obb_mid, obb_min, obb_max)

    t = vtk.vtkMatrix4x4()
    t.DeepCopy((uvw_i[0][0], uvw_i[0][1], uvw_i[0][2], 0,
                uvw_i[1][0], uvw_i[1][1], uvw_i[1][2], 0,
                uvw_i[2][0], uvw_i[2][1], uvw_i[2][2], 0,
                0,          0,          0,          1))

    transform = vtk.vtkTransform()
    transform.SetMatrix(t)
    transform.Inverse()  # inverse t --> inverse resampling
    transform.Update()

    imgdata = vtk_functions.reslice_imgdata(imgdata, transform)
    imgdata.SetSpacing(1, 1, 1)

    print("Transformed imagedata by t...")

    # Get bounds of rotated OBB to crop
    corner = uvw_i.dot(obb_corner)
    ax1 = uvw_i.dot(obb_max + obb_corner)
    ax2 = uvw_i.dot(obb_mid + obb_corner)
    ax3 = uvw_i.dot(obb_min + obb_corner)

    x_range = corner[0][0], np.max(np.array([ax1[0], ax2[0], ax3[0]]))
    y_range = corner[1][0], np.max(np.array([ax1[1], ax2[1], ax3[1]]))
    z_range = corner[2][0], np.max(np.array([ax1[2], ax2[2], ax3[2]]))

    new_bounds = x_range + y_range + z_range

    bte = bounds_to_extent(new_bounds, imgdata.GetOrigin(),
                           imgdata.GetSpacing())
    imgdata = vtk_functions.extract_voi(imgdata, bte[0], bte[1], bte[2],
                                        bte[3], bte[4], bte[5], padding)

    print("Cropped imagedata...")

    if visualize:
        r_sran = imgdata.GetScalarRange()
        rotated_polydata = vtk.vtkContourFilter()
        rotated_polydata.SetInputData(imgdata)
        rotated_polydata.SetValue(0, (r_sran[1] - r_sran[0])/2.0)
        rotated_polydata.Update()

        transform = vtk.vtkTransform()
        transform.RotateZ(180)
        transform.Update()

        tf = vtk.vtkTransformFilter()
        tf.SetTransform(transform)
        tf.SetInputConnection(rotated_polydata.GetOutputPort())
        tf.Update()

        rotated_polydata = tf

        print("\t(Visualize) Created rotated contour...")

        r_obb = vtk.vtkOBBTree()
        r_obb.SetDataSet(rotated_polydata.GetOutput())
        r_obb.SetMaxLevel(5)
        r_obb.BuildLocator()

        r_obb_corner = [0.0, 0.0, 0.0]
        r_obb_max = [0.0, 0.0, 0.0]
        r_obb_mid = [0.0, 0.0, 0.0]
        r_obb_min = [0.0, 0.0, 0.0]
        r_obb_size = [0.0, 0.0, 0.0]

        r_obb.ComputeOBB(rotated_polydata.GetOutput(), r_obb_corner, r_obb_max,
                         r_obb_mid, r_obb_min, r_obb_size)

        print("\t(Visualize) Derived OBB for rotated imagedata...")

        # MAPPERS + ACTORS
        polydata_mapper = vtk.vtkPolyDataMapper()
        polydata_mapper.SetInputConnection(polydata.GetOutputPort())
        polydata_mapper.SetScalarVisibility(False)
        polydata_actor = vtk.vtkActor()
        polydata_actor.GetProperty().SetColor(1, 1, 1,)
        polydata_actor.SetMapper(polydata_mapper)

        r_polydata_mapper = vtk.vtkPolyDataMapper()
        r_polydata_mapper.SetInputConnection(rotated_polydata.GetOutputPort())
        r_polydata_mapper.SetScalarVisibility(False)
        r_polydata_actor = vtk.vtkActor()
        r_polydata_actor.GetProperty().SetColor(1, 1, 1)
        r_polydata_actor.SetMapper(r_polydata_mapper)
        # r_polydata_actor.GetProperty().SetSpecular(0.0)
        r_polydata_actor.GetProperty().SetDiffuse(0.0)


        obb_poly = vtk.vtkPolyData()
        obb.GenerateRepresentation(0, obb_poly)
        obb_mapper = vtk.vtkPolyDataMapper()
        obb_mapper.SetInputData(obb_poly)
        obb_actor = vtk.vtkActor()
        obb_actor.SetMapper(obb_mapper)
        obb_actor.GetProperty().SetColor(1, 1, 1)
        obb_actor.GetProperty().LightingOff()


        obb_actor.GetProperty().SetRepresentationToWireframe()

        r_obb_poly = vtk.vtkPolyData()
        r_obb.GenerateRepresentation(0, r_obb_poly)
        r_obb_mapper = vtk.vtkPolyDataMapper()
        r_obb_mapper.SetInputData(r_obb_poly)
        r_obb_actor = vtk.vtkActor()
        r_obb_actor.SetMapper(r_obb_mapper)
        r_obb_actor.GetProperty().SetColor(1, 1, 1)
        r_obb_actor.GetProperty().LightingOff()
        r_obb_actor.GetProperty().SetRepresentationToWireframe()

        renderer = vtk.vtkRenderer()
        renderer.SetBackground(82/255,87/255,110/255)
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.AddRenderer(renderer)
        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)

        # Coordinate system
        axes = vtk.vtkAxesActor()
        axes.GetXAxisShaftProperty().SetColor(1,1,1)
        widget = vtk.vtkOrientationMarkerWidget()
        widget.SetOutlineColor(0.9300, 0.5700, 0.1300)
        widget.SetOrientationMarker(axes)
        widget.SetInteractor(renderWindowInteractor)
        widget.SetViewport(0.0, 0.0, 0.4, 0.4)
        widget.SetEnabled(1)
        widget.InteractiveOn()

        # renderer.AddActor(polydata_actor)
        # renderer.AddActor(obb_actor)
        renderer.AddActor(r_polydata_actor)
        renderer.AddActor(r_obb_actor)

        renderer.ResetCamera()
        renderWindow.SetSize(600, 600)
        renderWindow.Render()
        renderWindowInteractor.Start()

    np_data = vtk_functions.imgdata_to_arr(imgdata)

    return np_data, omega


def run(input_folder, output_folder, visualize=False):
    imgdata = vtk_functions.folder_to_imgdata(input_folder)

    np_data, omega = prepare_slices(imgdata, visualize)

    if is_upside_down(np_data[:, :, int(np_data.shape[2] / 2)]):
        np_data = np.rot90(np_data, k=2, axes=(0, 1))
        print("Finished flipping...")

    basename = os.path.basename(input_folder)

    if omega < 1:
        general.arr_to_imgseq(np_data, "%s/%s_prepared_%s" % (output_folder,
                                                              basename, omega))
    else:
        general.arr_to_imgseq(np_data, "%s/%s_prepared" % (output_folder,
                                                           basename))


if __name__ == "__main__":
    args = sys.argv[1:]
    input_folder = str(args[0])
    output_folder = args[1]

    visualize = True if len(args) > 2 else False

    run(input_folder, output_folder, visualize=visualize)
