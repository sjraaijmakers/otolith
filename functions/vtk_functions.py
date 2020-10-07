""" wrapper for vtk functions """


import vtk
from vtk.util import numpy_support
import numpy as np
import os
import general
import shutil
import cv2
from scipy.sparse import dok_matrix


""" IMGDATA """


def gauss_imgdata(imgdata, sigma=3):
    gauss = vtk.vtkImageGaussianSmooth()
    gauss.SetDimensionality(3)
    gauss.SetInputData(imgdata)
    gauss.SetStandardDeviations(sigma, sigma, sigma)
    gauss.SetRadiusFactors(sigma, sigma, sigma)
    gauss.Update()
    return gauss.GetOutput()


def resize_imgdata(imgdata, omega):
    resampler = vtk.vtkImageResample()
    resampler.SetInputData(imgdata)
    resampler.SetAxisMagnificationFactor(0, omega)
    resampler.SetAxisMagnificationFactor(1, omega)
    resampler.SetAxisMagnificationFactor(2, omega)
    resampler.SetInterpolationModeToCubic()
    resampler.Update()
    return resampler.GetOutput()


def pad_imgdata(imgdata, padding):
    extent = imgdata.GetExtent()
    pad = vtk.vtkImageConstantPad()
    pad.SetInputData(imgdata)
    pad.SetOutputWholeExtent(extent[0] - padding, extent[1] + padding,
                             extent[2] - padding, extent[3] + padding,
                             extent[4] - padding, extent[5] + padding)
    pad.Update()
    return pad.GetOutput()


def reslice_imgdata(imgdata, transform):
    res = vtk.vtkImageReslice()
    res.SetInputData(imgdata)
    res.SetAutoCropOutput(True)
    res.SetInformationInput(imgdata)
    res.SetResliceTransform(transform)
    res.SetOutputSpacing(1, 1, 1)
    res.SetInterpolationModeToLinear()
    res.Update()
    return res.GetOutput()


def center_imgdata(imgdata):
    center = vtk.vtkImageChangeInformation()
    center.SetInputData(imgdata)
    center.SetCenterImage(1)
    center.Update()
    return center.GetOutput()


# crop according to extent
def extract_voi(imgdata, xi, xf, yi, yf, zi, zf, padding=0):
    voi = vtk.vtkExtractVOI()
    voi.SetVOI(xi - padding, xf + padding, yi - padding, yf + padding,
               zi - padding, zf + padding)
    voi.SetInputData(imgdata)
    voi.SetSampleRate(1, 1, 1)
    voi.Update()
    return voi.GetOutput()


""" POLYDATA """


def read_ply(filename):
    reader = vtk.vtkPLYReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

# Write vtk polydata to .ply file
def write_ply(polydata, name):
    writer = vtk.vtkPLYWriter()
    writer.SetInputData(polydata)
    writer.SetFileName("%s" % name)
    writer.Write()


def write_vtk(polydata, filename):
    writer = vtk.vtkDataSetWriter()
    writer.SetInputData(polydata)
    writer.SetFileName(filename)
    writer.Write()


def add_arr_to_pd(polydata, data_array, name):
    vtk_da = numpy_support.numpy_to_vtk(data_array)
    vtk_da.SetName(name)
    polydata.GetPointData().AddArray(vtk_da)


def normals(polydata):
    normals = vtk.vtkPolyDataNormals()
    normals.SplittingOff()
    normals.SetInputData(polydata)
    normals.ComputePointNormalsOn()
    normals.Update()
    return normals.GetOutput()


def mean_curvature(polydata):
    curvatureFilter = vtk.vtkCurvatures()
    curvatureFilter.SetCurvatureTypeToMean()
    curvatureFilter.SetInputData(polydata)
    curvatureFilter.Update()
    return curvatureFilter.GetOutput()


def get_volume(polydata):
    mp = vtk.vtkMassProperties()
    mp.SetInputData(polydata)
    mp.Update()
    return mp.GetVolume()


def get_surface_area(polydata):
    mp = vtk.vtkMassProperties()
    mp.SetInputData(polydata)
    mp.Update()
    return mp.GetSurfaceArea()


def get_connect_vertices(polydata, vertex_id):
    connectedVertices = []

    cellIdList = vtk.vtkIdList()
    polydata.GetPointCells(vertex_id, cellIdList)

    for i in range(cellIdList.GetNumberOfIds()):
        pointIdList = vtk.vtkIdList()
        polydata.GetCellPoints(cellIdList.GetId(i), pointIdList)

        if pointIdList.GetId(0) != vertex_id:
            connectedVertices.append(pointIdList.GetId(0))
        else:
            connectedVertices.append(pointIdList.GetId(1))

    return connectedVertices


def get_distance_matrix(polydata):
    N = polydata.GetPoints().GetNumberOfPoints()
    d = dok_matrix((N, N))

    for i in range(N):
        neighbors = get_connect_vertices(polydata, i)
        d[i, neighbors] = 1
    return d


""" CONVERSIONS """


def imgdata_to_arr(imgdata):
    dims = imgdata.GetDimensions()
    vtk_data = imgdata.GetPointData().GetScalars()
    numpy_data = numpy_support.vtk_to_numpy(vtk_data)

    # why reshape + transpose?
    numpy_data = numpy_data.reshape(dims[2], dims[1], dims[0])
    numpy_data = numpy_data.transpose(1, 2, 0)
    # numpy_data = np.flip(numpy_data, axis=1)
    return numpy_data


def arr_to_imgseq(arr, folder, img_format="tif", verbose=False):
    if not os.path.exists(folder):
        os.makedirs(folder)
    # delete folder if it already exists
    else:
        shutil.rmtree(folder)
        os.makedirs(folder)

    for i in range(arr.shape[2]):
        name = folder + "/img_" + str(f'{i:04}' + "." + img_format)
        if verbose:
            print("Writing " + name + "...")
        cv2.imwrite(name, arr[:, :, i])

    print("Finished writing %s slices to %s" % (arr.shape[2], folder))


def arr_to_imgdata(array):
    array = array.transpose(2, 0, 1)
    array = np.flip(array, axis=1)

    data_array = numpy_support.numpy_to_vtk(array.ravel(), deep=True,
                                            array_type=vtk.VTK_DOUBLE)

    imgdata = vtk.vtkImageData()
    imgdata.SetDimensions(array.shape[2], array.shape[1], array.shape[0])
    imgdata.SetSpacing([1, 1, 1])
    imgdata.SetOrigin([0, 0, 0])
    imgdata.GetPointData().SetScalars(data_array)
    return imgdata


def imgdata_to_pd(imgdata):
    sran = imgdata.GetScalarRange()
    contourFilter = vtk.vtkContourFilter()
    contourFilter.SetInputData(imgdata)
    contourFilter.SetValue(0, (sran[1] - sran[0]) / 2)
    contourFilter.Update()
    return contourFilter.GetOutput()


def arr_to_pd(arr):
    imgdata = arr_to_imgdata(arr)
    return imgdata_to_pd(imgdata)


# Read folder to imagedata
def folder_to_imgdata(input_folder, verbose=False):
    z = general.count_files_in_folder(input_folder, "tif")

    if verbose:
        print("%s slices for %s" % (z, input_folder))

    p = len(str(z))

    file_prefix = input_folder + "/" + general.get_file_prefix(input_folder, z)

    if verbose:
        print("file prefix: %s" % file_prefix)

    reader = vtk.vtkTIFFReader()

    reader.SetFilePrefix(file_prefix)
    reader.SetFilePattern("%s%0" + str(p) + "d.tif")
    reader.SetFileDimensionality(3)
    reader.SetOrientationType(2)
    reader.Update()

    dims = reader.GetOutput().GetDimensions()
    reader.SetDataExtent((0, dims[0], 0, dims[1], 0, z - 1))
    reader.Update()

    return reader.GetOutput()


def folder_to_curv_arr(input_folder, sigma=3):
    imgdata = folder_to_imgdata(input_folder)

    if sigma > 0:
        padding = sigma * 2
        imgdata = pad_imgdata(imgdata, padding)
        imgdata = gauss_imgdata(imgdata, sigma=sigma)

    polydata = imgdata_to_pd(imgdata)

    return pd_to_curv_arr(polydata)


def pd_to_curv_arr(polydata):
    polydata = normals(polydata)

    mcFilter = vtk.vtkCurvatures()
    mcFilter.SetCurvatureTypeToMean()
    mcFilter.SetInputData(polydata)
    mcFilter.Update()

    polydata = mcFilter.GetOutput()

    curv_arr =  numpy_support.vtk_to_numpy(polydata.GetPointData().GetArray("Mean_Curvature"))

    return curv_arr