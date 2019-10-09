import numpy as np

from copy import copy
import os

from deeplabcut.utils.auxiliaryfunctions_3d import get_coord, unit_vector
from deeplabcut.pose_estimation_3d import triangulate_raw_2d_camera_coords


# TODO: Create a jupyter notebook with an implementation of this workflow

def get_basis(config_dict, image_dict, camera_pairs, clockwise=True, user_defined_axis=['x', 'z'], pixel_tolerance=2):
    """
    Parameters
    ----------
    config_dict : dict
        Dictionary where the key is name of the camera, and the value is the full path of the config.yaml file
        as a string.
    image_dict : dict
        Dictionary where the key is name of the camera, and the value is the full path to the image
        of the referance points taken with the camera
    camera_pairs : list-like
        List of cameras that are pairs. Pairs usually have their own deeplabcut 3D project
    user_defined_axis : dictionary with max length 2
        Dictionary where the key are the axis to be selected by the user, choose between 'x-axis', 'y-axis', 'z-axis'; 1-, 2-, 3-dimension.
        There must be two axis names, the third axis is defined using the cross product.
        
        The values of the keys is the direction of the defined axis. This is important for the calculation of the third axis (first right hand rule)
    pixel_tolerance : integer, float; default 2
        Defines the floor-tolerance for setting basis vector after coordinate components being set to 0. After being moved to the origin
    """ 
    VALID_AXIS_NAMES = ['x', 'y', 'z', 1, 2, 3]
    coord_labels = ['origin']
    for axis_name in user_defined_axis:
        if axis_name in VALID_AXIS_NAMES:
            coord_labels.append(axis_name)
        else:
            msg = "Invalid axis name, {}. Valid names: {}".format(axis_name, VALID_AXIS_NAMES)
            raise ValueError(msg)
            
    cam_coords = dict.fromkeys(image_dict)
    for cam_name, cam_img in image_dict.items():
        cam_coords[cam_name] = []
        max_i = len(coord_labels) - 1
        for coord_name in coord_labels:
            title = "%s: left mouse click add point" % coord_name
            cam_coords[cam_name].append(get_coord(cam_img, n=1, title=title))

    basis_of_pairs = {}
    basis_dict = dict.fromkeys(coord_labels)
    for cam1_name, cam2_name in camera_pairs:
        coords = triangulate_raw_2d_camera_coords(
            config_dict[(cam1_name, cam2_name)], cam1_coords=cam_coords[cam1_name], cam2_coords=cam_coords[cam2_name], coord_keys=coord_labels
        )

        basis_of_pairs[(cam1_name, cam2_name)] = copy(basis_dict)
        basis_of_pairs[(cam1_name, cam2_name)]['origin'] = np.array(coords['origin'])
        for user_axis in user_defined_axis:
            basis_of_pairs[(cam1_name, cam2_name)][user_axis] = unit_vector(
                remove_close_zero(coords[user_axis] - coords['origin'], tol=pixel_tolerance)
            )

    return basis_of_pairs

def get_cage_basis():
    CAMERAS = {
            'NorthWest': ['x', 'z', 'close'], 'NorthEast': ['x', 'z', 'close'],
            'EastNorth': ['y', 'z', 'far'], 'EastSouth': ['y', 'z', 'far'],
            'SouthEast': ['x', 'z', 'close'], 'SouthWest': ['x', 'z', 'close'],
            'WestSouth': ['y', 'z', 'far'], 'WestNorth': ['y', 'z', 'far']
        }
        camera_units = dict.fromkeys(CAMERAS)
        for camera, axis in CAMERAS.items():
            directions = []
            for direction in ('+', '-'):
                directions.append(get_coord(cam_img, n=1, title=direction+axis[0]))
            z = get_coord(cam_img, n=1, title='z-axis')
            if axis[2] == 'close':
                origin = direction[0] + (direction[1] - direction[0]) / 2
                camera_units[camera] = {
                    axis[0]: (direction[0] - origin) / unit_vector(direction[0] - origin)
                    'z': (origin - z) / unit_vector(origin - z)
                }
                camera_units[camera]['y'] = - np.cross(
                    camera_units[camera][axis[0]],
                    camera_units[camera]['z']
                    )
            else:
                origin = direction[1] + (direction[0] - direction[1]) / 2
                camera_units[camera] = {
                    axis[0]: (origin - direction[1]) / unit_vector(origin - direction[1]))
                    'z': (origin - z) / unit_vector(origin - z)
                }
                camera_units[camera]['x'] = np.cross(
                    camera_units[camera][axis[0]],
                    camera_units[camera]['z']
                    )
        return basis_of_pairs

def change_of_basis(column_matrix, x=None, y=None, z=None, origin=np.array((0, 0, 0)) ):
    """
    This function changes the basis of deeplabcut-triangulated that are 3D.
    
    Note: Two out of three axis should be defined. The third axis is the normal vector

    Parameters
    ----------
    column_matrix : numpy.array
        A 3D array that holds the coordinates that will have their basis changed
    x : numpy.array-like; default None
        A 3D row vector, that represents the basis of the new x axis
    y : numpy.array-like; default None
        A 3D row vector, that represents the basis of the new y axis
    z : numpy.array-like; default None
        A 3D row vector, that represents the basis of the new z axis
    origin : numpy.array-like; default np.array((0, 0, 0))
        A 3D row vector, that represents the origin

    Example
    -------
    With random basis vectors:
    >>> deeplabcut.change_of_basis(column_matrix, x=(2.4, 0, 0), y=(0,  5.3, 0), z=None, origin=np.array((0, 0, 0))).

    """
    basis_dict = {'x': x, 'y': y, 'z': z}
    known_basis = []
    last_axis = None
    for axis, basis in basis_dict.items():
        if basis is None:
            if last_axis is not None:
                msg = "Two out of three axis should be defined. The third axis is the normal vector"
                raise AttributeError(msg)
            last_axis = axis
        else:
            basis = np.asarray(basis)
            known_basis.append(axis)

    if basis_dict[known_basis[0]].shape != basis_dict[known_basis[1]].shape:
        msg = "The basis vectors (%s and %s) need to have the same shape" % known_basis
        raise AttributeError(msg)
    if basis_dict[known_basis[0]].shape != (3,):
        msg = "The basis vectors (%s and %s) can only have the shape (3,)\n" \
              "In other words, they have to be row vectors" % known_basis
        raise AttributeError(msg)

    # Make sure each basis vector are unit vectors
    basis_dict[known_basis[0]] = unit_vector(basis_dict[known_basis[0]])
    basis_dict[known_basis[1]] = unit_vector(basis_dict[known_basis[1]])

    basis_dict[last_axis] = np.cross(basis_dict[known_basis[0]], basis_dict[known_basis[1]])
    linear_transformer = np.array((tuple(basis_dict.values())))

    # Change basis, and return result
    return np.apply_along_axis(
        lambda v: np.dot(linear_transformer, v - np.asarray(origin)),
        1, column_matrix
    )
