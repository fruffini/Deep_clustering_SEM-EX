import numpy as np
import random

from easydict import EasyDict


def create_spheres_limited_sovrapposition(opt, number_of_spheres):
    spheres_dictionary = EasyDict()
    for i, j_center in enumerate(range(1, number_of_spheres+1)):
        radius1, center1 = generation_sphere_random(radius_range=opt.radius_range, space_dim=opt.space_dim, original_space=opt.image_dim) # Generated sphere in the space dimension
        if i != 0:
            for sphere in spheres_dictionary.values():
                boolean = True
                center2 = sphere[1]
                radius2 = sphere[0]
                while boolean:
                    total_dist = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2 + (center1[2] - center2[2]) ** 2)
                    if radius1 + radius2 < total_dist or total_dist > (radius1 + radius2)//2:
                        boolean = False
                    elif total_dist < (radius1 + radius2)//2:
                        radius1, center1 = generation_sphere_random(radius_range=opt.radius_range, space_dim=opt.space_dim, original_space=opt.image_dim)


        spheres_dictionary['sphere_{0}'.format(j_center)] = tuple((radius1, center1))
    return spheres_dictionary

def generation_sphere_random(radius_range, space_dim, original_space):
    """
    This function generates two variables that define a sphere inside the specified space (space_dim,space_dim,space_dim)
    :param radius_range: (list), range of possible radius values.
    :param space_dim:  (int), dimension of the space
    :return: radius, centers coordinates
    """
    (original_space - space_dim)//2
    return random.randint(radius_range[0], radius_range[1]) , [random.randint((original_space - space_dim)//2, original_space - (original_space - space_dim)//2) for el in range(0, 3)]


def sphere(shape, radius, position):
    """Generate an n-dimensional spherical mask."""
    # assume shape and position have the same length and contain ints
    # the units are pixels / voxels (px for short)
    # radius is a int or float in px
    assert len(position) == len(shape)
    n = len(shape)
    semisizes = (radius,) * len(shape)

    # genereate the grid for the support points
    # centered at the position indicated by position
    grid = [slice(-x0, dim - x0) for x0, dim in zip(position, shape)]
    position = np.ogrid[grid]
    # calculate the distance of all points from `position` center
    # scaled by the radius
    arr = np.zeros(shape, dtype=float)
    for x_i, semisize in zip(position, semisizes):
        # this can be generalized for exponent != 2
        # in which case `(x_i / semisize)`
        # would become `np.abs(x_i / semisize)`
        arr += (x_i / semisize) ** 2

    # the inner part of the sphere will have distance below or equal to 1
    arr[arr <= 1.0] = 1
    arr[arr > 1.0] = 0
    return arr








