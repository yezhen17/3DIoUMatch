# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Helper functions for calculating 2D and 3D bounding box IoU.

Collected and written by Charles R. Qi
Modified by Yezhen Cong, 2020
"""
from __future__ import print_function

import numpy as np
import numpy.testing as npt
import torch
from scipy.spatial import ConvexHull
try:
    from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
except:
    raise ImportError("please first install pcdet according to README.md")


def polygon_clip(subjectPolygon, clipPolygon):
   """ Clip a polygon with another polygon.

   Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python

   Args:
     subjectPolygon: a list of (x,y) 2d points, any polygon.
     clipPolygon: a list of (x,y) 2d points, has to be *convex*
   Note:
     **points have to be counter-clockwise ordered**

   Return:
     a list of (x,y) vertex point for the intersection polygon.
   """
   def inside(p):
      return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
 
   def computeIntersection():
      dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
      dp = [ s[0] - e[0], s[1] - e[1] ]
      n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
      n2 = s[0] * e[1] - s[1] * e[0] 
      n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
      return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
 
   outputList = subjectPolygon
   cp1 = clipPolygon[-1]
 
   for clipVertex in clipPolygon:
      cp2 = clipVertex
      inputList = outputList
      outputList = []
      s = inputList[-1]
 
      for subjectVertex in inputList:
         e = subjectVertex
         if inside(e):
            if not inside(s):
               outputList.append(computeIntersection())
            outputList.append(e)
         elif inside(s):
            outputList.append(computeIntersection())
         s = e
      cp1 = cp2
      if len(outputList) == 0:
          return None
   return(outputList)


def poly_area(x,y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0  


def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c


def box3d_vol_batch(corners):
    ''' corners: (n,8,3) no assumption on axis direction '''
    l = np.sqrt(np.linalg.norm(corners[:, 1, :] - corners[:, 2, :], axis=1))
    w = np.sqrt(np.linalg.norm(corners[:, 0, :] - corners[:, 1, :], axis=1))
    h = np.sqrt(np.linalg.norm(corners[:, 0, :] - corners[:, 4, :], axis=1))
    return l*w*h


def is_clockwise(p):
    x = p[:,0]
    y = p[:,1]
    return np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)) > 0


def box3d_iou(corners1, corners2):
    ''' Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU

    todo (rqi): add more description on corner points' orders.
    '''
    # corner points are in counter clockwise order
    rect1 = [(corners1[i,0], corners1[i,2]) for i in range(3,-1,-1)]
    rect2 = [(corners2[i,0], corners2[i,2]) for i in range(3,-1,-1)] 
    area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
    area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area/(area1+area2-inter_area)
    ymax = min(corners1[0,1], corners2[0,1])
    ymin = max(corners1[4,1], corners2[4,1])
    inter_vol = inter_area * max(0.0, ymax-ymin)
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d


def box3d_iou_batch_gpu(bbox1, bbox2):
    """ CUDA version compute 3D non-axis-aligned bounding box IoU.
    Input:
        bbox1: torch tensor, (x,y,z,dx,dy,dz,heading) this heading = -heading of Votenet
        bbox2: torch tensor, (x,y,z,dx,dy,dz,heading)
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU
    """
    return boxes_iou3d_gpu(bbox1, bbox2)


def boxes3d_iou_batch(batch_corners1, batch_corners2):
    '''
    Input:
        batch_corners1: numpy array (n,8,3), assume up direction is negative Y
        batch_corners2: numpy array (m,8,3), assume up direction is negative Y
    Output:
        batch_iou: 3D bounding box IoU (n,m)
    '''
    n = batch_corners1.shape[0]
    m = batch_corners2.shape[1] #suppose m < n

    vol_batch1 = box3d_vol_batch(batch_corners1) #n
    vol_batch2 = box3d_vol_batch(batch_corners2) #m

    y_max_batch1 = batch_corners1[:,0,1] #n
    y_min_batch1 = batch_corners1[:,4,1] #n
    y_max_batch2 = batch_corners2[:,0,1] #m
    y_min_batch2 = batch_corners2[:,4,1] #m

    batch_iou = np.zeros((n,m), dtype=np.float32)
    for i in range(m):
        rect2 = [(batch_corners2[i,k,0], batch_corners2[i,k,2]) for k in range(3,-1,-1)] #[((0,0),(0,2)),..., ((3,0),(3,2))]
        vol2 = vol_batch2[i]

        y_max = np.where(y_max_batch1-y_max_batch2[i]<0, y_max_batch1, y_max_batch2[i]) #n
        y_min = np.where(y_min_batch1-y_min_batch2[i]>0, y_min_batch1, y_min_batch2[i]) #n
        inter_y = np.where(y_max-y_min < 0., 0., y_max-y_min) #n
        inter_area = np.zeros((n), dtype=np.float32) #n
        for j in range(n):
            rect1 = [(batch_corners1[j,k,0], batch_corners1[j,k,2]) for k in range(3,-1,-1)]
            inter_area[j] = convex_hull_intersection(rect1, rect2)[1]
        inter_vol = inter_y * inter_area #n
        batch_iou[:,i] = inter_vol/(vol_batch1+vol2-inter_vol)

    return batch_iou


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two 2D bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def box2d_iou(box1, box2):
    ''' Compute 2D bounding box IoU.

    Input:
        box1: tuple of (xmin,ymin,xmax,ymax)
        box2: tuple of (xmin,ymin,xmax,ymax)
    Output:
        iou: 2D IoU scalar
    '''
    return get_iou({'x1':box1[0], 'y1':box1[1], 'x2':box1[2], 'y2':box1[3]}, \
        {'x1':box2[0], 'y1':box2[1], 'x2':box2[2], 'y2':box2[3]})


# -----------------------------------------------------------
# Convert from box parameters to 
# -----------------------------------------------------------
def rotz(t):
    """Rotation about the Z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    R = np.array([[c, -s, 0],
                  [s, c, 0],
                  [0, 0, 1]])
    return R


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                    [0,  1,  0],
                    [-s, 0,  c]])


def roty_batch(t):
    """Rotation about the y-axis.
    t: (x1,x2,...xn)
    return: (x1,x2,...,xn,3,3)
    """
    input_shape = t.shape
    output = np.zeros(tuple(list(input_shape)+[3,3]))
    c = np.cos(t)
    s = np.sin(t)
    output[...,0,0] = c
    output[...,0,2] = s
    output[...,1,1] = 1
    output[...,2,0] = -s
    output[...,2,2] = c
    return output


def rot_gpu(t):
    """Rotation about the upright axis.
    t: (x1,x2,...xn)
    return: (x1,x2,...,xn,3,3)
    """
    input_shape = t.shape
    output = torch.zeros(tuple(list(input_shape)+[3, 3])).cuda()
    c = torch.cos(t)
    s = torch.sin(t)
    output[..., 0, 0] = c
    output[..., 0, 1] = s
    output[..., 2, 2] = 1
    output[..., 1, 0] = -s
    output[..., 1, 1] = c
    return output


def get_3d_box_depth(box_size, heading_angle, center):
    ''' box_size is array(l,w,h), heading_angle is radius clockwise from pos x axis, center is xyz of box center
        output (8,3) array for 3D box cornders
        Similar to utils/compute_orientation_3d
            6 -------- 5
           /|         /|
          7 -------- 4 .
          | |        | |
          . 2 -------- 1
          |/         |/
          3 -------- 0

    '''
    R = rotz(heading_angle)
    l,w,h = box_size
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2];
    y_corners = [w/2,-w/2,-w/2, w/2, w/2,-w/2,-w/2, w/2];
    z_corners = [h/2, h/2, h/2, h/2,-h/2,-h/2,-h/2,-h/2];
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + center[0]
    corners_3d[1, :] = corners_3d[1, :] + center[1]
    corners_3d[2, :] = corners_3d[2, :] + center[2]
    corners_3d = np.transpose(corners_3d)
    return corners_3d


def get_3d_box(box_size, heading_angle, center):
    ''' box_size is array(l,w,h), heading_angle is radius clockwise from pos x axis, center is xyz of box center
        output (8,3) array for 3D box cornders
        Similar to utils/compute_orientation_3d
            6 -------- 5
           /|         /|
          7 -------- 4 .
          | |        | |
          . 2 -------- 1
          |/         |/
          3 -------- 0

    '''
    R = roty(heading_angle)
    l,w,h = box_size
    x_corners = [l/2, l/2,-l/2,-l/2, l/2, l/2,-l/2,-l/2];
    y_corners = [h/2, h/2, h/2, h/2,-h/2,-h/2,-h/2,-h/2];
    z_corners = [w/2,-w/2,-w/2, w/2, w/2,-w/2,-w/2, w/2];
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d[0,:] = corners_3d[0,:] + center[0];
    corners_3d[1,:] = corners_3d[1,:] + center[1];
    corners_3d[2,:] = corners_3d[2,:] + center[2];
    corners_3d = np.transpose(corners_3d)
    return corners_3d


def get_3d_box_batch(box_size, heading_angle, center):
    ''' box_size: [x1,x2,...,xn,3]
        heading_angle: [x1,x2,...,xn]
        center: [x1,x2,...,xn,3]
    Return:
        [x1,x3,...,xn,8,3]
    '''
    input_shape = heading_angle.shape
    R = roty_batch(heading_angle)
    l = np.expand_dims(box_size[...,0], -1) # [x1,...,xn,1]
    w = np.expand_dims(box_size[...,1], -1)
    h = np.expand_dims(box_size[...,2], -1)
    corners_3d = np.zeros(tuple(list(input_shape)+[8,3]))
    corners_3d[...,:,0] = np.concatenate((l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2), -1)
    corners_3d[...,:,1] = np.concatenate((h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2), -1)
    corners_3d[...,:,2] = np.concatenate((w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2), -1)
    tlist = [i for i in range(len(input_shape))]
    tlist += [len(input_shape)+1, len(input_shape)]
    corners_3d = np.matmul(corners_3d, np.transpose(R, tuple(tlist)))
    corners_3d += np.expand_dims(center, -2)
    return corners_3d


def box3d_iou_batch(corners1, corners2):
    ''' Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU

    TODO for simplicity this is aligned IoU
    '''
    # corner points are in counter clockwise order
    max_a = np.max(corners1, axis=2)
    max_b = np.max(corners2, axis=2)
    min_a = np.min(corners1, axis=2)
    min_b = np.min(corners2, axis=2)

    max_min = np.stack([min_a, min_b], axis=2).max(2)
    min_max = np.stack([max_a, max_b], axis=2).min(2)
    vol_a = (max_a - min_a).prod(axis=2)
    vol_b = (max_b - min_b).prod(axis=2)
    diff = np.max(np.stack([min_max - max_min, np.zeros(min_max.shape)], axis=2), axis=2)
    intersection = diff.prod(axis=2)
    # print(intersection.shape)
    union = vol_a + vol_b - intersection
    return 1.0 * intersection / (union + 1e-8)


def box3d_iou_gpu_axis_aligned(corners1, corners2):
    ''' Compute 3D bounding box IoU. Torch code that can be back propagated

    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU

    TODO for simplicity this is aligned IoU
    '''
    # corner points are in counter clockwise order
    max_a = torch.max(corners1, dim=2)[0]
    max_b = torch.max(corners2, dim=2)[0]
    min_a = torch.min(corners1, dim=2)[0]
    min_b = torch.min(corners2, dim=2)[0]

    max_min = torch.stack([min_a, min_b], dim=2).max(2)[0]
    min_max = torch.stack([max_a, max_b], dim=2).min(2)[0]
    vol_a = (max_a - min_a).prod(dim=2)
    vol_b = (max_b - min_b).prod(dim=2)
    diff = torch.max(torch.stack([min_max - max_min, torch.zeros(min_max.shape).cuda()], dim=2), dim=2)[0]
    intersection = diff.prod(dim=2)
    # print(intersection.shape)
    union = vol_a + vol_b - intersection
    return 1.0 * intersection / (union + 1e-8)


def corners3d_to_parameter(corners_3d):
    '''represent 8 corner points as box parameters (center, size, heading_angle)
            6 -------- 5
           /|         /|
          7 -------- 4 .
          | |        | |
          . 2 -------- 1
          |/         |/
          3 -------- 0
    :param corners_3d: (8,3) array for 3D box corners in upright camera frame
    :return: parameterized box: (7,) numpy array in depth frame
         box_size is array(l,w,h), heading_angle is radius clockwise from pos x axis, center is xyz of box center
    '''
    center = 0.5 * (corners_3d.max(0) + corners_3d.min(0))
    x_side = corners_3d[0]-corners_3d[3]
    y_side = corners_3d[0]-corners_3d[4]
    z_side = corners_3d[0]-corners_3d[1]
    l = np.linalg.norm(x_side)
    w = np.linalg.norm(z_side)
    h = np.linalg.norm(y_side)
    heading_angle = np.arccos(x_side[0]/l)

    box_params = np.zeros((7))
    box_params[0:3] = np.array([center[0], center[2], -center[1]])
    box_params[3:6] = np.array([l,w,h])
    box_params[6] = heading_angle

    return box_params


def check_valid_corners3d(corners_3d):
    ''' check if a predicted corners3d is a valid cube
            6 -------- 5
           /|         /|
          7 -------- 4 .
          | |        | |
          . 2 -------- 1
          |/         |/
          3 -------- 0
    :param corners_3d: (8,3) array for 3D box corners in upright camera frame
    :return: bool
    '''
    x_lines = np.vstack((corners_3d[0]-corners_3d[3], corners_3d[1]-corners_3d[2],
                         corners_3d[4]-corners_3d[7], corners_3d[5]-corners_3d[6])) #(4,3)
    y_lines = np.vstack((corners_3d[0]-corners_3d[4], corners_3d[1]-corners_3d[5],
                         corners_3d[3]-corners_3d[7], corners_3d[2]-corners_3d[6])) #(4,3)
    z_lines = np.vstack((corners_3d[0]-corners_3d[1], corners_3d[4]-corners_3d[5],
                         corners_3d[3]-corners_3d[2], corners_3d[7]-corners_3d[6])) #(4,3)

    x_lines_length = np.linalg.norm(x_lines, axis=1)
    y_lines_length = np.linalg.norm(y_lines, axis=1)
    z_lines_length = np.linalg.norm(z_lines, axis=1)

    lines_length= np.vstack((x_lines_length, y_lines_length, z_lines_length)).transpose() #(4,3)

    try:
        npt.assert_almost_equal(lines_length[0], np.array([0.,0.,0.]), decimal=1)
        print('\t\tWarning: this length of box [{0}] is almost zero...'.format(lines_length[0]))
        return False
    except:
        pass

    try:
        npt.assert_almost_equal(lines_length[0], lines_length[1], decimal=2)
        npt.assert_almost_equal(lines_length[0], lines_length[2], decimal=2)
        npt.assert_almost_equal(lines_length[0], lines_length[3], decimal=2)
        npt.assert_almost_equal(lines_length[1], lines_length[2], decimal=2)
        npt.assert_almost_equal(lines_length[1], lines_length[3], decimal=2)
        npt.assert_almost_equal(lines_length[2], lines_length[3], decimal=2)

        # in case it is a parallelepiped, check the three lines in one corner perpendicular with each other
        npt.assert_almost_equal((corners_3d[0] - corners_3d[4]) @ (corners_3d[0] - corners_3d[1]), 0, decimal=1)
        npt.assert_almost_equal((corners_3d[0] - corners_3d[4]) @ (corners_3d[0] - corners_3d[3]), 0, decimal=1)
        npt.assert_almost_equal((corners_3d[0] - corners_3d[1]) @ (corners_3d[0] - corners_3d[3]), 0, decimal=1)

        return True

    except:
        print('\t\tWarning: this box is not a valid cube...')
        return False


if __name__=='__main__':

    # Function for polygon ploting
    import matplotlib
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    import matplotlib.pyplot as plt
    def plot_polys(plist,scale=500.0):
        fig, ax = plt.subplots()
        patches = []
        for p in plist:
            poly = Polygon(np.array(p)/scale, True)
            patches.append(poly)

    pc = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.5)
    colors = 100*np.random.rand(len(patches))
    pc.set_array(np.array(colors))
    ax.add_collection(pc)
    plt.show()
 
    # Demo on ConvexHull
    points = np.random.rand(30, 2)   # 30 random points in 2-D
    hull = ConvexHull(points)
    # **In 2D "volume" is is area, "area" is perimeter
    print(('Hull area: ', hull.volume))
    for simplex in hull.simplices:
        print(simplex)

    # Demo on convex hull overlaps
    sub_poly = [(0,0),(300,0),(300,300),(0,300)]
    clip_poly = [(150,150),(300,300),(150,450),(0,300)] 
    inter_poly = polygon_clip(sub_poly, clip_poly)
    print(poly_area(np.array(inter_poly)[:,0], np.array(inter_poly)[:,1]))
    
    # Test convex hull interaction function
    rect1 = [(50,0),(50,300),(300,300),(300,0)]
    rect2 = [(150,150),(300,300),(150,450),(0,300)] 
    plot_polys([rect1, rect2])
    inter, area = convex_hull_intersection(rect1, rect2)
    print((inter, area))
    if inter is not None:
        print(poly_area(np.array(inter)[:,0], np.array(inter)[:,1]))
    
    print('------------------')
    rect1 = [(0.30026005199835404, 8.9408694211408424), \
             (-1.1571105364358421, 9.4686676477075533), \
             (0.1777082043006144, 13.154404877812102), \
             (1.6350787927348105, 12.626606651245391)]
    rect1 = [rect1[0], rect1[3], rect1[2], rect1[1]]
    rect2 = [(0.23908745901608636, 8.8551095691132886), \
             (-1.2771419487733995, 9.4269062966181956), \
             (0.13138836963152717, 13.161896351296868), \
             (1.647617777421013, 12.590099623791961)]
    rect2 = [rect2[0], rect2[3], rect2[2], rect2[1]]
    plot_polys([rect1, rect2])
    inter, area = convex_hull_intersection(rect1, rect2)
    print((inter, area))
