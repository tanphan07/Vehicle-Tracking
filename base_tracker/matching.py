import cv2
import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist
import base_tracker.kalman_filter as kalman_filter
from shapely.geometry import Polygon
from rtree import index

def merge_matches(m1, m2, shape):
    O, P, Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix(
        (np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix(
        (np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


# def ious(atlbrs, btlbrs):
#     """
#     Compute cost based on IoU
#     :type atlbrs: list[tlbr] | np.ndarray
#     :type atlbrs: list[tlbr] | np.ndarray

#     :rtype ious np.ndarray
#     """
#     ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
#     if ious.size == 0:
#         return ious

#     ious = bbox_ious(
#         np.ascontiguousarray(atlbrs, dtype=np.float),
#         np.ascontiguousarray(btlbrs, dtype=np.float)
#     )

#     return ious

# iou for segment
def ious_segment(mask1, mask2):
    """
    Compute cost based on IoU
    :type atlbrs: list[mask] | np.ndarray
    :type atlbrs: list[mask] | np.ndarray

    :rtype ious np.ndarray: 2D array of ious: len(mask1) x len(mask2)
    """
    # calculate intersection
    smooth = 1e-6
    ious = np.zeros((len(mask1), len(mask2)), dtype=np.float32)
    if ious.size == 0:
        return ious
    
    mask1 = np.ascontiguousarray(mask1)
    mask2 = np.ascontiguousarray(mask2)

    intersection = np.sum(np.logical_and(mask1[:, None, :, :], mask2[None, :, :, :]), axis=(2, 3))
    union = np.sum(np.logical_or(mask1[:, None, :, :], mask2[None, :, :, :]), axis=(2, 3))
    ious = (intersection + smooth) / (union + smooth)
    ious[np.isnan(ious)] = 0.0

    return ious

def ious_distance_segment(atracks, btracks):
    """
    Compute cost based on IoU segment
    :type atracks: list[Track]
    :type btracks: list[Track]

    :rtype cost_matrix np.ndarray
    """
    
    cost_matrix = np.zeros((len(atracks), len(btracks)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix

    amask = [track.mask for track in atracks]
    bmask = [track.mask for track in btracks]
    ious = ious_segment(amask, bmask)
    cost_matrix = 1 - ious

    return cost_matrix

def build_index(polygons):
    p_index = index.Index()
    for i, polygon in enumerate(polygons):
        p_index.insert(i, polygon.bounds)
    return p_index

def compute_pairwise_intersection(polygons1, polygons2):
    intersection_matrix = np.zeros((len(polygons1), len(polygons2)))

    index2 = build_index(polygons2)

    for i, polygon1 in enumerate(polygons1):
        potential_candidates = list(index2.intersection(polygon1.bounds))
        intersection_areas = []
        for j in potential_candidates:
            intersection = polygon1.intersection(polygons2[j])
            if intersection.is_valid:
                intersection_areas.append(intersection.area)
            else:
                intersection_areas.append(0.0)
        intersection_matrix[i, potential_candidates] = intersection_areas

    return intersection_matrix

def compute_pairwise_union(polygons1, polygons2):
    union_matrix = np.zeros((len(polygons1), len(polygons2)))

    index2 = build_index(polygons2)

    for i, polygon1 in enumerate(polygons1):
        potential_candidates = list(index2.intersection(polygon1.bounds))
        union_areas = []
        for j in potential_candidates:
            union = polygon1.union(polygons2[j])
            if union.is_valid:
                union_areas.append(union.area)
            else:
                union_areas.append(0.0)
        union_matrix[i, potential_candidates] = union_areas

    return union_matrix

def ious_base_contour(contour1, contour2):
    """
    Compute cost based on IoU
    :type contour1: list[contour] | np.ndarray
    :type contour2: list[contour] | np.ndarray

    :rtype ious np.ndarray
    """
    # calculate intersection
    smooth = 1e-6
    ious = np.zeros((len(contour1), len(contour2)), dtype=np.float32)
    if ious.size == 0:
        return ious
    

    contour1 = [Polygon(contour.reshape(-1, 2)) for contour in contour1]
    contour2 = [Polygon(contour.reshape(-1 ,2)) for contour in contour2]

    intersection = compute_pairwise_intersection(contour1, contour2)
    union = compute_pairwise_union(contour1, contour2)
    ious = (intersection) / (union + smooth)
    ious[np.isnan(ious)] = 0.0
    return ious
    
def ious_contour(atracks, btracks):

    ious = np.zeros((len(atracks), len(btracks)), dtype=np.float32)
    if ious.size == 0:
        return ious

    acontour = [track.contour for track in atracks]
    bcontour = [track.contour for track in btracks]
    ious = ious_base_contour(acontour, bcontour)

    cost_matrix = 1 - ious
    return cost_matrix

def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    # calculate intersection
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if ious.size == 0:
        return ious
    
    atlbrs = np.ascontiguousarray(atlbrs)
    btlbrs = np.ascontiguousarray(btlbrs)

    left = np.maximum(atlbrs[:, 0][:, np.newaxis], btlbrs[:, 0])
    top = np.maximum(atlbrs[:, 1][:, np.newaxis], btlbrs[:, 1])
    right = np.minimum(atlbrs[:, 2][:, np.newaxis], btlbrs[:, 2])
    bottom = np.minimum(atlbrs[:, 3][:, np.newaxis], btlbrs[:, 3])
    intersection = np.maximum(right - left, 0) * np.maximum(bottom - top, 0)

    # calculate union
    area_a = (atlbrs[:, 2] - atlbrs[:, 0]) * (atlbrs[:, 3] - atlbrs[:, 1])
    area_b = (btlbrs[:, 2] - btlbrs[:, 0]) * (btlbrs[:, 3] - btlbrs[:, 1])
    union = np.expand_dims(area_a, axis=1) + \
        np.expand_dims(area_b, axis=0) - intersection

    # calculate IoU
    ious = intersection / union
    ious[np.isnan(ious)] = 0.0
    return ious


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    
    # print(atlbrs)
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def v_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks]
        btlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray(
        [track.curr_feat for track in detections], dtype=np.float)
    # for i, track in enumerate(tracks):
    #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray(
        [track.smooth_feat for track in tracks], dtype=np.float)
    cost_matrix = np.maximum(0.0, cdist(
        track_features, det_features, metric))  # Nomalized features
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + \
            (1 - lambda_) * gating_distance
    return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(
        cost_matrix.shape[0], axis=0)
    #fuse_sim = fuse_sim * (1 + det_scores) / 2
    fuse_cost = 1 - fuse_sim
    return fuse_cost

def fuse_iou_segment(cost_matrix, tracks, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_seg = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = iou_seg * (1 + iou_sim) / 2
    fuse_cost = 1 - fuse_sim
    return fuse_cost

def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(
        cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost
