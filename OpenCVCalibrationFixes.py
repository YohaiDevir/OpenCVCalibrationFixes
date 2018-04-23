import cv2
import numpy as np
import matplotlib.pyplot as plt
# noinspection PyUnresolvedReferences
from matplotlib.cm import jet
import os
from os import path as osp
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

__author__ = 'Yohai Devir'

MAX_NEWTON_ITERATIONS = 40
N_BISECTION_ITERATIONS = 20

INITIAL_STEP_SIZE_PIX = 250
MIN_STEP_SIZE_PIX = 0.1

COS_TH_SIMILARITY = 0.98


class NewtonRaphsonUndistort:
    def __init__(self):
        pass

    # region Internals - Python implementations of OpenCV functions
    @staticmethod
    def icv_get_rectangles(camera_matrix, dist_coeffs, new_camera_matrix, img_size):
        """
        Python implementation of OpenCV's internal function icvGetRectangles with a bugfix.

        In general, it finds the mapping of an distorted image of img_size to the undisrted image space.

        :param camera_matrix: distorted image camera matrix
        :param dist_coeffs: distortion parameters
        :param new_camera_matrix: undistorted image's camera matrix, or None if it is not known
        :param img_size: following openCV conventions, img_size is X/Y
        :return: outer - bounding rectangle of all distorted image pixels,
                 inner - maximal rectengle that contain only pixels from the distorted image (without any padded pixels
        """

        n = 9
        pts = np.zeros((n * n, 2))
        pt_idx = 0
        for y in range(n):
            for x in range(n):
                pts[pt_idx, 0] = float(x) * (img_size[0] - 1) / (n - 1)  # following openCV conventions, img_size is X/Y
                pts[pt_idx, 1] = float(y) * (img_size[1] - 1) / (n - 1)
                pt_idx += 1

        res_pts, estimation_errors = \
            NewtonRaphsonUndistort.cv_undistort_points(pts, camera_matrix, dist_coeffs, new_camera_matrix)

        float_max = float(1e10)

        i_x0 = -float_max
        i_x1 = float_max
        i_y0 = -float_max
        i_y1 = float_max
        o_x0 = float_max
        o_x1 = -float_max
        o_y0 = float_max
        o_y1 = -float_max

        #  find the inscribed rectangle.
        # the code will likely not work with extreme rotation matrices (R) (>45%)

        pt_idx = 0
        for y in range(n):
            for x in range(n):
                p = res_pts[pt_idx]
                pt_idx += 1
                o_x0 = min(o_x0, p[0])
                o_x1 = max(o_x1, p[0])
                o_y0 = min(o_y0, p[1])
                o_y1 = max(o_y1, p[1])

                if x == 0:
                    i_x0 = max(i_x0, p[0])
                if x == n - 1:
                    i_x1 = min(i_x1, p[0])
                if y == 0:
                    i_y0 = max(i_y0, p[1])
                if y == n - 1:
                    i_y1 = min(i_y1, p[1])

        inner = {'x': i_x0, 'y': i_y0, 'width': i_x1 - i_x0, 'height': i_y1 - i_y0}
        outer = {'x': o_x0, 'y': o_y0, 'width': o_x1 - o_x0, 'height': o_y1 - o_y0}

        return inner, outer

    @staticmethod
    def cv_undistort_points(pts_distorted_pix, camera_matrix, dist_coeffs, new_camera_matrix):
        """
        Python implementation of OpenCV's cvUndistortPoints with a bugfix.
        cv_undistort_points takes a list of pixel location in a distorted image and returns their corresponding
        locations in an undistorted image

        For details: https://docs.opencv.org/3.1.0/da/d54/group__imgproc__transform.html


        :param pts_distorted_pix:
        :param camera_matrix:
        :param dist_coeffs:
        :type new_camera_matrix: np.ndarray | NoneType
        :return:
        """

        if new_camera_matrix is None:
            new_camera_matrix = np.eye(3)

        dist_coeffs = dist_coeffs.ravel()

        assert len(dist_coeffs) <= 5

        pts_undistorted_pix = np.zeros_like(pts_distorted_pix)

        dist_coeffs = np.hstack((dist_coeffs, np.zeros(14 - len(dist_coeffs))))
        k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, tilt_param0, tilt_param1 = dist_coeffs

        if tilt_param0 != 0 or tilt_param1 != 0:
            raise NotImplementedError("computeTiltProjectionMatrix was not implemented here in python")

        fx = camera_matrix[0][0]
        fy = camera_matrix[1][1]
        ifx = 1. / fx
        ify = 1. / fy
        cx = camera_matrix[0][2]
        cy = camera_matrix[1][2]

        pixel_to_mm = 0.5 * (fx + fy)
        initial_step_size_mm = INITIAL_STEP_SIZE_PIX / pixel_to_mm
        min_step_size_mm = MIN_STEP_SIZE_PIX / pixel_to_mm

        n = len(pts_distorted_pix)
        estimation_errors = np.ones(n) * -1
        for i in range(n):
            x_dist_pix = pts_distorted_pix[i, 0]
            y_dist_pix = pts_distorted_pix[i, 1]

            x_dist_mm = (x_dist_pix - cx) * ifx
            y_dist_mm = (y_dist_pix - cy) * ify
            loc_dist_target = np.array([x_dist_mm, y_dist_mm])

            loc_orig, final_error = NewtonRaphsonUndistort.undistort_single_pixel(
                loc_dist_target, dist_coeffs, initial_step_size_mm=initial_step_size_mm,
                min_step_size_mm=min_step_size_mm)
            estimation_errors[i] = final_error

            x_mm, y_mm = loc_orig

            xx_pix = new_camera_matrix[0, 0] * x_mm + new_camera_matrix[0, 1] * y_mm + new_camera_matrix[0, 2]
            yy_pix = new_camera_matrix[1, 0] * x_mm + new_camera_matrix[1, 1] * y_mm + new_camera_matrix[1, 2]
            ww_pix = 1. / (new_camera_matrix[2, 0] * x_mm + new_camera_matrix[2, 1] * y_mm + new_camera_matrix[2, 2])
            x_final = xx_pix * ww_pix
            y_final = yy_pix * ww_pix

            pts_undistorted_pix[i][0] = float(x_final)
            pts_undistorted_pix[i][1] = float(y_final)

        return pts_undistorted_pix, estimation_errors

    # endregion

    # region Internals - better implementation of single pixel undistortion
    @staticmethod
    def _distort_pixel_and_calc_error(loc_orig, loc_dist_target, k1, k2, k3, p1, p2):
        x_orig, y_orig = loc_orig[0], loc_orig[1]
        r2 = x_orig * x_orig + y_orig * y_orig
        r4 = r2 ** 2
        r6 = r2 ** 3
        a1 = 2 * x_orig * y_orig
        a2 = r2 + 2 * x_orig * x_orig
        a3 = r2 + 2 * y_orig * y_orig
        cdist = 1 + k1 * r2 + k2 * r4 + k3 * r6
        delta_x = p1 * a1 + p2 * a2
        delta_y = p1 * a3 + p2 * a1

        x_dist = x_orig * cdist + delta_x
        y_dist = y_orig * cdist + delta_y
        loc_distorted = np.array([x_dist, y_dist])

        error_xy = loc_distorted - loc_dist_target
        return loc_distorted, error_xy

    @staticmethod
    def _error_jacobian(loc_orig, k1, k2, k3, p1, p2):
        """
        Calculate the Jacobian of distorted location of loc_orig minus desired distorted location.

        :param loc_orig:
        :param k1:
        :param k2:
        :param k3:
        :param p1:
        :param p2:
        :return:
        """
        x_orig, y_orig = loc_orig[0], loc_orig[1]
        r2 = x_orig * x_orig + y_orig * y_orig
        r4 = r2 ** 2
        r6 = r2 ** 3

        cdist = 1 + k1 * r2 + k2 * r4 + k3 * r6

        common = (k1 + 2 * k2 * r2 + 3 * k3 * r4)
        d_error_x_dx = (cdist + 2 * p1 * y_orig + 6 * p2 * x_orig + 2 * x_orig * x_orig * common)
        d_error_y_dy = (cdist + 6 * p1 * y_orig + 2 * p2 * x_orig + 2 * y_orig * y_orig * common)
        d_error_x_dy = 2 * (p1 * x_orig + p2 * y_orig + x_orig * y_orig * common)
        d_error_y_dx = 2 * (p1 * x_orig + p2 * y_orig + y_orig * x_orig * common)

        jacobian = np.array([[d_error_x_dx, d_error_x_dy], [d_error_y_dx, d_error_y_dy]])
        return jacobian

    @staticmethod
    def _visualize_errors(k1, k2, k3, p1, p2, loc_dist_target):
        x_dist_target, y_dist_target = loc_dist_target[0], loc_dist_target[1]

        ex_x = [-1.0, 1.0]
        ex_y = [-1.0, 1.0]
        xx, yy = np.meshgrid(np.linspace(ex_x[0], ex_x[1], 1000), np.linspace(ex_y[0], ex_y[1], 1000))
        r2 = xx * xx + yy * yy
        r4 = r2 * r2
        r6 = r4 * r2
        a1 = 2 * xx * yy
        a2 = r2 + 2 * xx * xx
        a3 = r2 + 2 * yy * yy
        cdist = 1 + k1 * r2 + k2 * r4 + k3 * r6
        xd0 = xx * cdist + p1 * a1 + p2 * a2
        yd0 = yy * cdist + p1 * a3 + p2 * a1
        error = ((xd0 - x_dist_target) ** 2 + (yd0 - y_dist_target) ** 2) ** 0.5
        plt.subplot(221)
        plt.imshow(xd0 - x_dist_target, extent=[ex_x[0], ex_x[1], ex_y[1], ex_y[0]])
        plt.colorbar()
        plt.title("X error")
        plt.subplot(222)
        plt.imshow(yd0 - y_dist_target, extent=[ex_x[0], ex_x[1], ex_y[1], ex_y[0]])
        plt.colorbar()
        plt.title("Y error")
        plt.subplot(223)
        plt.imshow(error, extent=[ex_x[0], ex_x[1], ex_y[1], ex_y[0]])
        plt.colorbar()
        plt.title("compund error")

    # endregion

    @staticmethod
    def undistort_single_pixel(loc_dist_target, dist_coeffs, initial_step_size_mm=0.05, min_step_size_mm=1e-8):
        """
        Given a pixel location in a distorted image, find its location in the undistorted image.

        As the distortion function is hard to invert, this implementation iteratively searches for a location in the
        original image that is mapped by the distortion function to the desired location.

        OpenCV implementation (in cvUndistortPoints) seems to assume that such a location exists and that the distortion
        function is monotonous.
        Both assumptions may not hold as the distortion parameters calculation does not force monotonicity.
        In this implementation, I perform Newton Raphson iterations with binary search for a decrease in the error.

        :param loc_dist_target: pixel location in the distorted image
        :param dist_coeffs: distortion paramets as in openCV
        :param initial_step_size_mm:
        :param min_step_size_mm:
        :return:
        """

        dist_coeffs = np.hstack((dist_coeffs, np.zeros(14 - len(dist_coeffs))))
        k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, tilt_param0, tilt_param1 = dist_coeffs

        assert not np.any([k4, k5, k6, s1, s2, s3, s4, tilt_param0, tilt_param1]), \
            "NewtonRaphsonUndistort was given distortion params of more than 5 variables, which is unsuipported"

        # noinspection PyUnreachableCode
        # debug = True
        debug = False
        if debug:
            NewtonRaphsonUndistort._visualize_errors(k1, k2, k3, p1, p2, loc_dist_target)

        loc_orig = loc_dist_target  # initial guess is that there is no distortion
        step_size = initial_step_size_mm
        final_error = np.inf
        for ii in range(MAX_NEWTON_ITERATIONS):
            if debug:
                for sp in range(1, 4):
                    plt.subplot(2, 2, sp)
                    plt.plot(loc_orig[0], loc_orig[1], 'mx', markersize=7, markeredgewidth=2)

            # Performing NewtonRaphson iterations on f(x,y) = distort(x,y) - target_location(x,y)
            # Single variable explanation: https://en.wikipedia.org/wiki/Newton%27s_method#Description
            # Multivariate version explanation: http://fourier.eng.hmc.edu/e161/lectures/ica/node13.html
            loc_dist, err_vec = \
                NewtonRaphsonUndistort._distort_pixel_and_calc_error(loc_orig, loc_dist_target, k1, k2, k3, p1, p2)
            l2_error_before = np.linalg.norm(err_vec)

            jacobian = NewtonRaphsonUndistort._error_jacobian(loc_orig, k1, k2, k3, p1, p2)

            current_movement = -1 * np.linalg.inv(jacobian).dot(err_vec.T)
            current_movement_size = np.linalg.norm(current_movement)

            # when the derivatives near the solution are small, a single step may take us far away from the solution.
            # In order to avoid this, I enforce that the error must decrease in every step.
            should_break = False
            next_loc_orig = l2_error_after = None
            for step_size_iter in range(N_BISECTION_ITERATIONS):
                if step_size_iter == N_BISECTION_ITERATIONS - 1:
                    should_break = True
                    break
                next_loc_orig = (loc_orig.T + step_size * current_movement / current_movement_size).T
                _, error_xy_after = NewtonRaphsonUndistort._distort_pixel_and_calc_error(
                    next_loc_orig, loc_dist_target, k1, k2, k3, p1, p2)

                l2_error_after = np.linalg.norm(error_xy_after)
                if l2_error_after < l2_error_before:
                    final_error = l2_error_after
                    break
                step_size *= 0.5
                if debug:
                    print "decreasing step size"
            if should_break:
                break

            assert next_loc_orig is not None and l2_error_after is not None
            loc_orig = next_loc_orig

            if final_error < min_step_size_mm:
                # in this case we stop due to convergence
                final_error = 0
                break

            if step_size < min_step_size_mm:
                break

            if debug:
                print("iteration {:02d}, common: {:.4f}-->{:.4f} | step: {:.4f}".format(
                    ii, l2_error_before, l2_error_after, step_size))
        return loc_orig, final_error

    @staticmethod
    def get_optimal_new_camera_matrix(camera_matrix, dist_coeffs, img_size, alpha):
        """
        Python implementation of OpenCV's cvGetOptimalNewCameraMatrix with a bugfix.

        For details: docs.opencv.org/3.0-beta/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html

        :param camera_matrix:
        :param dist_coeffs:
        :param img_size:  following openCV conventions, img_size is X/Y
        :param alpha:
        :return:
        """

        # Get inscribed and circumscribed rectangles in normalized
        # (independent of camera matrix) coordinates
        inner, outer = NewtonRaphsonUndistort.icv_get_rectangles(camera_matrix, dist_coeffs, None, img_size)

        # Projection mapping inner rectangle to viewport
        fx0 = (img_size[0]) / inner['width']  # following openCV conventions, img_size is X/Y
        fy0 = (img_size[1]) / inner['height']
        cx0 = -fx0 * inner['x']
        cy0 = -fy0 * inner['y']

        # Projection mapping outer rectangle to viewport
        fx1 = (img_size[0]) / outer['width']  # following openCV conventions, img_size is X/Y
        fy1 = (img_size[1]) / outer['height']
        cx1 = -fx1 * outer['x']
        cy1 = -fy1 * outer['y']

        # Interpolate between the two optimal projections
        new_camera_matrix = np.zeros((3, 3))
        new_camera_matrix[0][0] = fx0 * (1 - alpha) + fx1 * alpha
        new_camera_matrix[1][1] = fy0 * (1 - alpha) + fy1 * alpha
        new_camera_matrix[0][2] = cx0 * (1 - alpha) + cx1 * alpha
        new_camera_matrix[1][2] = cy0 * (1 - alpha) + cy1 * alpha
        new_camera_matrix[2][2] = 1

        # # Commented out since not required
        # inner2, _ = icv_get_rectangles(camera_matrix, dist_coeffs, new_camera_matrix, img_size)
        # x0, y0, x1, y1 = inner2['x'], inner2['y'], inner2['x'] + inner2['width'], inner2['y'] + inner2['height']
        # x0 = max(0, min(img_size[1], x0))
        # x1 = max(0, min(img_size[1], x1))
        # y0 = max(0, min(img_size[0], y0))
        # y1 = max(0, min(img_size[0], y1))
        #
        # valid_pixels_roi = np.array([x0, y0, x1, y1])

        return new_camera_matrix, img_size


class CornerMapper:
    def __init__(self):
        pass

    # region Internals - ordering unordered set of detected corners
    @staticmethod
    def _map_non_internal_corners(c2, det_idx_to_corner, grid_size_xy):
        # map nn-internal corners to checkrboard by
        n_pts = c2.shape[0]
        corner_to_pixel = {corner: c2[det_idx] for det_idx, corner in det_idx_to_corner.items()}
        corner_to_width = {}
        corner_to_height = {}
        min_x = min_y = 99999
        max_x = max_y = -99999
        for (x, y) in corner_to_pixel.keys():
            if (x + 1, y) in corner_to_pixel and (x - 1, y) in corner_to_pixel:
                corner_to_width[(x, y)] = \
                    np.linalg.norm(corner_to_pixel[(x + 1, y)] - corner_to_pixel[(x - 1, y)]) * 0.5
            if (x, y + 1) in corner_to_pixel and (x, y - 1) in corner_to_pixel:
                corner_to_height[(x, y)] = \
                    np.linalg.norm(corner_to_pixel[(x, y + 1)] - corner_to_pixel[(x, y - 1)]) * 0.5
            min_x = min(x, min_x)
            min_y = min(y, min_y)
            max_x = max(x, max_x)
            max_y = max(y, max_y)
        xx, yy = np.meshgrid(np.arange(min_x, max_x + 1), np.arange(min_y, max_y + 1))
        missing_corners = np.vstack([(x, y) for x, y in zip(xx.ravel(), yy.ravel()) if (x, y) not in corner_to_pixel])
        poly = PolynomialFeatures(degree=3)
        missing_corners_w_poly = poly.fit_transform(missing_corners)  # type: np.array
        ##
        all_regressions = {}
        for key, data_dict in [('loc', corner_to_pixel),
                               ('width', corner_to_width),
                               ('height', corner_to_height)]:
            corner_indices = np.vstack(data_dict.keys())
            required_value = np.vstack(data_dict.values())

            corner_indices_w_poly = poly.fit_transform(corner_indices)

            regressor = LinearRegression()
            regressor.fit(corner_indices_w_poly, required_value)

            # noinspection PyArgumentList
            all_regressions[key] = regressor.predict(missing_corners_w_poly)
        ##
        predicted_locs = all_regressions['loc']
        predicted_width = all_regressions['width']
        predicted_height = all_regressions['height']
        unassigned_dets = sorted(list(set(np.arange(n_pts)) - set(det_idx_to_corner.keys())))
        dists = np.sum((predicted_locs.reshape(-1, 1, 2) - c2[unassigned_dets].reshape(1, -1, 2)) ** 2, axis=2) ** 0.5
        missed_corners_indices, detect_corner_indices = \
            np.where(dists < np.minimum(predicted_width, predicted_height).reshape(-1, 1) * 0.2)
        for missed_corner_idx, detect_corner_subidx in zip(missed_corners_indices, detect_corner_indices):
            detect_corner_idx = unassigned_dets[detect_corner_subidx]
            corner = tuple(missing_corners[missed_corner_idx])
            det_idx_to_corner[detect_corner_idx] = corner
            corner_to_pixel[corner] = c2[detect_corner_idx]
        ##
        should_transpose = np.argmax(grid_size_xy) != np.argmax([max_x - min_x + 1, max_y - min_y + 1])
        final_corners = {}
        for corner, pixel_location in corner_to_pixel.items():
            final_key = (corner[0] - min_x, corner[1] - min_y)
            if should_transpose:
                final_key = (final_key[1], final_key[0])
            final_corners[final_key] = pixel_location
        return final_corners

    @staticmethod
    def _map_internal_corners(c2, is_internal, neighbors_lists, with_vis=False):
        n_pts = c2.shape[0]
        det_idx_to_corner = {}
        n_supports = np.zeros(n_pts, np.int)
        is_ordered = np.zeros(n_pts, np.bool)
        first_point = np.where(is_internal)[0][0]  # type: int
        det_idx_to_corner[first_point] = (0, 0)
        is_ordered[first_point] = True
        stack = [first_point]
        if with_vis:
            plt.text(c2[first_point, 0], c2[first_point, 1],
                     "  {:d}\n  ({:d},{:d})".format(first_point, 0, 0), fontsize=12, color='m')
        offsets_4nn = [(0, -1),  # if neighbor is above
                       (+1, 0),  # if neighbor is to the right
                       (0, +1),  # if neighbor is below
                       (-1, 0)]  # if neighbor is to the left
        # first iteration, find the underlying grid for internal corners only
        for iter_guard in range(n_pts * 4):
            if len(stack) == 0:
                break

            curr_idx = stack.pop()
            assert curr_idx in det_idx_to_corner
            assert is_ordered[curr_idx]

            for n_idx, neighbor in enumerate(neighbors_lists[curr_idx]):
                if neighbor == -1:
                    continue

                # update the location of the corner
                if neighbor in det_idx_to_corner:
                    assert det_idx_to_corner[neighbor] == (det_idx_to_corner[curr_idx][0] + offsets_4nn[n_idx][0],
                                                           det_idx_to_corner[curr_idx][1] + offsets_4nn[n_idx][1])

                    if with_vis:
                        plt.plot(c2[[curr_idx, neighbor], 0], c2[[curr_idx, neighbor], 1], 'g', linewidth=3)
                    was_neighbor_processed = True
                else:
                    assert neighbor not in det_idx_to_corner
                    det_idx_to_corner[neighbor] = (det_idx_to_corner[curr_idx][0] + offsets_4nn[n_idx][0],
                                                   det_idx_to_corner[curr_idx][1] + offsets_4nn[n_idx][1])
                    if with_vis:
                        plt.text(c2[neighbor, 0], c2[neighbor, 1],
                                 "  {:d}\n  ({:d},{:d})".format(neighbor, int(det_idx_to_corner[neighbor][0]),
                                                                int(det_idx_to_corner[neighbor][1])),
                                 fontsize=12, color='m')

                    was_neighbor_processed = False
                n_supports[neighbor] += 1

                if with_vis:
                    plt.plot(c2[[curr_idx, neighbor], 0], c2[[curr_idx, neighbor], 1], 'g', linewidth=3)

                if not is_internal[neighbor]:
                    continue

                if not is_ordered[neighbor]:
                    assert not was_neighbor_processed

                    curr_idx_in_neighbor = np.where(neighbors_lists[neighbor] == curr_idx)[0]
                    assert len(curr_idx_in_neighbor) == 1
                    curr_idx_in_neighbor = curr_idx_in_neighbor[0]
                    neighbors_lists[neighbor] = \
                        [neighbors_lists[neighbor][(x + n_idx + 2 - curr_idx_in_neighbor) % 4] for x in range(4)]
                    assert neighbors_lists[neighbor].index(curr_idx) == (n_idx + 2) % 4
                    is_ordered[neighbor] = True

                if not was_neighbor_processed:
                    stack.append(neighbor)
        return det_idx_to_corner

    @staticmethod
    def _find_4nn_per_corner(c2):
        dists_mat = np.sum((c2.reshape(-1, 1, 2) - c2.reshape(1, -1, 2)) ** 2, axis=2) ** 0.5
        nearests_neighbors = np.argsort(dists_mat, axis=1)
        n_pts = c2.shape[0]
        is_internal = np.zeros(n_pts, np.bool)
        neighbors_lists = [[]] * n_pts
        for curr_idx in range(n_pts):
            point = c2[curr_idx]

            neigh_indices = nearests_neighbors[curr_idx, 1:5]
            dist_to_4nn = dists_mat[curr_idx, neigh_indices]

            offsets_4nn = c2[neigh_indices, :] - point[np.newaxis, :]
            assert min(dist_to_4nn) > 0
            directions = offsets_4nn / dist_to_4nn[:, np.newaxis]
            dot_prods = np.sum(directions.reshape(4, 1, 2) * directions.reshape(1, 4, 2), axis=2)
            is_internal[curr_idx] = np.all(np.min(dot_prods, axis=1) < -COS_TH_SIMILARITY)

            # arrange neighbors from the one above and clockwise
            top_most = np.argmax(directions.dot(np.array([[0], [-1]])))
            right_most = np.argmax(directions.dot(np.array([[1], [0]])))
            bottom_most = np.argmax(directions.dot(np.array([[0], [1]])))
            left_most = np.argmax(directions.dot(np.array([[-1], [0]])))
            if len({top_most, bottom_most, right_most, left_most}) != 4:
                assert not is_internal[curr_idx]
                neighbors_lists[curr_idx] = neigh_indices
            else:
                neighbors_lists[curr_idx] = neigh_indices[[top_most, right_most, bottom_most, left_most]]
        return is_internal, neighbors_lists

    @staticmethod
    def _visualize_cb(c2, img, is_internal, n_points):
        plt.clf()
        img_with_dets = img.copy()
        cv2.drawChessboardCorners(img_with_dets, n_points, c2.reshape(-1, 1, 2), False)
        plt.imshow(img_with_dets[:, :, ::-1])
        plt.plot(c2[np.where(is_internal), 0].ravel(), c2[np.where(is_internal), 1].ravel(), 'og', markersize=10)
        for curr_idx in range(c2.shape[0]):
            plt.text(c2[curr_idx, 0], c2[curr_idx, 1], "  {:d}\n".format(curr_idx), fontsize=12, color='r')
        plt.axis((0, img_with_dets.shape[1], img_with_dets.shape[0], 0))

    @staticmethod
    def draw_final_cb_corners(corners, objp, img, n_points):
        plt.clf()
        plt.imshow(img)
        n_detections = len(corners)
        for idx in range(n_detections):
            key = objp[idx, :2]
            next_h = np.where((objp[:, 0] == key[0] + 1) & (objp[:, 1] == key[1]))[0]
            next_v = np.where((objp[:, 0] == key[0]) & (objp[:, 1] == key[1] + 1))[0]
            if len(next_h) > 0:
                assert len(next_h) == 1
                plt.plot((corners[idx, 0, 0], corners[next_h, 0, 0]),
                         (corners[idx, 0, 1], corners[next_h, 0, 1]), 'g', linewidth=1)
            if len(next_v) > 0:
                assert len(next_v) == 1
                plt.plot((corners[idx, 0, 0], corners[next_v, 0, 0]),
                         (corners[idx, 0, 1], corners[next_v, 0, 1]), 'g', linewidth=1)
        for idx in range(n_detections):
            key = objp[idx, :2]
            color = jet(int((key[1] * n_points[0] + key[0]) / np.prod(n_points) * 255))
            plt.plot(corners[idx, 0, 0], corners[idx, 0, 1], 'o', markerfacecolor=color, markeredgecolor=color,
                     markersize='10')
            if idx < len(corners) - 1:
                plt.plot(corners[idx: idx + 2, 0, 0], corners[idx: idx + 2, 0, 1], color=color, linewidth=2)

    # endregion
    @staticmethod
    def find_partial_corner_mapping(corners, img, grid_size_xy, with_vis=False):
        c2 = corners.copy()
        c2 = c2.reshape(-1, 2)

        is_internal, neighbors_lists = CornerMapper._find_4nn_per_corner(c2)
        if with_vis:
            CornerMapper._visualize_cb(c2, img, is_internal, grid_size_xy)
        det_idx_to_corner = CornerMapper._map_internal_corners(c2, is_internal, neighbors_lists, with_vis)
        if with_vis:
            plt.ion()
            plt.show()
        final_corners = CornerMapper._map_non_internal_corners(c2, det_idx_to_corner, grid_size_xy)

        sorted_keys = sorted(final_corners.keys(), key=lambda x: (x[1], x[0]))
        partial_objp = np.array([[key[0], key[1], 0] for key in sorted_keys], np.float32)
        partial_corners = np.array([final_corners[key] for key in sorted_keys]).reshape(-1, 1, 2)
        return partial_corners, partial_objp


def find_corners_in_images_set(images, n_points, with_partial):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((np.prod(n_points), 3), np.float32)
    objp[:, :2] = np.mgrid[0:n_points[0], 0:n_points[1]].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    im_shape = None
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        im_shape = gray.shape[::-1]
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, n_points, None,
                                                 flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FILTER_QUADS)

        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # If found, add object points, image points (after refining them)
        if ret:
            imgpoints.append(corners)
            objpoints.append(objp)
            # # Draw and display the corners
            # plt.figure()
            # CornerMapper.draw_final_cb_corners(corners, objp, img, n_points)
            # plt.suptitle("{}: {}".format(osp.basename(fname), ret))
            # cv2.drawChessboardCorners(img, n_points, corners, ret)
            # imshow(img)
        elif with_partial:
            print (fname)
            with_vis = False
            partial_corners, partial_objp = CornerMapper.find_partial_corner_mapping(corners, img, n_points, with_vis)
            imgpoints.append(partial_corners)
            objpoints.append(partial_objp)

            # plt.figure()
            # CornerMapper.draw_final_cb_corners(partial_corners, partial_objp, img, n_points)
            # plt.suptitle("{}: {}".format(osp.basename(fname), ret))

    return im_shape, imgpoints, objpoints


def find_radial_distortion(images_path, n_cb_points, image_postfix, with_partial=False):
    """
    Find radial distortion parameters from a set of chckerboard images.

    :param with_partial: False to use openCV usual functionality. True to calibrate also using partial CB images.
    :param image_postfix: E.g. '.jpg' or '.png'
    :param images_path: Full path to a folder where the images are stored
    :param n_cb_points: amount of INNER CORNERS of the checkerboard. that is if the CB is held more or less on landscape
            orientation and has 10x8 tiles it should be (9, 7)
    :return:  The camera's K matrix and the len's radial distortion parameters

    """
    images_ = sorted([f for f in os.listdir(images_path) if f.endswith(image_postfix)])

    fullpath_images_ = [osp.join(images_path, f) for f in images_]
    if len(fullpath_images_) > 10:
        print "starting radial distortion calculation for folder" + images_path
        im_shape, imgpoints, objpoints = find_corners_in_images_set(fullpath_images_, n_cb_points, with_partial)
        n_images = len(objpoints)

        ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, im_shape, cameraMatrix=None, distCoeffs=None)
        print "Done"
    else:
        print "There are only {} images in folder (at least 10 needed)".format(len(fullpath_images_))
        mtx = np.eye(3)
        dist = np.zeros(5)
        n_images = 0

    return mtx, dist, n_images


def undistort_image(camera_matrix, dist_coeffs, image_dist):
    """
    Undistort image that has radial distortion.

    :param camera_matrix: a 3x3 internal camera matrix
    :param dist_coeffs: an array with the 5 radial distortion parameters
    :param image_dist: the distorted image
    :return: a corrected image
    """
    if np.all(np.equal(dist_coeffs, 0)):
        return image_dist, camera_matrix

    image_size = image_dist.shape[1::-1]
    new_camera_matrix, output_size = \
        NewtonRaphsonUndistort.get_optimal_new_camera_matrix(camera_matrix, dist_coeffs, image_size, alpha=0)
    # sanity check - verify that the new focal point is close to the center of the output image
    # It should be much closer to the image center than 0.5 of the minimal dimension
    new_focal_point = new_camera_matrix[:2, 2]
    focal_point_ctr_dist = np.linalg.norm(np.array(image_size, float) / 2 - new_focal_point)
    is_invalid_focal_point = focal_point_ctr_dist > 0.5 * min(image_size)
    if all([x == 0 for x in output_size]) or is_invalid_focal_point:
        # in some cases the getOptimalNewCameraMatrix fails to find optimal camera matrix.
        # The result of usig the original matrix is similar to the optimal but somewhat cropped.
        new_camera_matrix = camera_matrix
        image_corr = cv2.undistort(image_dist, camera_matrix, dist_coeffs)
    else:
        map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, np.eye(3), new_camera_matrix,
                                                 image_size, cv2.CV_32FC1)
        image_corr = cv2.remap(image_dist, map1, map2, interpolation=cv2.INTER_CUBIC)
    return image_corr, new_camera_matrix


def example__optimal_new_camera_matrix():
    dist_params_ = np.array([-0.3369753, 0.8365255, -0.001633824, 0.0007970532, -1.793908])
    camera_matrix_ = np.array([[4405.922, 0, 2318.92],
                               [0, 4389.491, 1622.885],
                               [0, 0, 1]])
    image_size = (4608, 3288)

    # pick any image, as the parameters are the important ones.
    # to be agnostic to your image - scale it to
    image_dist_unscaled = cv2.imread('/path/to/a/test/distorted/image')[:, :, ::-1]
    image_dist_ = cv2.resize(image_dist_unscaled, image_size)

    alpha_ = 0

    new_camera_matrix_, output_size_ = \
        NewtonRaphsonUndistort.get_optimal_new_camera_matrix(camera_matrix_, dist_params_, image_size, alpha=alpha_)
    map1, map2 = cv2.initUndistortRectifyMap(camera_matrix_, dist_params_, np.eye(3), new_camera_matrix_,
                                             image_size, cv2.CV_32FC1)
    image_corr_mine = cv2.remap(image_dist_, map1, map2, interpolation=cv2.INTER_CUBIC)

    new_camera_matrix_, _ = \
        cv2.getOptimalNewCameraMatrix(camera_matrix_, dist_params_, image_size, alpha=alpha_)
    map1, map2 = cv2.initUndistortRectifyMap(camera_matrix_, dist_params_, np.eye(3), new_camera_matrix_,
                                             image_size, cv2.CV_32FC1)
    image_corr_opencv = cv2.remap(image_dist_, map1, map2, interpolation=cv2.INTER_CUBIC)

    plt.subplot(131)
    plt.imshow(image_dist_)
    plt.title('distorted image')
    plt.subplot(132)
    plt.imshow(image_corr_mine)
    plt.title('suggested correction')
    plt.subplot(133)
    plt.imshow(image_corr_opencv)
    plt.title('opencv undistort result')
    plt.show()


def example__calibration_with_partial_checkerboards():
    folder = '/path/to/a/folder/containing/all/calibration/images'
    image_dist_ = cv2.imread('/path/to/a/test/distorted/image')
    n_cb_points_ = (9, 7)  # amount of internal corers in the checkerboard

    for with_p, ttl in [(False, 'without partial checkerboards'), (True, 'with partial checkerboards')]:
        plt.figure()
        camera_matrix_, dist_coeffs_, _ = find_radial_distortion(folder, n_cb_points_, '.jpg', with_partial=with_p)
        image_corr_, new_camera_matrix_ = undistort_image(camera_matrix_, dist_coeffs_, image_dist_)
        plt.clf()
        plt.title(ttl)
        plt.imshow(image_corr_)

    plt.figure(3)
    plt.subplot(122)
    plt.cla()
    plt.suptitle('distorted image')
    plt.imshow(image_dist_)

    plt.show()
