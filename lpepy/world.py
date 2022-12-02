import json

import numpy as np


class PointEstimate:
    """Hold on to sets of estimated points, and give back a combined est"""

    def __init__(self):
        self._point_estimates = []

    def add(self, new_points):
        if len(self._point_estimates) > 0:
            assert len(new_points) == len(self._point_estimates[0])
        new_points = [
            p if p is not None else (np.NaN, np.NaN, np.NaN) for p in new_points
        ]
        self._point_estimates.append(new_points)

    def _avg_pos(self, pts):
        return np.nanmean(pts, axis=0)

    def _dist(self, ref, pt):
        diff = ref - pt
        return np.sqrt(diff.dot(diff))

    def _ransacish(self, estimates):
        estimates = estimates[~np.isnan(estimates)].reshape(-1, estimates.shape[1])
        if len(estimates) == 0:
            return [np.NaN, np.NaN, np.NaN]
        if len(estimates) == 1:
            return estimates[0]
        n_samples = int(len(estimates) / 2 + 1)
        best_error = np.inf
        for i in range(50):  # TODO: Find better termination criteria
            np.random.shuffle(estimates)
            samples = estimates[: np.random.randint(2, n_samples + 1)]
            est = self._avg_pos(samples)

            if any(np.isnan(est)):
                # NaN > anything is False, so would consider all points as
                # inliers for outlier length test, but *also* NaN < anything
                # is False, so inliers would be empty, estimate would be NaN.
                continue

            distances = np.float64([self._dist(est, pt) for pt in estimates])
            outliers = estimates[distances >= 0.15]  # TODO, arbitrary value
            inliers = estimates[distances < 0.15]

            if len(outliers) < best_error:
                best_error = len(outliers)
                best_inliers = inliers
                if best_error == 0:
                    break

        return np.nanmean(best_inliers, axis=0)

    def combined(self):
        ests = np.float64(self._point_estimates)
        if len(ests) < 3:
            mean = np.nanmean(ests, axis=0)
            res = [None if np.isnan(v[0]) else v for v in mean]
        else:
            res = [None] * len(ests[0])
            for i, pt_ests in enumerate(ests.transpose((1, 0, 2))):
                new_est = self._ransacish(pt_ests)
                res[i] = new_est
        return res

    def json(self):
        pixels = {}
        for px, pt in enumerate(self.combined()):
            if pt is None:
                continue
            pixels[px] = list(pt)

        return json.dumps(pixels)
