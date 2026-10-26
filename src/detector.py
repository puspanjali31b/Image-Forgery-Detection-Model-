from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class DetectionResult:
    mask: np.ndarray
    confidence: float
    status: str
    method: str


class CopyMoveDetector:
    def __init__(self, max_keypoints: int = 2000, min_matches: int = 8):
        self.max_keypoints = max_keypoints
        self.min_matches = min_matches

    def detect(self, image: np.ndarray) -> DetectionResult:
        if image is None or image.size == 0:
            empty = np.zeros((1, 1), dtype=np.uint8)
            return DetectionResult(empty, 0.0, "No image provided", "none")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = self._resize_for_detection(gray)
        sift = cv2.SIFT_create(nfeatures=min(self.max_keypoints, 2000))
        keypoints, descriptors = sift.detectAndCompute(resized, None)

        if descriptors is not None and len(keypoints) >= self.min_matches:
            matches = self._match_keypoints(keypoints, descriptors)
            if len(matches) >= self.min_matches:
                mask = self._build_mask_from_matches(keypoints, matches, resized.shape)
                if np.count_nonzero(mask) > 0:
                    region_fraction = np.count_nonzero(mask) / float(mask.size)
                    confidence = float(np.clip(region_fraction * 3.0, 0.0, 0.99))
                    status = "Suspicious copy-move region detected"
                    return DetectionResult(mask=mask, confidence=confidence, status=status, method="SIFT + translation clustering")

        fallback = self._template_based_fallback(resized)
        if np.count_nonzero(fallback) > 0:
            region_fraction = np.count_nonzero(fallback) / float(fallback.size)
            confidence = float(np.clip(region_fraction * 2.0, 0.0, 0.99))
            return DetectionResult(fallback, confidence, "Suspicious copy-move region detected", "template matching fallback")

        return self._fallback_candidate(mask=np.zeros_like(resized, dtype=np.uint8), confidence=0.0)

    def _resize_for_detection(self, image_gray: np.ndarray) -> np.ndarray:
        height, width = image_gray.shape[:2]
        max_dim = 1200
        scale = min(1.0, max_dim / max(height, width))
        if scale < 1.0:
            new_shape = (int(height * scale), int(width * scale))
            return cv2.resize(image_gray, new_shape[::-1], interpolation=cv2.INTER_AREA)
        return image_gray

    def _match_keypoints(self, keypoints, descriptors):
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        knn_matches = matcher.knnMatch(descriptors, descriptors, k=2)
        filtered = []
        for first, second in knn_matches:
            if first.distance < 0.75 * second.distance:
                filtered.append(first)

        if len(filtered) < 8:
            return []

        vector_counts = {}
        for match in filtered:
            point_a = keypoints[match.queryIdx].pt
            point_b = keypoints[match.trainIdx].pt
            dx = point_b[0] - point_a[0]
            dy = point_b[1] - point_a[1]
            if abs(dx) < 8 and abs(dy) < 8:
                continue
            bucket = (round(dx / 25.0) * 25, round(dy / 25.0) * 25)
            vector_counts[bucket] = vector_counts.get(bucket, 0) + 1

        if not vector_counts:
            return []

        dominant = max(vector_counts, key=vector_counts.get)
        dominant_matches = []
        for match in filtered:
            point_a = keypoints[match.queryIdx].pt
            point_b = keypoints[match.trainIdx].pt
            dx = point_b[0] - point_a[0]
            dy = point_b[1] - point_a[1]
            bucket = (round(dx / 25.0) * 25, round(dy / 25.0) * 25)
            if bucket == dominant:
                dominant_matches.append(match)

        return dominant_matches

    def _build_mask_from_matches(self, keypoints, matches, image_shape) -> np.ndarray:
        rows, cols = image_shape[:2]
        source_points = []
        destination_points = []

        for match in matches:
            p1 = keypoints[match.queryIdx].pt
            p2 = keypoints[match.trainIdx].pt
            source_points.append(p1)
            destination_points.append(p2)

        if len(source_points) < 4:
            return np.zeros((rows, cols), dtype=np.uint8)

        source_array = np.float32(source_points)
        destination_array = np.float32(destination_points)

        buffer = 0
        x1 = max(0, int(np.min(source_array[:, 0]) - buffer))
        y1 = max(0, int(np.min(source_array[:, 1]) - buffer))
        x2 = min(cols - 1, int(np.max(source_array[:, 0]) + buffer))
        y2 = min(rows - 1, int(np.max(source_array[:, 1]) + buffer))

        if x2 <= x1 or y2 <= y1:
            return np.zeros((rows, cols), dtype=np.uint8)

        mask = np.zeros((rows, cols), dtype=np.uint8)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)

        if len(source_array) >= 8:
            hull = cv2.convexHull(source_array)
            filled = np.zeros((rows, cols), dtype=np.uint8)
            cv2.drawContours(filled, [np.int32(hull)], -1, 255, thickness=-1)
            mask = cv2.bitwise_or(mask, filled)

        return mask

    def _template_based_fallback(self, gray: np.ndarray) -> np.ndarray:
        height, width = gray.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        if float(np.std(gray)) < 8.0:
            return mask

        for patch_size in (24, 32, 48, 64, 96):
            if patch_size >= min(height, width):
                continue
            step = max(8, patch_size // 3)
            for y in range(0, height - patch_size + 1, step):
                for x in range(0, width - patch_size + 1, step):
                    patch = gray[y:y + patch_size, x:x + patch_size]
                    if np.std(patch) < 8.0:
                        continue
                    match_result = cv2.matchTemplate(gray, patch, cv2.TM_CCOEFF_NORMED)
                    _, best_value, _, best_loc = cv2.minMaxLoc(match_result)

                    if best_value < 0.96:
                        continue

                    match_x = best_loc[0]
                    match_y = best_loc[1]
                    if abs(match_x - x) < patch_size * 0.5 and abs(match_y - y) < patch_size * 0.5:
                        continue

                    cv2.rectangle(mask, (x, y), (x + patch_size, y + patch_size), 255, thickness=-1)
                    cv2.rectangle(mask, (match_x, match_y), (match_x + patch_size, match_y + patch_size), 255, thickness=-1)

        if np.count_nonzero(mask) == 0:
            return mask

        mask = self._largest_component(mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return self._largest_component(mask)

    def _largest_component(self, mask: np.ndarray) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
            return mask
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        refined = np.zeros_like(mask)
        refined[labels == largest_label] = 255
        return refined

    def _fallback_candidate(self, mask: np.ndarray, confidence: float) -> DetectionResult:
        return DetectionResult(mask=mask, confidence=confidence, status="No strong copy-move pattern detected", method="fallback")


def detect_copy_move(image: np.ndarray) -> DetectionResult:
    detector = CopyMoveDetector()
    return detector.detect(image)
