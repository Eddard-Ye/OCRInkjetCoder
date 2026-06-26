# -*- coding: utf-8 -*-
import unittest

import cv2
import numpy as np

from white_bg_segment import (
    WhiteBgSegmentConfig,
    apply_aux_segment_overlay,
    filter_ocr_boxes_by_white_rect,
    min_area_from_slider,
    ocr_box_inside_rect,
    segment_white_background,
)


class WhiteBgSegmentTest(unittest.TestCase):
    def _white_patch_frame(self) -> np.ndarray:
        img = np.zeros((200, 400, 3), dtype=np.uint8)
        img[:, :] = (40, 40, 40)
        cv2.rectangle(img, (80, 60), (320, 140), (230, 230, 230), thickness=-1)
        return img

    def test_segment_finds_largest_white_region(self) -> None:
        cfg = WhiteBgSegmentConfig(
            s_max=40,
            v_min=180,
            min_area=1000,
        )
        seg = segment_white_background(self._white_patch_frame(), cfg)
        self.assertTrue(seg.found)
        assert seg.rect_xyxy is not None
        x1, y1, x2, y2 = seg.rect_xyxy
        self.assertLessEqual(x1, 90)
        self.assertGreaterEqual(x2, 310)
        self.assertLessEqual(y1, 70)
        self.assertGreaterEqual(y2, 130)

    def test_filter_keeps_boxes_inside_rect(self) -> None:
        rect = [50, 50, 350, 150]
        inside = {"poly": [[100, 80], [180, 80], [180, 110], [100, 110]], "text": "ok"}
        outside = {"poly": [[10, 10], [30, 10], [30, 30], [10, 30]], "text": "bad"}
        kept, rejected = filter_ocr_boxes_by_white_rect([inside, outside], rect)
        self.assertEqual(len(kept), 1)
        self.assertEqual(rejected, 1)
        self.assertEqual(kept[0]["text"], "ok")

    def test_filter_rejects_all_when_no_rect(self) -> None:
        box = {"poly": [[10, 10], [30, 10], [30, 30], [10, 30]], "text": "x"}
        kept, rejected = filter_ocr_boxes_by_white_rect([box], None)
        self.assertEqual(kept, [])
        self.assertEqual(rejected, 1)

    def test_ocr_box_inside_rect_uses_polygon_bbox(self) -> None:
        rect = [0, 0, 100, 100]
        box = {"poly": [[20, 20], [80, 20], [80, 40], [20, 40]]}
        self.assertTrue(ocr_box_inside_rect(box, rect))
        box_out = {"poly": [[20, 20], [120, 20], [120, 40], [20, 40]]}
        self.assertFalse(ocr_box_inside_rect(box_out, rect))

    def test_aux_overlay_preserves_shape(self) -> None:
        img = self._white_patch_frame()
        cfg = WhiteBgSegmentConfig(s_max=40, v_min=180, min_area=1000)
        seg = segment_white_background(img, cfg)
        out = apply_aux_segment_overlay(img, seg)
        self.assertEqual(out.shape, img.shape)

    def test_min_area_slider_mapping(self) -> None:
        self.assertEqual(min_area_from_slider(50), 5000)


if __name__ == "__main__":
    unittest.main()
