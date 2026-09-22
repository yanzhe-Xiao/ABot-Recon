import pytest
import torch
import numpy as np
from PIL import Image
from viewer.streaming_api_server import (
    parse_devices,
    _sync_preprocess_frame,
    _sync_postprocess_slice,
)


def test_parse_devices_cpu():
    devs = parse_devices("cpu")
    assert devs == ["cpu"]


def test_parse_devices_cuda():
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        auto_devs = parse_devices("auto")
        assert len(auto_devs) == count
        assert all(d.startswith("cuda:") for d in auto_devs)

        explicit_devs = parse_devices("cuda:0")
        assert explicit_devs == ["cuda:0"]

        if count >= 2:
            multi_devs = parse_devices("0,1")
            assert multi_devs == ["cuda:0", "cuda:1"]
    else:
        assert parse_devices("auto") == ["cpu"]


def test_sync_preprocess_frame():
    # Test numpy array input
    img_arr = np.zeros((480, 640, 3), dtype=np.uint8)
    pil_img, tensor_chw, rgb_full = _sync_preprocess_frame(img_arr)
    assert isinstance(pil_img, Image.Image)
    assert tensor_chw.shape == (3, 280, 504)
    assert rgb_full.shape == (280, 504, 3)
    assert rgb_full.dtype == np.uint8

    # Test PIL Image input
    pil_img2, tensor_chw2, rgb_full2 = _sync_preprocess_frame(pil_img)
    assert tensor_chw2.shape == (3, 280, 504)


def test_sync_postprocess_slice():
    raw_world = torch.zeros((280, 504, 3), dtype=torch.float32)
    conf_map = torch.ones((280, 504), dtype=torch.float32)
    tensor_chw = torch.zeros((3, 280, 504), dtype=torch.float32)

    valid_pts, valid_rgb, thumb_b64 = _sync_postprocess_slice(
        raw_world=raw_world,
        conf_map=conf_map,
        static_mask=None,
        tensor_chw=tensor_chw,
        point_stride=4,
        confidence_threshold=0.1,
        need_thumbnail=False,
        pil_img=None,
    )
    expected_count = (280 // 4) * (504 // 4)
    assert valid_pts.shape == (expected_count, 3)
    assert valid_rgb.shape == (expected_count, 3)
    assert thumb_b64 is None  # not requested

    # Test with thumbnail
    pil_img = Image.new("RGB", (504, 280), color=(100, 150, 200))
    valid_pts, valid_rgb, thumb_b64 = _sync_postprocess_slice(
        raw_world=raw_world,
        conf_map=conf_map,
        static_mask=None,
        tensor_chw=tensor_chw,
        point_stride=4,
        confidence_threshold=0.1,
        need_thumbnail=True,
        pil_img=pil_img,
    )
    assert thumb_b64 is not None
    assert thumb_b64.startswith("data:image/jpeg;base64,")


def test_worker_session_load_balancing():
    import asyncio
    from unittest.mock import MagicMock
    from viewer.streaming_api_server import MultiSessionReconstructionManager, DeviceWorker

    # Mock two DeviceWorkers
    w0 = MagicMock(spec=DeviceWorker)
    w0.worker_id = 0
    w0.device = torch.device("cuda:0")
    w0.active_sessions = set()
    w0.paged_template = MagicMock()

    w1 = MagicMock(spec=DeviceWorker)
    w1.worker_id = 1
    w1.device = torch.device("cuda:1")
    w1.active_sessions = set()
    w1.paged_template = MagicMock()

    mgr = MultiSessionReconstructionManager.__new__(MultiSessionReconstructionManager)
    mgr.workers = [w0, w1]
    mgr.primary_worker = w0
    mgr.device = w0.device
    mgr.sessions = {}
    mgr.subscribers = {}
    mgr.meta_lock = asyncio.Lock()
    mgr.scene_locks = {}
    mgr.scene_dirty = {}
    mgr.dynamic_filter = False
    mgr.dynamic_interval = 2
    mgr.register_session_to_scene = MagicMock()
    mgr.broadcast_to_subscribers = MagicMock()

    # Create 5 sessions simulating 5 concurrent video streams
    s1 = mgr.get_or_create_session("stream_1")
    s2 = mgr.get_or_create_session("stream_2")
    s3 = mgr.get_or_create_session("stream_3")
    s4 = mgr.get_or_create_session("stream_4")
    s5 = mgr.get_or_create_session("stream_5")

    # Verify they were evenly balanced across the two GPU workers (3 and 2)
    assert len(w0.active_sessions) + len(w1.active_sessions) == 5
    assert abs(len(w0.active_sessions) - len(w1.active_sessions)) <= 1
    assert s1.worker in (w0, w1)
    assert s2.worker in (w0, w1)

