#!/usr/bin/env python3
"""
Real-time GPU & VRAM Performance Monitor for Video Stream Reconstruction (ABot-Recon)
====================================================================================
Monitors:
  - GPU compute utilization (%) and VRAM usage (Used / Total / %, Peak)
  - Video stream building active sessions from 8090 API (/api/sessions)
  - Running processes associated with video streaming & 3D reconstruction
  - Optional CSV logging for post-experiment memory profiling

Usage:
  python scripts/monitor_stream_gpu.py                  # Live rich dashboard
  python scripts/monitor_stream_gpu.py --port 8090      # Monitor with 8090 active stream status
  python scripts/monitor_stream_gpu.py --plain          # Plain text scrolling log (nohup friendly)
  python scripts/monitor_stream_gpu.py --csv gpu.csv    # Record time-series to CSV
  python scripts/monitor_stream_gpu.py --gpu 0          # Monitor only GPU 0
"""

import argparse
import csv
import datetime
import json
import os
import signal
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.layout import Layout
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def clean_gpu_name(raw_name: str) -> str:
    """Clean up verbose GPU model names for compact terminal display."""
    return raw_name.replace("NVIDIA GeForce ", "").replace("NVIDIA ", "").strip()


@dataclass
class GPUStats:
    index: int
    name: str
    util_gpu: int
    mem_used_mb: float
    mem_total_mb: float
    mem_percent: float
    temp_c: int
    power_w: float
    peak_mem_mb: float = 0.0
    history_utils: List[int] = field(default_factory=list)


@dataclass
class ProcessInfo:
    pid: int
    gpu_index: int
    used_mem_mb: float
    name: str
    cmdline: str
    is_recon: bool = False


class StreamGPUMonitor:
    def __init__(
        self,
        gpu_indices: Optional[List[int]] = None,
        port: int = 8090,
        interval: float = 1.0,
        csv_file: Optional[Path] = None,
        plain_mode: bool = False,
    ):
        if not HAS_PYNVML:
            raise RuntimeError("pynvml is required. Please install via: pip install pynvml / nvidia-ml-py")

        pynvml.nvmlInit()
        self.device_count = pynvml.nvmlDeviceGetCount()

        if gpu_indices is None:
            self.gpu_indices = list(range(self.device_count))
        else:
            self.gpu_indices = [i for i in gpu_indices if 0 <= i < self.device_count]

        self.handles: Dict[int, Any] = {}
        self.peaks: Dict[int, float] = {i: 0.0 for i in self.gpu_indices}
        self.history_utils: Dict[int, List[int]] = {i: [] for i in self.gpu_indices}

        for i in self.gpu_indices:
            self.handles[i] = pynvml.nvmlDeviceGetHandleByIndex(i)

        self.port = port
        self.interval = interval
        self.csv_file = csv_file
        self.plain_mode = plain_mode
        self.running = True
        self.start_time = time.time()

        if self.csv_file:
            self.csv_file.parent.mkdir(parents=True, exist_ok=True)
            self._init_csv()

    def _init_csv(self):
        with open(self.csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "gpu_index",
                "gpu_name",
                "util_percent",
                "mem_used_mb",
                "mem_total_mb",
                "mem_percent",
                "peak_mem_mb",
                "temp_c",
                "power_w",
                "active_streams_count",
                "recon_processes_count",
            ])

    def close(self):
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    def get_gpu_stats(self) -> List[GPUStats]:
        stats_list = []
        for i in self.gpu_indices:
            handle = self.handles[i]
            raw_name = pynvml.nvmlDeviceGetName(handle)
            name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)

            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used_mb = mem.used / (1024.0 * 1024.0)
            total_mb = mem.total / (1024.0 * 1024.0)
            mem_pct = (used_mb / total_mb * 100.0) if total_mb > 0 else 0.0

            if used_mb > self.peaks[i]:
                self.peaks[i] = used_mb

            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            except Exception:
                util = 0

            self.history_utils[i].append(util)
            if len(self.history_utils[i]) > 100:
                self.history_utils[i].pop(0)

            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temp = 0

            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            except Exception:
                power = 0.0

            stats_list.append(GPUStats(
                index=i,
                name=name,
                util_gpu=util,
                mem_used_mb=used_mb,
                mem_total_mb=total_mb,
                mem_percent=mem_pct,
                temp_c=temp,
                power_w=power,
                peak_mem_mb=self.peaks[i],
                history_utils=self.history_utils[i],
            ))
        return stats_list

    def get_running_processes(self) -> List[ProcessInfo]:
        procs: List[ProcessInfo] = []
        for i in self.gpu_indices:
            handle = self.handles[i]
            try:
                nvml_procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            except Exception:
                nvml_procs = []

            for p in nvml_procs:
                pid = p.pid
                used_mb = p.usedGpuMemory / (1024.0 * 1024.0) if p.usedGpuMemory else 0.0
                name = "unknown"
                cmdline = ""
                is_recon = False

                if HAS_PSUTIL:
                    try:
                        ps = psutil.Process(pid)
                        name = ps.name()
                        cmd_parts = ps.cmdline()
                        cmdline = " ".join(cmd_parts)
                        # Identify reconstruction / streaming / slam processes
                        recon_keywords = [
                            "streaming_api_server", "reconstruct", "match_and_fuse",
                            "optimize_fused", "abot_recon", "stream_camera", "torch", "python"
                        ]
                        if any(kw in cmdline.lower() for kw in recon_keywords):
                            is_recon = True
                    except Exception:
                        pass

                procs.append(ProcessInfo(
                    pid=pid,
                    gpu_index=i,
                    used_mem_mb=used_mb,
                    name=name,
                    cmdline=cmdline,
                    is_recon=is_recon,
                ))
        return procs

    def query_stream_sessions(self) -> Dict[str, Any]:
        """Query 8090 streaming API for active sessions and status."""
        if self.port <= 0:
            return {"status": "disabled", "active": []}
        url = f"http://127.0.0.1:{self.port}/api/sessions"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "GPUMonitor/1.0"})
            with urllib.request.urlopen(req, timeout=0.8) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    return {"status": "online", "active": data.get("active_sessions", [])}
        except Exception:
            pass
        return {"status": "offline", "active": []}

    def record_csv(self, stats: List[GPUStats], active_count: int, recon_proc_count: int):
        if not self.csv_file:
            return
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for s in stats:
                writer.writerow([
                    now_str,
                    s.index,
                    s.name,
                    s.util_gpu,
                    f"{s.mem_used_mb:.1f}",
                    f"{s.mem_total_mb:.1f}",
                    f"{s.mem_percent:.2f}",
                    f"{s.peak_mem_mb:.1f}",
                    s.temp_c,
                    f"{s.power_w:.1f}",
                    active_count,
                    recon_proc_count,
                ])

    def print_plain(self, stats: List[GPUStats], procs: List[ProcessInfo], session_data: Dict[str, Any]):
        now_str = datetime.datetime.now().strftime("%H:%M:%S")
        active_sessions = session_data.get("active", [])
        active_str = f"Streams: {len(active_sessions)}" if session_data["status"] == "online" else f"API:{session_data['status']}"

        for s in stats:
            used_gb = s.mem_used_mb / 1024.0
            total_gb = s.mem_total_mb / 1024.0
            peak_gb = s.peak_mem_mb / 1024.0
            display_name = clean_gpu_name(s.name)
            print(
                f"[{now_str}] GPU {s.index} ({display_name}): "
                f"Util: {s.util_gpu:3d}% | "
                f"VRAM: {used_gb:4.1f}/{total_gb:4.1f} GB ({s.mem_percent:5.1f}%, Peak: {peak_gb:4.1f}GB) | "
                f"Temp: {s.temp_c:2d}°C | Power: {s.power_w:4.0f}W | {active_str}"
            )

        recon_procs = [p for p in procs if p.is_recon]
        if recon_procs:
            p_desc = ", ".join([f"PID {p.pid}: {p.used_mem_mb:.0f}MB" for p in recon_procs[:3]])
            print(f"         └─ Recon Procs: {p_desc}")

    def render_rich(self, stats: List[GPUStats], procs: List[ProcessInfo], session_data: Dict[str, Any]) -> Layout:
        layout = Layout()
        gpu_table_size = len(stats) + 5
        active_list = session_data.get("active", [])
        sess_table_size = max(1, len(active_list)) + 5

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="gpu_table", size=gpu_table_size),
            Layout(name="sessions", size=sess_table_size),
            Layout(name="procs"),
        )

        # Header
        elapsed = int(time.time() - self.start_time)
        hrs, rem = divmod(elapsed, 3600)
        mins, secs = divmod(rem, 60)
        time_str = f"{hrs:02d}:{mins:02d}:{secs:02d}"
        header_text = Text.assemble(
            ("⚡ ABot-Recon 视频流建图 GPU / 显存实时监测器 ", "bold cyan"),
            (f"[运行时间: {time_str}] ", "bold yellow"),
            (f"[采样间隔: {self.interval}s]", "green"),
        )
        layout["header"].update(Panel(header_text, style="cyan"))

        # GPU Overview Table
        gpu_table = Table(title="🖥️ GPU & 显存占用状态 (实时 / 峰值)", expand=True)
        gpu_table.add_column("GPU", justify="center", style="bold white", width=7, no_wrap=True)
        gpu_table.add_column("型号", style="white", width=12, no_wrap=True)
        gpu_table.add_column("算力负载", justify="right", width=8, no_wrap=True)
        gpu_table.add_column("显存占用", justify="right", width=18, no_wrap=True)
        gpu_table.add_column("显存占比", justify="center", ratio=1, min_width=16, no_wrap=True)
        gpu_table.add_column("峰值显存", justify="right", style="bold magenta", width=10, no_wrap=True)
        gpu_table.add_column("温度", justify="center", width=6, no_wrap=True)
        gpu_table.add_column("功耗", justify="right", width=8, no_wrap=True)

        for s in stats:
            # Color coding
            util_color = "green" if s.util_gpu < 50 else ("yellow" if s.util_gpu < 85 else "red")
            mem_color = "green" if s.mem_percent < 60 else ("yellow" if s.mem_percent < 85 else "red bold")

            # Progress bar visualization for VRAM (compact 10 chars)
            bar_len = 10
            filled = int(bar_len * (s.mem_percent / 100.0))
            bar_str = "█" * filled + "░" * (bar_len - filled)

            used_gb = s.mem_used_mb / 1024.0
            total_gb = s.mem_total_mb / 1024.0
            peak_gb = s.peak_mem_mb / 1024.0
            display_name = clean_gpu_name(s.name)

            gpu_table.add_row(
                f"GPU {s.index}",
                display_name,
                f"[{util_color}]{s.util_gpu}%[/{util_color}]",
                f"[{mem_color}]{used_gb:.2f} / {total_gb:.2f} GB[/{mem_color}]",
                f"[{mem_color}]{bar_str} {s.mem_percent:.1f}%[/{mem_color}]",
                f"{peak_gb:.2f} GB",
                f"{s.temp_c}°C",
                f"{s.power_w:.1f} W",
            )
        layout["gpu_table"].update(gpu_table)

        # Video Streaming Sessions Table
        status_color = "green" if session_data["status"] == "online" else ("yellow" if session_data["status"] == "disabled" else "red")
        sess_title = f"🎥 视频流建图状态 (API 8090: [{status_color}]{session_data['status']}[/{status_color}] | 活动流: {len(active_list)})"
        sess_table = Table(title=sess_title, expand=True)
        sess_table.add_column("Session ID", style="bold cyan")
        sess_table.add_column("Scene ID", style="cyan")
        sess_table.add_column("设备/机器人", style="yellow")
        sess_table.add_column("接收帧数", justify="right")
        sess_table.add_column("处理帧数", justify="right")
        sess_table.add_column("点云点数", justify="right", style="green")

        if active_list:
            for item in active_list:
                sess_table.add_row(
                    str(item.get("session_id", "-")),
                    str(item.get("scene_id", "-")),
                    str(item.get("robot", "-")),
                    str(item.get("frames", item.get("frame_counter", "-"))),
                    str(item.get("processed", item.get("processed_counter", "-"))),
                    f"{item.get('points', 0):,}",
                )
        else:
            if session_data["status"] == "online":
                sess_table.add_row("[italic dim]当前无活动视频流正在建图 (空闲待命)", "-", "-", "-", "-", "-")
            else:
                sess_table.add_row(f"[italic dim]未连接到 8090 端口 ({session_data['status']})", "-", "-", "-", "-", "-")
        layout["sessions"].update(sess_table)

        # Running Processes Table
        proc_table = Table(title="⚙️ 占用显存的相关进程列表 (PID & 显存占用)", expand=True)
        proc_table.add_column("PID", style="bold white", justify="right", width=9, no_wrap=True)
        proc_table.add_column("GPU", justify="center", width=7, no_wrap=True)
        proc_table.add_column("显存占用", justify="right", style="magenta", width=12, no_wrap=True)
        proc_table.add_column("进程名", style="white", width=18, no_wrap=True)
        proc_table.add_column("启动命令 / 脚本", style="dim", ratio=1, no_wrap=True, overflow="ellipsis")

        for p in procs:
            tag = "[bold green][建图相关][/bold green] " if p.is_recon else ""
            proc_table.add_row(
                str(p.pid),
                f"GPU {p.gpu_index}",
                f"{p.used_mem_mb:.1f} MB",
                f"{tag}{p.name}",
                p.cmdline,
            )
        if not procs:
            proc_table.add_row("-", "-", "-", "[italic dim]未检测到活动计算进程", "-")
        layout["procs"].update(proc_table)

        return layout

    def run(self, duration: Optional[float] = None):
        def sig_handler(sig, frame):
            self.running = False

        signal.signal(signal.SIGINT, sig_handler)
        signal.signal(signal.SIGTERM, sig_handler)

        start = time.time()

        if not self.plain_mode and HAS_RICH and sys.stdout.isatty():
            console = Console()
            with Live(console=console, refresh_per_second=int(1.0 / max(0.2, self.interval)), screen=True) as live:
                while self.running:
                    stats = self.get_gpu_stats()
                    procs = self.get_running_processes()
                    sess = self.query_stream_sessions()

                    recon_count = len([p for p in procs if p.is_recon])
                    self.record_csv(stats, len(sess.get("active", [])), recon_count)

                    layout = self.render_rich(stats, procs, sess)
                    live.update(layout)

                    if duration and (time.time() - start) >= duration:
                        break
                    time.sleep(self.interval)
        else:
            # Plain terminal or redirected output mode
            print(f"=== Starting GPU & VRAM Stream Monitor (Interval: {self.interval}s) ===")
            if self.csv_file:
                print(f"Recording metrics to CSV: {self.csv_file}")
            while self.running:
                stats = self.get_gpu_stats()
                procs = self.get_running_processes()
                sess = self.query_stream_sessions()

                recon_count = len([p for p in procs if p.is_recon])
                self.record_csv(stats, len(sess.get("active", [])), recon_count)

                self.print_plain(stats, procs, sess)

                if duration and (time.time() - start) >= duration:
                    break
                time.sleep(self.interval)

        self.print_summary()

    def print_summary(self):
        print("\n" + "=" * 70)
        print("  📊 ABot-Recon 视频流建图 GPU 监测结束统计报告")
        print("=" * 70)
        elapsed = time.time() - self.start_time
        print(f"监测总时长: {elapsed:.1f} 秒 ({elapsed/60.0:.2f} 分钟)")

        for i in self.gpu_indices:
            peak_mb = self.peaks[i]
            utils = self.history_utils[i]
            avg_util = sum(utils) / max(1, len(utils))
            max_util = max(utils) if utils else 0
            print(f"GPU {i}:")
            print(f"  - 峰值显存 (Peak VRAM): {peak_mb:.1f} MB ({peak_mb/1024.0:.2f} GB)")
            print(f"  - 平均算力负载:       {avg_util:.1f}% (最高: {max_util}%)")

        if self.csv_file:
            print(f"历史数据已保存至: {self.csv_file.resolve()}")
        print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Monitor GPU utilization and VRAM consumption during video stream reconstruction."
    )
    parser.add_argument("-i", "--interval", type=float, default=1.0, help="Sampling interval in seconds (default: 1.0)")
    parser.add_argument("-g", "--gpu", type=str, default=None, help="Comma-separated GPU indices to monitor (e.g. '0' or '0,1')")
    parser.add_argument("-p", "--port", type=int, default=8090, help="Streaming API server port to query sessions (default: 8090, set 0 to disable)")
    parser.add_argument("--csv", type=str, default=None, help="Optional CSV file path to record metrics")
    parser.add_argument("--plain", action="store_true", help="Force plain text scrolling log mode instead of rich TUI dashboard")
    parser.add_argument("-d", "--duration", type=float, default=None, help="Optional monitoring duration in seconds")

    args = parser.parse_args()

    gpu_list = None
    if args.gpu:
        try:
            gpu_list = [int(x.strip()) for x in args.gpu.split(",")]
        except ValueError:
            print(f"Error: Invalid GPU specification '{args.gpu}', expected integers like '0' or '0,1'.")
            sys.exit(1)

    csv_path = Path(args.csv) if args.csv else None

    monitor = StreamGPUMonitor(
        gpu_indices=gpu_list,
        port=args.port,
        interval=args.interval,
        csv_file=csv_path,
        plain_mode=args.plain,
    )
    try:
        monitor.run(duration=args.duration)
    finally:
        monitor.close()


if __name__ == "__main__":
    main()
