#!/usr/bin/env bash
# ==============================================================================
# ABot-Recon 服务管理脚本
# 支持启动、停止、重启、状态查看及日志跟踪（8088 端口 3D 可视化器 & 8090 端口流式 API 服务）
# ==============================================================================

# 如果用户通过 sh 运行，自动切换到 bash 解释器
if [ -z "$BASH_VERSION" ]; then
    exec bash "$0" "$@"
fi

# 基础目录定位（确保在任何目录下执行均能正确指向项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
cd "$ROOT_DIR" || exit 1

# 配置项
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "$LOG_DIR"

PORT_8088=8088
PORT_8090=8090
DEFAULT_HOST="0.0.0.0"

PID_FILE_8088="${LOG_DIR}/viewer_8088.pid"
PID_FILE_8090="${LOG_DIR}/stream_8090.pid"
LOG_FILE_8088="${LOG_DIR}/viewer_8088.log"
LOG_FILE_8090="${LOG_DIR}/stream_8090.log"

# 终端颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# 检测 Python 解释器
detect_python() {
    # 1. 用户显式指定
    if [ -n "$PYTHON_EXEC" ] && [ -x "$PYTHON_EXEC" ]; then
        echo "$PYTHON_EXEC"
        return 0
    fi

    # 2. 如果当前已经激活了 abot-recon 环境
    if [ -n "$CONDA_PREFIX" ] && [[ "$CONDA_PREFIX" =~ abot-recon ]] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
        echo "${CONDA_PREFIX}/bin/python"
        return 0
    fi

    # 3. 检测常见的 abot-recon conda 专用环境路径
    local candidate_paths=(
        "/home/data/xyz/miniconda3/envs/abot-recon/bin/python"
        "$HOME/miniconda3/envs/abot-recon/bin/python"
        "$HOME/anaconda3/envs/abot-recon/bin/python"
        "/opt/conda/envs/abot-recon/bin/python"
        "/root/miniconda3/envs/abot-recon/bin/python"
    )
    for p in "${candidate_paths[@]}"; do
        if [ -x "$p" ]; then
            echo "$p"
            return 0
        fi
    done

    # 4. 尝试通过 conda info 查找 abot-recon 环境
    if command -v conda >/dev/null 2>&1; then
        local conda_env_path
        conda_env_path=$(conda info --envs 2>/dev/null | grep -E '\babot-recon\b' | awk '{print $NF}')
        if [ -n "$conda_env_path" ] && [ -x "${conda_env_path}/bin/python" ]; then
            echo "${conda_env_path}/bin/python"
            return 0
        fi
    fi

    # 5. 回退到当前 CONDA_PREFIX 或 PATH
    if [ -n "$CONDA_PREFIX" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
        echo "${CONDA_PREFIX}/bin/python"
        return 0
    fi

    if command -v python3 >/dev/null 2>&1; then
        command -v python3
    else
        command -v python
    fi
}

PYTHON_BIN="$(detect_python)"

# 自动选择最空闲的 GPU（若不可用则返回 cpu）
detect_best_device() {
    if [ -n "$DEVICE" ]; then
        echo "$DEVICE"
        return 0
    fi

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "cpu"
        return 0
    fi

    # 查找剩余显存最大的 GPU 序号
    local best_gpu
    best_gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null | sort -k2 -n -r | head -n1 | awk '{print $1}' | tr -d ',')
    if [ -n "$best_gpu" ]; then
        echo "cuda:${best_gpu}"
    else
        echo "cuda:0"
    fi
}

# 获取占用指定端口的所有 PID（多工具协同冗余检测）
get_pids_by_port() {
    local port="$1"
    local pids=()

    # 1. lsof 检测 (限定监听端，避免误伤连接该端口的客户端如 8088)
    if command -v lsof >/dev/null 2>&1; then
        for p in $(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null); do
            pids+=("$p")
        done
    fi

    # 2. fuser 检测
    if command -v fuser >/dev/null 2>&1; then
        for p in $(fuser "$port"/tcp 2>/dev/null | tr -s ' ' '\n' | grep -E '^[0-9]+$'); do
            pids+=("$p")
        done
    fi

    # 3. ss 检测
    if command -v ss >/dev/null 2>&1; then
        for p in $(ss -lptn "sport = :$port" 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2); do
            pids+=("$p")
        done
    fi

    # 4. netstat 检测
    if command -v netstat >/dev/null 2>&1; then
        for p in $(netstat -tlpn 2>/dev/null | grep ":$port " | awk '{print $7}' | cut -d'/' -f1 | grep -E '^[0-9]+$'); do
            pids+=("$p")
        done
    fi

    # 输出唯一去重的 PID
    printf "%s\n" "${pids[@]}" 2>/dev/null | sort -u | xargs
}

# 检查端口是否处于监听状态
is_port_listening() {
    local port="$1"
    local pids
    pids="$(get_pids_by_port "$port")"
    [ -n "$pids" ]
}

# 停止可能存在的后台托管守护进程（防止 omp/supervisor 守护进程自动无限重启）
stop_supervised_daemon() {
    local port="$1"
    if command -v omp >/dev/null 2>&1; then
        if [ "$port" = "8088" ]; then
            omp ps stop abot_viewer_8088 >/dev/null 2>&1 || true
            omp ps stop viewer_8088 >/dev/null 2>&1 || true
        elif [ "$port" = "8090" ]; then
            omp ps stop streaming_api_8090 >/dev/null 2>&1 || true
        fi
    fi
}

# ==============================================================================
# 服务控制函数
# ==============================================================================

# 启动 8088 端口（3D 可视化 Web 服务）
start_8088() {
    local host="${HOST:-$DEFAULT_HOST}"
    local port="${PORT_8088}"
    local device="${DEVICE_8088:-$(detect_best_device)}"

    echo -e "${CYAN}[8088] 检查 3D Visualizer 服务状态...${NC}"
    if is_port_listening "$port"; then
        local pids
        pids="$(get_pids_by_port "$port")"
        echo -e "${YELLOW}[8088] 端口已被占用 (PID: ${pids})，跳过启动。如需重启请执行 stop 8088 后再 start。${NC}"
        return 0
    fi

    echo -e "${CYAN}[8088] 正在启动 3D Visualizer (viewer/server.py)...${NC}"
    echo -e "       Host: ${host}, Port: ${port}, Device: ${device}"
    echo -e "       Python: ${PYTHON_BIN}"
    echo -e "       Log   : ${LOG_FILE_8088}"

    nohup "$PYTHON_BIN" viewer/server.py --host "$host" --port "$port" --device "$device" > "$LOG_FILE_8088" 2>&1 &
    local pid=$!
    echo "$pid" > "$PID_FILE_8088"

    # 等待验证启动并完成 GPU 预热 (最多等待 15 秒)
    local retries=15
    local started=0
    echo -n "       等待服务就绪与 GPU 预热..."
    while [ $retries -gt 0 ]; do
        sleep 1
        echo -n "."
        if is_port_listening "$port"; then
            started=1
            break
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            break
        fi
        ((retries--))
    done
    echo ""

    if [ $started -eq 1 ]; then
        echo -e "${GREEN}[8088] 3D Visualizer 启动成功! (PID: ${pid}, 访问地址: http://${host}:${port})${NC}"
    else
        echo -e "${RED}[8088] 启动失败或未在规定时间内就绪，请检查日志: ${LOG_FILE_8088}${NC}"
        tail -n 15 "$LOG_FILE_8088"
    fi
}

# 启动 8090 端口（实时流式重构 API 服务）
start_8090() {
    local host="${HOST:-$DEFAULT_HOST}"
    local port="${PORT_8090}"
    local device="${DEVICE_8090:-$(detect_best_device)}"

    echo -e "${CYAN}[8090] 检查 Streaming API 服务状态...${NC}"
    if is_port_listening "$port"; then
        local pids
        pids="$(get_pids_by_port "$port")"
        echo -e "${YELLOW}[8090] 端口已被占用 (PID: ${pids})，跳过启动。如需重启请执行 stop 8090 后再 start。${NC}"
        return 0
    fi

    echo -e "${CYAN}[8090] 正在启动 Streaming API (viewer/streaming_api_server.py)...${NC}"
    echo -e "       Host: ${host}, Port: ${port}, Device: ${device}"
    echo -e "       Python: ${PYTHON_BIN}"
    echo -e "       Log   : ${LOG_FILE_8090}"

    nohup "$PYTHON_BIN" viewer/streaming_api_server.py --host "$host" --port "$port" --device "$device" --dynamic-filter --dynamic-model yolo11m-seg.pt --dynamic-conf 0.12 --dynamic-dilate 15 > "$LOG_FILE_8090" 2>&1 &
    local pid=$!
    echo "$pid" > "$PID_FILE_8090"

    # 等待验证启动 (最多等待 15 秒)
    local retries=15
    local started=0
    echo -n "       等待流式服务就绪..."
    while [ $retries -gt 0 ]; do
        sleep 1
        echo -n "."
        if is_port_listening "$port"; then
            started=1
            break
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            break
        fi
        ((retries--))
    done
    echo ""

    if [ $started -eq 1 ]; then
        echo -e "${GREEN}[8090] Streaming API 服务启动成功! (PID: ${pid}, 访问地址: http://${host}:${port})${NC}"
    else
        echo -e "${RED}[8090] 启动失败或未在规定时间内就绪，请检查日志: ${LOG_FILE_8090}${NC}"
        tail -n 15 "$LOG_FILE_8090"
    fi
}

# 彻底停止并释放指定端口与资源
stop_port() {
    local port="$1"
    local name="$2"
    local pid_file="$3"
    local script_pattern="$4"

    echo -e "${CYAN}[${port}] 正在停止 ${name} 并释放资源...${NC}"

    # 1. 首先通知守护进程停止（防止守护系统检测到子进程退出后自动重启）
    stop_supervised_daemon "$port"

    # 2. 收集所有相关 PID (包含 PID 文件中的进程、端口占用进程、脚本匹配进程)
    local target_pids=()

    if [ -f "$pid_file" ]; then
        local file_pid
        file_pid="$(cat "$pid_file" 2>/dev/null | tr -d ' ')"
        if [ -n "$file_pid" ] && kill -0 "$file_pid" 2>/dev/null; then
            target_pids+=("$file_pid")
        fi
        rm -f "$pid_file"
    fi

    # 端口占用进程
    local port_pids
    port_pids="$(get_pids_by_port "$port")"
    for p in $port_pids; do
        if [ -n "$p" ]; then
            target_pids+=("$p")
        fi
    done

    # 进程名模糊匹配（防止端口处于卡死未释放状态）
    if [ -n "$script_pattern" ] && command -v pgrep >/dev/null 2>&1; then
        for p in $(pgrep -f "$script_pattern" 2>/dev/null); do
            if [ -n "$p" ]; then
                target_pids+=("$p")
            fi
        done
    fi

    # 移除重复 PID
    local unique_pids
    unique_pids=($(printf "%s\n" "${target_pids[@]}" 2>/dev/null | sort -u))

    if [ ${#unique_pids[@]} -gt 0 ]; then
        echo -e "       发现目标进程: ${unique_pids[*]}"

        # 3. 尝试优雅停止 (SIGTERM)
        for p in "${unique_pids[@]}"; do
            if kill -0 "$p" 2>/dev/null; then
                kill "$p" 2>/dev/null || true
            fi
        done

        # 4. 等待进程退出及资源释放 (最多 3 秒)
        local max_wait=3
        while [ $max_wait -gt 0 ]; do
            sleep 0.6
            local still_alive=0
            for p in "${unique_pids[@]}"; do
                if kill -0 "$p" 2>/dev/null; then
                    still_alive=1
                    break
                fi
            done
            if [ $still_alive -eq 0 ] && ! is_port_listening "$port"; then
                break
            fi
            ((max_wait--))
        done
    fi

    # 5. 若仍有进程存活或端口仍被占用，执行多级强制终止 (kill -9 & fuser -k)
    if is_port_listening "$port" || [ ${#unique_pids[@]} -gt 0 ]; then
        # 强制杀死已知目标 PID
        for p in "${unique_pids[@]}"; do
            if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then
                echo -e "${YELLOW}[${port}] 正在强制终止进程 PID $p (kill -9)...${NC}"
                kill -9 "$p" 2>/dev/null || true
            fi
        done

        # 内核级套接字强制释放 (fuser -k -9)
        if command -v fuser >/dev/null 2>&1; then
            fuser -k -9 "$port"/tcp >/dev/null 2>&1 || true
        fi

        # 再次扫描端口残留进程
        local remaining_pids
        remaining_pids="$(get_pids_by_port "$port")"
        for p in $remaining_pids; do
            if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then
                echo -e "${YELLOW}[${port}] 强制清理端口残留 PID $p...${NC}"
                kill -9 "$p" 2>/dev/null || true
            fi
        done
    fi

    sleep 0.5

    # 6. 最终确认
    if is_port_listening "$port"; then
        local lingering
        lingering="$(get_pids_by_port "$port")"
        echo -e "${RED}[${port}] 警告: 端口 ${port} 仍有占用 (PID: ${lingering})，请检查是否具有权限或执行 sudo lsof -i :${port}${NC}"
    else
        echo -e "${GREEN}[${port}] ${name} 已完全停止，端口与显存资源已成功释放。${NC}"
    fi
}

stop_8088() {
    stop_port "$PORT_8088" "3D Visualizer" "$PID_FILE_8088" "viewer/server.py"
}

stop_8090() {
    stop_port "$PORT_8090" "Streaming API" "$PID_FILE_8090" "viewer/streaming_api_server.py"
}

# 查看状态
show_status() {
    echo -e "\n${BOLD}=================== ABot-Recon 服务运行状态 ===================${NC}"

    # 检查 8088
    echo -e "\n${BOLD}1. 3D Visualizer (端口 ${PORT_8088}):${NC}"
    local pids_8088
    pids_8088="$(get_pids_by_port "$PORT_8088")"
    if [ -n "$pids_8088" ]; then
        echo -e "   状态: ${GREEN}运行中 (RUNNING)${NC}"
        echo -e "   PID : ${pids_8088}"
        echo -e "   地址: http://${DEFAULT_HOST}:${PORT_8088}"
        echo -e "   日志: ${LOG_FILE_8088}"
    else
        echo -e "   状态: ${RED}已停止 (STOPPED)${NC}"
    fi

    # 检查 8090
    echo -e "\n${BOLD}2. Streaming API (端口 ${PORT_8090}):${NC}"
    local pids_8090
    pids_8090="$(get_pids_by_port "$PORT_8090")"
    if [ -n "$pids_8090" ]; then
        echo -e "   状态: ${GREEN}运行中 (RUNNING)${NC}"
        echo -e "   PID : ${pids_8090}"
        echo -e "   地址: http://${DEFAULT_HOST}:${PORT_8090}"
        echo -e "   日志: ${LOG_FILE_8090}"
    else
        echo -e "   状态: ${RED}已停止 (STOPPED)${NC}"
    fi

    # 检查 GPU 进程
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo -e "\n${BOLD}3. GPU 占用概况:${NC}"
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | while IFS=, read -r g_pid g_name g_mem; do
            g_pid=$(echo "$g_pid" | tr -d ' ')
            g_name=$(echo "$g_name" | tr -d ' ')
            g_mem=$(echo "$g_mem" | tr -d ' ')
            # 判断是否属于我们的服务
            local tag=""
            if [[ " $pids_8088 " =~ " $g_pid " ]]; then
                tag=" [8088 Visualizer]"
            elif [[ " $pids_8090 " =~ " $g_pid " ]]; then
                tag=" [8090 Streaming API]"
            fi
            echo -e "   - PID: ${g_pid} | 进程: ${g_name} | 显存: ${g_mem}${tag}"
        done
    fi
    echo -e "${BOLD}===============================================================${NC}\n"
}

# 跟踪日志
show_logs() {
    local target="$1"
    case "$target" in
        8088)
            echo -e "${CYAN}正在跟踪 8088 日志 (${LOG_FILE_8088})... 按 Ctrl+C 退出${NC}"
            tail -n 50 -f "$LOG_FILE_8088"
            ;;
        8090)
            echo -e "${CYAN}正在跟踪 8090 日志 (${LOG_FILE_8090})... 按 Ctrl+C 退出${NC}"
            tail -n 50 -f "$LOG_FILE_8090"
            ;;
        *)
            echo -e "${YELLOW}请指定要查看的日志端口: ./manage_services.sh logs 8088 或 8090${NC}"
            ;;
    esac
}

# 帮助信息
show_help() {
    echo -e "${BOLD}ABot-Recon 服务管理脚本${NC}"
    echo -e "用法: $0 <action> [target] [options]"
    echo -e "  或者: $0 [target] <action>\n"
    echo -e "${BOLD}常用操作 (action):${NC}"
    echo -e "  ${GREEN}start${NC}   [all|8088|8090]   启动指定服务（默认 all，同时启动 8088 与 8090）"
    echo -e "  ${RED}stop${NC}    [all|8088|8090]   停止指定服务并彻底释放端口、进程与显存资源（默认 all）"
    echo -e "  ${YELLOW}restart${NC} [all|8088|8090]   重启指定服务（默认 all）"
    echo -e "  ${BLUE}status${NC}                    查看当前 8088、8090 运行状态与 GPU 占用"
    echo -e "  ${CYAN}logs${NC}    <8088|8090>       实时查看指定服务的日志输出\n"
    echo -e "${BOLD}支持的环境变量与选项参数:${NC}"
    echo -e "  DEVICE=<cuda:0|cuda:1|cpu>  指定流式与可视化服务运行的计算设备 (默认自动选择显存最空闲 GPU)"
    echo -e "  HOST=<host_ip>              指定监听地址 (默认: ${DEFAULT_HOST})"
    echo -e "  --device <device>           指定 GPU 设备 (例如: --device cuda:1)"
    echo -e "  --host <host>               指定监听主机 (例如: --host 127.0.0.1)\n"
    echo -e "${BOLD}示例:${NC}"
    echo -e "  $0 start                 # 自动选择最空闲 GPU 并同时启动 8088 和 8090"
    echo -e "  $0 stop                  # 同时停止 8088 和 8090 并释放所有端口与显存资源"
    echo -e "  $0 stop 8088             # 彻底停止 8088 端口（亦兼容 $0 8088 stop）"
    echo -e "  $0 stop 8090             # 彻底停止 8090 端口（亦兼容 $0 8090 stop）"
    echo -e "  $0 restart 8088          # 单独重启 8088 可视化服务"
    echo -e "  $0 status                # 查看当前端口与进程状态"
    echo -e "  $0 logs 8088             # 跟踪 8088 日志"
}

# ==============================================================================
# 参数解析与执行入口（支持灵活的参数顺序）
# ==============================================================================

ARG1="${1:-help}"
ARG2="${2:-}"

# 智能识别参数顺序（如 "stop 8088" 或 "8088 stop"）
ACTION=""
TARGET=""

if [[ "$ARG1" =~ ^(start|stop|kill|down|close|restart|status|logs|help|--help|-h)$ ]]; then
    ACTION="$ARG1"
    TARGET="${ARG2:-all}"
elif [[ "$ARG2" =~ ^(start|stop|kill|down|close|restart|status|logs|help|--help|-h)$ ]]; then
    ACTION="$ARG2"
    TARGET="$ARG1"
elif [[ "$ARG1" =~ ^(8088|8090|all)$ ]]; then
    TARGET="$ARG1"
    ACTION="status"
else
    ACTION="$ARG1"
    TARGET="${ARG2:-all}"
fi

# 同义词归一化
case "$ACTION" in
    kill|down|close)
        ACTION="stop"
        ;;
esac

# 弹出前两个参数后，处理其余选项
shift || true
shift || true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

case "$ACTION" in
    start)
        case "$TARGET" in
            8088)
                start_8088
                ;;
            8090)
                start_8090
                ;;
            all|"")
                start_8088
                start_8090
                ;;
            *)
                echo -e "${RED}未知目标: $TARGET (支持: all, 8088, 8090)${NC}"
                exit 1
                ;;
        esac
        ;;
    stop)
        case "$TARGET" in
            8088)
                stop_8088
                ;;
            8090)
                stop_8090
                ;;
            all|"")
                stop_8088
                stop_8090
                ;;
            *)
                echo -e "${RED}未知目标: $TARGET (支持: all, 8088, 8090)${NC}"
                exit 1
                ;;
        esac
        ;;
    restart)
        case "$TARGET" in
            8088)
                stop_8088
                sleep 1
                start_8088
                ;;
            8090)
                stop_8090
                sleep 1
                start_8090
                ;;
            all|"")
                stop_8088
                stop_8090
                sleep 1
                start_8088
                start_8090
                ;;
            *)
                echo -e "${RED}未知目标: $TARGET (支持: all, 8088, 8090)${NC}"
                exit 1
                ;;
        esac
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs "$TARGET"
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}未知操作: $ACTION${NC}"
        show_help
        exit 1
        ;;
esac
