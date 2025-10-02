#!/bin/bash

# MinerU Startup Script for Linux
# This script provides an easy way to start MinerU services

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
DEFAULT_VENV_PATH="./venv"
DEFAULT_HOST="0.0.0.0"
DEFAULT_WEB_PORT="7860"
DEFAULT_API_PORT="8000"
DEFAULT_SGLANG_PORT="30000"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}    MinerU Startup Script${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Function to activate virtual environment
activate_venv() {
    local venv_path=${1:-$DEFAULT_VENV_PATH}

    if [ -f "$venv_path/bin/activate" ]; then
        print_status "Activating virtual environment: $venv_path"
        source "$venv_path/bin/activate"
    elif [ -f "$venv_path/Scripts/activate" ]; then
        print_status "Activating virtual environment: $venv_path"
        source "$venv_path/Scripts/activate"
    else
        print_error "Virtual environment not found at: $venv_path"
        echo "Please specify the correct path with --venv-path option"
        exit 1
    fi
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check system requirements for Ubuntu 24.04
check_system_requirements() {
    # Check if we're on Ubuntu
    if [ -f /etc/os-release ]; then
        source /etc/os-release
        if [[ "$ID" == "ubuntu" ]]; then
            print_status "Detected Ubuntu $VERSION_ID"

            # Check for python3-venv package
            if ! dpkg -l | grep -q python3.*-venv; then
                print_warning "python3-venv package may not be installed"
                print_warning "If you encounter venv issues, run: sudo apt install python3.12-venv"
            fi
        fi
    fi
}

# Function to verify models are downloaded
check_models() {
    local config_file="$HOME/mineru.json"
    if [ ! -f "$config_file" ]; then
        print_warning "MinerU configuration not found at $config_file"
        print_warning "Models may not be downloaded. Run: mineru-models-download"
        return 1
    fi
    print_status "MinerU configuration found"
}

# Function to show help
show_help() {
    echo "MinerU Startup Script"
    echo ""
    echo "Usage: $0 [OPTIONS] COMMAND"
    echo ""
    echo "Commands:"
    echo "  web                 Start Gradio Web UI"
    echo "  api                 Start FastAPI server"
    echo "  sglang-server      Start SGLang server"
    echo "  all                 Start all services (web + api)"
    echo ""
    echo "Options:"
    echo "  --venv-path PATH    Path to virtual environment (default: ./venv)"
    echo "  --host HOST         Host to bind to (default: $DEFAULT_HOST)"
    echo "  --web-port PORT     Web UI port (default: $DEFAULT_WEB_PORT)"
    echo "  --api-port PORT     API server port (default: $DEFAULT_API_PORT)"
    echo "  --sglang-port PORT  SGLang server port (default: $DEFAULT_SGLANG_PORT)"
    echo "  --enable-sglang     Enable SGLang engine for web UI"
    echo "  --enable-examples   Enable example files in web UI"
    echo "  --max-pages NUM     Maximum pages to convert (default: 1000)"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 web                                    # Start web UI with defaults"
    echo "  $0 web --web-port 8080 --enable-sglang   # Start web UI on port 8080 with SGLang"
    echo "  $0 api --api-port 9000                    # Start API on port 9000"
    echo "  $0 all --venv-path /path/to/venv          # Start all services with custom venv"
    echo "  $0 sglang-server --sglang-port 31000      # Start SGLang server on port 31000"
}

# Parse command line arguments
VENV_PATH="$DEFAULT_VENV_PATH"
HOST="$DEFAULT_HOST"
WEB_PORT="$DEFAULT_WEB_PORT"
API_PORT="$DEFAULT_API_PORT"
SGLANG_PORT="$DEFAULT_SGLANG_PORT"
ENABLE_SGLANG=""
ENABLE_EXAMPLES=""
MAX_PAGES="1000"
COMMAND=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --venv-path)
            VENV_PATH="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --web-port)
            WEB_PORT="$2"
            shift 2
            ;;
        --api-port)
            API_PORT="$2"
            shift 2
            ;;
        --sglang-port)
            SGLANG_PORT="$2"
            shift 2
            ;;
        --enable-sglang)
            ENABLE_SGLANG="--enable-sglang-engine true"
            shift
            ;;
        --enable-examples)
            ENABLE_EXAMPLES="--enable-example true"
            shift
            ;;
        --max-pages)
            MAX_PAGES="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        web|api|sglang-server|all)
            COMMAND="$1"
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if command is provided
if [ -z "$COMMAND" ]; then
    print_error "No command specified"
    show_help
    exit 1
fi

print_header

# Check system requirements
check_system_requirements

# Activate virtual environment
activate_venv "$VENV_PATH"

# Check if MinerU commands are available
if ! command_exists mineru-gradio; then
    print_error "MinerU commands not found. Make sure MinerU is installed in the virtual environment."
    exit 1
fi

# Check if models are downloaded
check_models

# Function to start web UI
start_web() {
    print_status "Starting MinerU Web UI..."
    print_status "Host: $HOST"
    print_status "Port: $WEB_PORT"
    print_status "URL: http://$HOST:$WEB_PORT"

    # Fix for CUDA multiprocessing crash with fork() - use NVML-based CUDA check
    export PYTORCH_NVML_BASED_CUDA_CHECK=1

    local cmd="mineru-gradio --server-name $HOST --server-port $WEB_PORT --max-convert-pages $MAX_PAGES"

    if [ -n "$ENABLE_SGLANG" ]; then
        cmd="$cmd $ENABLE_SGLANG"
        print_status "SGLang engine enabled"
    fi

    if [ -n "$ENABLE_EXAMPLES" ]; then
        cmd="$cmd $ENABLE_EXAMPLES"
        print_status "Examples enabled"
    fi

    print_status "Command: $cmd"
    exec $cmd
}

# Function to start API
start_api() {
    print_status "Starting MinerU FastAPI server..."
    print_status "Host: $HOST"
    print_status "Port: $API_PORT"
    print_status "Docs URL: http://$HOST:$API_PORT/docs"

    # Fix for CUDA multiprocessing crash with fork() - use NVML-based CUDA check
    export PYTORCH_NVML_BASED_CUDA_CHECK=1

    local cmd="mineru-api --host $HOST --port $API_PORT"
    print_status "Command: $cmd"
    exec $cmd
}

# Function to start SGLang server
start_sglang() {
    print_status "Starting SGLang server..."
    print_status "Port: $SGLANG_PORT"

    # Fix for CUDA multiprocessing crash with fork() - use NVML-based CUDA check
    export PYTORCH_NVML_BASED_CUDA_CHECK=1

    local cmd="mineru-sglang-server --port $SGLANG_PORT"
    print_status "Command: $cmd"
    exec $cmd
}

# Function to start all services
start_all() {
    print_status "Starting all MinerU services..."
    print_warning "This will start multiple services in the background"

    # Start API in background
    print_status "Starting FastAPI server in background..."
    mineru-api --host "$HOST" --port "$API_PORT" &
    API_PID=$!

    # Wait a moment for API to start
    sleep 2

    # Start Web UI in foreground
    print_status "Starting Web UI..."
    start_web
}

# Execute command
case $COMMAND in
    web)
        start_web
        ;;
    api)
        start_api
        ;;
    sglang-server)
        start_sglang
        ;;
    all)
        start_all
        ;;
    *)
        print_error "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac