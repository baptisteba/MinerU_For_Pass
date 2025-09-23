# MinerU Startup Scripts

This directory contains easy-to-use startup scripts for MinerU services on both Linux and Windows platforms.

## Files

- `start_mineru.sh` - Linux/macOS Bash script
- `start_mineru.ps1` - Windows PowerShell script

## Prerequisites

1. MinerU must be installed in a Python virtual environment
2. The virtual environment should be located at `./.venv` (or specify custom path)
3. Required models should be downloaded (`mineru-models-download`)

## Linux/macOS Usage

### Make script executable (first time only)
```bash
chmod +x start_mineru.sh
```

### Basic usage
```bash
# Start Web UI (default: http://0.0.0.0:7860)
./start_mineru.sh web

# Start API server (default: http://0.0.0.0:8000)
./start_mineru.sh api

# Start SGLang server (default: port 30000)
./start_mineru.sh sglang-server

# Start both Web UI and API
./start_mineru.sh all
```

### Advanced options
```bash
# Custom ports and SGLang enabled
./start_mineru.sh web --web-port 8080 --enable-sglang --enable-examples

# Custom virtual environment path
./start_mineru.sh web --venv-path /path/to/your/venv

# API with custom port
./start_mineru.sh api --api-port 9000

# Help
./start_mineru.sh --help
```

## Windows Usage

### Basic usage (run in PowerShell)
```powershell
# Start Web UI (default: http://0.0.0.0:7860)
.\start_mineru.ps1 web

# Start API server (default: http://0.0.0.0:8000)
.\start_mineru.ps1 api

# Start SGLang server (default: port 30000)
.\start_mineru.ps1 sglang-server

# Start both Web UI and API
.\start_mineru.ps1 all
```

### Advanced options
```powershell
# Custom ports and SGLang enabled
.\start_mineru.ps1 web -WebPort 8080 -EnableSglang -EnableExamples

# Custom virtual environment path
.\start_mineru.ps1 web -VenvPath "C:\path\to\your\venv"

# Custom host (localhost only)
.\start_mineru.ps1 web -ServerHost "127.0.0.1"

# API with custom port
.\start_mineru.ps1 api -ApiPort 9000

# Help
.\start_mineru.ps1 help
```

## Common Options

| Option | Description | Default |
|--------|-------------|---------|
| Virtual environment path | Path to Python venv | `./.venv` |
| Host | Network interface to bind | `0.0.0.0` (all interfaces) |
| Web port | Gradio Web UI port | `7860` |
| API port | FastAPI server port | `8000` |
| SGLang port | SGLang server port | `30000` |
| Enable SGLang | Use SGLang engine in Web UI | `false` |
| Enable examples | Show example files in Web UI | `false` |
| Max pages | Maximum pages to convert | `1000` |

## Access URLs

Once started, you can access:

- **Web UI**: `http://localhost:7860` (or your custom port)
- **API Documentation**: `http://localhost:8000/docs` (or your custom port)
- **SGLang Server**: `http://localhost:30000` (or your custom port)

To access from other machines, replace `localhost` with your server's IP address.

## Troubleshooting

1. **"Virtual environment not found"**
   - Ensure the venv path is correct
   - Use `--venv-path` (Linux) or `-VenvPath` (Windows) to specify custom path

2. **"MinerU commands not found"**
   - Make sure MinerU is installed: `pip install -e .[core]`
   - Verify you're using the correct virtual environment

3. **"Port already in use"**
   - Use different ports with `--web-port`, `--api-port`, or `--sglang-port` options

4. **Permission denied (Linux)**
   - Make sure the script is executable: `chmod +x start_mineru.sh`

5. **PowerShell execution policy (Windows)**
   - If you get execution policy errors, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

## Examples

### Web UI with all features enabled
```bash
# Linux
./start_mineru.sh web --enable-sglang --enable-examples --max-pages 500

# Windows
.\start_mineru.ps1 web -EnableSglang -EnableExamples -MaxPages 500
```

### Production setup (custom ports)
```bash
# Linux
./start_mineru.sh all --web-port 80 --api-port 443

# Windows
.\start_mineru.ps1 all -WebPort 80 -ApiPort 443
```

### Development setup (localhost only)
```bash
# Linux
./start_mineru.sh web --host 127.0.0.1

# Windows
.\start_mineru.ps1 web -ServerHost "127.0.0.1"
```