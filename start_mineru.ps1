# MinerU Startup Script for Windows PowerShell
# This script provides an easy way to start MinerU services

param(
    [Parameter(Position=0, Mandatory=$true)]
    [ValidateSet("web", "api", "sglang-server", "all", "help")]
    [string]$Command,

    [string]$VenvPath = ".\.venv",
    [string]$ServerHost = "0.0.0.0",
    [int]$WebPort = 7860,
    [int]$ApiPort = 8000,
    [int]$SglangPort = 30000,
    [switch]$EnableSglang,
    [switch]$EnableExamples,
    [int]$MaxPages = 1000
)

# Colors for output
$Red = [ConsoleColor]::Red
$Green = [ConsoleColor]::Green
$Yellow = [ConsoleColor]::Yellow
$Blue = [ConsoleColor]::Blue
$White = [ConsoleColor]::White

function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor $Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor $Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor $Red
}

function Write-Header {
    Write-Host "================================" -ForegroundColor $Blue
    Write-Host "    MinerU Startup Script" -ForegroundColor $Blue
    Write-Host "================================" -ForegroundColor $Blue
}

function Show-Help {
    Write-Host @"
MinerU Startup Script for Windows

Usage: .\start_mineru.ps1 [COMMAND] [OPTIONS]

Commands:
  web                 Start Gradio Web UI
  api                 Start FastAPI server
  sglang-server      Start SGLang server
  all                 Start all services (web + api)
  help               Show this help message

Options:
  -VenvPath PATH      Path to virtual environment (default: .\.venv)
  -ServerHost HOST    Host to bind to (default: 0.0.0.0)
  -WebPort PORT       Web UI port (default: 7860)
  -ApiPort PORT       API server port (default: 8000)
  -SglangPort PORT    SGLang server port (default: 30000)
  -EnableSglang       Enable SGLang engine for web UI
  -EnableExamples     Enable example files in web UI
  -MaxPages NUM       Maximum pages to convert (default: 1000)

Examples:
  .\start_mineru.ps1 web                                     # Start web UI with defaults
  .\start_mineru.ps1 web -WebPort 8080 -EnableSglang        # Start web UI on port 8080 with SGLang
  .\start_mineru.ps1 api -ApiPort 9000                       # Start API on port 9000
  .\start_mineru.ps1 all -VenvPath "C:\path\to\venv"        # Start all services with custom venv
  .\start_mineru.ps1 sglang-server -SglangPort 31000         # Start SGLang server on port 31000

"@
}

function Test-Command {
    param([string]$CommandName)
    try {
        Get-Command $CommandName -ErrorAction Stop | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

function Activate-VirtualEnvironment {
    param([string]$Path)

    $activateScript = Join-Path $Path "Scripts\Activate.ps1"

    if (Test-Path $activateScript) {
        Write-Status "Activating virtual environment: $Path"
        try {
            & $activateScript
        }
        catch {
            Write-Error "Failed to activate virtual environment: $_"
            exit 1
        }
    }
    else {
        Write-Error "Virtual environment activation script not found at: $activateScript"
        Write-Host "Please specify the correct path with -VenvPath parameter"
        exit 1
    }
}

function Start-WebUI {
    Write-Status "Starting MinerU Web UI..."
    Write-Status "Host: $ServerHost"
    Write-Status "Port: $WebPort"
    Write-Status "URL: http://${ServerHost}:${WebPort}"

    $cmd = @("mineru-gradio", "--server-name", $ServerHost, "--server-port", $WebPort, "--max-convert-pages", $MaxPages)

    if ($EnableSglang) {
        $cmd += @("--enable-sglang-engine", "true")
        Write-Status "SGLang engine enabled"
    }

    if ($EnableExamples) {
        $cmd += @("--enable-example", "true")
        Write-Status "Examples enabled"
    }

    Write-Status "Command: $($cmd -join ' ')"

    try {
        & $cmd[0] $cmd[1..($cmd.Length-1)]
    }
    catch {
        Write-Error "Failed to start Web UI: $_"
        exit 1
    }
}

function Start-API {
    Write-Status "Starting MinerU FastAPI server..."
    Write-Status "Host: $ServerHost"
    Write-Status "Port: $ApiPort"
    Write-Status "Docs URL: http://${ServerHost}:${ApiPort}/docs"

    $cmd = @("mineru-api", "--host", $ServerHost, "--port", $ApiPort)
    Write-Status "Command: $($cmd -join ' ')"

    try {
        & $cmd[0] $cmd[1..($cmd.Length-1)]
    }
    catch {
        Write-Error "Failed to start API server: $_"
        exit 1
    }
}

function Start-SglangServer {
    Write-Status "Starting SGLang server..."
    Write-Status "Port: $SglangPort"

    $cmd = @("mineru-sglang-server", "--port", $SglangPort)
    Write-Status "Command: $($cmd -join ' ')"

    try {
        & $cmd[0] $cmd[1..($cmd.Length-1)]
    }
    catch {
        Write-Error "Failed to start SGLang server: $_"
        exit 1
    }
}

function Start-AllServices {
    Write-Status "Starting all MinerU services..."
    Write-Warning "This will start multiple services"

    # Start API in background
    Write-Status "Starting FastAPI server in background..."
    $apiJob = Start-Job -ScriptBlock {
        param($ServerHost, $ApiPort)
        & mineru-api --host $ServerHost --port $ApiPort
    } -ArgumentList $ServerHost, $ApiPort

    # Wait a moment for API to start
    Start-Sleep -Seconds 3

    # Check if API job is running
    if ($apiJob.State -eq "Running") {
        Write-Status "API server started successfully in background (Job ID: $($apiJob.Id))"
        Write-Status "Use 'Get-Job' and 'Stop-Job $($apiJob.Id)' to manage the background API server"
    } else {
        Write-Warning "API server may not have started correctly"
    }

    # Start Web UI in foreground
    Write-Status "Starting Web UI in foreground..."
    Start-WebUI
}

# Main execution
if ($Command -eq "help") {
    Show-Help
    exit 0
}

Write-Header

# Activate virtual environment
Activate-VirtualEnvironment -Path $VenvPath

# Check if MinerU commands are available
if (-not (Test-Command "mineru-gradio")) {
    Write-Error "MinerU commands not found. Make sure MinerU is installed in the virtual environment."
    Write-Host "You may need to run: pip install -e .[core]"
    exit 1
}

# Execute command
switch ($Command) {
    "web" { Start-WebUI }
    "api" { Start-API }
    "sglang-server" { Start-SglangServer }
    "all" { Start-AllServices }
    default {
        Write-Error "Unknown command: $Command"
        Show-Help
        exit 1
    }
}

Write-Status "Script completed."