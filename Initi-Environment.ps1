param(
    [switch]$Recreate
)

$venvPath = ".venv"
$requirementsPath = "requirements.txt"

Write-Host "Python Virtual Environment Setup Script"

# Check for Python
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Python not found. Please install Python 3.8+ and try again."
    exit 1
}

# Optionally delete existing venv
if ($Recreate -and (Test-Path $venvPath)) {
    Write-Host "Removing existing virtual environment..."
    Remove-Item $venvPath -Recurse -Force
}

# Create venv if not present
if (-not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment in $venvPath..."
    python -m venv $venvPath
} else {
    Write-Host "Virtual environment already exists."
}

# Activate venv
$activateScript = "$venvPath\\Scripts\\Activate.ps1"
if (Test-Path $activateScript) {
    Write-Host "Activating virtual environment..."
    & $activateScript

    # Install requirements
    if (Test-Path $requirementsPath) {
        Write-Host "Installing dependencies from $requirementsPath..."
        python -m pip install --upgrade pip
        pip install -r $requirementsPath
        Write-Host "Dependencies installed."
    } else {
        Write-Host "No requirements.txt found. Skipping dependency installation."
    }
} else {
    Write-Host "Failed to locate venv activation script."
    exit 1
}
