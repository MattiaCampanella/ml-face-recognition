param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("upload", "download", "download-logs", "download-checkpoints", "download-wandb", "sync-wandb", "push", "pull", "install-ssh-key")]
    [string]$Action,

    [Parameter(Mandatory = $false)]
    [string]$Path,

    [Parameter(Mandatory = $false)]
    [switch]$IncludeArtifacts
)

$CLUSTER_USER = "cmpmtt02p16h163d"
$CLUSTER_HOST = "gcluster.dmi.unict.it"
$SSH_TARGET = "${CLUSTER_USER}@${CLUSTER_HOST}"
$LOCAL = $PSScriptRoot
$PROJECT_NAME = Split-Path -Leaf $LOCAL
$REMOTE_PROJECT_DIR = "~/$PROJECT_NAME"
$REMOTE = "${SSH_TARGET}:${REMOTE_PROJECT_DIR}"
$SSH_KEY = Join-Path $HOME ".ssh\id_ed25519"
$SSH_PUBKEY = Join-Path $HOME ".ssh\id_ed25519.pub"

function Test-RemoteHasContent([string]$remotePath) {
    $cmd = 'if [ -d "' + $remotePath + '" ] && [ "$(find "' + $remotePath + '" -mindepth 1 -print -quit 2>/dev/null)" ]; then echo 1; else echo 0; fi'
    $probe = Invoke-SshCommand $cmd
    return (($probe | Out-String).Trim() -eq "1")
}

function Get-SshArgs {
    return @(
        '-i', $SSH_KEY,
        '-o', 'IdentitiesOnly=yes',
        '-o', 'PreferredAuthentications=publickey',
        '-o', 'PasswordAuthentication=no',
        '-o', 'BatchMode=yes',
        '-o', 'ConnectTimeout=15'
    )
}

function Get-ScpArgs {
    return @(
        '-i', $SSH_KEY,
        '-o', 'IdentitiesOnly=yes',
        '-o', 'PreferredAuthentications=publickey',
        '-o', 'PasswordAuthentication=no',
        '-o', 'BatchMode=yes',
        '-o', 'ConnectTimeout=15'
    )
}

function Invoke-SshCommand([string]$Command) {
    $sshArgs = Get-SshArgs
    return & ssh @sshArgs $SSH_TARGET $Command
}

function Invoke-ScpCommand([string[]]$Arguments) {
    $scpArgs = Get-ScpArgs
    return & scp @scpArgs @Arguments
}

function InstallSshKey {
    if (-not (Test-Path $SSH_PUBKEY)) {
        Write-Host "SSH public key not found: $SSH_PUBKEY" -ForegroundColor Red
        Write-Host "Generate one first with: ssh-keygen -t ed25519 -f $HOME\.ssh\id_ed25519 -C \"$CLUSTER_USER@$CLUSTER_HOST\"" -ForegroundColor Yellow
        return
    }

    $pubKey = (Get-Content $SSH_PUBKEY -Raw).Trim()
    if (-not $pubKey) {
        Write-Host "SSH public key file is empty: $SSH_PUBKEY" -ForegroundColor Red
        return
    }

    Write-Host "Installing SSH key on cluster..." -ForegroundColor Cyan
    Write-Host "You may need to enter the cluster password once, only for this first installation." -ForegroundColor Gray

    $keyBytes = [System.Text.Encoding]::UTF8.GetBytes($pubKey)
    $keyBase64 = [Convert]::ToBase64String($keyBytes)
    $remoteCommand = @'
mkdir -p ~/.ssh; umask 077; touch ~/.ssh/authorized_keys; key=$(printf '%s' '__KEY__' | base64 -d); grep -qxF -- "$key" ~/.ssh/authorized_keys || printf '%s\n' "$key" >> ~/.ssh/authorized_keys; chmod 700 ~/.ssh; chmod 600 ~/.ssh/authorized_keys
'@.Replace('__KEY__', $keyBase64)
    ssh $SSH_TARGET $remoteCommand
    if ($LASTEXITCODE -eq 0) {
        Write-Host "SSH key installed. Test with: ssh $SSH_TARGET" -ForegroundColor Green
    } else {
        Write-Host "SSH key installation failed." -ForegroundColor Red
    }
}

function Upload {
    Write-Host "Uploading project code to cluster: ${REMOTE_PROJECT_DIR}" -ForegroundColor Cyan

    Get-ChildItem -Path $LOCAL -Directory -Recurse -Filter "__pycache__" |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

    Invoke-SshCommand "mkdir -p $REMOTE_PROJECT_DIR/experiments/configs $REMOTE_PROJECT_DIR/experiments/runs $REMOTE_PROJECT_DIR/logs $REMOTE_PROJECT_DIR/data $REMOTE_PROJECT_DIR/data/splits $REMOTE_PROJECT_DIR/pretrained_weights" | Out-Null

    $codeItems = @(
        "src",
        "cluster",
        "README.md",
        "environment.yml",
        "INSTRUCTIONS.md",
        "LICENSE",
        "experiments/configs",
        "data/splits"
    )

    foreach ($item in $codeItems) {
        $localPath = Join-Path $LOCAL $item
        if (-not (Test-Path $localPath)) {
            Write-Host "[SKIP] $item (not found)" -ForegroundColor Yellow
            continue
        }

        if (Test-Path $localPath -PathType Container) {
            Write-Host "Copying folder: $item" -ForegroundColor Cyan
            Invoke-ScpCommand @('-r', "$localPath/.", "${REMOTE}/$item/")
        } else {
            Write-Host "Copying file: $item" -ForegroundColor Cyan
            Invoke-ScpCommand @($localPath, "${REMOTE}/")
        }
    }

    if ($IncludeArtifacts) {
        $localWeightsDir = Join-Path $LOCAL "pretrained_weights"
        if (Test-Path $localWeightsDir -PathType Container) {
            Write-Host "Copying folder: pretrained_weights" -ForegroundColor Cyan
            Invoke-ScpCommand @('-r', "$localWeightsDir/.", "${REMOTE}/pretrained_weights/")
        } else {
            Write-Host "[SKIP] pretrained_weights (not found)" -ForegroundColor Yellow
        }

        $localRunsDir = Join-Path $LOCAL "experiments\runs"
        if (Test-Path $localRunsDir -PathType Container) {
            Write-Host "Copying folder: experiments/runs" -ForegroundColor Cyan
            Invoke-ScpCommand @('-r', "$localRunsDir/.", "${REMOTE}/experiments/runs/")
        } else {
            Write-Host "[SKIP] experiments/runs (not found)" -ForegroundColor Yellow
        }
    } else {
        Write-Host "[SKIP] Large artifacts not uploaded by default. Use -IncludeArtifacts to copy pretrained_weights and experiments/runs." -ForegroundColor Yellow
    }

    Write-Host "[OK] Upload completed." -ForegroundColor Green
}

function Download-RemoteDir($remoteSubpath, $localDest) {
    New-Item -ItemType Directory -Force -Path $localDest | Out-Null

    $remoteListCmd = "cd $REMOTE_PROJECT_DIR && find `"$remoteSubpath`" -mindepth 1 -maxdepth 1 -print"
    $remoteEntries = @(Invoke-SshCommand $remoteListCmd | Where-Object { $_ -and $_.Trim() })

    foreach ($remoteEntry in $remoteEntries) {
        $remoteEntry = $remoteEntry.Trim()
        Write-Host "x $remoteEntry" -ForegroundColor Gray
        Invoke-ScpCommand @('-rq', "${SSH_TARGET}:${REMOTE_PROJECT_DIR}/$remoteEntry", $localDest)
    }
}

function Download {
    param([string]$What = "all")

    switch ($What) {
        "all" {
            Write-Host "Downloading all outputs from cluster..." -ForegroundColor Cyan
            New-Item -ItemType Directory -Force -Path "experiments/logs" | Out-Null
            New-Item -ItemType Directory -Force -Path "experiments/runs" | Out-Null
            New-Item -ItemType Directory -Force -Path "logs" | Out-Null
            Download-RemoteDir "experiments/logs" "experiments/logs"
            Download-RemoteDir "experiments/runs" "experiments/runs"
            Download-RemoteDir "logs" "logs"
        }
        "logs" {
            Write-Host "Downloading logs and figures..." -ForegroundColor Cyan
            New-Item -ItemType Directory -Force -Path "experiments/logs" | Out-Null
            New-Item -ItemType Directory -Force -Path "logs" | Out-Null
            Download-RemoteDir "experiments/logs" "experiments/logs"
            Download-RemoteDir "logs" "logs"
        }
        "checkpoints" {
            Write-Host "Downloading checkpoints..." -ForegroundColor Cyan
            New-Item -ItemType Directory -Force -Path "experiments/runs" | Out-Null
            Download-RemoteDir "experiments/runs" "experiments/runs"
        }
        "wandb" {
            Write-Host "Downloading wandb offline runs..." -ForegroundColor Cyan
            New-Item -ItemType Directory -Force -Path "experiments/logs" | Out-Null
            Download-RemoteDir "experiments/logs" "experiments/logs"
        }
    }

    Write-Host "[OK] Download completed." -ForegroundColor Green
}

function SyncWandb {
    Write-Host "Syncing wandb offline runs to wandb.ai..." -ForegroundColor Cyan

    $venvActivate = Join-Path $LOCAL ".venv\Scripts\Activate.ps1"
    if (-not (Get-Command wandb -ErrorAction SilentlyContinue)) {
        if (Test-Path $venvActivate) {
            & $venvActivate
        }
        if (-not (Get-Command wandb -ErrorAction SilentlyContinue)) {
            Write-Host "wandb CLI not found. Install it with: pip install wandb" -ForegroundColor Red
            return
        }
    }

    $logsDir = Join-Path $LOCAL "experiments\logs"
    if (-not (Test-Path $logsDir)) {
        Write-Host "No experiments/logs found. Run download first." -ForegroundColor Red
        return
    }

    $wandbDirs = Get-ChildItem -Path $logsDir -Recurse -Directory -Filter "wandb" |
        Where-Object { (Get-ChildItem -Path $_.FullName -Directory -Filter "offline-run-*" -ErrorAction SilentlyContinue).Count -gt 0 }

    foreach ($wdir in $wandbDirs) {
        Get-ChildItem -Path $wdir.FullName -Directory -Filter "offline-run-*" | ForEach-Object {
            Write-Host "Syncing $($_.Name)..." -ForegroundColor Gray
            & wandb sync --include-synced $_.FullName
        }
    }
}

function Push {
    if (-not $Path) {
        Write-Host "Usage: .\sync_cluster.ps1 -Action push -Path <file-or-folder>" -ForegroundColor Red
        return
    }

    $localPath = Join-Path $LOCAL $Path
    if (-not (Test-Path $localPath)) {
        Write-Host "Not found: $Path" -ForegroundColor Red
        return
    }

    $remotePath = $Path -replace '\\', '/'
    $remoteDir = Split-Path $remotePath
    if (Test-Path $localPath -PathType Container) {
        Invoke-SshCommand "mkdir -p $REMOTE_PROJECT_DIR/$remotePath" | Out-Null
    } elseif ($remoteDir) {
        Invoke-SshCommand "mkdir -p $REMOTE_PROJECT_DIR/$remoteDir" | Out-Null
    }

    if (Test-Path $localPath -PathType Container) {
        Invoke-ScpCommand @('-rq', "$localPath/.", "${REMOTE}/$remotePath/")
    } else {
        Invoke-ScpCommand @('-q', $localPath, "${REMOTE}/$remotePath")
    }

    Write-Host "Pushed $Path -> cluster" -ForegroundColor Green
}

function Pull {
    if (-not $Path) {
        Write-Host "Usage: .\sync_cluster.ps1 -Action pull -Path <file-or-folder>" -ForegroundColor Red
        return
    }

    $remotePath = $Path -replace '\\', '/'
    $localPath = Join-Path $LOCAL $Path
    $localDir = Split-Path $localPath
    if ($localDir) {
        New-Item -ItemType Directory -Force -Path $localDir | Out-Null
    }
    Invoke-ScpCommand @('-rq', "${REMOTE}/$remotePath", $localPath)
    Write-Host "Pulled $Path <- cluster" -ForegroundColor Green
}

switch ($Action) {
    "upload"               { Upload }
    "download"             { Download -What "all" }
    "download-logs"        { Download -What "logs" }
    "download-checkpoints"  { Download -What "checkpoints" }
    "download-wandb"       { Download -What "wandb" }
    "sync-wandb"           { SyncWandb }
    "push"                 { Push }
    "pull"                 { Pull }
    "install-ssh-key"      { InstallSshKey }
}