<#
.SYNOPSIS
Installs the folo CLI (and, with -Desktop, FOLO Desktop) from a GitHub release.

.DESCRIPTION
Downloads the Windows x64 archive of a release, verifies its SHA-256 against the published .sha256 file and
installs folo.exe into a user bin directory, which is added to the user PATH. The binary embeds FOLO Studio, so
`folo serve` works right away. Works in Windows PowerShell 5.1 and PowerShell 7.

Every parameter also reads an environment variable: FOLO_VERSION, FOLO_INSTALL_DIR, FOLO_REPO, FOLO_BASE_URL.
Private repositories need a token: GH_TOKEN or GITHUB_TOKEN, else the one from `gh auth token` when gh is logged in.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File scripts\install.ps1

.EXAMPLE
powershell -ExecutionPolicy Bypass -File scripts\install.ps1 -Version v0.2.0 -Desktop
#>
[CmdletBinding()]
param(
    # Release tag, e.g. v0.2.0, or "latest".
    [string]$Version = $(if ($env:FOLO_VERSION) { $env:FOLO_VERSION } else { 'latest' }),
    # Where folo.exe goes [default: %LOCALAPPDATA%\Programs\folo\bin].
    [string]$InstallDir = $(if ($env:FOLO_INSTALL_DIR) { $env:FOLO_INSTALL_DIR } else { Join-Path $env:LOCALAPPDATA 'Programs\folo\bin' }),
    # GitHub repository.
    [string]$Repo = $(if ($env:FOLO_REPO) { $env:FOLO_REPO } else { 'himent12/folo' }),
    # Download <BaseUrl>/<asset> instead of GitHub: an http(s) or file:// URL, or a local folder.
    [string]$BaseUrl = $env:FOLO_BASE_URL,
    # Also install FOLO Desktop (folo-desktop.exe).
    [switch]$Desktop,
    # Do not add the install directory to the user PATH.
    [switch]$NoModifyPath
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue' # the progress bar makes Invoke-WebRequest very slow in 5.1

function Say([string]$Message) { [Console]::Error.WriteLine("folo-install: $Message") }

# Downloads $Url to $Path, following redirects by hand so the Authorization header only goes to the host it was
# meant for (GitHub redirects asset downloads to a storage host that rejects it).
function Get-Download([string]$Url, [string]$Path, [hashtable]$Headers = @{}) {
    Add-Type -AssemblyName System.Net.Http
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    $handler = New-Object System.Net.Http.HttpClientHandler
    $handler.AllowAutoRedirect = $false
    $client = New-Object System.Net.Http.HttpClient($handler)
    try {
        $uri = [Uri]$Url
        $authHost = $uri.Host
        for ($hop = 0; $hop -lt 10; $hop++) {
            $req = New-Object System.Net.Http.HttpRequestMessage([System.Net.Http.HttpMethod]::Get, $uri)
            $req.Headers.TryAddWithoutValidation('User-Agent', 'folo-install') | Out-Null
            foreach ($k in $Headers.Keys) {
                if ($k -eq 'Authorization' -and $uri.Host -ne $authHost) { continue }
                $req.Headers.TryAddWithoutValidation($k, [string]$Headers[$k]) | Out-Null
            }
            $res = $client.SendAsync($req).GetAwaiter().GetResult()
            $code = [int]$res.StatusCode
            if ($code -ge 300 -and $code -lt 400 -and $res.Headers.Location) {
                $uri = New-Object Uri($uri, $res.Headers.Location)
                $res.Dispose()
                continue
            }
            if (-not $res.IsSuccessStatusCode) { throw "GET $uri answered $code $($res.ReasonPhrase)" }
            $out = [IO.File]::Create($Path)
            try { $res.Content.CopyToAsync($out).GetAwaiter().GetResult() } finally { $out.Dispose(); $res.Dispose() }
            return
        }
        throw "too many redirects for $Url"
    } finally {
        $client.Dispose()
    }
}

# The first field of a sha256sum-style line ("<hex>  <file>"), lowercased.
function Read-Checksum([string]$Path) {
    $line = (Get-Content -LiteralPath $Path -TotalCount 1)
    if (-not $line) { return '' }
    return ($line.Trim() -split '\s+')[0].ToLowerInvariant()
}

# True when $Dir is one of the ;-separated entries of $PathValue (case and trailing \ insensitive).
function Test-PathEntry([string]$PathValue, [string]$Dir) {
    $want = $Dir.TrimEnd('\')
    foreach ($entry in ($PathValue -split ';')) {
        if ($entry -and [Environment]::ExpandEnvironmentVariables($entry).TrimEnd('\') -ieq $want) { return $true }
    }
    return $false
}

# Copies $Source to $Dir\$Name. A running exe cannot be overwritten but can be renamed, so the old one is moved
# aside first and removed when possible (one still running is removed by a later install).
function Install-Exe([string]$Source, [string]$Dir, [string]$Name) {
    New-Item -ItemType Directory -Force -Path $Dir | Out-Null
    $dest = Join-Path $Dir $Name
    $tmp = "$dest.new"
    Copy-Item -LiteralPath $Source -Destination $tmp -Force
    Get-ChildItem -LiteralPath $Dir -Filter "$Name.old-*" | Remove-Item -Force -ErrorAction SilentlyContinue
    if (Test-Path -LiteralPath $dest) {
        $old = "$dest.old-" + [Guid]::NewGuid().ToString('N').Substring(0, 8)
        Move-Item -LiteralPath $dest -Destination $old
        Remove-Item -LiteralPath $old -Force -ErrorAction SilentlyContinue
    }
    Move-Item -LiteralPath $tmp -Destination $dest -Force
    return $dest
}

function Add-UserPath([string]$Dir) {
    $key = [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey('Environment', $true)
    try {
        # Read and write the raw value so entries like %USERPROFILE%\bin stay unexpanded.
        $current = [string]$key.GetValue('Path', '', [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
        if (Test-PathEntry $current $Dir) { return $false }
        $new = if ($current) { "$($current.TrimEnd(';'));$Dir" } else { $Dir }
        $key.SetValue('Path', $new, [Microsoft.Win32.RegistryValueKind]::ExpandString)
    } finally {
        $key.Dispose()
    }
    # Setting any user variable broadcasts WM_SETTINGCHANGE, so new terminals see the new PATH.
    [Environment]::SetEnvironmentVariable('FOLO_INSTALL_PATH_REFRESH', '1', 'User')
    [Environment]::SetEnvironmentVariable('FOLO_INSTALL_PATH_REFRESH', $null, 'User')
    return $true
}

function Main {
    $arch = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }
    switch ($arch) {
        'AMD64' { }
        'ARM64' { Say 'no native ARM64 build yet; installing the x64 build, which Windows on ARM runs emulated' }
        default { throw "no prebuilt folo for $arch Windows; build from source with cargo (see README.md)" }
    }
    $target = 'x86_64-pc-windows-msvc'
    $assets = @("folo-$target.zip")
    if ($Desktop) { $assets += "folo-desktop-$target.zip" }

    $token = if ($env:GH_TOKEN) { $env:GH_TOKEN } elseif ($env:GITHUB_TOKEN) { $env:GITHUB_TOKEN } else { $null }
    if (-not $token -and -not $BaseUrl -and (Get-Command gh -ErrorAction SilentlyContinue)) {
        # In 5.1 a native command's stderr turns into errors, which 'Stop' would make fatal.
        $prev = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            $token = (& gh auth token 2>$null | Out-String).Trim()
            if ($LASTEXITCODE -ne 0) { $token = $null }
        } catch {
            $token = $null
        } finally {
            $ErrorActionPreference = $prev
        }
    }

    $release = $null
    if ($BaseUrl) {
        $from = $BaseUrl
    } elseif ($token) {
        $api = "https://api.github.com/repos/$Repo/releases/" + $(if ($Version -eq 'latest') { 'latest' } else { "tags/$Version" })
        $headers = @{ Authorization = "Bearer $token"; Accept = 'application/vnd.github+json'; 'User-Agent' = 'folo-install' }
        try {
            $release = Invoke-RestMethod -UseBasicParsing -Uri $api -Headers $headers
        } catch {
            throw "could not read release $Version of $Repo (is it published, and can this token read the repository?): $_"
        }
        $from = "github.com/$Repo ($($release.tag_name))"
    } else {
        $from = "github.com/$Repo ($Version)"
    }

    # Fetches one release asset into $Path.
    function Get-Asset([string]$Name, [string]$Path) {
        if ($BaseUrl) {
            if ($BaseUrl -match '^https?://') { Get-Download "$($BaseUrl.TrimEnd('/'))/$Name" $Path; return }
            $dir = if ($BaseUrl -match '^file://') { ([Uri]$BaseUrl).LocalPath } else { $BaseUrl }
            Copy-Item -LiteralPath (Join-Path $dir $Name) -Destination $Path
        } elseif ($release) {
            $asset = $release.assets | Where-Object { $_.name -eq $Name } | Select-Object -First 1
            if (-not $asset) { throw "release $($release.tag_name) of $Repo has no asset $Name" }
            Get-Download $asset.url $Path @{ Authorization = "Bearer $token"; Accept = 'application/octet-stream' }
        } else {
            $dl = if ($Version -eq 'latest') { "https://github.com/$Repo/releases/latest/download" } else { "https://github.com/$Repo/releases/download/$Version" }
            try { Get-Download "$dl/$Name" $Path } catch { throw "download of $dl/$Name failed (a private repository needs GH_TOKEN or a logged-in gh): $_" }
        }
    }

    $tmp = Join-Path ([IO.Path]::GetTempPath()) ("folo-install-" + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $tmp | Out-Null
    try {
        foreach ($asset in $assets) {
            Say "downloading $asset from $from"
            $zip = Join-Path $tmp $asset
            Get-Asset $asset $zip
            Get-Asset "$asset.sha256" "$zip.sha256"
            $want = Read-Checksum "$zip.sha256"
            $got = (Get-FileHash -Algorithm SHA256 -LiteralPath $zip).Hash.ToLowerInvariant()
            if (-not $want -or $want -ne $got) { throw "checksum mismatch for ${asset}: expected $want, got $got" }
            Say "checksum ok ($got)"
            $unpacked = Join-Path $tmp ([IO.Path]::GetFileNameWithoutExtension($asset))
            Expand-Archive -LiteralPath $zip -DestinationPath $unpacked -Force
            $exe = if ($asset -like 'folo-desktop-*') { 'folo-desktop.exe' } else { 'folo.exe' }
            $bin = Get-ChildItem -LiteralPath $unpacked -Recurse -Filter $exe | Select-Object -First 1
            if (-not $bin) { throw "$asset does not contain $exe" }
            $dest = Install-Exe $bin.FullName $InstallDir $exe
            Say "installed $dest"
        }
    } finally {
        Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
    }

    $folo = Join-Path $InstallDir 'folo.exe'
    & $folo --version
    if ($LASTEXITCODE -ne 0) { Say "warning: $folo --version failed" }

    if (-not (Test-PathEntry $env:Path $InstallDir)) {
        if ($NoModifyPath) {
            Say "add $InstallDir to your PATH to run folo from any terminal"
        } else {
            if (Add-UserPath $InstallDir) { Say "added $InstallDir to your user PATH; open a new terminal to use it" }
            $env:Path = "$env:Path;$InstallDir"
        }
    }
}

# Dot-sourcing (`. .\install.ps1`) only defines the functions, for tests.
if ($MyInvocation.InvocationName -ne '.') { Main }
