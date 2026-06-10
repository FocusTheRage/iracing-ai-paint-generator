# Publish iracing-ai-paint-generator to GitHub and enable Pages.
# Prerequisites: GitHub CLI (gh) installed and authenticated — run: gh auth login

param(
    [string]$RepoName = "iracing-ai-paint-generator",
    [ValidateSet("public", "private")]
    [string]$Visibility = "public",
    [string]$ReleaseTag = "v1.0.0"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    Write-Error "GitHub CLI (gh) is not installed. Install from https://cli.github.com/"
}

$loggedIn = $true
try {
    gh auth status *> $null
} catch {
    $loggedIn = $false
}
if (-not $loggedIn) {
    Write-Host "Not logged in to GitHub. Complete the browser login when prompted..."
    gh auth login --hostname github.com --git-protocol https --web
}

$userJson = gh api user | ConvertFrom-Json
$username = $userJson.login
$userId = $userJson.id
$gitEmail = "$userId+$username@users.noreply.github.com"
git config user.name $username
git config user.email $gitEmail
Write-Host "GitHub user: $username (git author set to $gitEmail)"

# PowerShell treats git stderr as errors — parse remotes manually.
$remote = $null
$remoteLines = @(git remote 2>&1 | ForEach-Object { "$_" })
if ($remoteLines -contains "origin") {
    $remote = (git remote get-url origin 2>&1 | ForEach-Object { "$_" }) -join ""
}

if (-not $remote) {
    Write-Host "Creating repository $username/$RepoName ..."
    gh repo create $RepoName --$Visibility --source=. --remote=origin `
        --description "AI-assisted iRacing paint generator with UV template constraints"
} else {
    Write-Host "Remote already configured: $remote"
}

git branch -M main
git push -u origin main

Write-Host "Enabling GitHub Pages (Actions source)..."
try {
    gh api -X POST "repos/$username/$RepoName/pages" -f "build_type=workflow" *> $null
} catch {
    Write-Host "Pages already configured (or enable manually in repo Settings > Pages)."
}

gh workflow run pages.yml 2>$null

$zipUrl = "https://github.com/$username/$RepoName/archive/refs/heads/main.zip"
$pagesUrl = "https://$($username.ToLower()).github.io/$RepoName/"

Write-Host "Creating release $ReleaseTag (includes auto-generated source ZIP)..."
try {
    gh release view $ReleaseTag --repo "$username/$RepoName" *> $null
    Write-Host "Release $ReleaseTag already exists — skipping."
} catch {
    gh release create $ReleaseTag `
        --repo "$username/$RepoName" `
        --title "iRacing AI Paint Generator $ReleaseTag" `
        --notes "Initial public release. Download the ZIP below or clone the repo. See README for setup."
}

Write-Host ""
Write-Host "Repository:  https://github.com/$username/$RepoName"
Write-Host "Download ZIP: $zipUrl"
Write-Host "Releases:    https://github.com/$username/$RepoName/releases"
Write-Host "Pages URL:   $pagesUrl"
Write-Host "Check workflow: https://github.com/$username/$RepoName/actions"