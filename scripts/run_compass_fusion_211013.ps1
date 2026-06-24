param(
  [switch]$EpochAttitude,
  [switch]$VelocityAiding,
  [switch]$Check,
  [switch]$Verbose,
  [string]$Config = "configs\compass_fusion_211013.xml",
  [ValidateSet("", "loose", "tight", "mechanization")]
  [string]$Mode = "",
  [string]$Out = "",
  [double]$StartSow = 289371,
  [double]$EndSow = 293100
)

$Root = Split-Path $PSScriptRoot -Parent
Set-Location $Root

$py = if ($env:PYTHON) { $env:PYTHON } else { "D:\annaconda\envs\BraVL\python.exe" }
$dataset = "data\great_msf\MSF_20211013"

if (-not (Test-Path $dataset)) {
  New-Item -ItemType Directory -Force -Path "data\great_msf" | Out-Null
  tar -xf "conference\GREAT-MSF-main\sample_data\MSF_20211013.zip" -C "data\great_msf"
}

if (-not $Out) {
  $suffix = if ($EpochAttitude) { "ieepoch" } elseif ($VelocityAiding) { "velaid" } else { "ieinit" }
  $Out = "data\great_msf\compass_fusion_SEPT2860_loose_full_$suffix`_lever.csv"
}

$argsList = @(
  "run_compass_fusion.py",
  "--config", $Config,
  "--out", $Out,
  "--start-sow", "$StartSow",
  "--end-sow", "$EndSow",
  "--output-rate", "1"
)

if ($Mode) {
  $argsList += @("--mode", $Mode)
}
if ($EpochAttitude) {
  $argsList += @("--attitude-mode", "epoch")
}
if ($VelocityAiding) {
  $argsList += @("--velocity-attitude-aiding", "--velocity-attitude-min-speed", "8.0", "--velocity-attitude-gain", "1.0")
}
if ($Check) {
  $argsList += @("--check")
}
if ($Verbose) {
  $argsList += @("--verbose")
}

& $py @argsList
$runExit = $LASTEXITCODE
if (-not $Check -and $runExit -eq 0) {
  & $py "scripts\compare_compass_fusion.py" $Out --truth "$dataset\groundtruth\groundtruth_211013_ADIS.txt"
  exit $LASTEXITCODE
}
exit $runExit
