# Loads .env into the process environment, then runs the given command.
# Usage: powershell -File scripts\demo\run_with_env.ps1 <command> [args...]
$root = Resolve-Path "$PSScriptRoot\..\.."
Get-Content "$root\.env" | Where-Object { $_ -match '^[A-Za-z_][A-Za-z0-9_]*=' } | ForEach-Object {
    $k, $v = $_ -split '=', 2
    Set-Item "env:$k" $v
}
$cmd = $args[0]
$rest = @($args | Select-Object -Skip 1)
& $cmd @rest
exit $LASTEXITCODE
