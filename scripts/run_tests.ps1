# Run all tests with coverage
param(
    [switch]$Benchmarks,
    [switch]$Integration,
    [switch]$Unit,
    [switch]$GPU,
    [switch]$Coverage
)

$testArgs = @()

if ($Benchmarks) {
    $testArgs += "-m", "benchmark"
}
elseif ($Integration) {
    $testArgs += "-m", "integration"
}
elseif ($Unit) {
    $testArgs += "-m", "not (integration or benchmark or gpu)"
}
elseif ($GPU) {
    $testArgs += "-m", "gpu"
}

if ($Coverage) {
    $testArgs += "--cov=memory_gate",
    "--cov-report=term-missing:skip-covered",
    "--cov-report=html",
    "--cov-report=xml",
    "--cov-fail-under=85"
}

if ($testArgs.Count -eq 0) {
    # Run all tests by default
    pytest
}
else {
    pytest @testArgs
}
