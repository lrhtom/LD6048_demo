$ErrorActionPreference = "Stop"
Set-Location "E:\uk_lrean\LD6048_Programming using AI\end_pj\code"

if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
throw "未检测到 winget，请先安装 App Installer（Microsoft Store）后重试。"
}

winget install --id RProject.R --exact --silent --accept-package-agreements --accept-source-agreements

$rHome = Get-ChildItem "$env:ProgramFiles\R" -Directory | Sort-Object Name -Descending | Select-Object -First 1
if (-not $rHome) { throw "R 安装目录未找到。" }

$rscript1 = Join-Path $rHome.FullName "bin\Rscript.exe"
$rscript2 = Join-Path $rHome.FullName "bin\x64\Rscript.exe"

if (Test-Path $rscript1) {
$rscript = $rscript1
} elseif (Test-Path $rscript2) {
$rscript = $rscript2
} else {
throw "Rscript.exe 未找到。"
}

& $rscript ".\section4_nn_structure_plot.R"

if (Test-Path ".\img\ai_data_img\section4_nn_structure.png") {
Write-Host "完成：网络结构图已生成 -> img\ai_data_img\section4_nn_structure.png"
} else {
throw "脚本执行后未发现输出图片。"
}