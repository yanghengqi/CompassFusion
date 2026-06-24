$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $Root "src"
New-Item -ItemType Directory -Force -Path (Join-Path $Root "outputs") | Out-Null
& 'D:\annaconda\envs\BraVL\python.exe' `
  (Join-Path $Root "src\run_ppp.py") `
  --obs (Join-Path $Root "data_examples\great_msf_20211013\GNSS\SEPT2860.21O") `
  --nav (Join-Path $Root "data_examples\great_msf_20211013\GNSS\brdm2860.21p") `
  --sp3 (Join-Path $Root "data_examples\great_msf_20211013\products\sp3\COD0MGXFIN_20212860000_01D_05M_ORB.SP3") `
  --clk (Join-Path $Root "data_examples\great_msf_20211013\products\clk\COD0MGXFIN_20212860000_01D_30S_CLK.CLK") `
  --atx (Join-Path $Root "data_examples\great_msf_20211013\model\igs_absolute_14.atx") `
  --erp (Join-Path $Root "data_examples\great_msf_20211013\products\erp\COD0MGXFIN_20212860000_03D_12H_ERP.ERP") `
  --bia (Join-Path $Root "data_examples\great_msf_20211013\products\bia\COD0MGXFIN_20212860000_01D_01D_OSB.BIA") `
  --obx (Join-Path $Root "data_examples\great_msf_20211013\products\bia\COD0MGXFIN_20212860000_01D_15M_ATT.OBX") `
  --dcb-dir (Join-Path $Root "data_examples\great_msf_20211013\products\dcb") `
  --blq (Join-Path $Root "data_examples\great_msf_20211013\model\oceanload") `
  --station SEPT `
  --systems GEC `
  --max-epochs 600 `
  --filter-mode multipass `
  --out (Join-Path $Root "outputs\ppp_great_msf_example.csv")
