# 数据集说明

本目录放的是 `CompassFusion_core_code_v0_1` 随包携带的真实输入数据样例，不是算法输出结果。

## 数据集：great_msf_20211013

来源：当前工程已经用于 GNSS/INS 组合测试的 GREAT/MSF 2021-10-13 数据。

### GNSS 文件

- `GNSS/SEPT2860.21O`：流动站 RINEX 观测文件，动态站观测数据。
- `GNSS/R2932860.21o`：参考站/基站 RINEX 观测文件，可用于 RTK/PPK 或差分处理。
- `GNSS/brdm2860.21p`：广播星历导航文件，可用于 SPP、PPK 广播星历解算和伪距紧耦合输入生成。

### IMU 文件

- `IMU/smallimu_out_2.txt`：车载 IMU 原始数据，用于 INS 机械编排、GNSS/INS 松耦合和紧耦合实验。

### 真值文件

- `groundtruth/groundtruth_211013_GNSS.txt`：GNSS 轨迹真值/参考轨迹，主要用于 GNSS 位置结果对比。
- `groundtruth/groundtruth_211013_ADIS.txt`：GNSS/INS 参考轨迹，包含组合导航评估需要的位置、速度、姿态参考。

### 精密产品文件

- `products/sp3/COD0MGXFIN_20212860000_01D_05M_ORB.SP3`：CODE MGEX 精密轨道，用于 PPP/精密轨道解算。
- `products/clk/COD0MGXFIN_20212860000_01D_30S_CLK.CLK`：CODE MGEX 精密钟差，用于 PPP/精密钟差改正。
- `products/bia/COD0MGXFIN_20212860000_01D_01D_OSB.BIA`：CODE MGEX OSB 偏差产品，用于多频/多系统 PPP 偏差改正。
- `products/bia/COD0MGXFIN_20212860000_01D_15M_ATT.OBX`：CODE MGEX 姿态/ORBEX 辅助产品，用于需要卫星姿态信息的精密模型。
- `products/erp/COD0MGXFIN_20212860000_03D_12H_ERP.ERP`：地球自转参数产品。
- `products/dcb/`：CODE 月度 DCB 产品目录，用于 OSB 不完整时的码偏差补充/回退。

### 模型辅助文件

- `model/igs_absolute_14.atx`：天线相位中心模型。
- `model/oceanload`：海潮负荷模型。
- `model/poleut1_great`：地球自转参数。
- `model/leap_seconds`：闰秒文件。
- `model/ocean_tide`：海潮模型。
- `model/jpleph_de405_great`：JPL DE405 星历文件，供部分潮汐/天体模型使用。

### GREAT 原始 XML 示例

- `xml/great-211013-lcrtk.xml`：GREAT 松耦合 RTK 示例配置。
- `xml/great-211013-lcppp.xml`：GREAT 松耦合 PPP 示例配置。
- `xml/great-211013-tcrtk.xml`：GREAT 紧耦合 RTK 示例配置。
- `xml/great-211013-tcppp.xml`：GREAT 紧耦合 PPP 示例配置。

## CompassFusion 示例配置

包内配置文件：

- `configs/compass_fusion_great_msf_example.xml`

该配置已经把输入路径改为本包内的 `data_examples/great_msf_20211013/...`。在核心包根目录下运行时，可作为真实数据测试的起点。

示例：

```powershell
$env:PYTHONPATH = "$PWD\src"
& 'D:\annaconda\envs\BraVL\python.exe' src\run_compass_fusion.py --config configs\compass_fusion_great_msf_example.xml
```

输出默认写入：

- `outputs/compass_fusion_great_msf_loose.csv`

本包已经保留 `outputs/` 占位目录，用于写入示例输出。

## PPP 示例脚本

已提供一个使用随包精密产品的 PPP 示例脚本：

```powershell
scripts\run_ppp_great_msf_example.ps1
```

该脚本默认只跑前 600 个历元，输出到：

- `outputs/ppp_great_msf_example.csv`

## 注意

这套样例数据用于工程验证和软件演示。它不是完整数据仓库，但已经包含本样例日期可用的 SP3、CLK、OSB、ERP、ATX、EOP、海潮和 JPL 星历等常用产品。若切换日期或测站，需要替换为对应日期的数据和产品。



