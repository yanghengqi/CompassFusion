# CompassFusion Core Code v0.1

这是从 `F:\spp_standalone` 整理出来的核心代码版，用于后续继续开发、交付或单独归档。这个目录保留重要代码、配置、测试脚本、文档，并附带一套 GREAT/MSF 真实导航输入样例。

## 目录说明

- `src/`：主入口和核心 Python 包。
- `src/run_compass_fusion.py`：统一入口，读取 XML 配置后运行机械编排、松耦合或紧耦合。
- `src/compass/core/`：坐标、常量、类型等基础模块。
- `src/compass/gnss/`：SPP、PPP、PPK、卫星位置、精密产品、偏差模型等 GNSS 模块。
- `src/compass/ins/`：IMU 机械编排、松耦合、紧耦合核心实现。
- `src/compass/io/`：RINEX 读取与输入解析。
- `configs/`：可调参数配置模板。
- `scripts/`：真实数据导出、批量测试、结果对比和诊断脚本。
- `tests/`：当前主要回归测试。
- `docs/`：软件说明和发布记录。

## 推荐运行环境

当前工程验证使用：

```powershell
D:\annaconda\envs\BraVL\python.exe
```

## 快速运行

在本目录下运行：

```powershell
$env:PYTHONPATH = "$PWD\src"
& 'D:\annaconda\envs\BraVL\python.exe' src\run_compass_fusion.py --config configs\compass_fusion_211013.xml
```

也可以安装为可调用命令：

```powershell
& 'D:\annaconda\envs\BraVL\python.exe' -m pip install -e .
compass-fusion --config configs\compass_fusion_211013.xml
```

## 快速测试

```powershell
$env:PYTHONPATH = "$PWD\src"
& 'D:\annaconda\envs\BraVL\python.exe' -m pytest tests
```

注意：本包已经携带一套真实 RINEX、广播星历、IMU、真值和本样例日期的常用精密产品；若切换日期或测站，需要替换为对应日期的数据和产品。

## 当前定位

这个整理版是 `CompassFusion` 的核心代码包，适合作为后续发布版/工程版继续迭代。当前稳定功能以 GNSS、INS 机械编排、松耦合和伪距级紧耦合为主；PPP/RTK 载波相位紧耦合和模糊度固定仍属于后续增强方向。
## 随包示例数据

本包现在包含一套真实导航输入样例：`data_examples/great_msf_20211013/`。

里面包括流动站/基站 RINEX 观测、广播星历、精密轨道、精密钟差、OSB 偏差、IMU 原始数据、GNSS 真值、GNSS/INS 真值和常用模型辅助文件。详细说明见：`data_examples/DATASETS.md`。



