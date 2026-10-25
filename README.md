# machine-learning-in-c

一个用 C 语言实现的小型神经网络学习项目，参考《深度学习入门：基于 Python 的理论与实现》中的思路，同时练习矩阵运算、反向传播和 C 语言的内存管理。

项目已实现 MNIST 全连接网络的训练、模型保存与加载，以及基于 raylib 的手写数字画板。网络计算使用 CPU，无需安装 CUDA。

> 想先体验识别效果：完成环境准备和编译后，直接运行 `build-bat/bin/digit_app.exe`。仓库已有预训练模型，无需先转换数据或重新训练。

## 已实现的内容

- 基于连续内存和 `stride` 的矩阵表示及基本运算。
- Affine、Sigmoid 层的前向与反向传播。
- Softmax 与交叉熵损失、SGD 参数更新、mini-batch 训练。
- 通过层数组组合不同深度、宽度的全连接网络。
- MNIST IDX 数据读取、矩阵持久化和模型保存、加载。
- 可缩放的手写画板、实时预测和模型输入预览。
- 手写输入的前景裁剪、等比例缩放与灰度重心居中。

当前重点是全连接网络；卷积、池化和通用自动微分尚未实现。

## 运行环境

目前验证的环境如下：

| 项目 | 环境 |
| --- | --- |
| 操作系统与目标架构 | Windows x64 |
| 编译器 | MinGW-w64 GCC 14.2.0 |
| C 标准 | C11 |
| 图形库 | raylib 6.0，仓库已附带头文件和静态库 |
| 本文使用的终端 | PowerShell |

其他编译器版本尚未全面验证。当前构建脚本和附带的 raylib 库面向 Windows x64 MinGW 环境，不能直接视为 MSVC、32 位 Windows 或其他操作系统的构建包。

VS Code 是可选编辑器，不包含本项目需要的 GCC 编译器。只下载源码 ZIP，也仍然需要编译环境。

### 如果还没有 GCC

下面按 **Windows x64 + MinGW-Builds GCC 14.2.0** 给出一条完整的参考安装路线，与本项目当前已验证的工具链配置一致。已经能正常编译的用户可以跳过本节。

#### 第 1 步：下载并解压完整工具链

打开 [MinGW-Builds 14.2.0 发布页](https://github.com/niXman/mingw-builds-binaries/releases/tag/14.2.0-rt_v12-rev2)，在 **Assets** 中选择：

```text
x86_64-14.2.0-release-win32-seh-ucrt-rt_v12-rev2.7z
```

也可以点击[该工具链压缩包的直接下载链接](https://github.com/niXman/mingw-builds-binaries/releases/download/14.2.0-rt_v12-rev2/x86_64-14.2.0-release-win32-seh-ucrt-rt_v12-rev2.7z)。不要选择页面上的 `Source code`，它不是可直接使用的编译器。

文件名中的 `x86_64` 表示 64 位目标；`win32` 在这里表示线程模型，不表示这是 32 位编译器。

使用支持 `.7z` 的解压工具解压。若无法打开这个文件，可以先从 [7-Zip 官网](https://www.7-zip.org/download.html)安装 Windows x64 版本，再用它解压。

把解压得到的整个 `mingw64` 文件夹放到 `C:\`，使目录结构为：

```text
C:\mingw64\
├── bin\
│   ├── gcc.exe
│   ├── g++.exe
│   └── ...
├── include\
├── lib\
└── ...
```

关键是 **`C:\mingw64\bin\gcc.exe` 必须存在**。不要只复制 `gcc.exe` 或 `bin` 文件夹，也不要多套一层变成 `C:\mingw64\mingw64\bin\gcc.exe`。

如果 `C:\mingw64` 已经有其他工具链，不要直接覆盖它；可以使用别的目录，并把下文每一处 `$gccBin = ...` 都改为同一个实际的 `bin` 路径。

#### 第 2 步：打开 PowerShell，确认解压位置

按键盘上的 **Win** 键，输入 `PowerShell` 并打开。以下标记为 `powershell` 的代码块可以逐块复制到终端，粘贴后按 **Enter** 执行；示例输出不需要复制。设置当前用户的 PATH 无需以管理员身份运行。

先执行：

```powershell
$gccBin = 'C:\mingw64\bin'
Test-Path -LiteralPath (Join-Path $gccBin 'gcc.exe')
```

预期输出：

```text
True
```

如果输出 `False`，说明解压位置不对，或 `$gccBin` 与实际目录不一致。先找到真正的 `gcc.exe` 并修正路径，再继续。

接着直接用完整路径调用编译器，此时还不依赖 PATH：

```powershell
& (Join-Path $gccBin 'gcc.exe') --version
& (Join-Path $gccBin 'gcc.exe') -dumpmachine
```

第一条应显示包含 `14.2.0` 的 GCC 版本信息，第二条应输出：

```text
x86_64-w64-mingw32
```

命令开头的 `&` 是 PowerShell 的调用运算符，用来执行给定路径的程序。若目标是 `i686-w64-mingw32`，说明选到了 32 位工具链，应返回第 1 步检查下载的文件。

#### 第 3 步：让当前终端立即能使用 gcc

PATH 是系统查找命令时会搜索的目录列表。把编译器的 `bin` 放到当前终端的搜索路径前面：

```powershell
$gccBin = 'C:\mingw64\bin'
$env:Path = "$gccBin;$env:Path"
Get-Command gcc | Select-Object -ExpandProperty Source
gcc --version
gcc -dumpmachine
```

第一条查询应显示 `C:\mingw64\bin\gcc.exe`（或你自己设置的对应路径），后面应显示正确的版本和 64 位目标。

这次设置只对当前 PowerShell 及从它启动的程序生效。此时已可以在这个终端里进入项目目录并编译；若希望以后也能直接使用 `gcc`，继续下一步。

#### 第 4 步：保存到当前用户的 PATH

在同一个 PowerShell 中执行以下命令。它读取原有的**用户 PATH**，在尚未包含该目录时加入编译器路径，并保留原有条目：

```powershell
$gccBin = 'C:\mingw64\bin'
if (Test-Path -LiteralPath (Join-Path $gccBin 'gcc.exe')) {
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    if (($userPath -split ';') -notcontains $gccBin) {
        [Environment]::SetEnvironmentVariable('Path', "$gccBin;$userPath", 'User')
    }
} else {
    throw 'gcc.exe not found. Check $gccBin before updating PATH.'
}
```

执行成功时通常没有输出。这里保存的是当前用户的设置；当前终端已经通过第 3 步生效，重新打开的终端将读取保存后的环境。关于临时环境变量与持久设置的区别，可参考 [Microsoft PowerShell 环境变量说明](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_environment_variables)。

#### 第 5 步：重新打开终端，再验证一次

关闭当前终端程序，再从开始菜单重新打开 PowerShell。若使用 Windows Terminal 或 VS Code，请退出并重新启动整个应用，而不只是新增标签页或终端面板；已有应用可能仍保留旧环境。

在新终端执行：

```powershell
Get-Command gcc | Select-Object -ExpandProperty Source
gcc --version
gcc -dumpmachine
```

确认仍然指向正确的 `gcc.exe`，并输出 `x86_64-w64-mingw32`，即可进入下一节获取源码并编译。

如果仍提示找不到 `gcc`，可以先重复第 3 步，在当前终端继续编译，再检查持久设置是否成功。如果电脑上装过多个 GCC，可执行：

```powershell
Get-Command gcc -All | Select-Object Source
```

检查是否有旧版本或 32 位版本优先被选中。第 3 步会把本节选择的目录放到当前终端 PATH 的最前面，便于先确认使用的是哪一个编译器。

更多工具链来源见 [MinGW-w64 官方工具链列表](https://www.mingw-w64.org/downloads/)和 [GCC 官方二进制安装说明](https://gcc.gnu.org/install/binaries.html)。

### raylib 是否需要另外安装

通常不需要。仓库已经包含：

```text
raylib/
├── include/
│   ├── raylib.h
│   ├── raymath.h
│   └── rlgl.h
└── lib/
    └── libraylib.a
```

`build.bat` 已设置头文件和库的搜索路径，并链接 Windows 系统库。请保留这组配套文件，使用兼容的 64 位 MinGW-w64 工具链；不要只替换头文件或混入其他平台、其他编译器的库。

这里使用 raylib 静态库，生成的画板程序不需要另外提供 `raylib.dll`。不同工具链生成的程序可能有不同的运行库依赖，发布编译成品时仍需单独核验。

相关构建方式见 [raylib 官方 Windows 文档](https://github.com/raysan5/raylib/wiki/Working-on-Windows)。

## 获取源码与编译

安装 Git 后，可以执行：

```powershell
git clone https://github.com/Rex-YinRX/machine-learning-in-c.git
cd machine-learning-in-c
.\build.bat
```

没有 Git 也可以下载仓库 ZIP 并解压，然后在 PowerShell 中进入包含 `build.bat` 的目录，执行 `.\build.bat`。

**下文中的“项目根目录”，均指包含 `build.bat`、`CMakeLists.txt`、`programs`、`model` 和 `raylib` 的这个目录。** 当前脚本使用相对路径，必须先进入该目录再构建。请确认编译输出中没有错误，再运行生成的程序。

脚本会自动创建 `build-bat/bin/`，将四个程序统一输出到该目录。CMake 的产物保存在各自的构建目录中，例如 `build-release/bin/`，与脚本产物分开。

编译将生成四个程序：

| 程序 | 用途 |
| --- | --- |
| `build-bat/bin/digit_app.exe` | 加载已有模型，打开手写识别画板 |
| `build-bat/bin/load_model.exe` | 加载已有模型，在 MNIST 测试集上评估 |
| `build-bat/bin/matrix_demo.exe` | 从头训练网络，评估并保存模型 |
| `build-bat/bin/mnist.exe` | 将原始 IDX 文件转换为项目使用的矩阵数据 |

构建只编译程序，不会自动转换数据或启动训练。仓库忽略生成的 `.exe` 文件，需要在本机编译得到。

## 路线一：直接体验手写识别

在项目根目录执行：

```powershell
.\build-bat\bin\digit_app.exe
```

程序会加载 `model/mnist_model.bin`。此路线无需生成 `train_data` 或 `test_data`，也无需运行训练程序。

### 操作方式

| 操作 | 功能 |
| --- | --- |
| 按住鼠标左键拖动 | 绘制一位数字，绘制时自动更新预测 |
| `C` | 清空画布和当前预测 |
| `Esc` 或关闭窗口 | 退出程序 |
| 调整窗口大小 | 缩放画布、文字和预览 |

左上角显示预测数字，右下角的 **Model input** 显示预处理后真正送入网络的图像。

请每次只画一位 `0–9`。未写完时预测可能跳动；空白和很短的笔画会被过滤。预处理会裁剪前景、保持宽高比将最长边调整到 20 像素，再放入 `28×28` 图像并按灰度重心居中。目前没有倾斜校正，也不支持多位数字的分割识别。

## 路线二：评估已有模型

### 1. 准备数据

仓库已经包含四个解压后的 MNIST 原始文件，无需另行下载：

```text
data/
├── train-images.idx3-ubyte
├── train-labels.idx1-ubyte
├── t10k-images.idx3-ubyte
└── t10k-labels.idx1-ubyte
```

转换程序要求从 `data` 目录运行，且输出目录必须已经存在。在项目根目录执行以下 **PowerShell** 命令：

```powershell
New-Item -ItemType Directory -Force -Path .\data\train, .\data\test | Out-Null
Push-Location .\data
..\build-bat\bin\mnist.exe
Pop-Location
```

成功时会输出：

```text
train data saved
test data saved
mission finished
```

并生成：

```text
data/train/train_data
data/test/test_data
```

这两个文件合计约 220 MB，已加入 `.gitignore`。Git 不保留空目录，所以第一次克隆后也需要上面的创建目录步骤。重复转换会覆盖这两个生成文件。

每行数据由 784 个归一化到 `0～1` 的像素和最后一列数字标签组成：训练集 60000 行，测试集 10000 行。

### 2. 运行评估

回到项目根目录后执行：

```powershell
.\build-bat\bin\load_model.exe
```

仓库当前附带模型的一次本地复测结果：

```text
test_accuracy=96.77% (9677/10000)
```

这是原始 MNIST 测试集上的准确率，不代表鼠标手写输入也有同样的准确率。画板中的书写方式、笔画形状等会影响结果。

## 路线三：重新训练

先按“路线二”完成数据转换，再在项目根目录执行：

```powershell
.\build-bat\bin\matrix_demo.exe
```

**训练完成后会覆盖 `model/mnist_model.bin`。若需要保留仓库附带模型，请在训练前备份它。**

当前默认配置在 `programs/main.c` 中：

| 配置 | 默认值 |
| --- | --- |
| 网络宽度 | `784 → 200 → 80 → 10` |
| 隐藏层激活 | Sigmoid |
| 优化器 | SGD |
| 学习率 | `0.1` |
| Batch size | `1000` |
| Epoch 数 | `400` |

训练使用 CPU，400 轮可能需要较长时间，实际耗时取决于硬件和编译优化。可以修改 `programs/main.c` 中的训练参数后重新编译。当前训练循环按固定顺序分批遍历训练集。

训练日志中的 `loss` 是最后一个批次更新参数后重新计算的损失，`time` 是从训练开始累计的秒数。训练结束后还会输出训练集和测试集准确率。

## 代码结构

| 文件或目录 | 作用 |
| --- | --- |
| `include/matrix.h`、`src/matrix.c` | 矩阵表示、基础运算、矩阵保存与读取 |
| `include/layer.h`、`src/layer.c` | 网络层、损失和层类型分派 |
| `include/network.h`、`src/network.c` | 层组合、网络计算、参数更新、模型保存与加载 |
| `programs/main.c` | MNIST 训练入口 |
| `programs/load_model.c` | 已保存模型的评估入口 |
| `programs/digit_app.c` | raylib 画板、实时预测与显示 |
| `programs/digit_preprocess.h`、`programs/digit_preprocess.c` | 手写图像预处理 |
| `tools/mnist.c` | MNIST IDX 转换工具 |
| `model/mnist_model.bin` | 预训练模型 |
| `raylib/` | 随附的 raylib 头文件与静态库 |
| `build.bat` | Windows GCC 构建脚本，输出到 `build-bat/bin/` |
| `CMakeLists.txt` | 计算核心及三个命令行程序的 CMake 构建配置 |

多个入口文件各自包含 `main`，需要分别构建为不同的可执行程序。请使用 `build.bat`，或按其中对应目标的命令编译。

## 常见问题

| 现象 | 检查方式 |
| --- | --- |
| 提示找不到 `gcc` 或不是内部/外部命令 | 按[前面的逐步安装指引](#如果还没有-gcc)检查解压位置、PATH 和实际调用的编译器路径 |
| 找不到 `raylib.h` | 确认 `raylib/include` 文件完整，并从项目根目录运行构建脚本 |
| 找不到 `-lraylib` | 确认 `raylib/lib/libraylib.a` 存在，以及库搜索路径正确 |
| 提示库不兼容或出现链接符号错误 | 检查是否使用 64 位 MinGW-w64，以及头文件、库和工具链是否匹配 |
| 出现多个 `main` 的定义 | 分别构建四个入口，不要把所有 `.c` 一起链接 |
| 画板提示 `open model file failed` | 从项目根目录启动，并确认 `model/mnist_model.bin` 存在 |
| 转换程序提示 `open file failed` | 确认从 `data` 目录运行，四个 IDX 文件存在，且已经创建 `data/train`、`data/test` |
| 训练或评估提示找不到数据 | 先执行数据转换，并确认当前目录是项目根目录 |
| 实际手写识别不稳定 | 每次只写一位数字，观察右下角模型输入；模型尚不能保证适应所有人的笔迹 |

当前矩阵和模型二进制文件按本机的 `size_t`、`float` 内存表示和字节序写入，尚未设计为通用的跨平台交换格式。已有模型和生成数据应在兼容的 64 位环境中使用。

## 参考资料

- 《深度学习入门：基于 Python 的理论与实现》
- [MNIST 数据集镜像与说明：CVDF](https://github.com/cvdfoundation/mnist)
- [raylib](https://github.com/raysan5/raylib)
- [MinGW-w64](https://www.mingw-w64.org/)
