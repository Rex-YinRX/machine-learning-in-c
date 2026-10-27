# machine-learning-in-c

学习深度学习时一时兴起想到用 C 语言实现的小型神经网络学习项目。在AI大人的指导和有限帮助下也是把这个屎山代码给写出来了。

目前已实现 MNIST 全连接网络训练、模型保存与加载、手写数字识别画板，以及人工标注采集、平移/缩放/旋转增强和单一数据集导出。网络计算使用 CPU，无需 CUDA。项目使用 **CMake** 管理计算核心、公共数据模块和各个独立程序。

## 快速开始

以下命令使用 **Windows x64 + PowerShell + MinGW-w64 GCC**。先确认已经安装 CMake（至少 3.25）和完整 MinGW-w64 工具链，且 `cmake`、`gcc`、`g++`、`mingw32-make` 可以在终端调用。尚未安装时，请先看[环境准备](#环境准备)。

**本文主线使用 Windows x64 + MinGW-w64 GCC，新增采集和数据集导出工具已在这套环境验证。** MSVC 构建步骤和已验证范围见 [MSVC 构建步骤](#使用-msvc-构建)。

```powershell
git clone https://github.com/Rex-YinRX/machine-learning-in-c.git
cd machine-learning-in-c
cmake -S . -B build-release -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --parallel
```

没有 Git 也可以在仓库页面选择 **Code → Download ZIP**，解压后，在 PowerShell 中进入包含 `CMakeLists.txt` 的目录，再执行上面的两条 `cmake` 命令。

确认构建成功后，保持在项目根目录，运行画板：

```powershell
.\build-release\bin\digit_app.exe
```

仓库已包含预训练模型 `model/mnist_model.bin`，体验识别画板和启动采集画板无需转换 MNIST 数据或重新训练。默认启用 raylib；未提供本地源码时，配置会从 GitHub 下载，也可以[预先放置本地源码](#raylib-如何获取)。

## 已实现的功能

- 基于连续内存和 `stride` 的矩阵表示及基本运算。
- Affine、Sigmoid 层的前向与反向传播。
- Softmax 与交叉熵损失、SGD 参数更新、mini-batch 训练。
- 通过层数组组合不同深度、宽度的全连接网络。
- MNIST IDX 数据转换、矩阵保存与读取、模型保存与加载。
- 可缩放画板、实时预测和模型输入预览。
- 手写输入的前景裁剪、等比例缩放与灰度重心居中。
- 人工标签采集，原始笔迹及同编号增强包保存。
- 平移、缩放、旋转随机增强，合并导出单一 `N×785` 数据集。

当前统一层接口支持 Affine 和 Sigmoid；卷积、池化和通用自动微分尚未实现。

## 环境准备

| 工具 | 要求或本地验证环境 | 用途 |
| --- | --- | --- |
| 操作系统与目标架构 | Windows x64 | 本文命令和已有模型的使用环境 |
| CMake | 项目最低要求 3.25；下述 VS 2026 路线要求至少 4.2，本地使用 4.4.4 | 生成并调用构建系统 |
| MinGW-w64 工具链 | 本地使用 GCC 14.2.0，含 `gcc`、`g++`、`mingw32-make` | 编译、链接并执行构建任务 |
| MSVC 工具链 | 原有四个程序已验证 Visual Studio 2026、MSVC 19.51、x64；新增工具的验证范围见下文 | 编译、链接并通过 MSBuild 构建 |
| raylib | 未提供本地源码时下载 6.0，启用相关工具时参与构建 | 窗口、输入、绘图及数据工具的文件系统操作 |
| Git | 可选 | 克隆仓库；也可下载 ZIP |

核心代码使用 C11。raylib 的 CMake 配置还会检测 C++ 编译器，因此请安装完整工具链：MinGW 路线保留 `g++`，MSVC 路线安装 C++ 桌面开发组件。改用 CMake 后，编译器仍然需要单独安装；VS Code 只是可选编辑器。

当前 MinGW 本地环境为 GCC 14.2.0、CMake 4.4.4；原有四个程序另有 MSVC 构建验证记录。raylib 与项目一起使用所选工具链从源码编译。本文命令以 Windows PowerShell 为例；原有入口按平台选择默认路径，新工具使用正斜杠路径，其他操作系统的完整运行仍需验证。

以下 GCC 安装步骤供 MinGW 路线使用。选择 MSVC 的用户可以跳过 GCC 安装，准备 CMake 后按 [MSVC 构建步骤](#使用-msvc-构建)安装对应组件并构建。

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
│   ├── mingw32-make.exe
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

这次设置只对当前 PowerShell 及从它启动的程序生效。当前终端已能调用 GCC，完成后续 CMake 安装和构建前检查后即可构建。若希望以后也能直接使用 `gcc`，继续下一步。

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

确认仍然指向正确的 `gcc.exe`，并输出 `x86_64-w64-mingw32`，即可继续安装 CMake 并完成构建前检查。

如果仍提示找不到 `gcc`，可以先重复第 3 步，让当前终端能够调用 GCC，再检查持久设置是否成功。如果电脑上装过多个 GCC，可执行：

```powershell
Get-Command gcc -All | Select-Object Source
```

检查是否有旧版本或 32 位版本优先被选中。第 3 步会把本节选择的目录放到当前终端 PATH 的最前面，便于先确认使用的是哪一个编译器。

更多工具链来源见 [MinGW-w64 官方工具链列表](https://www.mingw-w64.org/downloads/)和 [GCC 官方二进制安装说明](https://gcc.gnu.org/install/binaries.html)。

### 安装 CMake

1. 打开 [CMake 官方下载页](https://cmake.org/download/)，选择稳定版本的 **Windows x64 Installer**。MinGW 路线至少需要 3.25；下述 Visual Studio 2026 路线至少需要 4.2。不要下载源码包或 ARM64 安装包。
2. 运行安装程序。如果提供 **Add CMake to the system PATH** 选项，选择为当前用户添加到 PATH。默认安装目录通常为 `C:\Program Files\CMake`。
3. 关闭并重新打开整个 PowerShell、Windows Terminal 或 VS Code，再验证：

```powershell
cmake --version
```

若已安装但命令仍找不到，可以在当前 PowerShell 临时添加安装目录，再验证：

```powershell
$cmakeBin = 'C:\Program Files\CMake\bin'
Test-Path -LiteralPath (Join-Path $cmakeBin 'cmake.exe')
```

先确认输出 `True`；安装在其他位置时修改 `$cmakeBin`。然后在同一个终端执行：

```powershell
$env:Path = "$cmakeBin;$env:Path"
cmake --version
```

这个设置只对当前终端生效。需要长期使用时，在 Windows 搜索“编辑账户的环境变量”，编辑用户的 `Path`，新增实际的 CMake `bin` 目录，并重新打开终端。

### MinGW 构建前检查

在同一个 PowerShell 中执行：

```powershell
cmake --version
gcc --version
gcc -dumpmachine
g++ --version
mingw32-make --version
Get-Command cmake, gcc, g++, mingw32-make | Select-Object Name, Source
```

确认命令均可执行，且 GCC 的目标是 `x86_64-w64-mingw32`。`gcc`、`g++` 和 `mingw32-make` 应来自同一套工具链。

`MinGW Makefiles` 使用 `mingw32-make` 执行构建规则。本指南从 Windows PowerShell 调用 CMake，不要在 Git Bash 或 MSYS shell 中照搬这一生成器的流程。参见 [CMake 的 MinGW Makefiles 说明](https://cmake.org/cmake/help/latest/generator/MinGW%20Makefiles.html)。

## CMake 构建说明

下文的“项目根目录”指包含 `CMakeLists.txt`、`include`、`src`、`programs`、`tools`、`data` 和 `model` 的目录。

使用 MinGW 时，在该目录执行：

```powershell
cmake -S . -B build-release -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --parallel
```

第一条命令配置项目并生成构建文件，第二条命令编译、链接程序。第一条成功后再执行第二条；构建出错时，应先解决错误，再运行程序。

| 参数 | 含义 |
| --- | --- |
| `-S .` | 当前目录是源码目录 |
| `-B build-release` | 将缓存、构建文件和产物放入 `build-release`，目录会自动创建 |
| `-G "MinGW Makefiles"` | 选择使用 `mingw32-make` 的构建方式 |
| `-DCMAKE_BUILD_TYPE=Release` | 使用开启优化的 Release 配置，适合训练和运行 |
| `--build build-release` | 构建此前配置的目录 |
| `--parallel` | 允许并行编译；也可写 `--parallel 4` 来限制并行任务数 |

命令语法见 [CMake 命令行文档](https://cmake.org/cmake/help/latest/manual/cmake.1.html)。这里只进行构建，不会自动转换数据、启动训练或打开画板。

### MinGW 生成的程序

| 路径 | 用途 |
| --- | --- |
| `build-release/bin/digit_app.exe` | 加载已有模型，打开手写识别画板 |
| `build-release/bin/load_model.exe` | 加载已有模型，在 MNIST 测试集上评估 |
| `build-release/bin/continue_train.exe` | 加载已有模型，在指定数据集继续训练后保存回原路径 |
| `build-release/bin/matrix_demo.exe` | 从头训练网络，评估并保存模型 |
| `build-release/bin/mnist.exe` | 将原始 IDX 文件转换为矩阵数据 |
| `build-release/bin/digit_collect.exe` | 手写样本采集、保存和导出 |
| `build-release/bin/digit_dataset.exe` | 从已采集原图生成合并数据集 |
| `build-release/lib/libnn_core.a` | 神经网络计算及矩阵读写的核心静态库 |
| `build-release/lib/libhandwritten_data.a` | 供采集、导出程序共用的数据保存和生成模块 |

**CMake 不会把 `data` 和 `model` 复制到 `build-release/bin`。** 识别、采集、导出、转换、训练和评估均以当前工作目录解释相对路径，因此应在项目根目录启动，不要先进入 `build-release/bin`。也请使用上表路径，避免误运行根目录中旧的 `.exe`。

修改 C 源码后，通常只需再次运行：

```powershell
cmake --build build-release --parallel
```

若要只编译画板及其依赖：

```powershell
cmake --build build-release --target digit_app --parallel
```

只编译采集和数据集导出工具时：

```powershell
cmake --build build-release --target digit_collect digit_dataset --parallel
```

切换编译器或生成器时，请使用新的构建目录。CMake 会缓存工具链信息，不能把之前的 Visual Studio 构建目录直接当作 MinGW 构建目录使用。

### 使用 MSVC 构建

原有的 `digit_app`、`load_model`、`matrix_demo`、`mnist` 已在 Windows x64、**Visual Studio 2026 + MSVC 19.51** 下验证。新增的 `digit_collect`、`digit_dataset` 和公共数据模块本轮仅用 MinGW 验证，尚未补做 MSVC 实测。如果采用 MSVC 路线，无需安装 GCC 或 `mingw32-make`。

1. 安装 Visual Studio 2026，或对应的 Visual Studio Build Tools。在 Visual Studio Installer 中选择 **“使用 C++ 的桌面开发”**，确认安装 MSVC x64/x86 构建工具和 Windows SDK。具体步骤见 [Microsoft C/C++ 安装说明](https://learn.microsoft.com/en-us/cpp/build/vscpp-step-0-installation)。
2. 准备 **CMake 4.2 或更新版本**，确认 `cmake --version` 可执行；`Visual Studio 18 2026` 生成器从 CMake 4.2 开始提供，见 [CMake 官方说明](https://cmake.org/cmake/help/latest/generator/Visual%20Studio%2018%202026.html)。
3. 在项目根目录的 PowerShell 中执行：

```powershell
cmake -S . -B build-msvc -G "Visual Studio 18 2026" -A x64
cmake --build build-msvc --config Release --parallel
```

`-A x64` 选择 64 位目标。Visual Studio 使用多配置构建目录，在构建阶段通过 `--config Release` 选择 Release；此路线不使用 `CMAKE_BUILD_TYPE` 来选择配置。

按当前 CMake 配置完整构建时，程序输出路径如下；新增工具的 MSVC 验证范围以上文为准：

| 路径 | 用途 |
| --- | --- |
| `build-msvc/bin/Release/digit_app.exe` | 手写识别画板 |
| `build-msvc/bin/Release/load_model.exe` | 已保存模型的评估 |
| `build-msvc/bin/Release/continue_train.exe` | 加载已有模型继续训练 |
| `build-msvc/bin/Release/matrix_demo.exe` | 从头训练并保存模型 |
| `build-msvc/bin/Release/mnist.exe` | MNIST 数据转换 |
| `build-msvc/bin/Release/digit_collect.exe` | 手写样本采集、保存和导出 |
| `build-msvc/bin/Release/digit_dataset.exe` | 单一数据集导出 |

保持在项目根目录，启动画板：

```powershell
.\build-msvc\bin\Release\digit_app.exe
```

下文运行示例默认使用 MinGW 的产物路径。使用 MSVC 时，将运行命令中的 `build-release\bin` 换成 `build-msvc\bin\Release`；配置和构建命令使用本节的 MSVC 版本。各程序仍从项目根目录运行，例如 `.\build-msvc\bin\Release\mnist.exe` 和 `.\build-msvc\bin\Release\load_model.exe`。

修改源码后再次执行 `cmake --build build-msvc --config Release --parallel`。如果只需要 MNIST 转换、训练、评估和继续训练这四个命令行程序，可在配置时添加 `-DBUILD_DIGIT_APP=OFF`。该开关也会关闭采集和数据集导出工具；即使 `digit_dataset` 不打开窗口，当前实现仍需要 raylib 的文件系统功能。

### raylib 如何获取

默认 `BUILD_DIGIT_APP=ON`，同时启用 `digit_app`、`digit_collect`、`digit_dataset` 和 `handwritten_data`。CMake 按以下顺序获取 raylib：

1. 如果存在 `third_party/raylib/CMakeLists.txt`，使用这份本地源码。
2. 否则通过 `FetchContent` 下载官方 raylib 6.0 源码压缩包，并核验配置中的 SHA256。
3. 将 raylib 与项目一起编译，供识别画板、采集画板和数据集导出工具使用；无需手动填写 `-I`、`-L` 或 Windows 系统库参数。

`third_party/raylib/` 已被 Git 忽略，不随克隆或仓库 ZIP 提供。首次下载失败或需要离线构建时，可以从 [raylib 6.0 源码压缩包](https://github.com/raysan5/raylib/archive/refs/tags/6.0.zip)下载并完整解压，将内容放成：

```text
third_party/
└── raylib/
    ├── CMakeLists.txt
    ├── CMakeOptions.txt
    ├── cmake/
    ├── src/
    └── ...
```

不要多套一层 `raylib-6.0` 目录，也不要只放 `raylib.h` 和预编译静态库。放置完整源码后，重新执行配置与构建命令即可。本地源码分支不会执行下载压缩包的 SHA256 校验，请使用官方对应版本。

默认配置将 raylib 构建为静态库，相关程序无需另附 `raylib.dll`。若以后分发编译成品，还需要核对所用工具链引入的其他运行库依赖。

### 只构建 MNIST 转换、训练和评估工具

只需要 `mnist`、`load_model`、`matrix_demo`、`continue_train` 时，可以在独立目录关闭该选项；这条路线不会下载或编译 raylib，也不会生成 `digit_app`、`digit_collect` 或 `digit_dataset`：

```powershell
cmake -S . -B build-core -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DBUILD_DIGIT_APP=OFF
cmake --build build-core --parallel
```

此时生成 `build-core/bin/mnist.exe`、`load_model.exe`、`matrix_demo.exe` 和 `continue_train.exe`。仅在后文的 MNIST 转换、训练、评估命令中将 `build-release` 替换为 `build-core`；采集与数据集导出命令需要启用完整配置。

## 程序参数

当前七个程序的调用形式如下：

```text
digit_app.exe [model_path]
load_model.exe [model_path test_data_path]
continue_train.exe model_path data_path epochs
matrix_demo.exe [train_data_path test_data_path model_output_path]
mnist.exe [data_dir]
digit_collect.exe [model_path [data_dir [augment_count]]]
digit_dataset.exe [data_dir [augment_count [seed]]]
```

方括号表示可选参数，实际运行时不要输入方括号：

- `load_model` 和 `matrix_demo` 的路径参数必须成组提供，分别是零个或两个、零个或三个。
- `continue_train` 必须提供模型路径、数据集路径和正整数 epoch 数；训练完成后保存回同一模型路径。
- `digit_app`、`mnist` 可以不传参数，或提供一个路径。
- `digit_collect`、`digit_dataset` 的参数从左到右依次提供；只修改增强数量时，也需要先提供前面的路径。

| 程序 | 省略全部参数时的默认值 |
| --- | --- |
| `digit_app.exe` | 模型：`model/mnist_model.bin` |
| `load_model.exe` | 模型：`model/mnist_model.bin`；测试数据：`data/test/test_data` |
| `matrix_demo.exe` | 训练数据：`data/train/train_data`；测试数据：`data/test/test_data`；模型输出：`model/mnist_model.bin` |
| `mnist.exe` | 数据目录：`data` |
| `digit_collect.exe` | 模型：`model/mnist_model.bin`；数据目录：`data/handwritten`；额外增强数量：`20` |
| `digit_dataset.exe` | 数据目录：`data/handwritten`；额外增强数量：`20`；随机种子：`1234` |

两个新增数据工具支持 `--help`：

```powershell
.\build-release\bin\digit_collect.exe --help
.\build-release\bin\digit_dataset.exe --help
```

运行时仍需写出程序的实际路径，例如 `.\build-release\bin\load_model.exe`。相对路径均以当前工作目录为基准；包含空格的路径请用双引号括起。

## 路线一：体验手写识别

从项目根目录执行：

```powershell
.\build-release\bin\digit_app.exe
```

不传参数时，程序加载 `model/mnist_model.bin`，无需转换 MNIST 数据。

| 操作 | 功能 |
| --- | --- |
| 按住鼠标左键拖动 | 绘制一位数字，实时更新预测 |
| `C` | 清空画布和预测 |
| `Esc` 或关闭窗口 | 退出 |
| 调整窗口大小 | 缩放画布、文字和预览 |

左上角显示预测数字，右下角 **Model input** 显示实际送入网络的图像。请每次只画一位 `0–9`；未写完时预测可能跳动，空白和很短的笔画会被过滤。

预处理会裁剪前景，保持宽高比把最长边调整为 20 像素，再放入 `28×28` 图像并按灰度重心居中。目前没有倾斜校正，也不支持多位数字分割。模型对不同笔迹的识别效果可能不同。

## 采集自己的手写数字与生成数据集

在项目根目录构建并启动采集画板：

```powershell
cmake -S . -B build-release -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --target digit_collect digit_dataset --parallel
.\build-release\bin\digit_collect.exe
```

采集画板会先加载模型用于实时预测，默认模型为 `model/mnist_model.bin`，模型缺失或无法加载时不会启动窗口。它保留绘图、实时预测、窗口缩放和模型输入预览：

- 鼠标左键绘制，数字键 `0`～`9` 设置真实标签。模型预测仅供参考，保存使用人工标签。
- `S` 保存原图及同编号增强包，默认是一份规范图加 20 份随机增强图。
- `C` 清空画布与标签，开始下一份。保存后画布保留，便于查看结果。
- `E` 将全部**已保存到 `raw` 的原图**导出为一份新的数据集，不包含当前尚未按 S 保存的画布。窗口显示结果，终端输出完整路径。

保存目录自动创建：

```text
data/handwritten/
├── raw/000001.cmat          # 1×785，原始784灰度值＋人工标签
├── augmented/000001.cmat    # 默认21×785，一份规范图＋20份增强图
└── datasets/000001/
    ├── dataset.cmat        # 所有来源合并后的 N×785 矩阵
    └── dataset_info.txt    # 参数、各标签原图数量和最终行数
```

原图和增强包共用编号，重启后跳过两个目录中已有的编号。空白、过小笔迹和未选择标签的输入不能保存。如果原图已保存而增强包生成或写入失败，窗口和终端会说明部分成功状态，原图会保留，之后仍可用 E 或命令行工具重新生成数据。

每次导出创建新的 `datasets/000002` 等目录，保留以前的数据集。`dataset_info.txt` 最后写入，作为完成标记；失败时可能留下没有该标记的快照目录。没有原图或发现损坏样本时，本次导出失败并在终端说明原因，已有原图不会被覆盖。

随机增强使用平移 `-2～2` 像素、缩放 `0.9～1.1`、旋转 `-10～10` 度，顺序为缩放、旋转、平移；正角在画面中顺时针。每个变体都从同一份规范图出发。原始笔迹始终保留，导出按原图文件名升序从 `raw` 重新生成，不依赖 `augmented` 缓存。

采集程序启动时用当前时间初始化随机数；E 导出会将随机种子重设为 `1234`，命令行导出可自行指定种子。增强包与最终数据集不保证逐条相同。

**只生成一份数据集，不划分训练集或验证集。** 每份原图贡献 `增强数量+1` 行。例如采集 10 份原图、每份额外增强 20 次，最终得到 `210×785` 的矩阵。前 784 列为灰度，最后一列为人工标签。

关闭画板后也可以直接运行导出工具；它不加载模型，也不打开窗口：

```powershell
.\build-release\bin\digit_dataset.exe
```

可选参数按顺序为：数据目录、每份原图的额外增强数量、随机种子。例如：

```powershell
.\build-release\bin\digit_dataset.exe ".\data\handwritten" 20 1234
```

增强数量支持 `0～100`，默认 20；设为 `0` 时，每份原图只生成一份规范图。种子默认 1234。相同原图、参数和种子在同一运行环境下可以重现数据内容。`--help` 查看用法。

采集画板的可选参数为模型路径、数据目录、增强数量：

```powershell
.\build-release\bin\digit_collect.exe ".\model\mnist_model.bin" ".\data\handwritten" 20
```

生成的矩阵使用现有 `CMAT` 格式，可由 `mat_load` 读取。采集和导出不会自动启动训练或覆盖已有模型。少量自采样本可用下文的 `continue_train` 继续训练已有模型，它会处理小数据集和尾批；原有 `matrix_demo` 的整批限制见重新训练章节。

`dataset_info.txt` 中的 `label_0_sources`～`label_9_sources` 统计的是各数字的原图份数，而非增强后的行数。这些采集文件和导出结果默认被 Git 忽略，不会加入仓库。

## 路线二：评估已有模型

### 1. 转换数据

仓库已包含四个解压后的 MNIST IDX 文件：

```text
data/
├── train-images.idx3-ubyte
├── train-labels.idx1-ubyte
├── t10k-images.idx3-ubyte
└── t10k-labels.idx1-ubyte
```

转换工具默认读取当前工作目录下的 `data`，即 `.\data`，并将结果写入其中的 `train`、`test` 子目录。输出目录需要提前创建，程序不会自行创建。在项目根目录执行：

```powershell
New-Item -ItemType Directory -Force -Path .\data\train, .\data\test | Out-Null

.\build-release\bin\mnist.exe
if ($LASTEXITCODE -ne 0) { throw 'MNIST conversion failed.' }
```

成功时输出：

```text
train data saved
test data saved
mission finished
```

并生成 `data/train/train_data` 和 `data/test/test_data`，合计约 220 MB。两者已被 Git 忽略，首次克隆后需要生成；重复转换会覆盖生成文件。

每行包含 784 个归一化到 `0～1` 的像素，最后一列为数字标签。训练集 60000 行，测试集 10000 行。

也可以通过命令行参数指定数据目录：

```powershell
.\build-release\bin\mnist.exe "D:\mnist-data"
```

`D:\mnist-data` 是示例路径，需要替换成自己的目录。该目录应包含上面列出的四个 IDX 文件，以及提前创建的 `train`、`test` 子目录。转换结果会写入该目录下的 `train/train_data` 和 `test/test_data`。

### 2. 执行评估

保持在项目根目录，执行：

```powershell
.\build-release\bin\load_model.exe
```

如果转换时使用了自定义数据目录，例如 `D:\mnist-data`，评估程序不会自动记住这个目录，需要同时指定模型和测试数据路径：

```powershell
.\build-release\bin\load_model.exe ".\model\mnist_model.bin" "D:\mnist-data\test\test_data"
```

请将示例中的数据路径替换为自己的实际路径。

附带模型的一次本地复测结果为：

```text
test_accuracy=96.77% (9677/10000)
```

这是原始 MNIST 测试集上的准确率，不代表鼠标手写输入也有相同准确率。

## 继续训练已有模型

`continue_train` 读取已保存模型和一份 `mat_save` / `CMAT` 格式的数据集，沿用 `programs/main.c` 的逐行打包方式，从已加载的参数继续进行 SGD 更新，不重新随机初始化网络。

重新配置并构建新增目标（以下为 MinGW 路线）：

```powershell
cmake -S . -B build-release -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --target continue_train --parallel
```

在项目根目录执行，例如使用第一次导出的手写数据集继续训练 10 个 epoch：

```powershell
.\build-release\bin\continue_train.exe ".\model\mnist_model.bin" ".\data\handwritten\datasets\000001\dataset.cmat" 10
```

将数据集路径替换成实际导出的快照；也可使用转换后的 `data/train/train_data`。每行最后一列为整数标签，前面的列为输入特征，必须与加载模型的输入维度匹配。对于当前 MNIST 模型，数据应为 `N×785`，标签范围为 `0～9`。

学习率为 `0.1`，每批最多 `1000` 行，位于 `programs/continue_train.c` 的宏定义中。数据不足 1000 行时仍会训练，末尾不足一批的样本也会参与；每个 epoch 按原数据顺序遍历全部样本。日志中的 `loss` 和 `final_last_batch_loss` 是最后一个批次在本轮更新后重新计算的损失，不是整份数据的平均损失。

**这个程序会覆盖传入的模型路径。需要保留原模型时，请先备份，或传入模型副本的路径。** 保存时先写入同目录的 `模型路径.tmp`，确认写入成功后替换原文件。若替换失败，训练结果会保留在终端提示的临时路径；已有同名 `.tmp` 不会被覆盖，需确认其内容并处理后再重试。

当前程序继续训练的是已有结构和参数；epoch 数必须是正整数，不会自动改变网络结构、划分验证集或混入其他训练数据。本轮已通过 MinGW-w64 GCC 14.2.0 和 MSVC 19.51 的 CMake 构建及小数据集数值验证。

## 路线三：重新训练

先完成数据转换，再从项目根目录执行：

```powershell
.\build-release\bin\matrix_demo.exe
```

**使用默认参数时，训练完成后会覆盖 `model/mnist_model.bin`；指定参数时，会写入或覆盖 `model_output_path`。需要保留已有模型时，请提前备份。**

默认训练配置位于 `programs/main.c`：

| 配置 | 默认值 |
| --- | --- |
| 网络宽度 | `784 → 200 → 80 → 10` |
| 隐藏层激活 | Sigmoid |
| 优化器 | SGD |
| 学习率 | `0.1` |
| Batch size | `1000` |
| Epoch 数 | `400` |

训练使用 CPU，400 轮需要较长时间，具体取决于硬件和优化配置。修改参数后执行 `cmake --build build-release --parallel` 重新编译，再启动训练。

原有 `matrix_demo` 训练循环按固定顺序遍历完整的 1000 行批次，末尾不足一批的数据会跳过；少于 1000 行时不会执行参数更新，不能直接用默认配置训练小型自采数据集。开始训练前应调整批量大小并完善尾批处理。日志中的 `loss` 是最后一个批次在更新参数后重新计算的损失，`time` 是从训练开始累计的秒数；训练结束后还会输出训练集和测试集准确率。

## 代码结构

| 文件或目录 | 作用 |
| --- | --- |
| `CMakeLists.txt` | 管理 `nn_core`、`handwritten_data` 两个公共静态库及七个独立程序，管理 raylib 依赖 |
| `include/` | 矩阵、层和网络的公共头文件 |
| `src/matrix.c` | 矩阵运算与矩阵保存、读取 |
| `src/layer.c` | 层计算、损失和层类型分派 |
| `src/network.c` | 层组合、前向反向、参数更新及模型保存、加载 |
| `programs/main.c` | MNIST 训练入口 |
| `programs/load_model.c` | 模型评估入口 |
| `programs/continue_train.c` | 已保存模型的继续训练入口 |
| `programs/digit_app.c` | raylib 画板与实时预测 |
| `programs/digit_preprocess.c`、`.h` | 前景规范化、双线性采样、平移/缩放/旋转和随机增强配置 |
| `tools/mnist.c` | MNIST IDX 转换入口 |
| `tools/digit_collect.c` | 图形采集入口：绘图、人工标签、保存和 E 键导出 |
| `tools/digit_dataset.c` | 解析导出参数、调用公共模块并打印结果的命令行入口 |
| `tools/handwritten_dataset.c`、`.h` | 原图/增强包保存、文件校验和数据集合并生成，供两个入口共用 |
| `data/` | 原始 IDX、本地转换矩阵及 `handwritten/` 下的采集/增强/导出数据 |
| `model/mnist_model.bin` | 预训练模型 |
| `third_party/raylib/` | 可选的本地 raylib 源码，不纳入 Git |
| `build-release/` | 本地 CMake 缓存、依赖和编译产物，不纳入 Git |
| `build-msvc/` | MSVC 构建目录，Release 程序位于 `bin/Release/`，不纳入 Git |

七个入口文件各自包含 `main`，CMake 将它们构建为独立程序；`handwritten_dataset.c` 是公共功能模块，没有 `main`。采集画板按 E 和运行 `digit_dataset` 都调用同一个导出实现。不要把所有 `.c` 一次链接到同一个可执行文件中。

## 常见问题

| 现象 | 处理方式 |
| --- | --- |
| 找不到 `cmake` | 安装 CMake，并将其 `bin` 加入 PATH，重新打开整个终端应用 |
| 找不到 `gcc`、`g++` 或提示没有编译器 | 按前面的工具链安装步骤检查完整工具链、PATH 和实际命令路径 |
| 找不到 `CMAKE_MAKE_PROGRAM` 或 `mingw32-make` | 检查完整 MinGW 工具链中的 `mingw32-make.exe` 是否存在并可在当前终端调用 |
| 找不到 `Visual Studio 18 2026` 生成器或安装实例 | 确认 CMake 至少为 4.2，且已安装 Visual Studio 2026 或对应 Build Tools 的 C++ 桌面开发组件 |
| 下载 raylib 失败 | 检查 GitHub 网络访问，或按上文放置完整本地源码后重新配置 |
| 生成器与已有缓存不匹配 | 使用未配置过的新构建目录，并同时修改配置与构建命令的目录参数 |
| 构建后没有 `digit_app.exe`、`digit_collect.exe` 或 `digit_dataset.exe` | 三者都受 `BUILD_DIGIT_APP` 控制；确认构建成功且该选项为 `ON`，需要时重新配置并添加 `-DBUILD_DIGIT_APP=ON` |
| 画板提示模型文件打开失败 | 检查报错中显示的实际模型路径；未传参数时，默认使用当前工作目录下的 `.\model\mnist_model.bin`，请确认文件存在 |
| 转换时提示文件打开失败 | 检查默认的 `.\data` 或传入的数据目录，并根据报错中的具体路径确认文件是否存在；确认该目录包含四个 IDX 文件，且已提前创建 `train`、`test` 子目录 |
| 训练或评估找不到数据 | 先完成转换，再从项目根目录启动对应程序 |
| 采集或导出提示没有原图 | 先选择真实标签并按 S 保存；E 和命令行只读取数据目录的 `raw/*.cmat`，不读取未保存画布 |
| 导出提示 Invalid raw sample | 按终端给出的路径检查样本；要求完整的 `1×785` CMAT、有限且在 `0～1` 的像素及整数标签 `0～9`，损坏样本会使本次导出失败 |
| 修改增强数量后旧增强包没有变化 | 修改参数不会重写已有样本；运行 E 或 `digit_dataset` 会从原图生成一份新的数据集 |
| 修改源码后运行效果没变 | 重新构建，并确认运行的是 `build-release/bin` 下的新程序 |

矩阵和模型二进制文件仍按本机 `size_t`、`float` 的内存表示和字节序写入，并非通用跨平台格式。CMake 管理构建过程，并不会自动改变数据格式或运行时路径；已有模型和数据应在兼容的 64 位环境下使用。

