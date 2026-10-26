# machine-learning-in-c

学习深度学习时一时兴起想到用 C 语言实现的小型神经网络学习项目。

目前已实现 MNIST 全连接网络训练、模型保存与加载，以及基于 raylib 的手写数字画板。网络计算使用 CPU，无需 CUDA。项目使用 **CMake** 管理计算核心、数据转换工具和各个可执行程序。

## 快速开始

以下命令使用 **Windows x64 + PowerShell + MinGW-w64 GCC**。先确认已经安装 CMake（至少 3.25）和完整 MinGW-w64 工具链，且 `cmake`、`gcc`、`g++`、`mingw32-make` 可以在终端调用。尚未安装时，请先看[环境准备](#环境准备)。

**Windows x64 下的 MinGW-w64 GCC 和 MSVC 均已验证可用。** 如果使用 MSVC，请按 [MSVC 构建步骤](#使用-msvc-构建)操作，两套工具链选择一套即可。

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

仓库已包含预训练模型 `model/mnist_model.bin`，体验画板无需转换数据或重新训练。首次配置默认会下载 raylib 源码，需要能够访问 GitHub；也可以[预先放置本地源码](#raylib-如何获取)。

## 已实现的功能

- 基于连续内存和 `stride` 的矩阵表示及基本运算。
- Affine、Sigmoid 层的前向与反向传播。
- Softmax 与交叉熵损失、SGD 参数更新、mini-batch 训练。
- 通过层数组组合不同深度、宽度的全连接网络。
- MNIST IDX 数据转换、矩阵保存与读取、模型保存与加载。
- 可缩放画板、实时预测和模型输入预览。
- 手写输入的前景裁剪、等比例缩放与灰度重心居中。

当前统一层接口支持 Affine 和 Sigmoid；卷积、池化和通用自动微分尚未实现。

## 环境准备

| 工具 | 要求或本地验证环境 | 用途 |
| --- | --- | --- |
| 操作系统与目标架构 | Windows x64 | 本文命令和已有模型的使用环境 |
| CMake | 项目最低要求 3.25；下述 VS 2026 路线要求至少 4.2，本地使用 4.4.4 | 生成并调用构建系统 |
| MinGW-w64 工具链 | 本地使用 GCC 14.2.0，含 `gcc`、`g++`、`mingw32-make` | 编译、链接并执行构建任务 |
| MSVC 工具链 | 已验证 Visual Studio 2026、MSVC 19.51、x64；可替代 MinGW-w64 | 编译、链接并通过 MSBuild 构建 |
| raylib | 默认下载 6.0 源码并参与构建 | 窗口、输入和绘图 |
| Git | 可选 | 克隆仓库；也可下载 ZIP |

核心代码使用 C11。raylib 的 CMake 配置还会检测 C++ 编译器，因此请安装完整工具链：MinGW 路线保留 `g++`，MSVC 路线安装 C++ 桌面开发组件。改用 CMake 后，编译器仍然需要单独安装；VS Code 只是可选编辑器。

**GCC 14.2.0 和上述 MSVC 环境均已验证可用。** raylib 与项目一起使用所选工具链从源码编译。其他工具链版本和操作系统仍需各自验证；当前程序中的资源路径采用 Windows 写法。

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

下文的“项目根目录”指包含 `CMakeLists.txt`、`include`、`src`、`programs`、`data` 和 `model` 的目录。

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
| `build-release/bin/matrix_demo.exe` | 从头训练网络，评估并保存模型 |
| `build-release/bin/mnist.exe` | 将原始 IDX 文件转换为矩阵数据 |
| `build-release/lib/libnn_core.a` | 供上述程序链接的神经网络核心静态库 |

**CMake 不会把 `data` 和 `model` 复制到 `build-release/bin`。** 使用默认参数时，画板、训练和评估都以当前工作目录查找资源，因此应在项目根目录通过完整相对路径启动，不要先进入 `build-release/bin`。也请使用上表路径，避免误运行根目录中旧的 `.exe`。

修改 C 源码后，通常只需再次运行：

```powershell
cmake --build build-release --parallel
```

若要只编译画板及其依赖：

```powershell
cmake --build build-release --target digit_app --parallel
```

切换编译器或生成器时，请使用新的构建目录。CMake 会缓存工具链信息，不能把之前的 Visual Studio 构建目录直接当作 MinGW 构建目录使用。

### 使用 MSVC 构建

Windows x64 下已验证 **Visual Studio 2026 + MSVC 19.51**。如果采用此路线，无需安装 GCC 或 `mingw32-make`。

1. 安装 Visual Studio 2026，或对应的 Visual Studio Build Tools。在 Visual Studio Installer 中选择 **“使用 C++ 的桌面开发”**，确认安装 MSVC x64/x86 构建工具和 Windows SDK。具体步骤见 [Microsoft C/C++ 安装说明](https://learn.microsoft.com/en-us/cpp/build/vscpp-step-0-installation)。
2. 准备 **CMake 4.2 或更新版本**，确认 `cmake --version` 可执行；`Visual Studio 18 2026` 生成器从 CMake 4.2 开始提供，见 [CMake 官方说明](https://cmake.org/cmake/help/latest/generator/Visual%20Studio%2018%202026.html)。
3. 在项目根目录的 PowerShell 中执行：

```powershell
cmake -S . -B build-msvc -G "Visual Studio 18 2026" -A x64
cmake --build build-msvc --config Release --parallel
```

`-A x64` 选择 64 位目标。Visual Studio 使用多配置构建目录，在构建阶段通过 `--config Release` 选择 Release；此路线不使用 `CMAKE_BUILD_TYPE` 来选择配置。

程序输出到 `build-msvc/bin/Release/`：

| 路径 | 用途 |
| --- | --- |
| `build-msvc/bin/Release/digit_app.exe` | 手写识别画板 |
| `build-msvc/bin/Release/load_model.exe` | 已保存模型的评估 |
| `build-msvc/bin/Release/matrix_demo.exe` | 从头训练并保存模型 |
| `build-msvc/bin/Release/mnist.exe` | MNIST 数据转换 |

保持在项目根目录，启动画板：

```powershell
.\build-msvc\bin\Release\digit_app.exe
```

下文训练、评估和转换示例默认使用 MinGW 的产物路径。使用 MSVC 时，将每一处 `build-release\bin` 替换为 `build-msvc\bin\Release`，其他步骤相同。转换和评估都从项目根目录运行，例如 `.\build-msvc\bin\Release\mnist.exe` 和 `.\build-msvc\bin\Release\load_model.exe`。

修改源码后再次执行 `cmake --build build-msvc --config Release --parallel`。如果只需要命令行工具，在本节配置命令末尾加上 `-DBUILD_DIGIT_APP=OFF`，再执行构建命令即可。

### raylib 如何获取

默认 `BUILD_DIGIT_APP=ON`，CMake 按以下顺序获取 raylib：

1. 如果存在 `third_party/raylib/CMakeLists.txt`，使用这份本地源码。
2. 否则通过 `FetchContent` 下载官方 raylib 6.0 源码压缩包，并核验配置中的 SHA256。
3. 将 raylib 与项目一起编译，再链接到画板程序；无需手动填写 `-I`、`-L` 或 Windows 系统库参数。

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

默认配置将 raylib 构建为静态库，画板无需另附 `raylib.dll`。若以后分发编译成品，还需要核对所用工具链引入的其他运行库依赖。

### 只构建训练和评估工具

不需要画板时，可以在独立目录关闭该选项；这条路线不会下载或编译 raylib：

```powershell
cmake -S . -B build-core -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DBUILD_DIGIT_APP=OFF
cmake --build build-core --parallel
```

此时生成 `build-core/bin/mnist.exe`、`load_model.exe` 和 `matrix_demo.exe`。执行后文命令时，将 `build-release` 替换为 `build-core`。

## 程序参数

画板、评估和训练程序支持以下参数格式：

```text
digit_app.exe [model_path]
load_model.exe [model_path test_data_path]
matrix_demo.exe [train_data_path test_data_path model_output_path]
```

方括号表示可选参数，实际运行时不要输入方括号。参数可以全部省略；评估和训练的参数必须成组提供，即 `load_model.exe` 要么不传参数，要么同时提供两个路径，`matrix_demo.exe` 要么不传参数，要么同时提供三个路径。画板可以不传参数，或提供一个模型路径。

| 程序 | 省略全部参数时使用的路径 |
| --- | --- |
| `digit_app.exe` | 模型：`.\model\mnist_model.bin` |
| `load_model.exe` | 模型：`.\model\mnist_model.bin`；测试数据：`.\data\test\test_data` |
| `matrix_demo.exe` | 训练数据：`.\data\train\train_data`；测试数据：`.\data\test\test_data`；模型输出：`.\model\mnist_model.bin` |

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

当前训练循环按固定顺序分批遍历数据。日志中的 `loss` 是最后一个批次在更新参数后重新计算的损失，`time` 是从训练开始累计的秒数；训练结束后还会输出训练集和测试集准确率。

## 代码结构

| 文件或目录 | 作用 |
| --- | --- |
| `CMakeLists.txt` | 构建核心库、三个命令行程序和可选画板，管理 raylib 依赖 |
| `include/` | 矩阵、层和网络的公共头文件 |
| `src/matrix.c` | 矩阵运算与矩阵保存、读取 |
| `src/layer.c` | 层计算、损失和层类型分派 |
| `src/network.c` | 层组合、前向反向、参数更新及模型保存、加载 |
| `programs/main.c` | MNIST 训练入口 |
| `programs/load_model.c` | 模型评估入口 |
| `programs/digit_app.c` | raylib 画板与实时预测 |
| `programs/digit_preprocess.c`、`.h` | 手写图像预处理 |
| `tools/mnist.c` | MNIST IDX 转换入口 |
| `data/` | 原始 IDX 文件及本地转换出的矩阵文件 |
| `model/mnist_model.bin` | 预训练模型 |
| `third_party/raylib/` | 可选的本地 raylib 源码，不纳入 Git |
| `build-release/` | 本地 CMake 缓存、依赖和编译产物，不纳入 Git |
| `build-msvc/` | MSVC 构建目录，Release 程序位于 `bin/Release/`，不纳入 Git |

不同入口文件各自包含 `main`。CMake 将它们构建为独立程序，不要把所有 `.c` 一次链接到同一个可执行文件中。

## 常见问题

| 现象 | 处理方式 |
| --- | --- |
| 找不到 `cmake` | 安装 CMake，并将其 `bin` 加入 PATH，重新打开整个终端应用 |
| 找不到 `gcc`、`g++` 或提示没有编译器 | 按前面的工具链安装步骤检查完整工具链、PATH 和实际命令路径 |
| 找不到 `CMAKE_MAKE_PROGRAM` 或 `mingw32-make` | 检查完整 MinGW 工具链中的 `mingw32-make.exe` 是否存在并可在当前终端调用 |
| 找不到 `Visual Studio 18 2026` 生成器或安装实例 | 确认 CMake 至少为 4.2，且已安装 Visual Studio 2026 或对应 Build Tools 的 C++ 桌面开发组件 |
| 下载 raylib 失败 | 检查 GitHub 网络访问，或按上文放置完整本地源码后重新配置 |
| 生成器与已有缓存不匹配 | 使用未配置过的新构建目录，并同时修改配置与构建命令的目录参数 |
| 构建后没有 `digit_app.exe` | 确认构建成功且 `BUILD_DIGIT_APP=ON`；需要时重新配置并显式添加 `-DBUILD_DIGIT_APP=ON` |
| 画板提示模型文件打开失败 | 检查报错中显示的实际模型路径；未传参数时，默认使用当前工作目录下的 `.\model\mnist_model.bin`，请确认文件存在 |
| 转换时提示文件打开失败 | 检查默认的 `.\data` 或传入的数据目录，并根据报错中的具体路径确认文件是否存在；确认该目录包含四个 IDX 文件，且已提前创建 `train`、`test` 子目录 |
| 训练或评估找不到数据 | 先完成转换，再从项目根目录启动对应程序 |
| 修改源码后运行效果没变 | 重新构建，并确认运行的是 `build-release/bin` 下的新程序 |

矩阵和模型二进制文件仍按本机 `size_t`、`float` 的内存表示和字节序写入，并非通用跨平台格式。CMake 管理构建过程，并不会自动改变数据格式或运行时路径；已有模型和数据应在兼容的 64 位环境下使用。

## 参考资料

- 《深度学习入门：基于 Python 的理论与实现》
- [MNIST 数据集镜像与说明：CVDF](https://github.com/cvdfoundation/mnist)
- [raylib 官方仓库](https://github.com/raysan5/raylib)
- [CMake 官方文档](https://cmake.org/documentation/)
- [MinGW-w64](https://www.mingw-w64.org/)
