# Carla-YOLO-DeepSort 多车跟踪系统

【第十次迭代更新：完善项目说明文档】

---

## 📋 项目简介

【第十次迭代更新：完善项目说明文档】

本项目基于 **YOLOv8 + DeepSORT** 实现自动驾驶仿真环境中的多车辆目标跟踪系统，在 CARLA 模拟器中实时进行车辆检测、跟踪、碰撞预警、车速分析、违章检测、车流量统计、车辆属性识别等功能。项目兼容 Python 3.8.10 + CARLA 0.9.13 + Windows 10 开发环境，提供完整的端到端车辆感知与行为分析解决方案。

### 核心技术栈

- **目标检测**: YOLOv8（Ultralytics），在 CARLA 数据集上训练
- **多目标跟踪**: DeepSORT（深度排序算法），结合卡尔曼滤波和匈牙利算法
- **仿真环境**: CARLA 0.9.13 开源自动驾驶模拟器
- **深度学习框架**: PyTorch + CUDA/cuDNN 加速
- **计算机视觉**: OpenCV 图像绘制与视频处理

---

## ⚙️ 软硬件环境

【第十次迭代更新：完善项目说明文档】

### 硬件环境

| 项目 | 推荐配置 |
|:---:|:---:|
| CPU | 多核处理器（推荐 8 核以上） |
| GPU | NVIDIA GPU（显存 ≥ 6GB，支持 CUDA） |
| 内存 | 8GB 以上 |
| 硬盘 | 50GB 可用空间（存放 CARLA 模拟器与模型权重） |

### 软件环境

| 项目 | 版本要求 |
|:---:|:---:|
| 操作系统 | Windows 10/11 |
| Python | 3.8.10 |
| CARLA Simulator | 0.9.13 |
| CUDA Toolkit | 11.x（与 PyTorch 版本对应） |
| cuDNN | 与 CUDA 版本匹配 |
| PyTorch | 1.13+（CUDA 版本需匹配） |
| Ultralytics | 8.0.150 |
| NumPy | 1.24.x |
| OpenCV | 4.8.x |

### 第三方依赖

```text
torch
torchvision
torchaudio
ultralytics==8.0.150
numpy
opencv-python
pygame
matplotlib
pyyaml
```

【第十次迭代更新：完善项目说明文档】

依赖安装命令：

```bash
pip install -r requirements.txt
```

或直接安装核心依赖：

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install ultralytics==8.0.150 numpy opencv-python
```

---

## 🔧 八大功能明细

【第十次迭代更新：完善项目说明文档】

### 模块一：车辆目标检测与多目标跟踪

**核心功能**：基于 YOLOv8 检测画面中的车辆目标，通过 DeepSORT 为每个检测到的车辆分配唯一跟踪 ID，实现跨帧连续追踪。

- **YOLOv8 目标检测**：调用已训练好的 yolov8n.pt 模型，对输入帧进行车辆目标检测（类别包括：轿车、SUV、面包车、货车等），输出检测框坐标、置信度与类别标签
- **DeepSORT 多目标跟踪**：以检测结果为输入，通过特征提取、卡尔曼滤波预测与匈牙利算法关联，为每个车辆分配稳定的 track_id，实现跨帧 ID 一致性
- **目标框可视化**：在画面上绘制每辆车的检测框，框上标注跟踪 ID、类别标签、置信度，颜色按 track_id 哈希生成以区分不同车辆

**代码入口**：[track.py](file:///C:/Users/86176/Desktop/多车跟踪/Carla-YOLO-DeepSort-Multi-Object-Tracking/track.py) 中的 `VehicleTracker.yolo_details()` 与 `VehicleTracker.process_frame()` 方法。

【第十次迭代更新：完善项目说明文档】

---

### 模块二：轨迹预测与碰撞预警

**核心功能**：基于车辆历史轨迹，预测未来数帧位置，检测车辆间潜在的碰撞风险，并在画面上高亮预警。

- **轨迹历史存储**：为每个 track_id 维护最近 10 帧中心点坐标列表，记录行驶轨迹
- **线性预测**：基于当前帧与上一帧速度向量，线性预测未来 3 帧的车辆位置
- **碰撞距离判定**：计算两两车辆预测位置的欧氏距离，当距离小于画面宽度 10%（可调）时判定为碰撞风险
- **预警可视化**：在画面顶部显示"COLLISION WARNING!"文字警告，将风险车辆的检测框加粗并标红

**代码入口**：[track.py](file:///C:/Users/86176/Desktop/多车跟踪/Carla-YOLO-DeepSort-Multi-Object-Tracking/track.py) 中的 `update_trajectory()`、`predict_future_position()`、`check_trajectory_collision()`、`draw_collision_warning()` 函数。

【第十次迭代更新：完善项目说明文档】

---

### 模块三：车速估算与超速报警

**核心功能**：利用车辆中心点位移和帧率换算出实时行驶速度，当超过限速阈值时发出超速报警。

- **速度计算**：基于相邻两帧中心点的像素位移，乘以像素-米换算系数（默认 0.1 m/px），除以帧间秒数得到速度（km/h）
- **限速判定**：默认限速 60 km/h（可调），超过该值即判定为超速
- **超速可视化**：超速车辆的检测框标红加粗，框上标注实时速度值，控制台输出车辆 ID 与当前速度
- **UI 面板显示**：顶部信息栏实时显示当前超速车辆数量

**代码入口**：[track.py](file:///C:/Users/86176/Desktop/多车跟踪/Carla-YOLO-DeepSort-Multi-Object-Tracking/track.py) 中的 `update_vehicle_trajectories()`、`calculate_speed()`、`estimate_vehicle_speeds()` 函数。

【第十次迭代更新：完善项目说明文档】

---

### 模块四：违章行为检测（逆行 + 拥堵）

**核心功能**：实时检测车辆的行驶方向异常（逆行）以及区域车流密度异常（拥堵）。

- **逆行判定**：基于车辆最近 8 帧中心点的 y 坐标变化方向，判断车辆是否逆向行驶，控制台输出警告信息
- **拥堵判定**：统计当前画面内车辆总数，当车辆数 ≥ 6（可调阈值）时判定为拥堵状态
- **违章可视化**：逆行车辆检测框标红加粗；顶部信息栏显示逆行车辆数和拥堵状态；画面左上角显示"逆行/拥堵状态"文字提示

**代码入口**：[track.py](file:///C:/Users/86176/Desktop/多车跟踪/Carla-YOLO-DeepSort-Multi-Object-Tracking/track.py) 中的 `update_violation_trajectories()`、`detect_violations()`、`draw_violation_warnings()` 函数。

【第十次迭代更新：完善项目说明文档】

---

### 模块五：车辆轨迹绘制

**核心功能**：在画面上绘制每辆车的运动轨迹线，直观展示车辆行驶路径。

- **轨迹点存储**：为每辆车维护最近 15 帧中心点坐标队列
- **连线绘制**：将轨迹点按时间顺序依次连线，绘制半透明轨迹线，颜色与对应车辆一致
- **起点标记**：在轨迹起点绘制实心圆点，便于识别轨迹起始位置

**代码入口**：[track.py](file:///C:/Users/86176/Desktop/多车跟踪/Carla-YOLO-DeepSort-Multi-Object-Tracking/track.py) 中的 `update_vehicle_path()`、`draw_vehicle_trajectories()` 函数。

【第十次迭代更新：完善项目说明文档】

---

### 模块六：车流量统计

**核心功能**：设置虚拟计数线（画面中下部横向线，默认 y=450），统计驶入与驶出车辆数量。

- **虚拟计数线**：在画面固定 y 坐标位置设置横向统计线
- **跨线判定**：跟踪每辆车的 y 坐标变化，当从上向下跨越计数线时计为"驶入"，从下向上跨越计为"驶出"，同一辆车仅计数一次
- **统计信息 UI**：左上角半透明面板显示总车流量、驶入车辆数、驶出车辆数三项实时统计

**代码入口**：[track.py](file:///C:/Users/86176/Desktop/多车跟踪/Carla-YOLO-DeepSort-Multi-Object-Tracking/track.py) 中的 `update_traffic_counting()`、`draw_traffic_counting_ui()` 函数。

【第十次迭代更新：完善项目说明文档】

---

### 模块七：车辆属性识别（车型 + 车身颜色）

**核心功能**：基于检测框长宽比与画面色彩分析，自动识别每辆车的车型与车身颜色并标注。

- **车型识别**：计算检测框高/宽比值，划分为 4 类：
  - 比值 < 0.7 → **轿车**
  - 比值 0.7 ~ 1.1 → **SUV**
  - 比值 1.1 ~ 1.5 → **面包车**
  - 比值 > 1.5 → **货车**
- **颜色识别**：截取车辆检测框中心区域（约占框 40%），统计 RGB 通道均值：
  - 亮度极高 → **白色**
  - 亮度极低 → **黑色**
  - RGB 均衡中等亮度 → **灰色**
  - R 显著高于其他通道 → **红色**
  - R、G 均高且 B 偏低 → **黄色**
  - B 显著高于其他通道 → **蓝色**
- **属性标注**：在检测框右下角小字显示"颜色+车型"文本（如"白色轿车"）
- **统计信息**：顶部信息栏实时显示各类车型数量与各色车辆占比

**代码入口**：[track.py](file:///C:/Users/86176/Desktop/多车跟踪/Carla-YOLO-DeepSort-Multi-Object-Tracking/track.py) 中的 `classify_car_type()`、`classify_car_color()`、`update_car_attr()`、`get_car_type_stats()`、`get_car_color_stats()`、`draw_car_attr_label()` 函数。

【第十次迭代更新：完善项目说明文档】

---

### 模块八：UI 信息面板增强

**核心功能**：在画面顶部和指定位置绘制半透明信息面板，集中展示所有感知模块的实时数据。

- **顶部信息栏**：画面顶端绘制黑色半透明横条（高度 40px，透明度 60%），均匀排列以下信息项：
  - 车辆总数（当前画面跟踪到的车辆数）
  - 碰撞预警次数（累计触发预警的次数）
  - 超速车辆数（当前帧速度 > 60 km/h 的车辆数）
  - 逆行车辆数（当前帧判定为逆行的车辆数）
  - 拥堵状态（正常/拥堵，拥堵项用红色突出显示）
  - 车型统计（各车型实时数量汇总）
  - 颜色统计（各色车辆占比百分比）
- **左上统计面板**：独立半透明面板显示车流量统计数据（总车流量、驶入数、驶出数）
- **统一视觉风格**：所有 UI 元素采用半透明背景 + 黄/白高亮文字，确保不遮挡核心检测画面同时信息可读

**代码入口**：[track.py](file:///C:/Users/86176/Desktop/多车跟踪/Carla-YOLO-DeepSort-Multi-Object-Tracking/track.py) 中的 `draw_ui_info_bar()`、`draw_traffic_counting_ui()`、`draw_violation_warnings()`、`draw_collision_warning()` 等绘制函数。

【第十次迭代更新：完善项目说明文档】

---

## 🚀 运行步骤

【第十次迭代更新：完善项目说明文档】

### 第 1 步：安装 CARLA 模拟器

1. 下载 **CARLA 0.9.13** 模拟器：https://carla.readthedocs.io/en/latest/start_quickstart/
2. 解压到本地目录（例如 `C:\carla_0.9.13`）
3. 确保目录结构中包含 `CarlaUE4.exe` 主程序

### 第 2 步：安装 Python 依赖

```bash
# 创建并激活虚拟环境（推荐）
conda create --name carla_track python=3.8
conda activate carla_track

# 安装 PyTorch（根据你的 CUDA 版本选择合适的安装命令）
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# 安装 YOLOv8 与其他计算机视觉依赖
pip install ultralytics==8.0.150
pip install numpy opencv-python
```

【第十次迭代更新：完善项目说明文档】

### 第 3 步：准备模型权重文件

项目所需模型权重文件已存放于 `weights/` 目录：

- `weights/yolov8n.pt` — YOLOv8 预训练检测模型
- `weights/best.pt` — 在 CARLA 数据集上训练的专用检测模型
- `deep_sort/deep/checkpoint/ckpt.t7` — DeepSORT 特征提取模型权重

> 如缺失权重文件可从百度网盘下载：链接: https://pan.baidu.com/s/1qubYmwj-uwLHeTeCifNjJw?pwd=ujqw 提取码: ujqw

【第十次迭代更新：完善项目说明文档】

### 第 4 步：启动 CARLA 模拟器

```bash
# 在 CARLA 安装目录下运行
CarlaUE4.exe
```

等待模拟器加载完毕，呈现 CARLA 默认城市街道场景。

### 第 5 步：运行跟踪系统

```bash
# 进入项目根目录
cd C:\Users\86176\Desktop\多车跟踪\Carla-YOLO-DeepSort-Multi-Object-Tracking

# 运行主程序（自动连接 CARLA 进行实时跟踪）
python track.py

# 或使用本地视频文件模式（跳过 CARLA 连接）
python track.py --video your_video.mp4

# 不显示画面窗口，仅后台处理
python track.py --video your_video.mp4 --no-display

# 保存输出视频
python track.py --video your_video.mp4 --save-output
```

程序启动后将自动：
1. 检测并打印当前运行环境与依赖状态
2. 加载 YOLOv8 模型与 DeepSORT 权重
3. 从 CARLA 模拟器（或视频文件）读取画面帧
4. 执行检测、跟踪、碰撞预警、车速估算、违章检测、轨迹绘制、车流量统计、属性识别八大功能模块
5. 输出带标注的跟踪画面窗口

按 `q` 键退出程序运行。

【第十次迭代更新：完善项目说明文档】

---

## 🎬 演示效果

【第十次迭代更新：完善项目说明文档】

### 功能演示一览

| 功能模块 | 演示效果说明 | 画面呈现 |
|:---:|:---:|:---:|
| 车辆检测与跟踪 | 多车辆实时检测框 + 唯一 ID 稳定分配 | 彩色检测框包围车辆，顶部标注 track_id |
| 碰撞预警 | 车辆接近时画面警告提示 | 风险车辆框标红加粗 + 顶部文字警告 |
| 车速估算 | 实时速度显示与超速标记 | 检测框显示"车辆ID 85.3km/h" |
| 违章检测 | 逆行与拥堵自动识别 | 逆行车辆标红 + 拥堵状态提示 |
| 轨迹绘制 | 车辆行驶路径可视化 | 彩色半透明连线跟随车辆移动 |
| 车流量统计 | 虚拟计数线实时统计 | 左上显示"总车流量/驶入/驶出" |
| 车辆属性识别 | 车型与颜色智能标注 | 框右下角小字"白色轿车"等 |
| UI 信息面板 | 顶部集中展示统计数据 | 黑色半透明横条 + 黄白高亮文本 |

### 输出示例

运行程序后将看到类似以下的控制台输出：

```
============================================================
YOLO + DeepSort 多车跟踪系统
【新增：轨迹预测+提前碰撞预警模块】
【新增：车速估算 + 超速报警】
【新增：违章行为检测】
【新增：UI界面增强模块】
【新增：车辆轨迹绘制模块】
【新增：第六模块 车流量统计】
【新增：第七模块 车辆属性识别】
============================================================
Python: 3.8.10
平台: win32
============================================================

[INFO] 使用设备: cuda
[INFO] YOLO 模型加载成功: weights/yolov8n.pt
[INFO] DeepSort 加载成功: deep_sort/deep/checkpoint/ckpt.t7
[INFO] 初始化完成，准备开始跟踪

[提前碰撞预警] 车辆ID:3 和 车辆ID:7 距离过近，存在碰撞风险
【超速警告】车辆 ID:5 当前速度：88.2 km/h
【逆行警告】车辆 ID:12 存在逆行行为
【异常停车警告】车辆ID:8 长时间静止，疑似违停/事故
[INFO] 已处理 30 帧
[INFO] 已处理 60 帧
...
```

画面窗口中将实时展示：车辆检测框、跟踪 ID、轨迹线、速度标注、碰撞/超速/逆行高亮、车流量面板、属性标注等八大功能的可视化效果。

> 项目自带示例输出视频 `output.mp4` 可供参考，运行时添加 `--save-output` 参数可保存当前演示视频。

【第十次迭代更新：完善项目说明文档】

---

## 📌 项目文件结构

【第十次迭代更新：完善项目说明文档】

```
Carla-YOLO-DeepSort-Multi-Object-Tracking/
├── track.py                              # 主文件（含八大功能模块的完整实现）
├── yolo_deepsort.py                      # YOLO+DeepSORT 备用实现
├── requirements.txt                      # Python 依赖清单
├── requirements_python38.txt             # Python 3.8 专用依赖
├── data.yaml                             # YOLO 数据集配置
├── output.mp4 / w_output.mp4             # 示例输出视频
├── README .MD                            # 项目说明文档（本文件）
├── 使用说明.md                           # 中文快速使用指南
├── weights/                              # 模型权重目录
│   ├── yolov8n.pt                       # YOLOv8 预训练模型
│   └── best.pt                          # CARLA 训练模型
├── deep_sort/                           # DeepSORT 实现
│   ├── configs/deep_sort.yaml           # 配置文件
│   └── deep/checkpoint/ckpt.t7          # 特征提取权重
└── runs/detect/train17/                 # YOLO 训练日志与评估结果
```

【第十次迭代更新：完善项目说明文档】

---

## 📚 参考资源

- CARLA 官方文档：https://carla.readthedocs.io/
- YOLOv8 官方仓库（Ultralytics）：https://github.com/ultralytics/ultralytics
- DeepSORT 论文：*Simple Online and Realtime Tracking with a Deep Association Metric*
- CARLA 数据集（Kaggle）：https://www.kaggle.com/datasets/alechantson/carladataset

【第十次迭代更新：完善项目说明文档】

---
