# 双足机器人PPO算法训练项目
本项目基于**近端策略优化（PPO）**算法在双足机器人仿真环境中训练智能体。该环境模拟一台拥有2条腿、4个关节的双足机器人，任务是控制智能体穿越起伏复杂地形，项目同时实现了普通训练模式与困难训练模式。

## 目录
0. [双足机器人仿真环境介绍](#双足机器人仿真环境介绍)
1. [项目目录结构](#项目目录结构)
2. [训练流程](#训练流程)
    - 基于PPO算法的普通模式、困难模式训练
3. [环境配置](#环境配置)
    - 3.1 make_env() 环境创建函数
    - 3.2 observe_model() 模型观测评估函数
4. [模型性能评估](#模型性能评估)
5. [训练日志与结果数据分析](#训练日志与结果数据分析)
6. [优化改进方案](#优化改进方案)
7. [环境依赖安装说明](#环境依赖安装说明)
8. [致谢](#致谢)

## 0. 双足机器人仿真环境介绍
本双足行走仿真环境基于Box2D物理引擎开发，模拟双足机器人在各类随机地形中自主行进。智能体需要自主完成机身平衡控制、四肢关节协调运动、持续向前移动，同时规避行进过程中的地形障碍。

- **观测空间**：包含24个连续型数值，涵盖机器人机身倾角、运动速度、各个关节角度以及前方激光雷达地形探测数据。
- **动作空间**：4个连续型数值，用于控制髋关节、膝关节的输出扭矩大小。
- **奖励规则**：机器人向前行进获得正向奖励；关节扭矩过大产生能耗惩罚，机器人摔倒会给予大额负向惩罚。
- **回合终止条件**：机器人摔倒失衡，或是达到最大运行步数（普通模式上限1600步，困难模式上限2000步）。

## 1. 项目目录结构
- **main.py**：主程序文件，包含普通模式、困难模式下完整的训练逻辑。
- **run.py**：命令行入口脚本，用于通过指令执行PPO模型的训练与模型评估。
- **benchmark.py**：模型基准评估脚本，可对普通、困难模式训练的模型做性能测试，输出CSV、Markdown格式评估报告。
- **env_utils.py**：环境工具脚本，用于配置仿真环境，可按需开启帧堆叠、视频录制、观测值与奖励归一化等功能。
- **logs/**：训练日志存放目录。
- **models/**：训练完成后的PPO模型保存目录。
- **videos/**：若开启视频录制功能，机器人运行过程视频会保存在该文件夹下。

## 2. 训练流程
### 普通训练模式
- **总训练步数**：100万步
- **仿真环境**：标准双足行走环境（`BipedalWalker-v3`）
- **训练优化手段**：向量化并行环境、奖励与观测值归一化、连续多帧堆叠、训练过程视频录制
- **模型结构**：采用多层感知器（MLP）作为策略网络的PPO算法

### 困难训练模式
- **总训练步数**：500万步
- **仿真环境**：高难度双足行走环境（`BipedalWalkerHardcore-v3`）
- **训练优化手段**：沿用普通模式全部优化配置，该模式地形复杂度更高，需要更长的训练时长让模型充分学习环境特征

本项目基于Stable Baselines3框架实现PPO算法，借助向量化环境技术实现多环境并行训练，大幅提升训练效率。

## 3. 环境配置
通过`env_utils.py`脚本中封装的两个核心函数完成仿真环境的初始化配置。

### 3.1 make_env() 环境创建函数
`make_env()`函数用于初始化训练、评估所用的仿真环境，支持多项自定义配置参数：

- **环境创建**：默认加载标准环境`BipedalWalker-v3`，传入参数`hardcore=True`即可切换为高难度行走环境`BipedalWalkerHardcore-v3`。
- **渲染模式**：支持`human`实时可视化渲染模式、`rgb_array`图像输出模式（多用于视频录制场景）。
- **视频录制**：设置`record_video=True`时，每间隔1000步自动录制机器人运行画面，并保存至指定文件夹。
- **运行监控**：为环境绑定监控器，自动记录每回合奖励、运行步数等性能指标，方便后续训练数据分析。
- **向量化运算**：基于`DummyVecEnv`实现多环境并行运行，加速模型训练。
- **观测值、奖励归一化**：通过`VecNormalize`对观测数据、回合奖励做归一化处理，稳定训练过程，提升智能体学习收敛速度。
- **帧堆叠**：默认堆叠连续4帧环境状态，为智能体提供时序运动特征信息，对机器人动态行走的策略学习至关重要。
- **观测值截断**：通过`clip_obs`参数（默认值10.0）截断异常观测数据，避免离群数据干扰模型训练。
- **命令行运行支持**：可通过`run.py`脚本在命令行执行训练、模型评估，支持普通/困难模式切换、视频录制、自定义训练步数等功能。

#### 使用示例
```python
env = make_env(env_name="BipedalWalker-v3", hardcore=True, record_video=True, use_monitor=True)
```

#### 命令行运行示例
训练普通模式模型：
```bash
python run.py --task train --mode normal --timesteps 100000 --model-name ppo_bipedalwalker
```

开启视频录制，训练困难模式模型：
```bash
python run.py --task train --mode hardcore --timesteps 200000 --model-name ppo_bipedalwalker_hardcore --record-video
```

加载已训练模型进行评估：
```bash
python run.py --task eval --mode normal --model-path models/ppo_bipedalwalker.zip --eval-episodes 5
```

模型评估并录制运行视频：
```bash
python run.py --task eval --mode normal --model-path models/ppo_bipedalwalker.zip --eval-episodes 3 --record-video
```

#### 双模型基准性能测试
```bash
python benchmark.py --normal-model-path models/ppo_bipedalwalker --hardcore-model-path models/ppo_bipedalwalker_hardcore --eval-episodes 5
```
基准测试结果会保存至 `reports/benchmark_results.csv` 与 `reports/benchmark_report.md`，同时可选择将评估过程视频保存到 `reports/videos/` 文件夹。

### 3.2 observe_model() 模型观测评估函数
该函数用于加载训练完成的PPO模型，并在对应仿真环境中完成性能测试，会自动同步训练阶段使用的归一化、帧堆叠等环境配置，保证训练与评估环境完全一致。

- **模型加载**：从指定路径读取本地保存的训练模型文件。
- **环境匹配**：根据困难模式配置，自动选用对应版本的双足行走仿真环境。
- **环境配置同步**：复用训练阶段的观测归一化、奖励归一化、帧堆叠等环境封装配置。
- **性能评估**：在指定回合数内运行模型，计算平均奖励与奖励标准差，量化模型的性能与稳定性。

#### 使用示例
```python
mean_reward, std_reward = observe_model(model_path='models/ppo_bipedalwalker_1M', n_eval_episodes=5, hardcore=False)
```

该套环境配置同时适配模型训练、模型效果评估场景，灵活支持视频录制、数据归一化、帧堆叠等进阶优化功能。

## 4. 模型性能评估
调用`observe_model()`函数，通过多轮测试回合对模型性能进行量化评估，同时可视化展示机器人的实际行走效果。

### 评估结果示例
- **普通模式**：平均奖励 248.39，奖励标准差 ±112.10
- **困难模式（300万训练步）**：平均奖励 -28.23，奖励标准差 ±24.82
- **困难模式（500万训练步）**：平均奖励 -10.66，奖励标准差 ±3.91
- **困难模式（700万训练步）**：平均奖励 -5.45，奖励标准差 ±2.10

从结果可以看出：智能体在标准普通环境下行走表现良好，但在高难度地形场景中性能仍存在明显不足，需要进一步增加训练步数或是调优超参数来优化模型效果。

## 5. 训练日志与结果数据分析
本项目针对困难模式下500万训练步数产生的日志数据开展分析：
- **奖励变化趋势**：训练过程中奖励值存在小幅波动，但整体随着训练迭代逐步趋于稳定。
- **单回合运行步数趋势**：随着训练推进，机器人单次存活的运行总步数持续上升，仅存在少量回落情况。
- **数据相关性**：回合奖励与单回合运行步数呈现强正相关（相关系数0.89），说明机器人存活时间越久，获得的累计奖励越高。

项目借助`pandas`、`matplotlib`工具绘制奖励变化曲线、回合步数移动平均线等可视化图表，直观呈现模型训练收敛过程。

## 6. 优化改进方案
为进一步提升智能体的行走性能，给出如下优化建议：
- **调整学习率**：适当降低学习率，让模型参数迭代更加平稳，提升训练过程的稳定性。
- **重构奖励规则**：优化奖励函数设计，优先引导机器人维持机身平衡、延长存活时间，而非仅以向前移动作为主要奖励依据。
- **提升智能探索能力**：引入ε-贪心策略、好奇心驱动探索等算法，让智能体探索更多样的行走策略。
- **延长训练时长**：增加总训练步数，让智能体积累更多环境交互经验，优化策略网络决策效果。

### 6.5 学习曲线可视化（learning_curve.py）
项目新增`learning_curve.py`脚本，可一键完成「短样本训练→日志读取→训练曲线绘图」全流程，无需额外配置即可直接运行：

```bash
python learning_curve.py                       # 默认训练5000步
python learning_curve.py 20000                 # 自定义训练总步数
python learning_curve.py 5000 demo.png         # 自定义步数 + 指定结果图片保存名称
```

执行流程：
1. 调用`env_utils.py`中的监控封装器创建`BipedalWalker-v3`环境，自动生成`logs/*.monitor.csv`格式训练日志；
2. 基于PPO算法+多层感知器策略网络完成指定步数训练（CPU环境下训练20000步约耗时50秒）；
3. 自动读取日志文件，在控制台输出Markdown格式训练统计表格，包含总回合数、平均奖励、最高奖励、末期平均奖励、单回合平均运行步数；
4. 借助matplotlib生成双栏可视化折线图：左图为「回合奖励+滚动均值曲线」，右图为「单回合步数+滚动均值曲线」。

下图为20000步训练、80个回合后的可视化结果：可以看到单回合运行步数从初始1600步逐步收敛至100步左右，体现出智能体正在自主学习规避无效探索行为。

![学习曲线](result_learning_curve.png)

## 7. 环境依赖安装说明
执行下方命令，通过项目依赖配置文件一键安装所有运行所需第三方库：
```bash
pip install -r requirements.txt
```

### 依赖清单
- Python 3.8 及以上版本
- gymnasium：强化学习仿真环境框架
- stable-baselines3：PPO强化学习算法开源框架
- pandas、matplotlib：训练日志数据处理与可视化绘图

## 8. 多模型对比评估
项目新增`compare_models.py`工具脚本，可对多个训练完成的PPO模型做横向对比评估，自动生成CSV数据表、Markdown评估报告与性能对比柱状图。

### 使用示例
```bash
python compare_models.py --model-paths models/ppo_bipedalwalker models/ppo_bipedalwalker_hardcore --labels normal hardcore --mode normal --eval-episodes 5
```

### 可选参数说明
- `--model-paths`：至少传入两个待对比的模型文件路径，使用空格分隔；
- `--labels`：和模型路径一一对应的模型名称标签；
- `--mode`：可选参数`normal`（普通模式）或`hardcore`（困难模式）；
- `--eval-episodes`：单模型评估的回合数量；
- `--output-dir`：评估报告输出文件夹，默认路径为`comparison_reports`；
- `--record-video`：是否录制模型评估过程的运行视频；
- `--video-folder`：评估视频保存目录，默认路径为`comparison_reports/videos`。

### 输出文件内容
- `comparison_reports/comparison_results.csv`
- `comparison_reports/comparison_report.md`
- `comparison_reports/comparison_plot.png`

### 8.5 批量评估与奖励分布箱线图绘制
新增`batch_eval_boxplot.py`工具脚本，对多个模型进行多组重复测试，绘制各模型回合奖励分布的箱线图，直观展示模型性能稳定性。

### 使用示例
```bash
python batch_eval_boxplot.py --model-paths models/ppo_bipedalwalker models/ppo_bipedalwalker_hardcore --labels normal hardcore --mode normal --eval-episodes 5 --trials 10
```

### 输出文件内容
- `comparison_reports/batch_boxplot_results.csv`（每行包含模型标签、模型路径、单回合奖励）
- `comparison_reports/batch_boxplot.png`（奖励分布箱线图）

## 9. 致谢
本项目基于奥列格·克利莫夫的开源仿真环境二次开发，依托Stable Baselines3框架完成PPO算法的训练部署，在此向所有开源项目贡献者表示感谢。