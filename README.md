# video-scene-cut-detect

## 题面

自动拆镜头后，运动镜头被切成十几段、渐变转场又完全漏掉：直方图不归一，亮帧距离偏大；卡方距离不开方，阈值不可比；只比颜色不比边缘，运动镜头误切；不做最短镜头约束，一秒切十次；合并短镜头取平均边界，切点漂移；渐变检测用固定窗，长短转场都漏；越界切点不校验，越界访问；乱序切点不报错，导出失败；批量共用直方图，片之间串；bins=0 不校验，除零。scenedetect/cuts.py 是核心，tests/test_cuts.py 15个红。要修的是：按 8 个函数的 docstring 契约修复 `scenedetect/cuts.py`，消除上述 10 个缺陷，同时保持 `tests/test_cuts.py` 不变。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 音视频处理 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 运行方式

在仓库根目录执行：

```powershell
cd D:\gsb-20261119\video-scene-cut-detect
python -m pytest tests -q
```

初始状态必须恰好是 **15 failed**，且没有 collection / import / syntax error。输出尾部形如：

```text
15 failed in 0.xxs
```

15 条测试都会因 `NotImplementedError` 失败；这是预期的红测，不代表测试或环境损坏。修复 `scenedetect/cuts.py` 后应变为 `15 passed`。

## 目录

```text
video-scene-cut-detect/
├── README.md
├── conftest.py
├── .gitignore
├── scenedetect/
│   ├── __init__.py
│   └── cuts.py
└── tests/
    └── test_cuts.py
```

## 测试不可修改

`tests/test_cuts.py` 是固定的 15 条验收测试，不得修改、删除、跳过或弱化断言；只能修改 `scenedetect/cuts.py`（必要时可改 `scenedetect/__init__.py` 的导出）。测试覆盖 10 个缺陷各一条，以及相同帧距离为 0（1e-6）、全黑与全白距离为 1（1e-6）、`bins=0` 抛 `SceneError`、单帧片返回空切点、批量全成功时顺序不变等组合场景。
