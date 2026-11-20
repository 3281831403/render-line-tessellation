# render-line-tessellation

## 题面

把 UI 描边换成几何线段后，拐角长刺、曲线还有折线感：expand_segment 的法线方向取反，四边形朝内；miter_join 不设上限，尖角出现长刺；tessellate_bezier 固定 16 段，曲率大时折线明显；细分容差不参与，近距离也不细分；dash_pattern 的 offset 不取模，负 offset 全部丢失；snap_to_pixel 用 round 做半像素对齐，线宽奇偶不一致；退化段（长度为 0）不剔除，法线为 NaN；validate_mesh 不查索引越界，越界写；tessellate_batch 共用顶点缓冲，线之间串；线宽为 0 不报错，输出空网格。linetess/tessellator.py 是核心，tests/test_tessellator.py 15个红。要修的是：修复上述 10 个缺陷，使 15 个测试全部通过。

## 运行方式

在仓库根目录执行：

```bat
python -m pytest tests -q
```

初始状态必须恰好为 `15 failed`，且没有 collection/import/syntax error。修好 `linetess/tessellator.py` 后，同一命令应变为 `15 passed`。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 图形渲染 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约束

`tests/` 是评测用例，不可修改；只允许修改 `linetess/tessellator.py`。实现仅使用 Python 标准库（`math`），离线运行，不使用随机数；所有坐标均为屏幕空间像素，线宽以像素计，几何比较容差为 `1e-6`。
