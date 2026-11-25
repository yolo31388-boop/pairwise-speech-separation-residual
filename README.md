# pairwise-speech-separation-residual

## 领域

游戏、图形与多媒体

## 主题

实时人声分离与伴奏残留

## 说明

多分辨率窗、相位连续重建、推理延迟补偿。

初始状态下 `av/speech_separate.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
