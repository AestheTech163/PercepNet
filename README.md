# PercepNet
0. 该模型是 2020 年（好像是）亚马逊在 DNS 挑战赛提交的轻量级降噪模型，当年夺得了轻量级的第一名，由传统 dsp 算法结合深度学习，足够复杂，但非常值得学习。
1. 纯 Python 对轻量级降噪模型 PercepNet 的复现，不包括落地代码以及 相关dsp算法的 c 代码重写。
2. 因此，本份代码不是面向落地的。
3. 其中 dsp 算法的 音高跟踪，请参考其它开源代码，如：Rnnoise 内置的一份实现，轻量级、可用，但不是最好的。

原论文：A Perceptually-Motivated Approach for Low-Complexity, Real-Time Enhancement of Fullband Speech

其中：对核心的量 音高滤波强度 r 的计算，使用的 推导出来的严格公式，而非论文上的近似公式。见：https://www.jianshu.com/p/80ab3b6ad638
