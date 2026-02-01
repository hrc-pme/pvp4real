
<h1 align="center">PVP4Real：人類介入下的資料高效率學習</h1>

<p align="center">
  <img src="assets/PVP4Real_Teaser.png" alt="PVP4Real" width="100%">
</p>

本專案包含模擬人類介入的訓練與評估腳本，對應 ICRA 2025 論文與網頁：
<a href="https://metadriverse.github.io/pvp4real/"><b>Webpage</b></a> |
<a href="https://github.com/metadriverse/pvp4real"><b>Code</b></a> |
<a href="https://arxiv.org/pdf/2503.04969"><b>PDF</b></a>

## 專案大致架構

- [pvp4real/pvp](pvp4real/pvp) ：主要訓練與算法實作
  - [pvp4real/pvp/experiments](pvp4real/pvp/experiments) ：各種實驗入口（含 MetaDrive）
  - [pvp4real/pvp/sb3](pvp4real/pvp/sb3) ：內建的 RL/控制演算法
  - [pvp4real/pvp/utils](pvp4real/pvp/utils) ：工具與輔助模組
- [pvp4real/scripts](pvp4real/scripts) ：一鍵啟動實驗的 shell 腳本
- [docker](docker) ：容器化環境（CPU 版本）
- [assets](assets) ：專案圖片與資源

## 如何使用


### 啟動實驗

常用腳本在 [pvp4real/scripts](pvp4real/scripts)：

```bash
bash scripts/metadrive_simhuman_pvp4real.sh
```

或直接執行訓練入口：

```bash
python pvp/experiments/metadrive/train_pvp_metadrive_fakehuman.py \
  --exp_name="pvp4real" \
  --bc_loss_weight=1.0
```
