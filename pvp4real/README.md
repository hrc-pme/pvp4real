# PVP4Real 訓練模組說明

本目錄包含 PVP4Real 的核心訓練程式碼與實驗腳本。

## 📂 目錄結構

### `pvp/experiments/`
實驗入口與環境設定
- **MetaDrive 實驗**：`metadrive/` 目錄包含各種訓練入口
  - `train_pvp_metadrive_fakehuman.py`：PVP/PVP4Real/BC 訓練
  - `train_haco_metadrive_fakehuman.py`：HACO 算法訓練
  - `train_td3_metadrive.py`：TD3 基線訓練
  - `train_ppo_metadrive.py`：PPO 基線訓練
  - `train_eil_metadrive_fakehuman.py`：EIL 算法訓練
- **環境配置**：`egpo/fakehuman_env.py` 提供模擬人類介入環境
- **專家策略**：`egpo/` 包含預訓練專家模型載入

### `pvp/sb3/`
Stable-Baselines3 強化學習算法實作
- **核心算法**：
  - `td3/`：Twin Delayed DDPG 算法
  - `ppo/`：Proximal Policy Optimization
  - `haco/`：Human-AI Cooperation 算法
  - `sac/`：Soft Actor-Critic
  - `ddpg/`、`dqn/`、`a2c/` 等其他算法
- **通用元件**：
  - `common/buffers.py`：經驗回放緩衝區
  - `common/policies.py`：策略網路
  - `common/callbacks.py`：訓練回呼（檢查點、評估）
  - `common/monitor.py`：環境監控與記錄
  - `common/wandb_callback.py`：Weights & Biases 整合

### `pvp/utils/`
工具函式與評估配置
- `train_eval_config.py`：訓練與評估超參數
- `expert_common.py`：專家策略相關工具
- `shared_control_monitor.py`：共享控制數據監控

### `scripts/`
批次實驗啟動腳本（詳見下表）

---

## 🔬 訓練腳本比較

| 腳本名稱 | 訓練算法 | 訓練入口 | 關鍵參數 | WandB | 用途說明 |
|---------|---------|---------|---------|-------|---------|
| `metadrive_simhuman_pvp4real.sh` | **PVP4Real** | `train_pvp_metadrive_fakehuman.py` | `--bc_loss_weight=1.0`<br>`--with_human_proxy_value_loss=True` | ❌ | PVP + BC loss + 人類代理價值損失（論文主方法） |
| `metadrive_simhuman_pvp.sh` | **PVP** | `train_pvp_metadrive_fakehuman.py` | `--bc_loss_weight=0.0` | ❌ | 純 PVP（無 BC loss） |
| `metadrive_simhuman_bc.sh` | **BC** | `train_pvp_metadrive_fakehuman.py` | `--only_bc_loss=True`<br>`--free_level=-10000.0` | ❌ | 純行為複製（無 RL） |
| `metadrive_simhuman_haco.sh` | **HACO** | `train_haco_metadrive_fakehuman.py` | - | ❌ | HACO 人機協作算法 |
| `metadrive_simhuman_ppo.sh` | **PPO** | `train_ppo_metadrive.py` | - | ❌ | PPO 基線方法 |
| `metadrive_simhuman_td3.sh` | **TD3** | `train_td3_metadrive.py` | - | ❌ | TD3 基線方法 |
| `metadrive_simhuman_eil.sh` | **EIL** | `train_eil_metadrive_fakehuman.py` | - | ✅ | EIL 模仿學習 |
| `metadrive_simulation_hgdagger.sh` | **HGDagger** | `train_pvp_metadrive_fakehuman.py` | `--only_bc_loss=True` | ✅ | HGDagger 迭代式專家修正 |

### 共同特性
- **並行訓練**：每個腳本啟動 8 個進程（不同 seed）
- **Seeds**：`[0, 100, 200, 300, 400, 500, 600, 700]`
- **背景執行**：使用 `nohup` 並重導向輸出至 `.log` 檔
- **GPU 分配**：透過 `CUDA_VISIBLE_DEVICES` 指定 GPU

### 關鍵差異
1. **損失函數**：PVP4Real 混合 RL + BC，BC 只用模仿，PPO/TD3 純 RL
2. **人類介入**：PVP 系列考慮人類接管，PPO/TD3 無人類介入
3. **訓練入口**：不同算法使用不同訓練腳本
4. **監控工具**：部分腳本整合 WandB 線上監控

---

## 🚀 使用範例

### 啟動 PVP4Real 訓練
```bash
cd pvp4real/scripts
bash metadrive_simhuman_pvp4real.sh
```

### 監控訓練進度
```bash
# 即時檢視 log
tail -f metadrive_simhuman_pvp4real_seed0.log

# 使用 TensorBoard
tensorboard --logdir=/home/zhenghao/pvp/runs --host=0.0.0.0
```

### 停止訓練
```bash
# 在容器內執行
pkill -f train_pvp_metadrive_fakehuman
```

---

## 📊 輸出結果

訓練結果儲存於 `/home/zhenghao/pvp/runs/`：
- `models/`：模型檢查點（每 500 步儲存）
- `data/`：共享控制數據
- TensorBoard 日誌檔
