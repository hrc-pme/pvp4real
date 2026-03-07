# Offline Training Configuration Guide

## 概述

本目錄包含 PVP4Real 離線訓練的兩種配置模式：

- **`config.yaml`**: 從頭訓練 (Train from Scratch)
- **`config.resume.yaml`**: 繼續訓練 (Resume Training)

---

## Train from Scratch (`config.yaml`)

### 使用時機
第一次訓練，使用純人類介入(human intervention)的 ROS bag 資料。

### 資料特性
- **資料來源**: 完全由人類遙控的 bag 檔案
- **動作特性**: `actions_novice == actions_behavior`（所有動作都是人類執行）
- **Buffer 內容**: 僅包含人類示範資料

### Loss 設定

```yaml
training:
  is_resume_training: false
  
  pvp:
    with_human_proxy_value_loss: true   # ✅ 開啟
    with_agent_proxy_value_loss: false  # ❌ 關閉
    add_bc_loss: true                   # ✅ 開啟
    agent_data_ratio: 0.0
```

### 設定緣由

1. **`with_human_proxy_value_loss: true`**  
   - 推動 Q(obs, teleop_action) → `+q_value_bound`
   - 學習人類動作的高價值

2. **`with_agent_proxy_value_loss: false`** (必須關閉)  
   - 在 scratch 階段，所有資料都是人類執行的
   - `novice_action == behavior_action`，兩者相同
   - 如果開啟會導致**梯度衝突**：
     - Human proxy loss 推 Q(obs, action) → +bound
     - Agent proxy loss 推 Q(obs, action) → -bound
     - 兩個 loss 作用在同一個 (obs, action) pair 上，互相抵消

3. **`add_bc_loss: true`**  
   - Actor 透過 BC (Behavioral Cloning) 模仿人類動作
   - 是主要的行為學習機制

---

## Resume Training (`config.resume.yaml`)

### 使用時機
從 online HITL (Human-in-the-Loop) checkpoint 繼續訓練。

### 資料特性
- **資料來源**: Online HITL 階段產生的 checkpoint + buffer
- **動作特性**: `actions_novice != actions_behavior`
  - `novice`: Agent 自主決策的動作
  - `behavior`: 實際執行的動作（人類介入時為遙控動作）
- **Buffer 內容**: 混合了 agent 自主資料和人類示範資料

### Loss 設定

```yaml
training:
  is_resume_training: true
  
  checkpoint:
    resume_from: "models/offline/0002/chkpt-5000.zip"  # 必填
    
  buffer:
    resume_from: "models/offline/0002"  # 選填
  
  pvp:
    with_human_proxy_value_loss: true   # ✅ 開啟
    with_agent_proxy_value_loss: true   # ✅ 開啟 (關鍵差異!)
    add_bc_loss: true                   # ✅ 開啟
```

### 設定緣由

1. **`with_human_proxy_value_loss: true`**  
   - 推動 Q(obs, teleop_action) → `+q_value_bound`
   - 繼續強化人類動作的高價值

2. **`with_agent_proxy_value_loss: true`** (現在安全開啟)  
   - 推動 Q(obs, novice_action) → `-q_value_bound`
   - 降低 agent 自主動作的價值（這些動作導致需要人類介入）
   - **為什麼現在可以開啟**：
     - Replay buffer 包含真實的 agent 自主資料
     - `novice_action ≠ behavior_action`，兩者不同
     - 不會產生梯度衝突

3. **`add_bc_loss: true`**  
   - Actor 繼續學習模仿人類動作
   - 結合 value loss 引導，形成完整的 PVP 訓練

### Resume 必要設定

```yaml
checkpoint:
  resume_from: "models/offline/0002/chkpt-5000.zip"  # 必須指定

buffer:
  resume_from: "models/offline/0002"  # 可選，推薦使用
```

- **checkpoint.resume_from**: 必填，指向要繼續訓練的模型 `.zip` 檔案
- **buffer.resume_from**: 選填，指向包含 `buffer_human-*.pkl` 和 `buffer_replay-*.pkl` 的目錄

---

## 核心差異總結

| 項目 | Scratch | Resume | 差異原因 |
|------|---------|--------|----------|
| **is_resume_training** | `false` | `true` | 訓練模式標記 |
| **with_agent_proxy_value_loss** | ❌ `false` | ✅ `true` | Scratch 時會導致梯度衝突 |
| **資料來源** | Human-only bag | HITL checkpoint + buffer | Resume 含 agent 自主資料 |
| **agent_data_ratio** | `0.0` | (未設定) | Scratch 無 agent 資料 |
| **checkpoint.resume_from** | `null` | **必須指定** | Resume 需載入模型 |
| **buffer.resume_from** | `null` | 推薦指定 | Resume 可載入 buffer |

---

## 使用範例

### Scratch 訓練
```bash
cd pvp4real/pvp4real
python scripts/train/offline/train.py --config scripts/train/offline/config.yaml
```

### Resume 訓練
```bash
# 1. 編輯 config.resume.yaml，設定 resume_from 路徑
# 2. 執行訓練
cd pvp4real/pvp4real
python scripts/train/offline/train.py --config scripts/train/offline/config.resume.yaml
```

---

## 注意事項

⚠️ **梯度衝突警告**  
不要在 scratch 訓練時開啟 `with_agent_proxy_value_loss`！這會導致：
- Human proxy: Q(obs, action) → +bound
- Agent proxy: Q(obs, action) → -bound  
因為此時 action 相同，造成梯度相互抵消，訓練無效。

✅ **正確流程**  
1. 用 `config.yaml` 從 human bag 開始訓練
2. 部署到 online HITL 環境收集 agent 自主資料
3. 用 `config.resume.yaml` 載入 checkpoint 繼續訓練（完整 PVP loss）
