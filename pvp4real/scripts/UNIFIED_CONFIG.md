# 統一配置系統使用說明

已完成統一配置系統的建置，所有訓練腳本現在從 `settings.yml` 讀取配置。


## 使用方式

### 基本使用
直接修改 `scripts/settings.yml`，所有腳本會自動套用新設定：

```yaml
# 開啟 WandB
visualization:
  wandb_enabled: true
  wandb_project: "my_project"
  
# 開啟渲染（需要 X11）
  render_enabled: true

# 修改訓練參數
training:
  seeds: [0, 42, 100]  # 只跑 3 個 seed
  total_timesteps: 100000
  batch_size: 2048
```

### 執行訓練
```bash
cd pvp4real/scripts
bash metadrive_simhuman_pvp4real.sh
```

## 測試配置

```bash
# 檢視會使用的 seeds
python3 scripts/load_settings.py seeds

# 檢視生成的參數（for seed=0）
python3 scripts/load_settings.py args 0
```
