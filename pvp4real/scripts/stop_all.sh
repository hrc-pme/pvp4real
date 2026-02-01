#!/bin/bash
# 中斷所有訓練進程的腳本

echo "正在尋找運行中的訓練進程..."

# 訓練入口點列表
TRAINING_SCRIPTS=(
    "train_pvp_metadrive_fakehuman"
    "train_haco_metadrive_fakehuman"
    "train_eil_metadrive_fakehuman"
    "train_td3_metadrive"
    "train_ppo_metadrive"
    "train_hgdagger_metadrive"
)

# 檢查是否有訓練進程正在運行
found_processes=false
for script in "${TRAINING_SCRIPTS[@]}"; do
    if pgrep -f "$script" > /dev/null; then
        found_processes=true
        echo "找到 $script 進程"
    fi
done

if [ "$found_processes" = false ]; then
    echo "沒有找到運行中的訓練進程"
    exit 0
fi

echo ""
echo "正在中斷所有訓練進程..."

# 中斷所有訓練進程
for script in "${TRAINING_SCRIPTS[@]}"; do
    pkill -f "$script" 2>/dev/null
done

# 等待進程完全終止
sleep 2

# 檢查是否還有殘留進程
remaining=false
for script in "${TRAINING_SCRIPTS[@]}"; do
    if pgrep -f "$script" > /dev/null; then
        remaining=true
        echo "警告: $script 進程仍在運行，嘗試強制終止..."
        pkill -9 -f "$script" 2>/dev/null
    fi
done

if [ "$remaining" = true ]; then
    sleep 1
fi

echo ""
echo "✓ 所有訓練進程已中斷"
