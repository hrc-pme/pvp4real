#!/usr/bin/env python3
"""
統一配置讀取工具
從 settings.yml 讀取配置並生成命令列參數
"""
import os
import sys
import yaml
from pathlib import Path


def load_settings():
    """載入 settings.yml 配置"""
    script_dir = Path(__file__).parent
    settings_path = script_dir / "settings.yml"
    
    if not settings_path.exists():
        print(f"Warning: {settings_path} not found, using defaults")
        return get_default_settings()
    
    with open(settings_path, 'r') as f:
        return yaml.safe_load(f)


def get_default_settings():
    """預設配置"""
    return {
        'visualization': {
            'wandb_enabled': False,
            'wandb_project': 'pvp4real',
            'wandb_team': '',
            'render_enabled': False,
            'render_seed_id': None
        },
        'training': {
            'num_parallel': 8,
            'seeds': [0, 100, 200, 300, 400, 500, 600, 700],
            'total_timesteps': 50000,
            'batch_size': 1024,
            'learning_starts': 10,
            'save_freq': 500,
            'eval_freq': 150,
            'n_eval_episodes': 50
        },
        'logging': {
            'log_dir': '/home/zhenghao/pvp',
            'background_mode': True
        }
    }


def build_common_args(settings, seed=None):
    """生成通用命令列參數"""
    args = []
    
    # 基本訓練參數
    training = settings.get('training', {})
    if seed is not None:
        args.append(f"--seed={seed}")
    args.append(f"--batch_size={training.get('batch_size', 1024)}")
    args.append(f"--learning_starts={training.get('learning_starts', 10)}")
    args.append(f"--save_freq={training.get('save_freq', 500)}")
    
    # 日誌目錄
    logging = settings.get('logging', {})
    args.append(f"--log_dir={logging.get('log_dir', '/home/zhenghao/pvp')}")
    
    # WandB 配置
    viz = settings.get('visualization', {})
    if viz.get('wandb_enabled', False):
        args.append("--wandb")
        if viz.get('wandb_project'):
            args.append(f"--wandb_project={viz['wandb_project']}")
        if viz.get('wandb_team'):
            args.append(f"--wandb_team={viz['wandb_team']}")
    
    # 渲染配置：只有指定的 seed 才渲染
    if viz.get('render_enabled', False):
        render_seed = viz.get('render_seed_id')
        # 如果未指定 render_seed_id，使用第一個 seed
        if render_seed is None:
            render_seed = training.get('seeds', [0])[0]
        # 只有當前 seed 等於 render_seed 時才啟用渲染
        if seed is not None and seed == render_seed:
            args.append("--render")
    
    return ' '.join(args)


if __name__ == '__main__':
    # 測試用途
    settings = load_settings()
    
    if len(sys.argv) > 1 and sys.argv[1] == 'seeds':
        # 輸出 seeds 陣列（給 bash 使用）
        seeds = settings['training']['seeds']
        print(' '.join(map(str, seeds)))
    elif len(sys.argv) > 1 and sys.argv[1] == 'num_parallel':
        # 輸出並行數量
        print(settings['training']['num_parallel'])
    elif len(sys.argv) > 1 and sys.argv[1] == 'args':
        # 輸出通用參數
        seed = int(sys.argv[2]) if len(sys.argv) > 2 else None
        print(build_common_args(settings, seed))
    else:
        # 完整輸出配置
        import json
        print(json.dumps(settings, indent=2))
