"""
消融实验 - 批量训练脚本
一键训练所有消融实验所需的模型
"""

import os
import sys
import subprocess
import time
from datetime import datetime


def run_training(model_type, use_astar, episodes=3000):
    """运行单个训练"""
    cmd = [
        sys.executable, 
        'train_ablation.py',
        '--model', model_type,
        '--episodes', str(episodes)
    ]
    
    if not use_astar:
        cmd.append('--no-astar')
    
    print(f"\n{'='*60}")
    print(f"开始训练: {model_type} {'(有A*)' if use_astar else '(无A*)'}")
    print(f"{'='*60}")
    
    start_time = time.time()
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    elapsed = time.time() - start_time
    
    print(f"\n训练完成，耗时: {elapsed/60:.1f} 分钟")
    return result.returncode == 0


def main():
    print("=" * 60)
    print("消融实验 - 批量训练")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 训练配置
    experiments = [
        # (模型类型, 是否使用A*, 训练回合数)
        ('basic', True, 3000),      # 基础PPO + A*
        ('attention', False, 3000), # 双重注意力PPO（无A*）
        # ('full', True, 3000),     # 完整模型（如果需要重新训练）
    ]
    
    results = []
    
    for model_type, use_astar, episodes in experiments:
        success = run_training(model_type, use_astar, episodes)
        results.append((model_type, use_astar, success))
    
    # 打印总结
    print("\n" + "=" * 60)
    print("训练总结")
    print("=" * 60)
    
    for model_type, use_astar, success in results:
        status = "✓ 成功" if success else "✗ 失败"
        astar_str = "有A*" if use_astar else "无A*"
        print(f"  {model_type} ({astar_str}): {status}")
    
    print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 运行评估
    print("\n" + "=" * 60)
    print("开始评估所有算法...")
    print("=" * 60)
    
    subprocess.run([sys.executable, 'evaluate_all.py', '--trials', '50'])


if __name__ == "__main__":
    main()
