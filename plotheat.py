import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_data_for_heatmap(levels_path):
    """加载所有set的数据用于热图绘制"""
    all_sets_data = []
    set_names = []
    
    # 遍历所有set文件夹
    for set_folder in sorted(Path(levels_path).glob('set_*')):
        json_file = set_folder / 'ddqn_experiment0.json'
        if json_file.exists():
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    steps = data.get('data', {}).get('steps', [])
                    all_sets_data.append(steps)
                    set_names.append(set_folder.name)
                    print(f"Loaded {len(steps)} steps from {json_file}")
            except Exception as e:
                print(f"Error reading {json_file}: {e}")
    
    # 确保所有数据长度一致
    min_length = min(len(steps) for steps in all_sets_data)
    all_sets_data = [steps[:min_length] for steps in all_sets_data]
    
    return np.array(all_sets_data), set_names, min_length

def plot_performance(data):
    """绘制优化后的性能曲线（只分析前240关）"""
    plt.figure(figsize=(15, 6))
    
    # 只取前240关的数据
    data = data[:, :240]
    mean_steps = np.mean(data, axis=0)
    std_steps = np.std(data, axis=0)
    x = np.arange(len(mean_steps))
    
    # 绘制平均值曲线
    plt.plot(x, mean_steps, 'b-', label='Average Steps', linewidth=2)
    
    # 绘制标准差区域
    plt.fill_between(x, 
                    mean_steps - std_steps, 
                    mean_steps + std_steps, 
                    color='b', 
                    alpha=0.2, 
                    label='Standard Deviation')
    
    # 设置更合理的y轴范围
    y_min = max(0, np.min(mean_steps - std_steps))
    y_max = np.percentile(mean_steps + std_steps, 95)
    plt.ylim(y_min, y_max)
    
    # 设置网格和标签
    plt.grid(True, alpha=0.3)
    plt.xlabel('Level Number')
    plt.ylabel('Steps')
    plt.title('Average Performance Across Sets (First 240 Levels)')
    plt.legend()
    
    # 设置x轴范围和刻度
    plt.xlim(0, 240)
    xticks = np.arange(0, 241, 20)  # 每20关显示一个刻度
    plt.xticks(xticks)
    
    # 移除顶部和右侧边框
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    
    # 添加统计信息
    info_text = f'Mean Steps: {np.mean(mean_steps):.2f}\n'
    info_text += f'Min Steps: {np.min(mean_steps):.2f}\n'
    info_text += f'Max Steps: {np.max(mean_steps):.2f}\n'
    info_text += f'Median Steps: {np.median(mean_steps):.2f}'
    
    # 在图的右上角添加统计信息
    plt.text(0.95, 0.95, info_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    return plt.gcf()

def plot_combined_analysis(data, set_names):
    """绘制优化后的组合分析图（只分析前240关）"""
    fig = plt.figure(figsize=(15, 12))
    
    # 只取前240关的数据
    data = data[:, :240]
    
    gs = plt.GridSpec(2, 1, height_ratios=[1, 1])
    
    # 热图
    ax1 = plt.subplot(gs[0])
    vmax = np.percentile(data, 95)
    vmin = np.percentile(data, 5)
    
    sns.heatmap(data, 
                cmap='YlOrRd',
                xticklabels=20,  # 每20关显示一个刻度
                yticklabels=set_names,
                vmin=vmin,
                vmax=vmax,
                cbar_kws={'label': 'Steps'},
                robust=True,
                ax=ax1)
    
    ax1.set_xlabel('Level Number')
    ax1.set_ylabel('Set')
    ax1.set_title('Steps per Level across Different Sets (First 240 Levels)')
    
    # 性能曲线
    ax2 = plt.subplot(gs[1])
    mean_steps = np.mean(data, axis=0)
    std_steps = np.std(data, axis=0)
    x = np.arange(len(mean_steps))
    
    ax2.plot(x, mean_steps, 'b-', label='Average Steps', linewidth=2)
    ax2.fill_between(x, 
                     mean_steps - std_steps, 
                     mean_steps + std_steps, 
                     color='b', 
                     alpha=0.2, 
                     label='Standard Deviation')
    
    y_min = max(0, np.min(mean_steps - std_steps))
    y_max = np.percentile(mean_steps + std_steps, 95)
    ax2.set_ylim(y_min, y_max)
    
    ax2.set_xlim(0, 240)
    xticks = np.arange(0, 241, 20)
    ax2.set_xticks(xticks)
    
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Level Number')
    ax2.set_ylabel('Average Steps')
    ax2.set_title('Average Performance Across Sets (First 240 Levels)')
    ax2.legend()
    
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    # 添加统计信息
    info_text = f'Mean Steps: {np.mean(mean_steps):.2f}\n'
    info_text += f'Min Steps: {np.min(mean_steps):.2f}\n'
    info_text += f'Max Steps: {np.max(mean_steps):.2f}\n'
    info_text += f'Median Steps: {np.median(mean_steps):.2f}'
    
    ax2.text(0.95, 0.95, info_text,
             transform=ax2.transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def main():
    levels_path = "./levels"  # 请修改为你的实际路径
    
    # 加载数据
    data, set_names, num_levels = load_data_for_heatmap(levels_path)
    print(f"Processed {num_levels} levels from {len(set_names)} sets")
    
    # 绘制组合分析图
    fig = plot_combined_analysis(data, set_names)
    
    # 保存图表
    plt.savefig('level_analysis_240.pdf', bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()