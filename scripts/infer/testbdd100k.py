import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 第一步：诊断标签数据
# ============================================================

def diagnose_label_data(label_path):
    """
    诊断标签数据的格式和内容
    """
    print(f"\n{'='*60}")
    print(f"诊断标签文件: {os.path.basename(label_path)}")
    print(f"{'='*60}")

    # 读取标签
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    if label is None:
        print(f"❌ 无法读取文件!")
        return None

    print(f"✓ 图像尺寸: {label.shape}")
    print(f"✓ 数据类型: {label.dtype}")

    # 统计像素值分布
    unique_values = np.unique(label)
    print(f"✓ 唯一像素值: {unique_values}")

    for val in unique_values:
        count = np.sum(label == val)
        percentage = count / label.size * 100
        print(f"  - 值 {val}: {count} 像素 ({percentage:.2f}%)")

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 原始标签
    axes[0].imshow(label, cmap='gray')
    axes[0].set_title('原始标签')
    axes[0].axis('off')

    # 使用阈值 128 二值化
    binary_128 = (label > 128).astype(np.uint8) * 255
    axes[1].imshow(binary_128, cmap='gray')
    axes[1].set_title('阈值>128 二值化')
    axes[1].axis('off')

    # 使用阈值 0 二值化（检测任何非零值）
    binary_0 = (label > 0).astype(np.uint8) * 255
    axes[2].imshow(binary_0, cmap='gray')
    axes[2].set_title('阈值>0 二值化')
    axes[2].axis('off')

    plt.tight_layout()
    output_path = label_path.replace('.png', '_diagnosis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 诊断图像已保存到: {output_path}")
    plt.close()

    return label

# ============================================================
# 使用示例
# ============================================================

# 1. 诊断几个标签文件
da_dir = r"E:\detected sys\lane_detect\data\bdd100k\segments\train"
sample_files = [f for f in os.listdir(da_dir) if f.endswith('.png')][:3]

for fname in sample_files:
    label_path = os.path.join(da_dir, fname)
    diagnose_label_data(label_path)

print("\n" + "="*60)
print("诊断完成！请检查生成的 _diagnosis.png 文件")
print("="*60)


# ============================================================
# 第二步：修复后的数据集类（根据诊断结果选择）
# ============================================================

class BDDLaneDrivableDataset_Fixed:
    """
    修复后的数据集类 - 提供多种标签处理方式
    """

    def __init__(self, img_dir, da_dir, ll_dir,
                 img_size=(640, 360),
                 da_label_mode='auto'):
        """
        da_label_mode: 标签处理模式
            - 'auto': 自动检测（推荐）
            - 'binary_0_255': 标签是 0 和 255
            - 'binary_0_1': 标签是 0 和 1
            - 'multiclass_0_1_2': 标签是 0, 1, 2（合并1和2为前景）
            - 'threshold_128': 使用阈值 128
        """
        self.img_dir = img_dir
        self.da_dir = da_dir
        self.ll_dir = ll_dir
        self.img_size = img_size
        self.da_label_mode = da_label_mode

        # 获取图像文件列表
        import glob
        exts = ["*.jpg", "*.png", "*.jpeg"]
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(img_dir, ext)))
        self.img_paths = sorted(files)

        assert len(self.img_paths) > 0, f"未找到图像: {img_dir}"

        # 如果是自动模式，检测标签格式
        if self.da_label_mode == 'auto':
            self.da_label_mode = self._detect_label_format()

        print(f"[数据集] {img_dir}")
        print(f"  - 图像数量: {len(self.img_paths)}")
        print(f"  - DA标签模式: {self.da_label_mode}")

    def _detect_label_format(self):
        """自动检测标签格式"""
        # 随机采样几个标签检测
        sample_size = min(10, len(self.img_paths))

        for i in range(sample_size):
            name = os.path.splitext(os.path.basename(self.img_paths[i]))[0]
            da_path = self._find_label(self.da_dir, name)

            if da_path:
                da = cv2.imread(da_path, cv2.IMREAD_GRAYSCALE)
                if da is not None:
                    unique_vals = np.unique(da)

                    # 判断标签类型
                    if len(unique_vals) == 2:
                        if 255 in unique_vals:
                            return 'binary_0_255'
                        elif max(unique_vals) <= 1:
                            return 'binary_0_1'
                    elif len(unique_vals) == 3 and max(unique_vals) <= 2:
                        return 'multiclass_0_1_2'

        # 默认使用阈值方法
        print("⚠️  无法自动检测标签格式，使用默认阈值128")
        return 'threshold_128'

    def _process_da_label(self, da):
        """根据模式处理 drivable area 标签"""
        if self.da_label_mode == 'binary_0_255':
            # 标签是 0 和 255
            return (da > 127).astype(np.int64)

        elif self.da_label_mode == 'binary_0_1':
            # 标签已经是 0 和 1
            return da.astype(np.int64)

        elif self.da_label_mode == 'multiclass_0_1_2':
            # 标签是 0, 1, 2，将 1 和 2 都视为可驾驶区域
            return (da > 0).astype(np.int64)

        elif self.da_label_mode == 'threshold_128':
            # 使用阈值 128
            return (da > 128).astype(np.int64)

        else:
            raise ValueError(f"未知的标签模式: {self.da_label_mode}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        import torch

        img_path = self.img_paths[idx]
        name = os.path.splitext(os.path.basename(img_path))[0]

        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(img_path)
        img = cv2.resize(img, self.img_size)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_chw = img_rgb.transpose(2, 0, 1)

        # 读取 drivable area 标签
        da_path = self._find_label(self.da_dir, name)
        da = cv2.imread(da_path, cv2.IMREAD_GRAYSCALE)
        if da is None:
            raise FileNotFoundError(da_path)
        da = cv2.resize(da, self.img_size, interpolation=cv2.INTER_NEAREST)
        da = self._process_da_label(da)  # 使用新的处理方法

        # 读取 lane 标签
        ll_path = self._find_label(self.ll_dir, name)
        ll = cv2.imread(ll_path, cv2.IMREAD_GRAYSCALE)
        if ll is None:
            raise FileNotFoundError(ll_path)
        ll = cv2.resize(ll, self.img_size, interpolation=cv2.INTER_NEAREST)
        ll = (ll > 128).astype(np.int64)

        img_tensor = torch.from_numpy(img_chw).float()
        da_tensor = torch.from_numpy(da).long()
        ll_tensor = torch.from_numpy(ll).long()

        return img_tensor, da_tensor, ll_tensor, name

    def _find_label(self, label_dir, name):
        for ext in [".png", ".jpg", ".jpeg"]:
            p = os.path.join(label_dir, name + ext)
            if os.path.exists(p):
                return p
        return None


# ============================================================
# 第三步：可视化预测结果（用于验证修复效果）
# ============================================================

def visualize_predictions(model, dataset, device, num_samples=5, save_dir='./debug_output'):
    """
    可视化模型预测结果，检查是否为填充区域
    """
    import torch
    import torch.nn.functional as F

    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            img, da_gt, ll_gt, name = dataset[i]

            # 预测
            img_batch = img.unsqueeze(0).to(device)
            da_logits, ll_logits = model(img_batch)

            # 转换为概率和预测
            da_prob = F.softmax(da_logits, dim=1)[0, 1].cpu().numpy()
            da_pred = (da_prob > 0.5).astype(np.uint8)

            ll_prob = F.softmax(ll_logits, dim=1)[0, 1].cpu().numpy()
            ll_pred = (ll_prob > 0.5).astype(np.uint8)

            # 可视化
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))

            # 原图
            img_vis = img.permute(1, 2, 0).numpy()
            axes[0, 0].imshow(img_vis)
            axes[0, 0].set_title('原始图像')
            axes[0, 0].axis('off')

            # DA Ground Truth
            axes[0, 1].imshow(da_gt.numpy(), cmap='gray', vmin=0, vmax=1)
            axes[0, 1].set_title('DA Ground Truth')
            axes[0, 1].axis('off')

            # DA 预测概率
            axes[0, 2].imshow(da_prob, cmap='hot', vmin=0, vmax=1)
            axes[0, 2].set_title('DA 预测概率')
            axes[0, 2].axis('off')

            # DA 预测结果
            axes[0, 3].imshow(da_pred, cmap='gray', vmin=0, vmax=1)
            axes[0, 3].set_title('DA 预测结果')
            axes[0, 3].axis('off')

            # Lane Ground Truth
            axes[1, 1].imshow(ll_gt.numpy(), cmap='gray', vmin=0, vmax=1)
            axes[1, 1].set_title('Lane Ground Truth')
            axes[1, 1].axis('off')

            # Lane 预测概率
            axes[1, 2].imshow(ll_prob, cmap='hot', vmin=0, vmax=1)
            axes[1, 2].set_title('Lane 预测概率')
            axes[1, 2].axis('off')

            # Lane 预测结果
            axes[1, 3].imshow(ll_pred, cmap='gray', vmin=0, vmax=1)
            axes[1, 3].set_title('Lane 预测结果')
            axes[1, 3].axis('off')

            # 叠加显示
            overlay = img_vis.copy()
            overlay[da_pred > 0] = overlay[da_pred > 0] * 0.5 + np.array([0, 1, 0]) * 0.5
            axes[1, 0].imshow(overlay)
            axes[1, 0].set_title('DA 叠加显示')
            axes[1, 0].axis('off')

            plt.tight_layout()
            save_path = os.path.join(save_dir, f'{name}_visualization.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"✓ 已保存: {save_path}")

print("\n✓ 诊断和修复代码准备完成！")
print("\n使用步骤：")
print("1. 运行诊断代码检查标签格式")
print("2. 根据诊断结果，在训练代码中使用 BDDLaneDrivableDataset_Fixed")
print("3. 使用 visualize_predictions 验证修复效果")
