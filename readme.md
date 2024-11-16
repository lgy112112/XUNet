
<div align="center">
  <h1>XUNet: <br>UNet Cross Over Features</h1>
</div>



<p align="center">
  <img src="images/cover.png" alt="封面图片" width="600">
</p>

<p align="center">
  
<p align="center">
  <img src="https://img.shields.io/badge/%E6%88%91%E5%A5%BD%E7%B4%AF-blue?style=for-the-badge&logo=coffeescript" alt="我好累">
  <img src="https://img.shields.io/badge/%E6%88%91%E5%A5%BD%E9%A5%BF-orange?style=for-the-badge&logo=fastapi" alt="我好饿">
  <img src="https://img.shields.io/badge/%E6%94%AF%E6%8C%81-NMIXX-ff69b4?style=for-the-badge&logo=musicbrainz" alt="支持NMIXX">
  <img src="https://img.shields.io/badge/%E7%8C%AB%E7%8C%AB-%E5%8A%A0%E6%B2%B9%7E-yellow?style=for-the-badge&logo=github" alt="猫猫加油~">
  <a href="images/alipay-qrcode.jpg">
    <img src="https://img.shields.io/badge/%E6%8A%95%E5%96%82%E4%B8%BB%E6%92%AD-%E5%8D%95%E5%87%BB%E6%94%AF%E6%8C%81-9cf?style=for-the-badge&logo=alipay" alt="投喂主播">
  </a>
</p>

---

## 这是什么？

**XUNet**（[CrossBasicUNet](cross_basic_unet.py)）是我基于 MONAI 实现的经典 U-Net 的扩展模型。其核心思想是通过跨层特征共享，将所有编码器层的特征整合到每一层解码器中，从而更充分地利用多尺度的特征信息。

XUNet 的主要改进体现在以下方面：
1. **跨层特征共享**：与传统 U-Net 每层解码器只使用对应编码器特征不同，XUNet 在解码阶段融合了所有编码器层的特征。这一设计通过插值对齐不同分辨率的特征，使解码器具备更强的上下文感知能力。
2. **模块化设计**：引入了 `UpCatAll` 模块，用于动态调整和拼接特征，为解码器提供全尺度的编码器特征支持。

---

## 安装指南

1. **克隆仓库**

   ```bash
   git clone https://github.com/lgy112112/XUNet.git
   cd XUNet
   ```

2. **安装依赖**

   ```bash
   pip install -r requirements.txt
   cd
   ```

---

## 运行顺序

请按照以下顺序运行项目中的 Jupyter Notebook 文件：

1. **模型介绍**

   - [introduce.ipynb](introduce.ipynb)
   - 该 Notebook 详细介绍了 CrossBasicUNet（即 XUNet）的结构和创新点，并与原始的 BasicUNet 进行了对比。

2. **数据准备**

   - [train_data_get.ipynb](train_data_get.ipynb)
   - 该 Notebook 用于获取和预处理训练数据，为模型训练做好准备。

3. **模型训练与比较**

   - [train_comparison.ipynb](train_comparison.ipynb)
   - 该 Notebook 训练 BasicUNet 和 CrossBasicUNet，并对比两者在测试集上的性能。

---

## 实验结果

### 训练效果

下图展示了模型在训练过程中的性能曲线。可以看出，CrossBasicUNet（XUNet）的表现均优于原始的 BasicUNet。

![训练效果](images/image.png)

### 测试集评估

在测试集上，我使用 Dice 系数（Dice Score）作为评估指标，结果如下：

| 模型名称      | Dice 系数 |
|---------------|-----------|
| **BasicUNet** | 0.76994   |
| **CrossUNet** | 0.79180   |

---

## 日后更新

### 日后更新计划

- [ ] **优化共享特征的方式**  
  - 探索更加高效的特征融合策略（如注意力机制或特征加权）。  
  - 在共享特征的同时，避免冗余信息的引入，进一步提升模型性能。

- [ ] **减少参数量的同时保证训练效率**  
  - 引入轻量化模块（如深度可分离卷积或稀疏卷积）。  
  - 设计更高效的跳跃连接机制，在性能和参数量之间找到平衡点。

---

## 支持与反馈

感谢您对 XUNet 的关注！如有任何问题或建议，欢迎提交 [Issues](https://github.com/lgy112112/XUNet/issues)。如果您对我的工作感兴趣，请为本项目点个 ⭐️！
