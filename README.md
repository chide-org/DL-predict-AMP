# DL-predict-AMP
深度学习预测抗菌肽

# 数据集

**6K数据集v0**是第一版AMP正负样本数据集，其中包含3656条抗菌肽，2469条非抗菌肽，所有肽的长度在6-15内，文件包含肽序列，肽长度等信息，数据集其他信息如下：

- **AMP-6~15-单体-无N修饰-常规20AA-溶血和细胞毒活性测试.xlsx**：3656条AMP肽，来自DBAASP数据库，长度6~15，单体肽，无N末端修饰，无不寻常氨基酸，包含溶血和细胞毒活性测试。
- **uniprotkb_length_6_TO_15_NOT_keyword_KW_2025_07_01.xlsx**：2469条非AMP肽，来自uniport数据库，长度6-15，人工审核，蛋白名非AMP，基因名非cAMP，排除关键词：antifungal，Fungicide，Antimicrobial，Antibiotic，Antiviral protein。
