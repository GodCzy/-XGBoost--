# 排放预测与优化框架

本项目展示了一个最小化的工作流程，使用机器学习预测排放，并在排放超出阈值时进行优化。

## 功能
- **数据预处理**：支持 CSV/SQLite 数据加载，缺失值填补、异常值剔除、特征
  工程以及数据质量报告。
- **EmissionPredictor**：结合 RandomForest 与 XGBoost 并报告常见指标。
- **过程监控**：当排放超过用户设定的阈值时触发优化。
- **粒子群优化**：调节工艺参数。
- **演示脚本**：生成合成数据并展示完整流程。

## 安装
1. 确保已安装 Python 3.12+。
2. 安装依赖：
   ```bash
   pip install numpy pandas scikit-learn xgboost shap
   ```
   `shap` 库为可选项，未安装时将跳过基于 SHAP 的解释。

## 使用方法
运行演示脚本：
```bash
python main.py
```
脚本将训练模型、打印评估指标、尝试计算 SHAP 值，并在排放超过阈值时运行 PSO 优化器。

## 测试
运行测试以确保核心组件的稳定性：
```bash
pytest
```

## 项目结构
- `data_preprocessing.py`：数据加载、清洗、特征工程和质量报告工具。
- `emission_predictor.py`：集成模型封装。
- `monitoring.py`：排放阈值监控与优化触发。
- `optimization.py`：简单的粒子群优化实现。
- `main.py`：端到端演示脚本。
- `requirements.txt`：Python 依赖列表。

## 许可证
未指定许可证。
