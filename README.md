# 排放预测与优化框架

本项目展示了一个最小化的工作流程，使用机器学习预测排放，并在排放超出阈值时进行优化。

## 功能
- **数据预处理**：支持 CSV/SQLite 数据加载，缺失值填补、异常值剔除、特征工程以及数据质量报告。
- **EmissionPredictor**：结合 RandomForest、XGBoost 与 MLP，支持自适应组合策略与多指标评估。
- **自适应监控**：实时监测排放，支持异常检测、自适应阈值与告警，可与优化器闭环联动。
- **参数优化与实验管理**：提供粒子群、贝叶斯与遗传算法，并记录实验结果及性能曲线。
- **演示脚本**：生成合成数据并展示完整流程。
- **Flask API 服务**：对外提供排放预测接口以及元数据、指标、特征解释、优化实验与监控样例等完整信息，可用于部署或可视化集成。
- **Dockerfile 与 CI**：通过 GitHub Actions 自动化格式检查、冒烟测试和镜像构建。

## 安装
1. 确保已安装 Python 3.12+。
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
   `shap` 库为可选项，未安装时将跳过基于 SHAP 的解释。

## 使用方法
运行演示脚本：
```bash
python main.py
```
脚本将训练模型、打印详细评估指标、尝试计算组合 SHAP 值，并在排放超过阈值时运行 PSO 优化器。

## 测试
运行测试以确保核心组件的稳定性：
```bash
pytest
```

## API 服务
启动 API 服务：
```bash
python service.py
```
浏览器访问 `http://localhost:8000` 可打开交互式前端页面，浏览数据质量报告、模型指标、特征影响、优化实验与监控记录，并生成不同策略下的预测结果。

发送预测请求示例：
```bash
curl -X POST "http://localhost:8000/predict?strategy=self_adaption&all_strategies=true" \
  -H "Content-Type: application/json" \
  -d '{"electricity":100,"gdp":50,"coal":30}'
```

更多 API 端点：
- `GET /metadata`：返回可用特征、组合策略及权重。
- `GET /metrics`：返回基础模型与组合模型的多项指标。
- `GET /feature-insights`：返回置换重要性与 SHAP 汇总结果。
- `GET /optimization`：返回优化实验结果与历史。
- `GET /monitor-sample`：返回过程监控示例日志。
使用 Docker 运行服务：
```bash
docker build -t emission-service .
docker run -p 8000:8000 emission-service
```

## 项目结构
- `data_preprocessing.py`：数据加载、清洗、特征工程和质量报告工具。
- `emission_predictor.py`：集成模型封装。
- `monitoring.py`：排放阈值监控与优化触发。
- `optimization.py`：粒子群、贝叶斯及遗传算法实现。
- `experiment_manager.py`：实验记录与可视化工具。
- `main.py`：端到端演示脚本。
- `service.py`：Flask API 服务。
- `tests/`：单元测试用例。
- `requirements.txt`：Python 依赖列表。

## 许可证
未指定许可证。

