import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import linregress
import shap

# 设置可视化样式
plt.rcParams['font.sans-serif'] = [
    'Microsoft YaHei',   # 微软雅黑（Windows）
    'SimHei',            # 黑体（Windows）
    'Songti SC',         # 宋体（macOS）
    'WenQuanYi Zen Hei', # 文泉驿（Linux）
    'FangSong'           # 仿宋
]
plt.rcParams['axes.unicode_minus'] = False
class CarbonEmissionEnsemble:
    def __init__(self, xgb_params=None, rf_params=None):
        self.xgb_params = xgb_params or {
            'n_estimators': 150,
            'learning_rate': 0.1,
            'max_depth': 5,
            'random_state': 42
        }

        self.rf_params = rf_params or {
            'n_estimators': 200,
            'max_depth': 8,
            'random_state': 42
        }

        self.xgb = XGBRegressor(**self.xgb_params)
        self.rf = RandomForestRegressor(**self.rf_params)
        self.adaptive_weights = None

        self.mse1 = None
        self.mse2 = None

    def train(self, X_train, y_train):
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.045, random_state=42)

        self.xgb.fit(X_tr, y_tr)
        self.rf.fit(X_tr, y_tr)

        self._train_adaptive_weights(X_val, y_val)

        self.xgb.fit(X_train, y_train)
        self.rf.fit(X_train, y_train)

        self.mse1 = mean_squared_error(y_train, self.xgb.predict(X_train))
        self.mse2 = mean_squared_error(y_train, self.rf.predict(X_train))

    def _train_adaptive_weights(self, X_val, y_val):
        xgb_pred = self.xgb.predict(X_val)
        rf_pred = self.rf.predict(X_val)

        best_score = float('inf')
        best_weights = (0.5, 0.5)
        for w1 in np.linspace(0, 1, 21):
            combined = w1 * xgb_pred + (1 - w1) * rf_pred
            score = mean_squared_error(y_val, combined)
            if score < best_score:
                best_score = score
                best_weights = (w1, 1 - w1)
        self.adaptive_weights = best_weights

    def predict(self, X, method='self-adaption'):
        xgb_pred = self.xgb.predict(X)
        rf_pred = self.rf.predict(X)

        if method == 'equal':
            return 0.5 * xgb_pred + 0.5 * rf_pred
        elif method == 'residual':
            return self._residual_weighted(xgb_pred, rf_pred)
        elif method == 'self-adaption':
            return self.adaptive_weights[0] * xgb_pred + self.adaptive_weights[1] * rf_pred
        else:
            raise ValueError("可用方法: 'equal', 'residual', 'self-adaption'")

    def _residual_weighted(self, pred1, pred2):
        total = self.mse1 + self.mse2 + 1e-8
        return (self.mse2 / total) * pred1 + (self.mse1 / total) * pred2


def clean_data(df):
    numeric_cols = ['全社会用电量(亿千瓦时)', '常住人口(万人)', 'GDP总量(亿元)',
                    '城镇化率(%)', '粗钢产量(万吨)', '煤炭使用量(万吨)', '碳排放()']

    for col in numeric_cols:
        df[col] = df[col].astype(str).str.replace(r'[^\d.Ee+-]', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.interpolate(method='linear')
    df = df.dropna(subset=numeric_cols)
    return df


def emae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def emape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


if __name__ == "__main__":
    try:
        df = pd.read_csv("F:\\model-based\\data\\444.csv", encoding='utf-8-sig')
        df = clean_data(df)
    except FileNotFoundError:
        raise SystemExit("错误：数据文件不存在，请检查路径")

    features = ["全社会用电量(亿千瓦时)", "常住人口(万人)", "GDP总量(亿元)",
                "城镇化率(%)", "粗钢产量(万吨)", "煤炭使用量(万吨)"]
    target = "碳排放()"
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.045, random_state=42)

    assert not X_train.empty, "训练集不能为空"
    assert not X_test.empty, "测试集不能为空"

    ensemble = CarbonEmissionEnsemble()
    ensemble.train(X_train.values, y_train.values)

    model_filename = "carbon_emission_ensemble_model.pkl"
    joblib.dump(ensemble, model_filename)
    print(f"\n模型已成功保存为 {model_filename}")

    joblib.dump(ensemble.xgb, "xgb_model.pkl")
    joblib.dump(ensemble.rf, "rf_model.pkl")
    joblib.dump(ensemble.adaptive_weights, "ensemble_weights.pkl")

    print("\n模型保存完成：")
    print(f"- XGBoost模型已保存为 xgb_model.pkl")
    print(f"- 随机森林模型已保存为 rf_model.pkl")
    print(f"- 组合权重已保存为 ensemble_weights.pkl")

    results = {}
    methods = ['equal', 'residual', 'self-adaption']

    print("\n" + "=" * 30 + " 模型评估结果 " + "=" * 30)
    for method in methods:
        pred = ensemble.predict(X_test.values, method=method)
        results[method] = {
            'RMSE': np.sqrt(mean_squared_error(y_test, pred)),
            'R2': r2_score(y_test, pred),
            'EMAE': emae(y_test, pred),
            'EMAPE': emape(y_test, pred)
        }
        print(f"{method.upper():<12} | RMSE: {results[method]['RMSE']:.4f} | R²: {results[method]['R2']:.4f} | EMAE: {results[method]['EMAE']:.4f} | EMAPE: {results[method]['EMAPE']:.4f}%")

    # 单模型评估
    print("\n" + "=" * 30 + " 单模型评估结果 " + "=" * 30)
    xgb_pred = ensemble.xgb.predict(X_test.values)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
    xgb_r2 = r2_score(y_test, xgb_pred)
    xgb_emae = emae(y_test, xgb_pred)
    xgb_emape = emape(y_test, xgb_pred)
    print(f"XGBOOST    | RMSE: {xgb_rmse:.4f} | R²: {xgb_r2:.4f} | EMAE: {xgb_emae:.4f} | EMAPE: {xgb_emape:.4f}%")

    rf_pred = ensemble.rf.predict(X_test.values)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_r2 = r2_score(y_test, rf_pred)
    rf_emae = emae(y_test, rf_pred)
    rf_emape = emape(y_test, rf_pred)
    print(f"RANDOM FOREST | RMSE: {rf_rmse:.4f} | R²: {rf_r2:.4f} | EMAE: {rf_emae:.4f} | EMAPE: {rf_emape:.4f}%")

    # 单模型的预测相关性关系图
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(y_test, xgb_pred, alpha=0.5)
    plt.title('XGBoost 预测相关性关系图')
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    slope, intercept, _, _, _ = linregress(y_test, xgb_pred)
    formula = f'$y = {slope:.4f}x + {intercept:.4f}$'
    plt.text(0.05, 0.9, f'$R^2$: {xgb_r2:.4f}\n{formula}', transform=plt.gca().transAxes)

    plt.subplot(1, 2, 2)
    plt.scatter(y_test, rf_pred, alpha=0.5)
    plt.title('Random Forest 预测相关性关系图')
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    slope, intercept, _, _, _ = linregress(y_test, rf_pred)
    formula = f'$y = {slope:.4f}x + {intercept:.4f}$'
    plt.text(0.05, 0.9, f'$R^2$: {rf_r2:.4f}\n{formula}', transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.show()

    # 单模型预测结果对比图
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values, label='真实值', marker='o')
    plt.plot(xgb_pred, label='XGBoost 预测值', marker='s')
    plt.plot(rf_pred, label='Random Forest 预测值', marker='^')
    plt.title('单模型预测结果对比图')
    plt.xlabel('样本编号')
    plt.ylabel('碳排放值')
    plt.legend()
    plt.show()

    # 组合模型的预测相关性关系图
    plt.figure(figsize=(18, 6))
    for i, method in enumerate(methods, 1):
        combined_pred = ensemble.predict(X_test.values, method=method)
        rmse = results[method]['RMSE']
        r2 = results[method]['R2']
        plt.subplot(1, 3, i)
        plt.scatter(y_test, combined_pred, alpha=0.5)
        plt.title(f'{method.upper()} 组合模型预测相关性关系图')
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        slope, intercept, _, _, _ = linregress(y_test, combined_pred)
        formula = f'$y = {slope:.4f}x + {intercept:.4f}$'
        plt.text(0.05, 0.9, f'$R^2$: {r2:.4f}\n{formula}', transform=plt.gca().transAxes)
    plt.tight_layout()
    plt.show()

    # 组合模型预测结果对比图
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values, label='真实值', marker='o')
    for method in methods:
        combined_pred = ensemble.predict(X_test.values, method=method)
        plt.plot(combined_pred, label=f'{method.upper()} 组合模型预测值', marker='s')
    plt.title('组合模型预测结果对比图')
    plt.xlabel('样本编号')
    plt.ylabel('碳排放值')
    plt.legend()
    plt.show()

    # 赋权权值可视化
    weights = {
        'equal': (0.5, 0.5),
        'residual': (ensemble.mse2 / (ensemble.mse1 + ensemble.mse2 + 1e-8),
                     ensemble.mse1 / (ensemble.mse1 + ensemble.mse2 + 1e-8)),
        'self-adaption': ensemble.adaptive_weights
    }

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, [weights[method][0] for method in methods], width, label='XGBoost')
    rects2 = ax.bar(x + width/2, [weights[method][1] for method in methods], width, label='Random Forest')

    ax.set_ylabel('权重')
    ax.set_title('不同组合方法下的模型权重')
    ax.set_xticks(x)
    ax.set_xticklabels([method.upper() for method in methods])
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.4f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.show()

    # ==========================
    # SHAP特征解释分析（自适应组合模型）
    # ==========================
    import shap  # 添加SHAP库导入（如果尚未导入）

    print("\n" + "=" * 30 + " 自适应组合模型SHAP分析 " + "=" * 30)

    # 1. 初始化SHAP解释器（使用训练集样本加速计算）
    sample_idx = np.random.choice(X_train.shape[0], 200, replace=False)  # 选择200个样本
    X_sample = X_train.iloc[sample_idx]

    explainer_xgb = shap.TreeExplainer(ensemble.xgb)
    explainer_rf = shap.TreeExplainer(ensemble.rf)

    # 2. 计算基模型SHAP值
    shap_values_xgb = explainer_xgb.shap_values(X_sample)
    shap_values_rf = explainer_rf.shap_values(X_sample)

    # 3. 计算自适应组合SHAP值（关键步骤）
    w_xgb, w_rf = ensemble.adaptive_weights  # 获取自适应权重
    combined_shap = w_xgb * shap_values_xgb + w_rf * shap_values_rf  # 加权组合

    # ==========================
    # 特征重要性对比可视化
    # ==========================
    plt.figure(figsize=(18, 6))

    # 子图1：XGBoost特征重要性
    plt.subplot(131)
    shap.summary_plot(
        shap_values_xgb, X_sample, feature_names=features,
        plot_type="bar", color='#FFA07A', show=False
    )
    plt.title("XGBoost特征重要性", fontsize=12)

    # 子图2：随机森林特征重要性
    plt.subplot(132)
    shap.summary_plot(
        shap_values_rf, X_sample, feature_names=features,
        plot_type="bar", color='#20B2AA', show=False
    )
    plt.title("随机森林特征重要性", fontsize=12)

    # 子图3：自适应组合模型特征重要性
    plt.subplot(133)
    shap.summary_plot(
        combined_shap, X_sample, feature_names=features,
        plot_type="bar", color='#4B0082', show=False  # 紫色代表自适应组合
    )
    plt.title("自适应组合模型特征重要性", fontsize=12)

    plt.suptitle("基模型与自适应组合模型特征重要性对比", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

    # ==========================
    # 特征依赖关系分析（自适应权重）
    # ==========================
    plt.figure(figsize=(12, 8))

    for idx, feature in enumerate(features):
        plt.subplot(2, 3, idx + 1)

        # 绘制特征值与SHAP值的关系
        shap.dependence_plot(
            idx, combined_shap, X_sample,
            feature_names=features, interaction_index=None,
            show=False, title=f"{feature}的影响"
        )

        # 添加自适应权重标注
        plt.text(0.95, 0.95,
                 f"XGB权重: {w_xgb:.2%}\nRF权重: {w_rf:.2%}",
                 transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.8),
                 ha='right', va='top')

    plt.suptitle("自适应组合模型特征依赖关系", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

    # ==========================
    # 单样本决策过程可视化（自适应组合）
    # ==========================
    sample_idx = 50  # 选择第50个样本进行解释
    plt.figure(figsize=(10, 4))

    # 绘制SHAP力图
    shap.force_plot(
        base_value=w_xgb * explainer_xgb.expected_value + w_rf * explainer_rf.expected_value,
        shap_values=combined_shap[sample_idx],
        features=X_sample.iloc[sample_idx],
        feature_names=features,
        matplotlib=True, text_rotation=15
    )

    # 添加预测结果标注
    true_value = y_train.iloc[sample_idx]
    xgb_pred = ensemble.xgb.predict([X_sample.iloc[sample_idx]])
    rf_pred = ensemble.rf.predict([X_sample.iloc[sample_idx]])
    combined_pred = w_xgb * xgb_pred + w_rf * rf_pred

    plt.title(f"样本{sample_idx}决策过程\n"
              f"真实值: {true_value:.1f}\n"
              f"组合预测: {combined_pred[0]:.1f} = "
              f"XGB({xgb_pred[0]:.1f})*{w_xgb:.2f} + "
              f"RF({rf_pred[0]:.1f})*{w_rf:.2f}",
              fontsize=10)

    plt.tight_layout()
    plt.show()

    # ==========================
    # 特征影响方向分析（小提琴图）
    # ==========================
    plt.figure(figsize=(14, 8))
    shap.summary_plot(combined_shap, X_sample, feature_names=features, plot_type="violin")
    plt.title("自适应组合模型特征影响分布", fontsize=14)
    plt.tight_layout()
    plt.show()

    # ==========================
    # 权重分配对比可视化（新增）
    # ==========================
    plt.figure(figsize=(10, 5))
    methods = ['EQUAL', 'RESIDUAL', 'SELF-ADAPTION']
    weights = [
        (0.5, 0.5),  # 等值赋权
        (ensemble.mse2 / (ensemble.mse1 + ensemble.mse2 + 1e-8),  # 残差赋权
         ensemble.mse1 / (ensemble.mse1 + ensemble.mse2 + 1e-8)),
        ensemble.adaptive_weights  # 自适应赋权
    ]

    for i, (method, (w1, w2)) in enumerate(zip(methods, weights)):
        plt.subplot(1, 3, i + 1)
        plt.bar(['XGB', 'RF'], [w1, w2], color=['#FFA07A', '#20B2AA'])
        plt.title(method)
        plt.ylim(0, 1)
        plt.text(0, w1, f'{w1:.2%}', ha='center', va='bottom', color='white', fontweight='bold')
        plt.text(1, w2, f'{w2:.2%}', ha='center', va='bottom', color='white', fontweight='bold')

    plt.suptitle("不同组合策略的模型权重对比", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

    # 组合模型箱型图
    combined_predictions = [ensemble.predict(X_test.values, method=method) for method in methods]

    plt.figure(figsize=(10, 6))
    plt.boxplot(combined_predictions, labels=[method.upper() for method in methods])
    plt.title('组合模型预测结果箱型图')
    plt.ylabel('碳排放预测值')
    plt.show()