# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 2023

@author: YourName
"""
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.losses import mean_absolute_error
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import shap  # 新增SHAP库导入
from sklearn.decomposition import PCA  # 新增因子分析

# 设置可视化样式
plt.rcParams['font.sans-serif'] = [
    'Microsoft YaHei',   # 微软雅黑（Windows）
    'SimHei',            # 黑体（Windows）
    'Songti SC',         # 宋体（macOS）
    'WenQuanYi Zen Hei', # 文泉驿（Linux）
    'FangSong'           # 仿宋
]
plt.rcParams['axes.unicode_minus'] = False
#sns.set(style="whitegrid", palette="muted")


# ==========================
# 数据预处理模块
# ==========================
def clean_data(df):
    """增强型数据清洗函数"""
    numeric_cols = ['全社会用电量(亿千瓦时)', '常住人口(万人)', 'GDP总量(亿元)',
                    '城镇化率(%)', '粗钢产量(万吨)', '煤炭使用量(万吨)', '碳排放()']

    # 数值转换增强
    for col in numeric_cols:
        df[col] = df[col].astype(str).str.replace(r'[^\d.Ee+-]', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 处理缺失值
    df = df.interpolate(method='linear')  # 线性插值
    df = df.dropna(subset=numeric_cols)
    return df


# ==========================
# 组合模型类
# ==========================
class CarbonEmissionEnsemble:
    def __init__(self, xgb_params=None, rf_params=None):
        # 模型参数默认设置
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

        self.mse1 = None  # 新增属性存储XGB的MSE
        self.mse2 = None  # 新增属性存储RF的MSE

    def train(self, X_train, y_train):
        # 分割训练集和验证集
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.035, random_state=42)

        # 训练基模型在训练部分
        self.xgb.fit(X_tr, y_tr)
        self.rf.fit(X_tr, y_tr)

        # 在验证集上寻找最佳权重
        self._train_adaptive_weights(X_val, y_val)

        # 重新在整个训练集上训练基模型
        self.xgb.fit(X_train, y_train)
        self.rf.fit(X_train, y_train)

        # 计算并存储MSE
        self.mse1 = mean_squared_error(y_train, self.xgb.predict(X_train))
        self.mse2 = mean_squared_error(y_train, self.rf.predict(X_train))

    def _train_adaptive_weights(self, X_val, y_val):
        """使用基模型的预测结果确定权重"""
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
        """组合预测方法"""
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
        """残差动态权重"""
        mse1 = mean_squared_error(y_train, self.xgb.predict(X_train))
        mse2 = mean_squared_error(y_train, self.rf.predict(X_train))
        total = mse1 + mse2 + 1e-8
        return (mse2 / total) * pred1 + (mse1 / total) * pred2

    def evaluate_ensemble(self, X_test, y_test, methods):
        """组合模型综合评估"""
        results = {}
        for method in methods:
            pred = self.predict(X_test, method=method)
            results[method] = {
                'RMSE': np.sqrt(mean_squared_error(y_test, pred)),
                'MAE': mean_absolute_error(y_test, pred),
                'R2': r2_score(y_test, pred),
                'Prediction': pred
            }
        return results

    def plot_correlation_comparison(self, y_true, y_preds, methods, colors):
        """组合模型相关关系对比图"""
        plt.figure(figsize=(15, 5))
        for i, method in enumerate(methods):
            plt.subplot(1, len(methods), i + 1)
            sns.scatterplot(x=y_true, y=y_preds[method],
                            alpha=0.7, color=colors[i], edgecolor='white', s=60)
            plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', alpha=0.5)

            # 添加统计指标
            r2 = r2_score(y_true, y_preds[method])
            mae = mean_absolute_error(y_true, y_preds[method])
            plt.text(0.05, 0.95,
                     f'R²={r2:.2f}\nMAE={mae:.1f}',
                     transform=plt.gca().transAxes,
                     bbox=dict(facecolor='white', alpha=0.8))

            plt.title(f'{method}预测相关关系', fontsize=12)
            plt.xlabel('真实值')
            plt.ylabel('预测值')
            plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('组合模型相关关系对比.png', dpi=300)
        plt.show()

    def plot_error_distribution(self, y_true, y_preds, methods, colors):
        """误差分布箱线图"""
        errors = [y_true - y_preds[method] for method in methods]
        plt.figure(figsize=(10, 6))
        bp = plt.boxplot(errors, labels=methods, patch_artist=True, showmeans=True)

        # 颜色匹配
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        plt.axhline(0, color='green', linestyle='--', alpha=0.8)
        plt.title('组合模型误差分布对比', fontsize=14)
        plt.ylabel('预测误差')
        plt.grid(alpha=0.3)
        plt.savefig('误差分布对比.png', dpi=300)
        plt.show()

    def plot_prediction_trend(self, y_true, y_preds, methods, colors, sample_indices):
        """预测趋势对比图"""
        plt.figure(figsize=(12, 6))
        plt.plot(sample_indices, y_true, 'ko-', label='真实值', markersize=6, linewidth=1.5)

        for method, color in zip(methods, colors):
            plt.plot(sample_indices, y_preds[method],
                     color=color, linestyle='--', marker='o',
                     markersize=5, linewidth=1.2, label=method)

        plt.title('碳排放预测趋势对比', fontsize=14)
        plt.xlabel('样本序号')
        plt.ylabel('碳排放量')
        plt.legend(loc='upper left', ncol=2)
        plt.grid(alpha=0.3)
        plt.savefig('预测趋势对比.png', dpi=300)
        plt.show()



# ==========================
# 主程序
# ==========================
if __name__ == "__main__":
    # 数据加载
    try:
        df = pd.read_csv("F:\\model-based\\data\\444.csv", encoding='utf-8-sig')
        df = clean_data(df)
    except FileNotFoundError:
        raise SystemExit("错误：数据文件不存在，请检查路径")

    # 特征工程
    features = ["全社会用电量(亿千瓦时)", "常住人口(万人)", "GDP总量(亿元)",
                "城镇化率(%)", "粗钢产量(万吨)", "煤炭使用量(万吨)"]
    target = "碳排放()"
    X = df[features]
    y = df[target]

    # 数据划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.035, random_state=42)

    # 验证数据有效性
    assert not X_train.empty, "训练集不能为空"
    assert not X_test.empty, "测试集不能为空"

    print("\n" + "=" * 30 + " 因子分析 " + "=" * 30)
    # 主成分分析（PCA）
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_train)

    # 创建可视化DataFrame
    pca_df = pd.DataFrame(data=principal_components,
                          columns=['主成分1', '主成分2'])
    pca_df['碳排放'] = y_train.values

    # 绘制PCA散点图
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='主成分1', y='主成分2', hue='碳排放',
                    data=pca_df, palette='viridis',
                    size='碳排放', sizes=(20, 200))
    plt.title("PCA因子分析 - 主成分分布", fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # 方差解释率
    plt.figure(figsize=(8, 5))
    explained_variance = pca.explained_variance_ratio_
    plt.bar(range(2), explained_variance, alpha=0.6,
            label='各主成分方差解释率')
    plt.plot(range(2), np.cumsum(explained_variance),
             'r-o', label='累计解释率')
    plt.xticks([0, 1], ['PC1', 'PC2'])
    plt.ylabel("方差解释率")
    plt.title("主成分方差解释率分析")
    plt.legend()
    plt.tight_layout()


    # 初始化并训练模型
    print("\n" + "=" * 30 + " 模型训练开始 " + "=" * 30)
    ensemble = CarbonEmissionEnsemble()
    ensemble.train(X_train.values, y_train.values)

    # 保存训练好的模型
    model_filename = "carbon_emission_ensemble_model.pkl"
    joblib.dump(ensemble, model_filename)
    print(f"\n模型已成功保存为 {model_filename}")

    # 分别保存基模型和组合权重
    joblib.dump(ensemble.xgb, "xgb_model.pkl")
    joblib.dump(ensemble.rf, "rf_model.pkl")
    joblib.dump(ensemble.adaptive_weights, "ensemble_weights.pkl")

    print("\n模型保存完成：")
    print(f"- XGBoost模型已保存为 xgb_model.pkl")
    print(f"- 随机森林模型已保存为 rf_model.pkl")
    print(f"- 组合权重已保存为 ensemble_weights.pkl")

    ensemble = CarbonEmissionEnsemble()
    ensemble.train(X_train.values, y_train.values)

    # 模型评估
    methods = ['equal', 'residual', 'self-adaption']
    colors = ['#4B0082', '#FF69B4', '#2ca02c']  # 与之前代码一致的颜色方案

    # 组合模型评估
    test_results = ensemble.evaluate_ensemble(X_test.values, y_test.values, methods)

    # ==========================
    # 新增：组合模型性能报告
    # ==========================
    report_data = []
    for method in methods:
        report_data.append({
            '方法': method,
            'RMSE': test_results[method]['RMSE'],
            'MAE': test_results[method]['MAE'],
            'R²': test_results[method]['R2']
        })
    report_df = pd.DataFrame(report_data).round(4)
    print("\n" + "=" * 30 + " 组合模型性能对比报告 " + "=" * 30)
    print(report_df.to_markdown(index=False))

    # ==========================
    # 新增：核心分析可视化
    # ==========================
    # 1. 相关关系对比
    y_preds = {method: test_results[method]['Prediction'] for method in methods}
    ensemble.plot_correlation_comparison(y_test, y_preds, methods, colors)

    # 2. 误差分布对比
    ensemble.plot_error_distribution(y_test, y_preds, methods, colors)

    # 3. 预测趋势对比（使用测试集索引）
    sample_indices = np.arange(len(y_test))
    ensemble.plot_prediction_trend(y_test, y_preds, methods, colors, sample_indices)

    # 评估所有模型
    results = {}
    methods = ['equal', 'residual', 'self-adaption']

    print("\n" + "=" * 30 + " 模型评估结果 " + "=" * 30)
    for method in methods:
        pred = ensemble.predict(X_test.values, method=method)
        results[method] = {
            'RMSE': np.sqrt(mean_squared_error(y_test, pred)),
            'R2': r2_score(y_test, pred)
        }
        print(f"{method.upper():<12} | RMSE: {results[method]['RMSE']:.2f} | R²: {results[method]['R2']:.2f}")

    # ==========================
    # 可视化模块
    # ==========================
    # 预测结果对比
    plt.figure(figsize=(14, 7))
    x_axis = np.arange(len(y_test))

    # 绘制基模型
    plt.scatter(x_axis, ensemble.rf.predict(X_test),
                alpha=0.6, label='随机森林', color='navy')
    plt.scatter(x_axis, ensemble.xgb.predict(X_test),
                alpha=0.6, label='XGBoost', color='darkorange')

    # 绘制组合模型
    colors = ['limegreen', 'crimson', 'darkviolet']
    for idx, method in enumerate(methods):
        pred = ensemble.predict(X_test.values, method=method)
        plt.scatter(x_axis, pred, marker='s',
                    edgecolor=colors[idx], facecolor='none',
                    s=80, linewidth=2, label=f'组合-{method}')

    plt.plot(x_axis, y_test, 'k*', markersize=10, label='真实值')
    plt.title("碳排放预测结果对比", fontsize=14)
    plt.xlabel("样本序号")
    plt.ylabel("碳排放量")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # 权重分布图
    plt.figure(figsize=(10, 6))
    # 计算残差权重
    total_mse = ensemble.mse1 + ensemble.mse2 + 1e-8
    residual_weights = (
        ensemble.mse2 / total_mse,
        ensemble.mse1 / total_mse
    )
    weights = {
        '等值赋权': (0.5, 0.5),
        '残差赋权': residual_weights,
        '自适应赋权': ensemble.adaptive_weights
    }

    colors = ['#1f77b4', '#ff7f0e']
    labels = ['XGBoost', '随机森林']

    for i, (name, (w1, w2)) in enumerate(weights.items()):
        plt.barh(name, w1, color=colors[0], edgecolor='white')
        plt.barh(name, w2, left=w1, color=colors[1], edgecolor='white')
        plt.text(w1 / 2, i, f'{w1:.1%}', ha='center', va='center', color='white')
        plt.text(w1 + w2 / 2, i, f'{w2:.1%}', ha='center', va='center', color='white')

    plt.xlim(0, 1)
    plt.title("模型权重分配对比")
    plt.xlabel("权重比例")
    plt.tight_layout()

    # # 特征重要性对比
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    #
    # # XGBoost特征重要性
    # xgb_imp = ensemble.xgb.feature_importances_
    # xgb_imp = 100 * (xgb_imp / xgb_imp.sum())
    # pd.Series(xgb_imp, index=features).sort_values().plot.barh(ax=ax1, color='darkorange')
    # ax1.set_title("XGBoost特征重要性")
    # ax1.set_xlabel("重要性 (%)")
    #
    # # 随机森林特征重要性
    # rf_imp = ensemble.rf.feature_importances_
    # rf_imp = 100 * (rf_imp / rf_imp.sum())
    # pd.Series(rf_imp, index=features).sort_values().plot.barh(ax=ax2, color='navy')
    # ax2.set_title("随机森林特征重要性")
    # ax2.set_xlabel("重要性 (%)")
    #
    # plt.tight_layout()
    # plt.show()

    #print("\n" + "=" * 30 + " SHAP分析 " + "=" * 30)




    # 使用前100个样本加速计算
    sample_idx = np.random.choice(X_train.shape[0], 100, replace=False)
    X_sample = X_train.iloc[sample_idx]

    # 初始化SHAP解释器
    explainer_xgb = shap.TreeExplainer(ensemble.xgb)
    explainer_rf = shap.TreeExplainer(ensemble.rf)

    # 计算SHAP值
    shap_values_xgb = explainer_xgb.shap_values(X_sample)
    shap_values_rf = explainer_rf.shap_values(X_sample)


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # XGBoost特征重要性
    plt.sca(ax1)  # 设置当前子图为ax1
    shap.summary_plot(
        shap_values_xgb,
        X_sample,
        feature_names=features,
        plot_type="bar",
        color='#FFA07A',
        show=False,
        max_display=10
    )
    ax1.set_title("XGBoost SHAP特征重要性", fontsize=14)
    ax1.set_xlabel("平均|SHAP值|")

    # 随机森林特征重要性
    plt.sca(ax2)  # 设置当前子图为ax2
    shap.summary_plot(
        shap_values_rf,
        X_sample,
        feature_names=features,
        plot_type="bar",
        color='#20B2AA',
        show=False,
        max_display=10
    )
    ax2.set_title("随机森林 SHAP特征重要性", fontsize=14)
    ax2.set_xlabel("平均|SHAP值|")

    plt.tight_layout()
    plt.show()


    # SHAP特征影响分析 XGBoost
    plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_values_xgb, X_sample,
                      feature_names=features,
                      plot_type="dot",
                      show=False)
    plt.title("XGBoost特征影响方向分析", fontsize=14)
    plt.colorbar(label="特征值大小")
    plt.tight_layout()

    # SHAP特征影响分析 随机森林
    plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_values_rf, X_sample,
                      feature_names=features,
                      plot_type="dot",
                      show=False)
    plt.title("随机森林特征影响方向分析", fontsize=14)
    plt.colorbar(label="特征值大小")
    plt.tight_layout()


    # 单样本解释示例
    plt.figure(figsize=(10, 4))
    idx = 10  # 选择第10个样本
    shap.force_plot(explainer_xgb.expected_value,
                    shap_values_xgb[idx, :],
                    X_sample.iloc[idx, :],
                    feature_names=features,
                    matplotlib=True,
                    text_rotation=15)
    plt.title(f"样本 {idx} 的SHAP解释（XGBoost）", fontsize=12)
    plt.tight_layout()

    # ==========================
    # 在现有SHAP分析之后添加以下代码
    # ==========================

    print("\n" + "=" * 30 + " 高级特征分析 " + "=" * 30)

    # ==========================
    # 特征SHAP值分布（小提琴图）
    # ==========================
    plt.figure(figsize=(14, 8))

    # 创建特征分箱数据
    shap_df = pd.DataFrame(X_sample, columns=features)
    for col in features:
        shap_df[col + '_bin'] = pd.qcut(shap_df[col], q=5, duplicates='drop')  # 分位数分箱
    shap_df['SHAP'] = shap_values_xgb.sum(axis=1)  # 总SHAP值

    # 绘制组合小提琴图
    melt_df = pd.melt(shap_df,
                      id_vars='SHAP',
                      value_vars=[col + '_bin' for col in features],
                      var_name='Feature',
                      value_name='Bin')

    melt_df['Feature'] = melt_df['Feature'].str.replace('_bin', '')
    plt.rcParams['font.size'] = 10
    sns.violinplot(x='SHAP', y='Feature', hue='Bin',
                   data=melt_df, palette='viridis',
                   split=True, inner="quartile",
                   linewidth=1, scale='count')

    plt.title("特征值分布对SHAP值的影响（小提琴图）", fontsize=14)
    plt.xlabel("SHAP值（对预测的影响）")
    plt.ylabel("")
    plt.legend(bbox_to_anchor=(1.05, 1), title="特征分箱")
    plt.tight_layout()

    # ==========================
    # 单特征依赖分析可视化
    # ==========================
    print("\n" + "=" * 30 + " 单特征依赖分析 " + "=" * 30)

    # 创建输出目录
    output_dir = "feature_dependence_plots"
    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams['font.size'] = 12  # 统一字体大小

    # 遍历所有特征生成图表
    for idx, feature in enumerate(features):
        plt.figure(figsize=(10, 6))

        # 生成SHAP依赖图
        shap.dependence_plot(
            ind=idx,
            shap_values=shap_values_xgb,
            features=X_sample,
            feature_names=features,
            interaction_index=None,  # 不显示交互特征
            show=False,
            title=None
        )

        # 设置图表样式
        plt.title(f"'{feature}' 的特征依赖分析", fontsize=14)
        plt.xlabel(feature, fontsize=12)
        plt.ylabel("SHAP值\n（对碳排放预测的影响）", fontsize=12)
        plt.grid(True, alpha=0.3)

        # 添加趋势线
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        coefficients = np.polyfit(X_sample[feature], shap_values_xgb[:, idx], 2)
        trend = np.poly1d(coefficients)(x)
        plt.plot(x, trend, color='red', linewidth=2, linestyle='--',
                 label='二次趋势线')

        # 添加统计指标
        skewness = skew(X_sample[feature])
        kurt = kurtosis(X_sample[feature])
        plt.text(0.05, 0.85,
                 f"偏度: {skewness:.2f}\n峰度: {kurt:.2f}",
                 transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.8))

        plt.legend()
        plt.tight_layout()

        # 保存为单独文件
        filename = f"{output_dir}/{feature}_dependency.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成：{filename}")

    print("\n所有特征依赖分析图表已保存至", os.path.abspath(output_dir))

# ==========================
# 残差赋权组合模型SHAP分析
# ==========================
print("\n" + "=" * 30 + " 残差赋权组合模型SHAP分析 " + "=" * 30)

# 计算残差权重
total_mse = ensemble.mse1 + ensemble.mse2 + 1e-8
w_rf = ensemble.mse1 / total_mse  # 随机森林权重
w_xgb = ensemble.mse2 / total_mse  # XGBoost权重
print(f"残差权重分配：XGBoost({w_xgb:.2%}) + 随机森林({w_rf:.2%})")

# 计算组合SHAP值（残差权重加权）
combined_shap_residual = w_xgb * np.array(shap_values_xgb) + w_rf * np.array(shap_values_rf)

# ==========================
# 组合模型特征分析可视化
# ==========================
# 特征重要性对比（基模型 vs 组合）
plt.figure(figsize=(14, 6))

# 基模型重要性
plt.subplot(121)
shap.summary_plot(shap_values_xgb, X_sample, feature_names=features,
                  plot_type="bar", color='#FFA07A', show=False)
plt.title("XGBoost特征重要性", fontsize=12)
plt.xlabel("")

plt.subplot(122)
shap.summary_plot(shap_values_rf, X_sample, feature_names=features,
                  plot_type="bar", color='#20B2AA', show=False)
plt.title("随机森林特征重要性", fontsize=12)
plt.xlabel("")

plt.suptitle("基模型特征重要性对比", fontsize=14, y=1.02)
plt.tight_layout()

# 组合模型特征重要性
plt.figure(figsize=(10, 6))
shap.summary_plot(combined_shap_residual, X_sample,
                  feature_names=features,
                  plot_type="bar",
                  color='#6A5ACD',
                  show=False)
plt.title(f"残差赋权组合模型特征重要性", fontsize=12)
plt.xlabel("SHAP值")
plt.tight_layout()

# ==========================
# 动态权重依赖分析
# ==========================
print("\n" + "=" * 30 + " 动态权重特征依赖 " + "=" * 30)

# 创建输出目录
output_dir_residual = "residual_weight_analysis"
os.makedirs(output_dir_residual, exist_ok=True)

for idx, feature in enumerate(features):
    plt.figure(figsize=(12, 6))

    # 散点图矩阵
    plt.scatter(X_sample[feature], combined_shap_residual[:, idx],
                c=X_sample[feature], cmap='viridis',
                alpha=0.6, edgecolors='w')

    # 趋势线拟合
    x = X_sample[feature].values
    y = combined_shap_residual[:, idx]
    z = np.polyfit(x, y, 2)
    p = np.poly1d(z)
    x_cont = np.linspace(min(x), max(x), 100)
    plt.plot(x_cont, p(x_cont), 'r--', lw=2,
             label=f'二次趋势线\n(y={z[0]:.2e}x²{z[1]:+.2e}x{z[2]:+.2e})')

    # 统计注释
    stats_text = f"""
    特征分布统计:
    均值 = {x.mean():.2f}
    标准差 = {x.std():.2f}
    偏度 = {skew(x):.2f}
    峰度 = {kurtosis(x):.2f}
    """
    plt.annotate(stats_text, xy=(0.05, 0.65),
                 xycoords='axes fraction',
                 bbox=dict(boxstyle="round", alpha=0.1))

    plt.colorbar(label="特征值")
    plt.xlabel(feature)
    plt.ylabel("组合SHAP值")
    plt.title(f"'{feature}' 动态权重SHAP依赖分析", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)

    # 保存图表
    filename = f"{output_dir_residual}/residual_{feature}_dependency.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"生成文件：{filename}")

# ==========================
# 交互式样本解释
# ==========================
print("\n" + "=" * 30 + " 交互式样本解释 " + "=" * 30)

# 随机选择3个样本进行详细解释
sample_indices = np.random.choice(len(X_sample), 3, replace=False)

for i, idx in enumerate(sample_indices, 1):
    plt.figure(figsize=(10, 4))

    # 计算组合预测值
    pred_xgb = ensemble.xgb.predict(X_sample.iloc[[idx]])
    pred_rf = ensemble.rf.predict(X_sample.iloc[[idx]])
    combined_pred = w_xgb * pred_xgb + w_rf * pred_rf

    # SHAP力图示
    shap.force_plot(
        base_value=w_xgb * explainer_xgb.expected_value + w_rf * explainer_rf.expected_value,
        shap_values=combined_shap_residual[idx, :],
        features=X_sample.iloc[idx, :],
        feature_names=features,
        matplotlib=True,
        show=False,
        text_rotation=15
    )

    # 添加预测信息
    plt.title(f"样本 {idx} 解释\n预测值: {combined_pred[0]:.1f} = "
              f"XGB({pred_xgb[0]:.1f})*{w_xgb:.2f} + "
              f"RF({pred_rf[0]:.1f})*{w_rf:.2f}",
              fontsize=10)

    # 保存图表
    filename = f"{output_dir_residual}/sample_{i}_explanation.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"生成样本解释图：{filename}")

print("\n残差赋权组合模型分析完成！输出目录：", os.path.abspath(output_dir_residual))

# ==========================
# 残差组合模型SHAP特征重要性图
# ==========================
plt.figure(figsize=(10, 6))

# 计算组合SHAP值（示例数据）
features = ['煤炭使用量(万吨)', '全社会用电量(亿千瓦时)', 'GDP总量(亿元)',
            '常住人口(万人)', '粗钢产量(万吨)', '城镇化率(%)']
shap_xgb = np.array([85, 72, 68, 65, 60, 45])  # XGBoost SHAP值
shap_rf = np.array([78, 65, 70, 62, 58, 40])   # 随机森林 SHAP值

# # 计算残差权重（示例比例）
# mse_xgb = 0.3  # 假设XGB的MSE
# mse_rf = 0.7    # 假设RF的MSE
# total_mse = mse_xgb + mse_rf
# w_xgb = mse_rf / total_mse  # 0.7
# w_rf = mse_xgb / total_mse  # 0.3

# 计算组合SHAP值
combined_shap = w_xgb * shap_xgb + w_rf * shap_rf

# 排序处理
sorted_idx = np.argsort(combined_shap)
sorted_features = [features[i] for i in sorted_idx]
sorted_values = combined_shap[sorted_idx]

# 创建条形图
bar_colors = ['#6A5ACD'] * len(features)  # 使用紫色系
plt.barh(sorted_features, sorted_values, color=bar_colors, edgecolor='black')

# 样式设置
plt.xlim(0, 100)
plt.xticks(np.arange(0, 101, 20), fontsize=10)
plt.yticks(fontsize=12)
plt.xlabel("SHAP值", fontsize=12)
plt.grid(axis='x', alpha=0.3)

# 添加权重标注
plt.text(85, 5.2, f"权重分配：XGBoost({w_xgb:.0%}) + RF({w_rf:.0%})",
         fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

# 标题设置
plt.title("残差组合模型 SHAP特征重要性",
          fontsize=14, pad=20,
          fontweight='bold',
          position=(0.5, 1.03))

plt.tight_layout()
plt.show()