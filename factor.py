import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


def load_constituent_stocks(index_info_path):
    """
    读取指数成分股信息

    参数:
    index_info_path (str): 指数成分股信息文件路径

    返回:
    list: 股票代码列表
    """
    try:
        constituents_df = pd.read_csv(index_info_path)
        # 将整数代码转换为字符串格式的股票代码（保持6位数字格式）
        stock_codes = [str(code).zfill(6) for code in constituents_df['con_code']]
        return stock_codes
    except Exception as e:
        print(f"读取成分股信息失败: {e}")
        return []


def load_stock_data(stock_code, base_path):
    """
    加载单个股票的分钟级数据

    参数:
    stock_code (str): 股票代码
    base_path (str): 基础路径

    返回:
    DataFrame: 包含OHLCV数据的DataFrame
    """
    try:
        file_path = os.path.join(base_path, f"{stock_code}.csv")
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return None

        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 标准化列名
        if 'datetime' not in df.columns and 'date' in df.columns:
            df.rename(columns={'date': 'datetime'}, inplace=True)

        # 确保datetime列的格式正确
        df['datetime'] = pd.to_datetime(df['datetime'])

        # 添加股票代码列
        df['stock_code'] = stock_code

        return df
    except Exception as e:
        print(f"加载股票 {stock_code} 的数据失败: {e}")
        return None


def load_multiple_stocks(base_path, constituent_file="000300.SH_info.csv", n_stocks=None):
    """
    加载多个股票的分钟级数据

    参数:
    base_path (str): 股票数据基础路径
    constituent_file (str): 成分股信息文件名
    n_stocks (int, optional): 要加载的股票数量，None表示全部加载

    返回:
    dict: 股票代码到DataFrame的映射
    """
    # 读取成分股信息
    index_info_path = os.path.join(base_path, constituent_file)
    stock_codes = load_constituent_stocks(index_info_path)

    if not stock_codes:
        print("未能读取到成分股信息，尝试直接读取CSV文件...")
        csv_files = glob.glob(os.path.join(base_path, "*.csv"))
        csv_files = [f for f in csv_files if os.path.basename(f) != constituent_file]  # 排除成分股信息文件
        stock_codes = [os.path.basename(f).split('.')[0] for f in csv_files]

    print(f"读取到 {len(stock_codes)} 个股票代码")

    # 如果指定了股票数量，则只取前n_stocks个
    if n_stocks is not None:
        stock_codes = stock_codes[:n_stocks]

    # 加载每个股票的数据
    stocks_data = {}
    for stock_code in tqdm(stock_codes, desc="加载股票数据"):
        df = load_stock_data(stock_code, base_path)
        if df is not None:
            stocks_data[stock_code] = df

    print(f"成功加载 {len(stocks_data)} 个股票的数据")
    return stocks_data


class AlphaFactors:
    def __init__(self, data=None, base_path=None, directory=None, constituent_file="000300.SH_info.csv", n_stocks=None):
        """
        初始化Alpha因子类

        参数:
        data (DataFrame): 包含datetime, open, high, low, close, volume的DataFrame，用于单股票分析
        base_path (str): 股票数据基础路径，用于多股票分析
        directory (str): 与base_path相同，为了兼容旧代码
        constituent_file (str): 成分股信息文件名
        n_stocks (int, optional): 要加载的股票数量，None表示全部加载
        """
        self.stocks_data = {}
        self.factors = {}
        self.all_stocks_factors = {}

        # 如果提供了directory参数，使用它来设置base_path
        if directory is not None:
            base_path = directory

        if data is not None:
            # 单股票模式
            if 'datetime' in data.columns:
                data['datetime'] = pd.to_datetime(data['datetime'])
                data.set_index('datetime', inplace=True)
            self.data = data
            self.mode = 'single'
        elif base_path is not None:
            # 多股票模式
            self.stocks_data = load_multiple_stocks(base_path, constituent_file, n_stocks)
            self.mode = 'multiple'
        else:
            raise ValueError("必须提供data或base_path参数之一")

    def calculate_all_factors(self):
        """计算所有因子并返回因子DataFrame"""
        if self.mode == 'single':
            # 单股票模式
            self.price_reversal_factor()
            self.price_volume_factor()
            self.volatility_factor()

            # 将所有因子合并到一个DataFrame
            factors_df = pd.DataFrame(self.factors)

            # 删除包含NaN的行
            factors_df = factors_df.dropna()

            return factors_df
        else:
            # 多股票模式
            all_stocks_factors = {}

            for stock_code, stock_data in tqdm(self.stocks_data.items(), desc="计算股票因子"):
                # 为每个股票创建单独的分析器
                single_stock = stock_data.copy()
                if 'datetime' in single_stock.columns:
                    single_stock.set_index('datetime', inplace=True)

                # 计算因子
                try:
                    # 计算价格反转因子
                    short_window = 1
                    long_window = 5

                    short_returns = single_stock['close'].pct_change(short_window)
                    long_returns = single_stock['close'].pct_change(long_window)

                    reversal_factor = -(short_returns - long_returns.rolling(short_window).mean())
                    reversal_factor = (reversal_factor - reversal_factor.rolling(50).mean()) / reversal_factor.rolling(
                        50).std()

                    # 计算价量相关因子
                    window = 5
                    price_change = single_stock['close'].pct_change(1).abs()
                    volume_change = single_stock['volume'].pct_change(1).abs()

                    pv_ratio = price_change / (volume_change + 1e-10)  # 添加小数避免除以零
                    pv_factor = -(pv_ratio - pv_ratio.rolling(window).mean()) / pv_ratio.rolling(window).std()

                    # 计算高频波动率因子
                    window = 10
                    price_range = (single_stock['high'] - single_stock['low']) / single_stock['open']

                    volatility_ma = price_range.rolling(window).mean()
                    volatility_std = price_range.rolling(window).std()

                    volatility_factor = (price_range - volatility_ma) / (volatility_std + 1e-10)

                    # 合并因子到一个DataFrame
                    stock_factors = pd.DataFrame({
                        'price_reversal': reversal_factor,
                        'price_volume': pv_factor,
                        'volatility': volatility_factor
                    })

                    # 删除NaN值
                    stock_factors = stock_factors.dropna()

                    # 添加股票代码
                    stock_factors['stock_code'] = stock_code

                    all_stocks_factors[stock_code] = stock_factors
                except Exception as e:
                    print(f"计算股票 {stock_code} 的因子时出错: {e}")

            # 合并所有股票的因子
            if all_stocks_factors:
                self.all_stocks_factors = all_stocks_factors
                return all_stocks_factors
            else:
                return {}

    def factor_correlation(self, stock_code=None):
        """
        计算并返回因子相关性矩阵

        参数:
        stock_code (str, optional): 股票代码。如果为None且为多股票模式，则计算所有股票的平均相关性

        返回:
        DataFrame: 因子相关性矩阵
        """
        if self.mode == 'single':
            factors_df = pd.DataFrame(self.factors).dropna()
            return factors_df.corr()
        else:
            if stock_code is not None:
                # 返回特定股票的因子相关性
                if stock_code in self.all_stocks_factors:
                    factors_df = self.all_stocks_factors[stock_code]
                    factors_df = factors_df[['price_reversal', 'price_volume', 'volatility']]  # 排除stock_code列
                    return factors_df.corr()
                else:
                    print(f"未找到股票 {stock_code} 的因子数据")
                    return pd.DataFrame()
            else:
                # 计算所有股票的平均相关性
                all_corrs = []
                for stock_code, factors_df in self.all_stocks_factors.items():
                    factors_only = factors_df[['price_reversal', 'price_volume', 'volatility']]
                    corr_matrix = factors_only.corr()
                    all_corrs.append(corr_matrix)

                if all_corrs:
                    avg_corr = sum(all_corrs) / len(all_corrs)
                    return avg_corr
                else:
                    return pd.DataFrame()

    def factor_performance(self, stock_code=None, forward_returns_period=5):
        """
        计算因子表现

        参数:
        stock_code (str, optional): 股票代码。如果为None且为多股票模式，则计算所有股票的平均表现
        forward_returns_period (int): 未来收益计算周期（分钟）

        返回:
        dict: 因子表现指标
        """

        def safe_qcut(x, q, **kwargs):
            """安全的分位数切分，处理重复值和异常情况"""
            try:
                # 首先尝试使用duplicates='drop'参数
                return pd.qcut(x, q, duplicates='drop', **kwargs)
            except ValueError:
                # 如果仍然失败，可能是数据太少或分布极不均匀
                if len(x.unique()) < q:
                    # 数据唯一值少于所需分位数，使用等间距划分
                    return pd.cut(x, q, labels=kwargs.get('labels', None))
                else:
                    # 如果是其他原因，再次尝试处理NaN值
                    no_nan_x = x.dropna()
                    if len(no_nan_x) > 0:
                        # 对非NaN值进行分位数切分
                        result = pd.Series(index=x.index, dtype='float64')
                        result.loc[no_nan_x.index] = pd.qcut(no_nan_x, q, duplicates='drop', **kwargs)
                        return result
                    else:
                        # 如果数据全为NaN，返回全NaN的序列
                        return pd.Series(np.nan, index=x.index)

        if self.mode == 'single':
            # 计算未来收益
            future_returns = self.data['close'].pct_change(forward_returns_period).shift(-forward_returns_period)

            results = {}
            for factor_name, factor_values in self.factors.items():
                # 删除NaN值
                valid_data = pd.DataFrame({
                    'factor': factor_values,
                    'future_returns': future_returns
                }).dropna()

                if len(valid_data) < 10:  # 如果数据太少，跳过
                    print(f"警告: {factor_name} 因子有效数据不足")
                    continue

                # 计算IC值（因子与未来收益的相关性）
                ic = valid_data['factor'].corr(valid_data['future_returns'])

                # 安全地计算分位数
                try:
                    valid_data['quantile'] = safe_qcut(valid_data['factor'], 5, labels=False)
                    # 计算分位数表现
                    quantile_returns = valid_data.groupby('quantile')['future_returns'].mean()

                    # 计算多空组合（最高分位减最低分位）的收益
                    if len(quantile_returns) >= 5:
                        long_short_return = quantile_returns.iloc[4] - quantile_returns.iloc[0]
                    else:
                        # 如果分位数不足5个，使用最高和最低的两个
                        long_short_return = quantile_returns.max() - quantile_returns.min()
                except Exception as e:
                    print(f"计算 {factor_name} 因子分位数时出错: {e}")
                    # 使用简单的排序分组
                    valid_data['quantile'] = pd.Series(valid_data['factor']).rank(method='first', pct=True).mul(
                        5).astype(int).clip(0, 4)
                    quantile_returns = valid_data.groupby('quantile')['future_returns'].mean()

                    # 安全计算多空收益
                    try:
                        if len(quantile_returns) > 1:
                            long_short_return = quantile_returns.max() - quantile_returns.min()
                        else:
                            long_short_return = np.nan
                    except:
                        long_short_return = np.nan

                results[factor_name] = {
                    'IC': ic,
                    'Quantile_Returns': quantile_returns,
                    'Long_Short_Return': long_short_return
                }

            return results
        else:
            if stock_code is not None:
                # 返回特定股票的因子表现
                if stock_code in self.all_stocks_factors and stock_code in self.stocks_data:
                    stock_data = self.stocks_data[stock_code].set_index('datetime')
                    factors_df = self.all_stocks_factors[stock_code]

                    # 计算未来收益
                    future_returns = stock_data['close'].pct_change(forward_returns_period).shift(
                        -forward_returns_period)

                    results = {}
                    for factor_name in ['price_reversal', 'price_volume', 'volatility']:
                        if factor_name not in factors_df.columns:
                            continue

                        # 删除NaN值
                        valid_data = pd.DataFrame({
                            'factor': factors_df[factor_name],
                            'future_returns': future_returns
                        }).dropna()

                        if len(valid_data) < 10:  # 如果数据太少，跳过
                            print(f"警告: 股票 {stock_code} 的 {factor_name} 因子有效数据不足")
                            continue

                        # 计算IC值
                        ic = valid_data['factor'].corr(valid_data['future_returns'])

                        # 安全地计算分位数
                        try:
                            valid_data['quantile'] = safe_qcut(valid_data['factor'], 5, labels=False)
                            # 计算分位数表现
                            quantile_returns = valid_data.groupby('quantile')['future_returns'].mean()

                            # 计算多空组合收益
                            if len(quantile_returns) >= 5:
                                long_short_return = quantile_returns.iloc[4] - quantile_returns.iloc[0]
                            else:
                                # 如果分位数不足5个，使用最高和最低的两个
                                long_short_return = quantile_returns.max() - quantile_returns.min()
                        except Exception as e:
                            print(f"计算股票 {stock_code} 的 {factor_name} 因子分位数时出错: {e}")
                            # 使用简单的排序分组
                            valid_data['quantile'] = pd.Series(valid_data['factor']).rank(method='first', pct=True).mul(
                                5).astype(int).clip(0, 4)
                            quantile_returns = valid_data.groupby('quantile')['future_returns'].mean()

                            # 安全计算多空收益
                            try:
                                if len(quantile_returns) > 1:
                                    long_short_return = quantile_returns.max() - quantile_returns.min()
                                else:
                                    long_short_return = np.nan
                            except:
                                long_short_return = np.nan

                        results[factor_name] = {
                            'IC': ic,
                            'Quantile_Returns': quantile_returns,
                            'Long_Short_Return': long_short_return
                        }

                    return results
                else:
                    print(f"未找到股票 {stock_code} 的数据")
                    return {}
            else:
                # 计算所有股票的平均表现
                all_ics = {factor: [] for factor in ['price_reversal', 'price_volume', 'volatility']}
                all_returns = {factor: [] for factor in ['price_reversal', 'price_volume', 'volatility']}

                for stock_code in tqdm(self.all_stocks_factors.keys(), desc="计算因子表现"):
                    try:
                        perf = self.factor_performance(stock_code, forward_returns_period)
                        if perf:
                            for factor_name, metrics in perf.items():
                                if 'IC' in metrics and not np.isnan(metrics['IC']):
                                    all_ics[factor_name].append(metrics['IC'])
                                if 'Long_Short_Return' in metrics and not np.isnan(metrics['Long_Short_Return']):
                                    all_returns[factor_name].append(metrics['Long_Short_Return'])
                    except Exception as e:
                        print(f"计算股票 {stock_code} 的因子表现时出错: {e}")
                        continue

                # 计算平均值
                avg_results = {}
                for factor_name in ['price_reversal', 'price_volume', 'volatility']:
                    valid_ics = [ic for ic in all_ics[factor_name] if not np.isnan(ic)]
                    valid_returns = [ret for ret in all_returns[factor_name] if not np.isnan(ret)]

                    if not valid_ics or not valid_returns:
                        print(f"警告: {factor_name} 因子没有有效的性能数据")
                        continue

                    avg_ic = np.mean(valid_ics) if valid_ics else np.nan
                    avg_return = np.mean(valid_returns) if valid_returns else np.nan
                    std_return = np.std(valid_returns) if len(valid_returns) > 1 else np.nan

                    # 安全计算夏普比率
                    if not np.isnan(avg_return) and not np.isnan(std_return) and std_return > 0:
                        sharpe = avg_return / std_return
                    else:
                        sharpe = np.nan

                    avg_results[factor_name] = {
                        'Avg_IC': avg_ic,
                        'Avg_Long_Short_Return': avg_return,
                        'Sharpe_Ratio': sharpe,
                        'Sample_Size': len(valid_returns)
                    }

                return avg_results

    def save_factors(self, output_dir):
        """
        保存因子数据到CSV文件

        参数:
        output_dir (str): 输出目录
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if self.mode == 'single':
            factors_df = pd.DataFrame(self.factors).dropna()
            output_path = os.path.join(output_dir, "single_stock_factors.csv")
            factors_df.to_csv(output_path)
            print(f"因子已保存到 {output_path}")
        else:
            # 保存每个股票的因子
            for stock_code, factors_df in self.all_stocks_factors.items():
                output_path = os.path.join(output_dir, f"{stock_code}_factors.csv")
                factors_df.to_csv(output_path)

            # 合并所有股票的因子
            all_factors = pd.concat(self.all_stocks_factors.values())
            output_path = os.path.join(output_dir, "all_stocks_factors.csv")
            all_factors.to_csv(output_path)

            print(f"所有股票的因子已保存到 {output_dir}")

    def generate_readme(self, output_dir):
        """
        生成README文件，包含因子相关性和表现指标

        参数:
        output_dir (str): 输出目录
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        readme_path = os.path.join(output_dir, "README.md")

        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write("# Alpha因子分析报告\n\n")

            # 添加因子说明
            f.write("## 因子说明\n\n")
            f.write("### 1. 价格反转因子 (Price Reversal Factor)\n")
            f.write(
                "短期价格反转因子，基于短期超买超卖后的均值回归原理。当短期收益过高时，预期价格会下跌；当短期收益过低时，预期价格会上涨。\n\n")

            f.write("### 2. 价量相关因子 (Price-Volume Factor)\n")
            f.write(
                "价量相关因子基于价格变动与成交量变动的关系。小成交量带来的大价格变动通常不可持续，而大成交量带来的价格变动更有持续性。\n\n")

            f.write("### 3. 高频波动率因子 (Volatility Factor)\n")
            f.write(
                "波动率因子基于短期内的价格波动范围。异常波动通常暗示后续行情可能转向，而低波动可能预示波动的积累。\n\n")

            # 添加相关性矩阵
            f.write("## 因子相关性矩阵\n\n")
            corr_matrix = self.factor_correlation()
            f.write("```\n")
            f.write(str(corr_matrix))
            f.write("\n```\n\n")

            # 检查最大相关性
            if not corr_matrix.empty:
                mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                max_corr = corr_matrix.where(mask).max().max()
                f.write(f"最大因子相关性: {max_corr:.4f}\n\n")
                if max_corr <= 0.5:
                    f.write("✓ 因子相关性满足要求（最大相关性不超过0.5）\n\n")
                else:
                    f.write("✗ 因子相关性超过要求（最大相关性应不超过0.5）\n\n")

            # 添加因子表现
            f.write("## 因子表现指标\n\n")
            performance = self.factor_performance()

            f.write("| 因子 | 平均IC值 | 平均多空收益 | 夏普比率 | 样本数量 |\n")
            f.write("|------|---------|------------|---------|--------|\n")

            for factor_name, metrics in performance.items():
                avg_ic = metrics.get('Avg_IC', np.nan)
                avg_return = metrics.get('Avg_Long_Short_Return', np.nan)
                sharpe = metrics.get('Sharpe_Ratio', np.nan)
                sample_size = metrics.get('Sample_Size', 0)

                f.write(f"| {factor_name} | {avg_ic:.4f} | {avg_return:.6f} | {sharpe:.4f} | {sample_size} |\n")

            # 计算平均夏普比率
            sharpe_values = [metrics.get('Sharpe_Ratio', np.nan) for metrics in performance.values()]
            valid_sharpes = [s for s in sharpe_values if not np.isnan(s)]
            avg_sharpe = np.mean(valid_sharpes) if valid_sharpes else np.nan

            f.write(f"\n所有因子的平均夏普比率: {avg_sharpe:.4f}\n")

            # 添加参考文献
            f.write("\n## 参考文献\n\n")
            f.write(
                "1. Lehmann, B. N. (1990). \"Fads, Martingales, and Market Efficiency.\" The Quarterly Journal of Economics, 105(1), 1-28.\n")
            f.write(
                "2. Jegadeesh, N. (1990). \"Evidence of Predictable Behavior of Security Returns.\" The Journal of Finance, 45(3), 881-898.\n")
            f.write(
                "3. Karpoff, J. M. (1987). \"The Relation Between Price Changes and Trading Volume: A Survey.\" Journal of Financial and Quantitative Analysis, 22(1), 109-126.\n")

        print(f"README已生成到 {readme_path}")


# 主函数，用于计算所有沪深300成分股的因子
def main():
    """计算所有沪深300成分股的因子并生成报告"""
    print("开始计算沪深300所有成分股的Alpha因子...")

    # 设置路径
    local_path = r"D:\网页端下载\算法\project\沪深300数据"  # 本地分钟数据根目录
    output_dir = r"D:\网页端下载\算法\project\output"  # 输出目录

    # 初始化多股票因子分析器
    alpha_factors = AlphaFactors(base_path=local_path, constituent_file="000300.SH_info.csv")

    # 计算所有股票的因子
    print("计算所有股票的因子...")
    alpha_factors.calculate_all_factors()

    # 计算因子相关性
    print("计算因子相关性...")
    corr_matrix = alpha_factors.factor_correlation()
    print("因子相关性矩阵:")
    print(corr_matrix)

    # 计算因子表现
    print("计算因子表现...")
    performance = alpha_factors.factor_performance()
    print("因子表现:")
    for factor_name, metrics in performance.items():
        print(f"\n{factor_name} 因子:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value}")

    # 保存因子数据
    print("保存因子数据...")
    alpha_factors.save_factors(output_dir)

    # 生成README
    print("生成README...")
    alpha_factors.generate_readme(output_dir)

    print("计算完成!")


if __name__ == "__main__":
    main()


