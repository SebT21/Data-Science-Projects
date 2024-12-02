#Code by Sebastian Tampu (1004928572)
#Using parts of the rotational momentum code from class

import pandas as pd
import numpy as np
import yfinance as yf
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import statsmodels.api as sm
import math

#function to get historical ETF data
def get_etf_data(stock_list, start_date, end_date):
    main_df = pd.DataFrame()
    for stock in stock_list:
        df = yf.download(stock, start=start_date, end=end_date)
        df.drop(['Close', 'High', 'Low', 'Open', 'Volume'], axis=1, inplace=True)
        df.rename(columns={'Adj Close': stock}, inplace=True)
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer', rsuffix='_dup')  #handle any overlap
            main_df.drop(columns=[col for col in main_df.columns if '_dup' in col], inplace=True)  #remove duplicate columns
    return main_df

#function to detrend price series
def detrend_price(series):
    length = len(series)
    x = np.arange(length)
    y = np.array(series.values)
    x_const = sm.add_constant(x)
    model = sm.OLS(y, x_const)
    result = model.fit()
    y_hat = result.params[0] + result.params[1] * x
    resid = y - y_hat
    resid = resid + abs(resid.min() + 1 / 10 * resid.min())
    return resid

#function to perform hierarchical clustering
def hierarchical_cluster(combined_metrics, stock_list, original_etfs):
    Z = linkage(combined_metrics, 'ward')
    plt.figure(figsize=(10, 7))
    dendro = dendrogram(Z, labels=stock_list)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('ETF')
    plt.ylabel('Distance')
    ax = plt.gca()
    xlbls = ax.get_xmajorticklabels()
    for lbl in xlbls:
        if lbl.get_text() in original_etfs:
            lbl.set_color('red')
    plt.show()
    
    clusters = fcluster(Z, t=3, criterion='maxclust')
    score = silhouette_score(combined_metrics, clusters)
    print(f'Hierarchical Clustering Silhouette Score: {score}')
    
    clustered_etfs = {i: [] for i in np.unique(clusters)}
    for i, cluster in enumerate(clusters):
        clustered_etfs[cluster].append(stock_list[i])
    
    return clustered_etfs, clusters

#function to perform k-means clustering
def kmeans_cluster(combined_metrics, stock_list, original_etfs, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(combined_metrics)
    clusters = kmeans.labels_
    score = silhouette_score(combined_metrics, clusters)
    print(f'K-Means Clustering Silhouette Score: {score}')
    
    clustered_etfs = {i: [] for i in np.unique(clusters)}
    for i, cluster in enumerate(clusters):
        clustered_etfs[cluster].append(stock_list[i])
    
    # Plot K-means clustering result
    plt.figure(figsize=(10, 7))
    for i in range(n_clusters):
        cluster_indices = np.where(clusters == i)
        plt.scatter(combined_metrics[cluster_indices, 0], combined_metrics[cluster_indices, 1], label=f'Cluster {i + 1}')
        for index in cluster_indices[0]:
            color = 'red' if stock_list[index] in original_etfs else 'black'
            plt.text(combined_metrics[index, 0], combined_metrics[index, 1], stock_list[index], fontsize=8, color=color)
    
    plt.title('K-Means Clustering')
    plt.xlabel('Mean Return')
    plt.ylabel('Volatility')
    plt.legend()
    plt.show()
    
    return clustered_etfs, clusters

#function to implement the rotational momentum program
def rotational_momentum(dfP, dfAP, clusters, stock_list, lookback=20, short_term_weight=0.3, long_term_weight=0.4, volatility_weight=0.4, delay=1):
    Aperiods = lookback
    Bperiods = 3 * Aperiods + ((3 * Aperiods) // 20) * 2
    Speriods = Aperiods
    Zperiods = 200
    MAperiods = 200
    
    #calculate returns, volatility, and other metrics
    dfA = dfP.pct_change(periods=Aperiods - 1).dropna()
    dfB = dfP.pct_change(periods=Bperiods - 1).dropna()
    dfR = dfP.pct_change(periods=1).dropna()
    dfS = dfR.rolling(window=Speriods).std() * math.sqrt(252)
    dfZ = (dfP - dfP.rolling(window=Zperiods).mean()) / dfP.rolling(window=Zperiods).std()
    dfMA = dfP.rolling(window=MAperiods).mean()
    dfDetrend = dfAP.apply(detrend_price).dropna()
    
    #rank ETFs based on calculated metrics
    dfA_ranks = dfA.rank(axis=1, ascending=False)
    dfB_ranks = dfB.rank(axis=1, ascending=False)
    dfS_ranks = dfS.rank(axis=1, ascending=True)
    
    #apply weights to ranks
    dfA_ranks = dfA_ranks.multiply(short_term_weight)
    dfB_ranks = dfB_ranks.multiply(long_term_weight)
    dfS_ranks = dfS_ranks.multiply(volatility_weight)
    dfAll_ranks = dfA_ranks.add(dfB_ranks, fill_value=0).add(dfS_ranks, fill_value=0)
    
    #choose ETFs based on ranks, handle NaN values
    dfAll_ranks.fillna(0, inplace=True)
    dfChoice = dfAll_ranks.idxmax(axis=1).ffill()
    dfChoice = pd.get_dummies(dfChoice).reindex(columns=stock_list, fill_value=0)
    
    #calculate returns based on chosen ETFs
    dfPLog = dfAP.apply(np.log)
    dfPLogShift = dfPLog.shift(1)
    dfPRR = dfPLog.subtract(dfPLogShift, fill_value=0)
    dfDetrendRR = dfDetrend.apply(np.log).subtract(dfDetrend.shift(1), fill_value=0)
    dfPRR = dfPRR.multiply(dfChoice.shift(delay))
    dfDetrendRR = dfDetrendRR.multiply(dfChoice.shift(delay))
    dfPRR['ALL_R'] = dfPRR.sum(axis=1)
    dfDetrendRR['ALL_R'] = dfDetrendRR.sum(axis=1)
    
    #calculate cumulative returns
    dfPRR['I'] = np.cumprod(1 + dfPRR['ALL_R']).clip(lower=1e-10, upper=1e10)
    dfDetrendRR['I'] = np.cumprod(1 + dfDetrendRR['ALL_R']).clip(lower=1e-10, upper=1e10)
    
    #calculate sharpe ratio
    sharpe = (dfPRR['ALL_R'].mean() / dfPRR['ALL_R'].std()) * math.sqrt(252)
    
    return sharpe, dfPRR['I']

if __name__ == "__main__":
    #ETFs
    stock_list = [
        "UUP", "FDN", "IBB", "IEZ", "IGV", "IHE", "IHF", "IHI", "ITA", "ITB", "IYJ", "IYT", "IYW", "IYZ",
        "KBE", "KCE", "KIE", "PBJ", "PBS", "SMH", "VNQ", "BIL", "TIP", "IEI", "IEF", "TLH", "TLT", "SHY",
        "SPY", "QQQ", "DIA", "XLK", "XLE", "XLF", "XLY", "XLU", "XLI", "XLV", "XLP", "XLB", "XLRE", "XRT",
        "XTL", "XTN", "XOP", "XHB", "IEMG", "EFA", "EEM", "VWO", "VNQI", "VEA", "VTI", "VTV", "VO", "VB",
        "BND", "HYG", "LQD", "BIV", "SHV", "TBT", "GLD", "SLV", "USO", "XME", "MUB", "PFF", "REZ", "KRE",
        "XBI", "IGF", "SOXX", "IYR", "SCHH", "IJS", "IWM"
    ]

    original_etfs = set([
        "UUP", "FDN", "IBB", "IEZ", "IGV", "IHE", "IHF", "IHI", "ITA", "ITB", "IYJ", "IYT", "IYW", "IYZ",
        "KBE", "KCE", "KIE", "PBJ", "PBS", "SMH", "VNQ", "BIL", "TIP", "IEI", "IEF", "TLH", "TLT", "SHY"
    ])

    start_date = "2016-01-30"
    end_date = "2020-01-30"

    #get ETF data
    dfP = get_etf_data(stock_list, start_date, end_date)
    dfAP = dfP.copy()

    #calculate returns and volatility
    returns = dfP.pct_change().dropna()
    volatility = returns.std()
    combined_metrics = np.vstack((returns.mean(), volatility)).T

    #perform hierarchical clustering
    hierarchical_clusters, hierarchical_labels = hierarchical_cluster(combined_metrics, stock_list, original_etfs)

    #perform k-means clustering
    kmeans_clusters, kmeans_labels = kmeans_cluster(combined_metrics, stock_list, original_etfs)

    #separate out new ETFs from the original lists for hierarchical clustering
    new_etfs_hierarchical = [etf for cluster in hierarchical_clusters.values() for etf in cluster if etf not in original_etfs]
    
    #separate out new ETFs from the original lists for k-means clustering
    new_etfs_kmeans = [etf for cluster in kmeans_clusters.values() for etf in cluster if etf not in original_etfs]

    print("New ETFs found similar to the original lists (Hierarchical Clustering):")
    print(new_etfs_hierarchical)
    
    print("New ETFs found similar to the original lists (K-Means Clustering):")
    print(new_etfs_kmeans)

    #sets of parameters for shuffling
    lookbacks = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
    holding_periods = ["2W-FRI", "4W-FRI", "6W-FRI", "8W-FRI", "10W-FRI", "12W-FRI", "14W-FRI", "16W-FRI", "18W-FRI", "20W-FRI", "22W-FRI", "24W-FRI"]
    weights = [
        (0.3, 0.4, 0.4), (0.2, 0.5, 0.3), (0.4, 0.3, 0.3), (0.5, 0.3, 0.2), 
        (0.3, 0.2, 0.5), (0.4, 0.4, 0.2), (0.3, 0.5, 0.2), (0.2, 0.3, 0.5), 
        (0.5, 0.2, 0.3), (0.3, 0.3, 0.4), (0.2, 0.4, 0.4), (0.4, 0.2, 0.4)
    ]

    best_sharpe_ratio = -np.inf
    best_params = None
    best_cumulative_returns = None

    #iterate through parameters
    for lookback in lookbacks:
        for holding_period in holding_periods:
            for short_term_weight, long_term_weight, volatility_weight in weights:
                sharpe, cumulative_returns = rotational_momentum(
                    dfP, dfAP, hierarchical_labels, stock_list,
                    lookback=lookback,
                    short_term_weight=short_term_weight,
                    long_term_weight=long_term_weight,
                    volatility_weight=volatility_weight
                )
                if sharpe > best_sharpe_ratio:
                    best_sharpe_ratio = sharpe
                    best_params = (lookback, holding_period, short_term_weight, long_term_weight, volatility_weight)
                    best_cumulative_returns = cumulative_returns

    print(f'Best Sharpe Ratio: {best_sharpe_ratio}')
    print(f'Best Parameters: Lookback={best_params[0]}, Holding Period={best_params[1]}, Weights={best_params[2:]}')

    #plot cumulative returns for best configuration
    plt.figure()
    best_cumulative_returns.plot()
    plt.title('Cumulative Returns for Best Configuration')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.show()