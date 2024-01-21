# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from pandas_datareader import data as pdr
from scipy.stats import norm, t
import seaborn as sns
import scipy.optimize as sc
import plotly.graph_objects as go

pd.set_option('display.max_columns', None)

# Setting up time period 
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days = 180)

def get_stock_data(stocks, start, end):
    stockData = pdr.get_data_moex(stocks, start, end)
    stockData = stockData['CLOSE']
    returns = stockData.pct_change()
    return returns

# Choosing stocks from MOEX (Moscow Stock Exchange)
stock_list = ['SBER', 'GAZP', 'LKOH', 'SBERP', 'GMKN', 'SNGSP', 'ROSN', 'NVTK', 'YNDX', 'PLZL']

portfolio_returns = [get_stock_data(stock, startDate, endDate) for stock in stock_list]

# gathering all returns to a single DataFrame
returns_df = pd.concat([portfolio_returns], keys=["GAZP", "SBER", 'LKOH', 'SBERP', 'GMKN', 'SNGSP', 'ROSN', 'NVTK', 'YNDX', 'PLZL'], axis=1)

# Getting data about covariation of portfolio and mean returns of portfolio
covMatrix = returns_df.cov()
meanReturns = returns_df.mean()

# Setting weights based on popularity of stocks in MOEX amoung investors
weights = np.array([0.043, 0.067, 0.16,  0.0, 0.442, 0.0, 0.171, 0.018,0.049, 0.05])

mc_sims = 400
T  = 180
meanM = np.full(shape=(T, len(weights)), fill_value=meanReturns)
meanM = meanM.T
portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)

# Take average portfolio of Gazinvest investor
initialPortfolio = 1000000

for m in range(0, mc_sims):
    Z = np.random.normal(size=(T, len(weights)))
    L = np.linalg.cholesky(covMatrix)
    dailyReturns = meanM + np.inner(L, Z)
    portfolio_sims[:,m] = np.cumprod(np.inner(weights, dailyReturns.T) + 1) * initialPortfolio

plt.plot(portfolio_sims)
plt.ylabel('Объем портфеля (руб.)')
plt.xlabel('Дни')
plt.title('Симуляции Монте-Карло для портфеля минимального риска')
# plt.show()

# Calculating VaR and CVaR coefficients
def mcVaR(returns, alpha=5):
    if isinstance(returns, pd.Series):
        return np.percentile(returns, alpha)
    else:
        raise TypeError("Expected pandas series ")
    
def mcCVaR(returns, alpha=5):
    if isinstance(returns, pd.Series):
        belowVaR = returns <= mcVaR(returns, alpha=alpha)
        return returns[belowVaR].mean()
    else:
        raise TypeError("Expected pandas series ")

portfolioResults = pd.Series(portfolio_sims[-1, :])

VaR = initialPortfolio - mcVaR(portfolioResults, alpha=5)
CVaR = mcCVaR(portfolioResults, alpha=5)

print('VaR ${}'.format(round(VaR, 2)))
print('CVaR ${}'.format(round(CVaR, 2)))


# CVaR VaR, USING PARAMETRIC AND HISTORICAL METHODS
std = np.sqrt( np.dot(weights.T, np.dot(covMatrix, weights)) ) * np.sqrt(100)
returns_df['portfolio'] = returns_df.dot(weights)
returnsDF = returns_df.dropna()
returnsHVaR = np.sum(meanReturns*weights)*100

def historicalVaR(returns, alpha=5):
    """
    Read in a pandas dataframe of returns / a pandas series of returns
    Output the percentile of the distribution at the given alpha confidence level
    """
    if isinstance(returns, pd.Series):
        return np.percentile(returns, alpha)
    # A passed user-defined-function will be passed a Series for evaluation.
    elif isinstance(returns, pd.DataFrame):
        return returns.aggregate(historicalVaR, alpha=alpha)
    else:
        raise TypeError("Expected returns to be dataframe or series")


def historicalCVaR(returns, alpha=5):
    """
    Read in a pandas dataframe of returns / a pandas series of returns
    Output the CVaR for dataframe / series
    """
    if isinstance(returns, pd.Series):
        belowVaR = returns <= historicalVaR(returns, alpha=alpha)
        return returns[belowVaR].mean()
    # A passed user-defined-function will be passed a Series for evaluation.
    elif isinstance(returns, pd.DataFrame):
        return returns.aggregate(historicalCVaR, alpha=alpha)
    else:
        raise TypeError("Expected returns to be dataframe or series")



# MAIN PROCESS IS HERE
Time = 100
hVaR = -historicalVaR(returnsDF['portfolio'], alpha=5)*np.sqrt(Time)
hCVaR = -historicalCVaR(returnsDF['portfolio'], alpha=5)*np.sqrt(Time)


InitialInvestment = 1000000
print('Expected Portfolio Return:      ', round(InitialInvestment*returnsHVaR,2))
print('Value at Risk 95th CI    :      ', round(InitialInvestment*hVaR,2))
print('Conditional VaR 95th CI  :      ', round(InitialInvestment*hCVaR,2))


def var_parametric(portofolioReturns, portfolioStd, distribution='normal', alpha=5, dof=6):
    # because the distribution is symmetric
    if distribution == 'normal':
        VaR = norm.ppf(1-alpha/100)*portfolioStd - portofolioReturns
    elif distribution == 't-distribution':
        nu = dof
        VaR = np.sqrt((nu-2)/nu) * t.ppf(1-alpha/100, nu) * portfolioStd - portofolioReturns
    else:
        raise TypeError("Expected distribution type 'normal'/'t-distribution'")
    return VaR
def cvar_parametric(portofolioReturns, portfolioStd, distribution='normal', alpha=5, dof=6):
    if distribution == 'normal':
        CVaR = (alpha/100)**-1 * norm.pdf(norm.ppf(alpha/100))*portfolioStd - portofolioReturns
    elif distribution == 't-distribution':
        nu = dof
        xanu = t.ppf(alpha/100, nu)
        CVaR = -1/(alpha/100) * (1-nu)**(-1) * (nu-2+xanu**2) * t.pdf(xanu, nu) * portfolioStd - portofolioReturns
    else:
        raise TypeError("Expected distribution type 'normal'/'t-distribution'")
    return CVaR

normVaR = var_parametric(returnsHVaR, std)
normCVaR = cvar_parametric(returnsHVaR, std)
tVaR = var_parametric(returnsHVaR, std, distribution='t-distribution')
tCVaR = cvar_parametric(returnsHVaR, std, distribution='t-distribution')
print("Normal VaR 95th CI       :      ", round(InitialInvestment*normVaR,2))
print("Normal CVaR 95th CI      :      ", round(InitialInvestment*normCVaR,2))
print("t-dist VaR 95th CI       :      ", round(InitialInvestment*tVaR,2))
print("t-dist CVaR 95th CI      :      ", round(InitialInvestment*tCVaR,2))


def portfolioPerformance(weights, meanReturns, covMatrix):
    returns = np.sum(meanReturns*weights)*124
    std = np.sqrt(
            np.dot(weights.T,np.dot(covMatrix, weights)))*np.sqrt(124)
    return returns, std

def negativeSR(weights, meanReturns, covMatrix, riskFreeRate = 0.75):
    pReturns, pStd = portfolioPerformance(weights, meanReturns, covMatrix)
    return - (pReturns - riskFreeRate)/pStd
riskFreeRate = 0.075
negativeSR = ((returnsHVaR - riskFreeRate)/std) * -1
# negSR = -(returnsHVaR - riskFreeRate)/std
# print(negativeSR, negSR)
def maxSR(meanReturns, covMatrix, riskFreeRate = 0.075, constrainSet = (0,1)):
    "Minimaze the negative SR, by altering the weights of the portfolio"
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, riskFreeRate)
    constraints  = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constrainSet
    bounds = tuple(bound for asset in range(numAssets))
    weightsSharp = np.array([1./numAssets])
    result = sc.minimize(negativeSR, numAssets*weightsSharp,
                         args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

result = maxSR(meanReturns, covMatrix)
maxSR, maxweights = result['fun'], result['x']
print(maxSR, maxweights)

# Calculating Varience of portfolio
def portfolioVariance(weights, meanReturns, covMatrix):
    return portfolioPerformance(weights, meanReturns, covMatrix)[1]


def minimizeVariance(meanReturns, covMatrix, constraintSet=(0,1)):
    """Minimize the portfolio variance by altering the
     weights/allocation of assets in the portfolio"""
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    resultV = sc.minimize(portfolioVariance, numAssets*[1./numAssets], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return resultV

minVarRes = minimizeVariance(meanReturns, covMatrix)
minVar, minVarweights = minVarRes['fun'], minVarRes['x']
print('Minimize portfolio = ',minVar, minVarweights)
minVarsum = np.sum(minVarweights)
print(minVarsum)

# Efficient frontier
def portfolioReturn(weights, meanReturns, covMatrix):
    return portfolioPerformance(weights, meanReturns, covMatrix)[0]


def efficientOpt(meanReturns, covMatrix, returnTarget, constraintSet=(0, 1)):
    """For each returnTarget, we want to optimise the portfolio for min variance"""
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)

    constraints = ({'type': 'eq', 'fun': lambda x: portfolioReturn(x, meanReturns, covMatrix) - returnTarget},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    effOpt = sc.minimize(portfolioVariance, numAssets * [1. / numAssets], args=args, method='SLSQP', bounds=bounds,
                         constraints=constraints)
    return effOpt

def calculatedResults(meanReturns, covMatrix, riskFreeRate = 0.075, constraintSet = (0,1)):
    """Read in mean, cov matrix, and other financial information
        Output, Max SR , Min Volatility, efficient frontier """
    # Max Sharpe Ratio Portfolio
    maxSR_Portfolio = maxSR(meanReturns, covMatrix)
    maxSR_returns, maxSR_std = portfolioPerformance(maxSR_Portfolio['x'], meanReturns, covMatrix)
    maxSR_returns, maxSR_std = round(maxSR_returns * 100, 2), round(maxSR_std * 100, 2)
    maxSR_allocation = pd.DataFrame(maxSR_Portfolio['x'], index=meanReturns.index, columns=['allocation'])
    maxSR_allocation.allocation = [round(i * 100, 0) for i in maxSR_allocation.allocation]

    # Min Volatility Portfolio
    minVol_Portfolio = minimizeVariance(meanReturns, covMatrix)
    minVol_returns, minVol_std = portfolioPerformance(minVol_Portfolio['x'], meanReturns, covMatrix)
    minVol_returns, minVol_std = round(minVol_returns * 100, 2), round(minVol_std * 100, 2)
    minVol_allocation = pd.DataFrame(minVol_Portfolio['x'], index=meanReturns.index, columns=['allocation'])
    minVol_allocation.allocation = [round(i * 100, 0) for i in minVol_allocation.allocation]

    # Efficient Frontier
    efficientList = []
    targetReturns = np.linspace(minVol_returns, maxSR_returns, 20)
    for target in targetReturns:
        efficientList.append(efficientOpt(meanReturns, covMatrix, target)['fun'])

    return maxSR_returns, maxSR_std, maxSR_allocation, minVol_returns, minVol_std, minVol_allocation, efficientList, targetReturns

def EF_graph(meanReturns, covMatrix, riskFreeRate=0, constraintSet=(0, 1)):
    """Return a graph ploting the min vol, max sr and efficient frontier"""
    maxSR_returns, maxSR_std, maxSR_allocation, minVol_returns, minVol_std, minVol_allocation, efficientList, targetReturns = calculatedResults(
        meanReturns, covMatrix, riskFreeRate, constraintSet)

    # Max SR
    MaxSharpeRatio = go.Scatter(
        name='Maximium Sharpe Ratio',
        mode='markers',
        x=[maxSR_std],
        y=[maxSR_returns],
        marker=dict(color='red', size=14, line=dict(width=3, color='black'))
    )

    # Min Vol
    MinVol = go.Scatter(
        name='Mininium Volatility',
        mode='markers',
        x=[minVol_std],
        y=[minVol_returns],
        marker=dict(color='green', size=14, line=dict(width=3, color='black'))
    )

    # Efficient Frontier
    EF_curve = go.Scatter(
        name='Efficient Frontier',
        mode='lines',
        x=[round(ef_std * 100, 2) for ef_std in efficientList],
        y=[round(target * 100, 2) for target in targetReturns],
        line=dict(color='black', width=4, dash='dashdot')
    )

    data = [MaxSharpeRatio, MinVol, EF_curve]

    layout = go.Layout(
        title='Portfolio Optimisation with the Efficient Frontier',
        yaxis=dict(title='Annualised Return (%)'),
        xaxis=dict(title='Annualised Volatility (%)'),
        showlegend=True,
        legend=dict(
            x=0.75, y=0, traceorder='normal',
            bgcolor='#E2E2E2',
            bordercolor='black',
            borderwidth=2),
        width=800,
        height=600)

    fig = go.Figure(data=data, layout=layout)
    return fig.show()


def calculatedResults1(meanReturns, covMatrix, riskFreeRate = 0.075, constraintSet = (0,1)):
    """Read in mean, cov matrix, and other financial information
        Output, Max SR , Min Volatility, efficient frontier """
    # Max Sharpe Ratio Portfolio
    maxSR_Portfolio = maxSR(meanReturns, covMatrix)
    maxSR_returns, maxSR_std = portfolioPerformance(maxSR_Portfolio['x'], meanReturns, covMatrix)
    maxSR_returns, maxSR_std = round(maxSR_returns * 100, 2), round(maxSR_std * 100, 2)
    maxSR_allocation = pd.DataFrame(maxSR_Portfolio['x'], index=meanReturns.index, columns=['allocation'])
    maxSR_allocation.allocation = [round(i * 100, 0) for i in maxSR_allocation.allocation]

    # Min Volatility Portfolio
    minVol_Portfolio = minimizeVariance(meanReturns, covMatrix)
    minVol_returns, minVol_std = portfolioPerformance(minVol_Portfolio['x'], meanReturns, covMatrix)
    minVol_returns, minVol_std = round(minVol_returns * 100, 2), round(minVol_std * 100, 2)
    minVol_allocation = pd.DataFrame(minVol_Portfolio['x'], index=meanReturns.index, columns=['allocation'])
    minVol_allocation.allocation = [round(i * 100, 0) for i in minVol_allocation.allocation]

    # Efficient Frontier
    efficientList = []
    targetReturns = np.linspace(minVol_returns, maxSR_returns, 20)
    for target in targetReturns:
        efficientList.append(efficientOpt(meanReturns, covMatrix, target)['fun'])
    return maxSR_std, maxSR_returns
maxSRstd, maxSRreturns = calculatedResults1(meanReturns, covMatrix)



num_iterations = 10000
simulation_res = np.zeros((4+len(meanReturns)-1,num_iterations))

for i in range(num_iterations):
    # Выбрать случайные веса и нормализовать, чтоб сумма равнялась 1
    weiGhts = np.array(np.random.random(10))
    weiGhts /= np.sum(weiGhts)

    # Вычислить доходность и стандартное отклонение
    portfolio_return = np.sum(meanReturns * weiGhts)
    portfolio_std_dev = np.sqrt(np.dot(weiGhts.T, np.dot(covMatrix, weiGhts)))

    # Сохранить все полученные значения в массив
    simulation_res[0, i] = portfolio_return
    simulation_res[1, i] = portfolio_std_dev

    # Вычислить коэффициент Шарпа и сохранить
    simulation_res[2, i] = simulation_res[0, i] / simulation_res[1, i]

    # Сохранить веса
    for j in range(len(weiGhts)):
        simulation_res[j + 3, i] = weiGhts[j]

# сохраняем полученный массив в DataFrame для построения данных и анализа.
sim_frame = pd.DataFrame(simulation_res.T, columns=['ret', 'stdev', 'sharpe',"GAZP", "SBER", 'LKOH', 'SBERP', 'GMKN', 'SNGSP', 'ROSN', 'NVTK', 'YNDX', 'PLZL'])
max_sharpe = sim_frame.iloc[sim_frame['sharpe'].idxmax()]

# узнать минимальное стандартное отклонение
min_std = sim_frame.iloc[sim_frame['stdev'].idxmin()]

fig, ax = plt.subplots(figsize=(10, 10))

#Создать разноцветный график  scatter plot для различных значений коэффициента Шарпо по оси x и стандартного отклонения по оси y

plt.scatter(sim_frame.stdev,sim_frame.ret,c=sim_frame.sharpe,cmap='inferno')
plt.xlabel('Дисперсия')
plt.ylabel('Доходность')
plt.ylim(0,0.0040)
plt.xlim(0,0.024)
plt.colorbar(label='Коэффициент Шарпа')

plt.scatter(max_sharpe[1],max_sharpe[0],marker=(5,1,0),color='r',s=600)

plt.scatter(min_std[1],min_std[0],marker=(5,1,0),color='b',s=600)

plt.scatter(std, matplotret,marker=(5,1,0),color='g',s=600)

# plt.show()
