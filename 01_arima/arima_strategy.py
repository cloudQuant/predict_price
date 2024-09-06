import datetime
import pandas as pd
import backtrader as bt  # backtrader
from backtrader.comminfo import ComminfoFuturesPercent, ComminfoFuturesFixed  # 期货交易的手续费用，按照比例或者按照金额

# from backtrader.plot.plot import run_cerebro_and_plot  # 个人编写，非backtrader自带
import pyfolio as pf

### 编写相应的策略,每个策略逻辑需要单独编写，回测和实盘直接运行策略类就行

class MACDStrategy(bt.Strategy):
    # 策略作者
    author = 'yunjinqi'
    # 策略的参数
    params = (("period_me1", 10),
              ("period_me2", 20),
              ("period_dif", 9),

              )

    # log相应的信息
    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or bt.num2date(self.datas[0].datetime[0])
        print('{}, {}'.format(dt.isoformat(), txt))

    # 初始化策略的数据
    def __init__(self):
        # 基本上常用的部分属性变量
        self.bar_num = 0  # next运行了多少个bar
        self.current_date = None  # 当前交易日
        # 计算macd指标
        self.ema_1 = bt.indicators.ExponentialMovingAverage(self.datas[0].close, period=self.p.period_me1)
        self.ema_2 = bt.indicators.ExponentialMovingAverage(self.datas[0].close, period=self.p.period_me2)
        self.dif = self.ema_1 - self.ema_2
        self.dea = bt.indicators.ExponentialMovingAverage(self.dif, period=self.p.period_dif)
        self.macd = (self.dif - self.dea) * 2
        # 保存现在持仓的合约是哪一个
        self.holding_contract_name = None

    def prenext(self):
        # 由于期货数据有几千个，每个期货交易日期不同，并不会自然进入next
        # 需要在每个prenext中调用next函数进行运行
        self.next()
        # pass

    # 在next中添加相应的策略逻辑
    def next(self):
        # 每次运行一次，bar_num自然加1,并更新交易日
        self.current_date = bt.num2date(self.datas[0].datetime[0])
        self.bar_num += 1
        # self.log(f"{self.bar_num},{self.datas[0]._name},{self.broker.getvalue()}")
        # self.log(f"{self.ema_1[0]},{self.ema_2[0]},{self.dif[0]},{self.dea[0]},{self.macd[0]}")
        data = self.datas[0]
        self.log(f"close = {data.close[0]}")

    def notify_order(self, order):

        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Rejected:
            self.log(f"Rejected : order_ref:{order.ref}  data_name:{order.p.data._name}")

        if order.status == order.Margin:
            self.log(f"Margin : order_ref:{order.ref}  data_name:{order.p.data._name}")

        if order.status == order.Cancelled:
            self.log(f"Concelled : order_ref:{order.ref}  data_name:{order.p.data._name}")

        if order.status == order.Partial:
            self.log(f"Partial : order_ref:{order.ref}  data_name:{order.p.data._name}")

        if order.status == order.Completed:
            if order.isbuy():
                self.log(
                    f" BUY : data_name:{order.p.data._name} price : {order.executed.price} , cost : {order.executed.value} , commission : {order.executed.comm}")

            else:  # Sell
                self.log(
                    f" SELL : data_name:{order.p.data._name} price : {order.executed.price} , cost : {order.executed.value} , commission : {order.executed.comm}")

    def notify_trade(self, trade):
        # 一个trade结束的时候输出信息
        if trade.isclosed:
            self.log('closed symbol is : {} , total_profit : {} , net_profit : {}'.format(
                trade.getdataname(), trade.pnl, trade.pnlcomm))
            # self.trade_list.append([self.datas[0].datetime.date(0),trade.getdataname(),trade.pnl,trade.pnlcomm])

        if trade.isopen:
            self.log('open symbol is : {} , price : {} '.format(
                trade.getdataname(), trade.price))

    def stop(self):
        # 策略停止的时候输出信息
        pass

    # 准备配置策略


cerebro = bt.Cerebro()
# 参数设置
data_kwargs = dict(
    fromdate=datetime.datetime(2010, 1, 1),
    todate=datetime.datetime(2020, 12, 31),
    timeframe=bt.TimeFrame.Minutes,
    compression=1,
    dtformat=('%Y-%m-%d %H:%M:%S'),  # 日期和时间格式
    tmformat=('%H:%M:%S'),  # 时间格式
    datetime=0,
    high=3,
    low=4,
    open=1,
    close=2,
    volume=5,
    openinterest=6)

data = pd.read_csv("./中金所期货合约数据.csv", index_col=0)
data = data[data['variety'] == "T"]
data['datetime'] = pd.to_datetime(data['date'], format="%Y%m%d")
data = data.dropna()
# print(data)
# 根据持仓量加权合成指数合约
result = []
for index, df in data.groupby("datetime"):
    # print(df)
    # print(df.columns)
    total_open_interest = df['open_interest'].sum()
    open = (df['open']*df['open_interest']).sum()/total_open_interest
    high = (df['high'] * df['open_interest']).sum() / total_open_interest
    low = (df['low'] * df['open_interest']).sum() / total_open_interest
    close = (df['close'] * df['open_interest']).sum() / total_open_interest
    volume = (df['volume'] * df['open_interest']).sum() / total_open_interest
    open_interest = df['open_interest'].mean()
    result.append([index, open, high, low, close, volume, open_interest])
index_df = pd.DataFrame(result, columns=['datetime', 'open', 'high', 'low', 'close', 'volume', 'openinterest'])
# 加载指数合约
index_df.index = pd.to_datetime(index_df['datetime'])
index_df = index_df.drop(["datetime"],axis=1)
feed = bt.feeds.PandasDirectData(dataname=index_df)
cerebro.adddata(feed, name='index')
# print(index_df)
# 设置合约的交易信息，佣金设置为2%%，保证金率为10%，杠杆按照真实的杠杆来
comm = ComminfoFuturesPercent(commission=0.0002, margin=0.1, mult=10)
cerebro.broker.addcommissioninfo(comm, name="index")
# 加载具体合约数据
for symbol, df in data.groupby("symbol"):
    df.index = pd.to_datetime(df['datetime'])
    df = df[['open', 'high', 'low', 'close', 'volume', 'open_interest']]
    df.columns = ['open', 'high', 'low', 'close', 'volume', 'openinterest']
    feed = bt.feeds.PandasDirectData(dataname=df)
    cerebro.adddata(feed, name=symbol)
    # 设置合约的交易信息，佣金设置为2%%，保证金率为10%，杠杆按照真实的杠杆来
    comm = ComminfoFuturesPercent(commission=0.0002, margin=0.1, mult=10)
    cerebro.broker.addcommissioninfo(comm, name=symbol)
# 添加策略
cerebro.addstrategy(MACDStrategy)
cerebro.broker.setcash(1000000)
cerebro.addanalyzer(bt.analyzers.TotalValue, _name='_TotalValue')
cerebro.addanalyzer(bt.analyzers.PyFolio)
# 运行回测
results = cerebro.run()

pyfoliozer = results[0].analyzers.getbyname('pyfolio')
returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()
pf.create_full_tear_sheet(
    returns,
    positions=positions,
    transactions=transactions,
    # gross_lev=gross_lev,
    live_start_date='2019-01-01',
)

