from matplotlib import pyplot as plt, ticker as m_ticker, dates as m_dates
from mplfinance.original_flavor import candlestick_ohlc
from datetime import datetime, date, timedelta
import pandas_datareader.data as web
import concurrent.futures
from tqdm import tqdm
import pandas as pd
import numpy as np
import indicators
import stocklist
import logging
import time
import os


class Logger:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s:%(levelname)s:%(name)s:%(funcName)s:%(message)s")
        file_handler = logging.FileHandler("stock_error.log")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def log_error(self, err_msg):
        self.logger.warning(err_msg)


class Stock:

    obx_stocks = []
    stock_folder = '/Stockmarked/'
    start_limit = str(date.today() - timedelta(days=365))
    stock_data = pd.DataFrame([[0, 0, 0, 0, 0, 0, 0]], columns=[
        'Stock', 'Price', 'RSI', 'MACD', 'abs(MACD - EMA9)', 'MACD norm', 'RSI mean change'])
    logger = Logger()

    if not os.path.exists(os.getcwd() + stock_folder):
        os.makedirs(os.getcwd() + stock_folder)

    @staticmethod
    def pull_stocks(stock):
        try:
            data = web.DataReader(
                name=stock, data_source='yahoo', start=Stock.start_limit)
            data.sort_index(inplace=True)

        except Exception as e:
            Stock.logger.log_error(
                'Could not pull stock {}. Error {} \n'.format(stock, e))
            data = pd.DataFrame([0], columns=['empty'])
        return data

    @staticmethod
    def save_stocks_to_file(data, stock):
        file_name = os.getcwd() + Stock.stock_folder + stock + '.csv'
        data.to_csv(file_name)

    @staticmethod
    def pull_and_save_stocks(stock):
        data = Stock.pull_stocks(stock)
        Stock.save_stocks_to_file(data, stock)

    @staticmethod
    def get_rsi_mean(stock_data_raw, i):
        n_days = 5
        rsi = indicators.rsi_func(stock_data_raw.iloc[1:i, 1])
        rsi_mean_change = np.sum(np.diff(rsi[-n_days:-1]))/n_days
        rsi_mean_change = rsi[-1] - rsi[-2]
        return '{:.6f}'.format(rsi_mean_change)

    @staticmethod
    def get_macd_norm(stock_data_raw, i):
        _, _, macd = indicators.macd_calc(stock_data_raw.iloc[1:i, 1])
        return macd[-1] / min(macd) if macd[-1] >= 0 else macd[-1] / min(macd)

    @staticmethod
    def browse_stocks(stocks):
        for stock in tqdm(stocks):
            try:
                file_name = os.getcwd() + Stock.stock_folder + stock + '.csv'
                stock_data_raw = pd.read_csv(file_name)
                for i in range(len(stock_data_raw.iloc[:, 1])):
                    try:
                        macd_norm = Stock.get_macd_norm(stock_data_raw, i)
                        rsi_mean_change = Stock.get_rsi_mean(stock_data_raw, i)
                    except Exception as e:
                        pass
                price = stock_data_raw.iloc[-1, 1]
                temp_pd = pd.DataFrame([[stock, price, macd_norm, rsi_mean_change]],
                                       columns=['Stock', 'Price', 'MACD norm', 'RSI mean change'])
                Stock.stock_data = Stock.stock_data.append(temp_pd)

            except Exception as e:
                Stock.logger.log_error(
                    'Could not read stock file {} with error {}'.format(stock, e))

        Stock.stock_data.to_csv(os.getcwd() + Stock.stock_folder +
                                'stock_data_test' + '.csv')


class Plotter():
    start_lim = m_dates.date2num(datetime.strptime(
        str(date.today() - timedelta(days=365)), '%Y-%m-%d'))
    end_lim = m_dates.date2num(datetime.strptime(
        str(datetime.now().strftime('%Y-%m-%d')), '%Y-%m-%d'))
    m_stock = Stock()

    @staticmethod
    def get_mov_avg(close_price):
        mov_avg_20 = indicators.moving_average(close_price, window=20)
        mov_avg_60 = indicators.moving_average(close_price, window=60)
        mov_avg_100 = indicators.moving_average(close_price, window=100)
        return mov_avg_20, mov_avg_60, mov_avg_100

    @staticmethod
    def set_ax_properties(ax, ax_rsi=None):
        ax.tick_params(axis='both', colors='w')
        for loc in ('bottom', 'top', 'left', 'right'):
            ax.spines[loc].set_color('#5998ff')
        ax.set_facecolor('#07000d')
        if ax_rsi:
            ax_rsi.axhline(70, color='#8f2020', linewidth=0.5)
            ax_rsi.axhline(30, color='#386d13', linewidth=0.5)
            ax_rsi.axhline(50, color='white', linewidth=0.5, linestyle=':')

    @staticmethod
    def graph_rsi(stock, dates, close_price):
        ax_rsi = plt.subplot2grid(
            shape=(7, 1), loc=(0, 0), rowspan=1, colspan=1)
        rsi = indicators.rsi_func(close_price)

        ax_rsi.plot(dates, rsi, '#c1f9f7', linewidth=1)
        ax_rsi.fill_between(dates, rsi, 70, where=(rsi
                                                   >= 70), interpolate=True,  facecolor='#8f2020', edgecolor='#8f2020')
        ax_rsi.fill_between(dates, rsi, 30, where=(rsi
                                                   <= 30), interpolate=True, facecolor='#386d13', edgecolor='#386d13')
        ax_rsi.set_yticks([30, 50, 70])
        ax_rsi.text(0.015, 0.95, 'RSI (14)', va='top',
                    color='w', transform=ax_rsi.transAxes)
        Plotter.set_ax_properties(ax=ax_rsi, ax_rsi=ax_rsi)

        plt.title('{} Stock'.format(stock), color='w')
        plt.ylabel('RSI')

        return ax_rsi

    @staticmethod
    def graph_candlesticks(stock, quotes, dates, close_price, ax_rsi):
        ax_can_sticks = plt.subplot2grid(shape=(7, 1), loc=(
            1, 0), rowspan=4, sharex=ax_rsi, colspan=1)
        mov_avg_20, mov_avg_60, mov_avg_100 = Plotter.get_mov_avg(close_price)
        candlestick_ohlc(ax_can_sticks, quotes, width=0.75,
                         colorup='#53C156', colordown='#ff1717')
        ax_can_sticks.plot(dates[-len(mov_avg_20):], mov_avg_20,
                           '#e1edf9', label='20 SMA', linewidth=1)
        ax_can_sticks.plot(dates[-len(mov_avg_60):], mov_avg_60,
                           '#4ee6fd', label='60 SMA', linewidth=1)
        try:
            ax_can_sticks.plot(dates[-len(mov_avg_100):], mov_avg_100,
                               'red', label='100 SMA', linewidth=1)
        except Exception as e:
            stock.logger.log_error(
                'Not enough stock data to plot 100MA for stock {}'.format(stock))
        plt.setp(ax_can_sticks .get_xticklabels(), visible=False, size=8)
        plt.gca().yaxis.set_major_locator(m_ticker.MaxNLocator(prune='upper'))
        plt.ylabel('Price and Volume')
        plt.ylim(min(close_price[-300:-1])*0.8, max(close_price[-300:-1])*1.1)
        plt.grid()
        plt.legend(loc=9, ncol=2, borderaxespad=0,
                   fancybox=True, prop={'size': 7}, framealpha=0.4)
        Plotter.set_ax_properties(ax_can_sticks)
        return ax_can_sticks

    @staticmethod
    def graph_volume(volume, dates, ax_can_sticks):
        ax_vol = ax_can_sticks.twinx()
        ax_vol.fill_between(dates, 0, volume,
                            facecolor='#00ffe8', alpha=0.5)
        ax_vol.axes.yaxis.set_ticklabels([])
        ax_vol.set_ylim(0, 2*volume.max())

        Plotter.set_ax_properties(ax_vol)

    @staticmethod
    def graph_ppo(dates, ax_can_sticks, close_price):
        ax_ppo = plt.subplot2grid(shape=(7, 1), loc=(
            5, 0), sharex=ax_can_sticks, rowspan=1, colspan=1)
        nema = 9

        emaslow, _, macd = indicators.macd_calc(close_price)
        ema9 = indicators.exp_moving_average(macd, nema)
        ppo = macd/emaslow*100
        ppo_ema9 = ema9/emaslow*100

        ax_ppo.plot(dates, ppo, color='#4ee6fd', linewidth=2)
        ax_ppo.plot(dates, ppo_ema9, color='#e1edf9', linewidth=1)
        ax_ppo.text(0.015, 0.95, 'PPO (12,26,9)', va='top',
                    color='w', transform=ax_ppo.transAxes)
        ax_ppo.fill_between(dates, ppo-ppo_ema9, 0, alpha=0.5,
                            facecolor='#00ffe8', edgecolor='#00ffe8')
        ax_ppo.yaxis.set_major_locator(
            m_ticker.MaxNLocator(nbins=5, prune='upper'))

        Plotter.set_ax_properties(ax_ppo)

    @staticmethod
    def graph_obv(dates, ax_can_sticks, existingData):
        ax_obv = plt.subplot2grid(shape=(7, 1), loc=(
            6, 0), sharex=ax_can_sticks, rowspan=1, colspan=1)

        obv = indicators.on_balance_volume(existingData)

        ax_obv.plot(dates, obv['obv'],
                    color='#4ee6fd', linewidth=1.5)
        ax_obv.plot(dates, obv['obv_ema21'],
                    color='white', linewidth=1)
        ax_obv.text(0.015, 0.95, 'OBV (21)', va='top',
                    color='w', transform=ax_obv.transAxes)

        plt.ylim(min(obv['obv'].iloc[-300:-1]), max(obv['obv'].iloc[-300:-1]))

        ax_obv.yaxis.set_major_locator(
            m_ticker.MaxNLocator(nbins=5, prune='upper'))

        Plotter.set_ax_properties(ax_obv)

    @staticmethod
    def graph_candlestick_volume_show(stock, stock_data):
        close_price = stock_data.loc[:, "Close"]
        volume = stock_data.loc[:, "Volume"]
        dates = stock_data.loc[:, "Date"]
        stock_data["Date"] = m_dates.date2num(dates)
        quotes = [tuple(x) for x in stock_data[[
            'Date', 'Open', 'High', 'Low', 'Close']].values]

        # Expands plottet window, weird
        _, _ = plt.subplots(facecolor='#07000d')

        ax_rsi = Plotter.graph_rsi(stock, dates, close_price)
        ax_can_sticks = Plotter.graph_candlesticks(
            stock, quotes, dates, close_price, ax_rsi)
        Plotter.graph_volume(volume, dates, ax_can_sticks)
        Plotter.graph_ppo(dates, ax_can_sticks, close_price)
        Plotter.graph_obv(dates, ax_can_sticks, stock_data)

        plt.subplots_adjust(hspace=0.0, bottom=0.1,
                            top=0.94, right=0.96, left=0.06)
        plt.xlim(left=Plotter.start_lim, right=Plotter.end_lim)

        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        # plt.gcf().autofmt_xdate()
        plt.show()

    @staticmethod
    def graph_data_show(stocks):
        for stock in stocks:
            try:
                print('Stock {}'.format(stock))
                file_name = os.getcwd() + Plotter.m_stock.stock_folder + stock + '.csv'
                stock_data = pd.read_csv(file_name, parse_dates=[
                    "Date"], date_parser=lambda x: datetime.strptime(x, '%Y-%m-%d'))
                Plotter.graph_candlestick_volume_show(
                    stock, stock_data)
            except Exception as e:
                Plotter.m_stock.logger.log_error(
                    'Could not read stock file {} with error {}'.format(stock, e))

    @staticmethod
    def plot_macd_change(num_stocks):
        pd.options.display.float_format = '{:.5f}'.format
        try:
            file_name = os.getcwd() + Plotter.m_stock.stock_folder + 'stock_data_test.csv'
            stock_data = pd.read_csv(file_name)
        except Exception as e:
            Plotter.m_stock.logger.log_error(
                'Could not read stock file with error {}'.format(e))

        stock_data_macd_ema9 = stock_data.sort_values(
            by='MACD norm', ascending=False)
        print('Generating plot for following stocks sorted after MACD norm')
        print(stock_data_macd_ema9[['Stock', 'MACD norm']].iloc[0:num_stocks])
        Plotter.graph_data_show(
            stock_data_macd_ema9['Stock'].iloc[0:num_stocks])

    @staticmethod
    def plot_RSI_change(num_stocks):
        pd.options.display.float_format = '{:.5f}'.format
        try:
            file_name = os.getcwd() + Plotter.m_stock.stock_folder + 'stock_data_test.csv'
            stock_data = pd.read_csv(file_name)
        except Exception as e:
            Plotter.m_stock.logger.log_error(
                'Could not read stock file with error {}'.format(e))

        stock_data_macd_ema9 = stock_data.sort_values(
            by='RSI mean change', ascending=False)
        print('Generating plot for the following stocks sorted after RSI')
        print(stock_data_macd_ema9[[
            'Stock', 'RSI mean change']].iloc[0:num_stocks])
        Plotter.graph_data_show(
            stock_data_macd_ema9['Stock'].iloc[0:num_stocks])


class UserInput():
    num_stock_to_show = 25
    plotter = Plotter()

    @staticmethod
    def tqdm_parallel_map(executor, fn, stocks):
        futures_list = []
        futures_list += [executor.submit(fn, stock) for stock in stocks]
        for f in tqdm(concurrent.futures.as_completed(futures_list), total=len(futures_list)):
            f.result()

    @staticmethod
    def user_input():
        alternative = input(
            "Valg 1-6: \n 1: MACD norm filter \n 2: RSI change filter \n 3: Pull new stock data \n 4: Plot stocks \n 5: Update stocks \n 6: Exit \n >> ")
        try:
            while int(alternative) >= 1 or int(alternative) <= 6:
                if int(alternative) == 1:
                    UserInput.plotter.plot_macd_change(
                        UserInput.num_stock_to_show)
                elif int(alternative) == 2:
                    UserInput.plotter.plot_RSI_change(
                        UserInput.num_stock_to_show)
                elif int(alternative) == 3:
                    start_time = time.process_time()
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        UserInput.tqdm_parallel_map(
                            executor, UserInput.plotter.m_stock.pull_and_save_stocks, stocklist.get_stocks_from_file())

                    intermediate_time = round(
                        time.process_time() - start_time, 1)
                    intermedate_start_time = time.process_time()

                    UserInput.plotter.m_stock.browse_stocks(
                        stocklist.get_stocks_from_file())

                    int_time = round(time.process_time() -
                                     intermedate_start_time, 1)
                    execution_time = round(time.process_time() - start_time, 1)
                    print(
                        f"Pulling and saving took {intermediate_time} seconds")
                    print(f"Store and browse took {int_time} seconds")
                    print(f"Total execution time: {execution_time} seconds")
                elif int(alternative) == 4:
                    stocks_own = ['KOA.OL', 'SHLF.OL', 'ENTRA.OL', 'EQNR.OL', 'PEN.OL', 'DNB.OL',
                                  'NHY.OL', 'PHO.OL', 'FRO.OL', 'HUNT.OL', 'AKERBP.OL', 'AKSO.OL',
                                  'B2H.OL', 'ODL.OL', 'KID.OL', 'KAHOOT-ME.OL']
                    stocks_watch = []

                    UserInput.plotter.graph_data_show(stocks_own)
                    UserInput.plotter.graph_data_show(stocks_watch)
                elif int(alternative) == 5:
                    stocklist.dump_to_file()

                elif int(alternative) == 6:
                    exit()

                alternative = input(
                    "Valg 1-6: \n 1: MACD norm filter \n 2: RSI change filter \n 3: Pull new stock data \n 4: Plot stocks \n 5: Update stocks \n 6: Exit \n >> ")

        except Exception as e:
            print(f"{alternative} is not valid input, program failes with error {e}")


def main():
    user = UserInput()
    user.user_input()


if __name__ == '__main__':
    main()
