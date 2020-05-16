import pandas as pd
import pandas_datareader.data as web
import numpy as np
from datetime import datetime, date, timedelta
from matplotlib import pyplot as plt, ticker as m_ticker, dates as m_dates
from mpl_finance import candlestick_ohlc
import math
import os
from tqdm import tqdm
import indicators
import concurrent.futures
import time
import multiprocessing
import logging


class Logger:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s:%(levelname)s:%(name)s:%(message)s")
        file_handler = logging.FileHandler("stock_error.log")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def log_error(self, err_msg):
        self.logger.warning(err_msg)


class Stock:

    obx_stocks = (['VOW.OL', 'FIVasdadadEPG.OLa', 'ASC.OL', 'AFG.OL', 'AKER.OL', 'AKERBP.OL', 'AKSO.OL', 'ARCHER.OL', 'ARCUS.OL',
                   'ASETEK.OL', 'ATEA.OL',  'AUSS.OL', 'AVANCE.OL', 'AWDR.OL', 'AXA.OL', 'B2H.OL', 'BAKKA.OL', 'BGBIO.OL',
                   'BIOTEC.OL', 'BON.OL', 'BDRILL.OL', 'BRG.OL', 'BOUVET.OL', 'BWLPG.OL', 'BWO.OL', 'COV.OL', 'CRAYON.OL', 'DNB.OL',
                   'DNO.OL', 'DOF.OL', 'EAM.OL', 'EIOF.OL', 'EMGS.OL', 'ELE.OL', 'ELK.OL', 'ENTRA.OL', 'EQNR.OL', 'EPR.OL', 'TIETOO.OL', 'FJORD.OL',
                   'FKRAFT.OL', 'FLNG.OL', 'FRO.OL', 'FUNCOM.OL', 'GJF.OL', 'GOGL.OL', 'GOD.OL', 'GSF.OL', 'HYARD.OL', 'HELG.OL', 'HEX.OL', 'HIDDN.OL',
                   'HBC.OL', 'HUNT.OL', 'IDEX.OL', 'IOX.OL', 'ITE.OL', 'JAEREN.OL', 'KAHOOT-ME.OL', 'KID.OL', 'KIT.OL', 'KOMP.OL', 'KOA.OL', 'KOG.OL',
                   'KVAER.OL', 'LSG.OL', 'MSEIS.OL', 'MEDI.OL',  'MOWI.OL', 'MPCC.OL', 'MULTI.OL', 'NAPA.OL', 'NAVA.OL', 'NEL.OL', 'NEXT.OL', 'NORBIT.OL',
                   'NOM.OL', 'NANO.OL', 'NOD.OL', 'NHY.OL', 'NORTH.OL', 'NODL.OL', 'NRS.OL', 'NAS.OL', 'NPRO.OL', 'NRC.OL', 'OCY.OL', 'OTS.OL', 'ODL.OL',
                   'ODFB.OL', 'OET.OL', 'OLT.OL', 'ORK.OL', 'OTELLO.OL', 'PARB.OL', 'PCIB.OL', 'PEN.OL', 'PGS.OL', 'PHLY.OL', 'PHO.OL', 'PLCS.OL',
                   'PLT.OL', 'PRS.OL', 'PROTCT.OL', 'QEC.OL', 'RAKP.OL', 'REC.OL', 'SDSD.OL', 'SALM.OL', 'SALMON.OL', 'SADG.OL', 'SAS-NOK.OL',
                   'SBANK.OL', 'SSO.OL', 'SCHA.OL', 'SCHB.OL', 'SBX.OL', 'SDRL.OL', 'SSG.OL', 'SBO.OL', 'SHLF.OL', 'SKUE.OL', 'SOLON.OL',
                   'SOFF.OL', 'SBVG.OL', 'NONG.OL', 'MING.OL', 'SRBANK.OL', 'SOAG.OL', 'SPOL.OL', 'MORG.OL', 'SOR.OL', 'SVEG.OL', 'SPOG.OL',
                   'SBLK.OL', 'SNI.OL', 'STB.OL', 'STRONG.OL', 'SUBC.OL', 'TRVX.OL', 'TEL.OL', 'TGS.OL', 'SSC.OL', 'THIN.OL', 'TOM.OL',
                   'TOTG.OL', 'TRE.OL', 'VEI.OL', 'VISTIN.OL', 'WALWIL.OL', 'WWI.OL', 'XXL.OL', 'YAR.OL', 'ZAL.OL'])

    stock_folder = '/Stockmarked/'
    start_limit = str(date.today() - timedelta(days=365))
    stock_counter = 0
    stock_data = pd.DataFrame([[0, 0, 0, 0, 0, 0, 0]], columns=[
        'Stock', 'Price', 'RSI', 'MACD', 'abs(MACD - EMA9)', 'MACD norm', 'RSI mean change'])
    logger = Logger()

    if not os.path.exists(os.getcwd() + stock_folder):
        os.makedirs(os.getcwd() + stock_folder)

    def pull_stocks(self, stock):
        try:
            data = web.DataReader(
                name=stock, data_source='yahoo', start=self.start_limit)
            data.sort_index(inplace=True)

        except Exception as e:
            self.logger.log_error(
                'Could not pull stock {}. Error {} \n'.format(stock, e))
            data = pd.DataFrame([0], columns=['empty'])
        return data

    def save_stocks_to_file(self, data, stock):
        file_name = os.getcwd() + self.stock_folder + stock + '.csv'
        data.to_csv(file_name)

    def pull_and_save_stocks(self, stock):
        self.stock_counter += 1
        print(str(self.stock_counter) + "/" +
              str(len(self.obx_stocks)), end=" ")
        print(f'Pulling and saving stock {stock}')
        data = self.pull_stocks(stock)
        self.save_stocks_to_file(data, stock)

    def browse_stocks(self, stocks):
        n_days, nema = 5, 9
        for stock in tqdm(stocks):
            try:
                file_name = os.getcwd() + self.stock_folder + stock + '.csv'
                existingData = pd.read_csv(file_name)
                for i in range(len(existingData.iloc[:, 1])):
                    try:
                        rsi = indicators.rsi_func(existingData.iloc[1:i, 1])

                        _, _, macd = indicators.macd_calc(
                            existingData.iloc[1:i, 1])
                        min_macd = min(macd)
                        max_macd = max(macd)
                        if macd[-1] >= 0:
                            macd_norm = macd[-1] / min_macd
                        elif macd[-1] < 0:
                            macd_norm = (macd[-1] / min_macd)
                        ema9 = indicators.exp_moving_average(macd, nema)
                        rsi_mean_change = np.sum(
                            np.diff(rsi[-n_days:-1]))/n_days
                        rsi_mean_change = rsi[-1] - rsi[-2]
                        rsi_mean_change = '{:.6f}'.format(rsi_mean_change)
                    except Exception as e:
                        pass
                price = existingData.iloc[-1, 1]
                temp_pd = pd.DataFrame([[stock, price, rsi[-1], macd[-1], abs(macd[-1]-ema9[-1]), macd_norm, rsi_mean_change]],
                                       columns=['Stock', 'Price', 'RSI', 'MACD', 'abs(MACD - EMA9)', 'MACD norm', 'RSI mean change'])
                self.stock_data = self.stock_data.append(temp_pd)
                self.stock_data.to_csv(os.getcwd() + self.stock_folder +
                                       'stock_data_test' + '.csv')

            except Exception as e:
                self.logger.log_error(
                    'Could not read stock file {} with error {}'.format(stock, e))


class Plotter():
    start_lim = m_dates.date2num(datetime.strptime(
        str(date.today() - timedelta(days=365)), '%Y-%m-%d'))
    end_lim = m_dates.date2num(datetime.strptime(
        str(datetime.now().strftime('%Y-%m-%d')), '%Y-%m-%d'))

    def __init__(self):
        self.stock = Stock()
        self.logger = Logger()

    def set_spines(self, ax):
        ax.spines['bottom'].set_color('#5998ff')
        ax.spines['top'].set_color('#5998ff')
        ax.spines['left'].set_color('#5998ff')
        ax.spines['right'].set_color('#5998ff')

    def set_ticks(self, ax):
        ax.tick_params(axis='y', colors='w')
        ax.tick_params(axis='x', colors='w')

    def graph_rsi(self, stock, dates, close_price):
        SP = len(dates)
        rsi = indicators.rsi_func(close_price)
        ax_rsi = plt.subplot2grid(
            shape=(7, 1), loc=(0, 0), rowspan=1, colspan=1)

        ax_rsi.plot(dates[-SP:], rsi[-SP:], '#c1f9f7', linewidth=1)
        ax_rsi.axhline(70, color='#8f2020', linewidth=0.5)
        ax_rsi.axhline(30, color='#386d13', linewidth=0.5)
        ax_rsi.axhline(50, color='white', linewidth=0.5, linestyle=':')
        ax_rsi.fill_between(dates[-SP:], rsi[-SP:], 70, where=(rsi[-SP:]
                                                               >= 70), facecolor='#8f2020', edgecolor='#8f2020')
        ax_rsi.fill_between(dates[-SP:], rsi[-SP:], 30, where=(rsi[-SP:]
                                                               <= 30), facecolor='#386d13', edgecolor='#386d13')
        ax_rsi.set_yticks([30, 50, 70])
        ax_rsi.set_facecolor('#07000d')
        ax_rsi.yaxis.label.set_color('w')
        ax_rsi.text(0.015, 0.95, 'RSI (14)', va='top',
                    color='w', transform=ax_rsi.transAxes)
        self.set_ticks(ax_rsi)

        plt.setp(ax_rsi.get_xticklabels(), visible=False, size=8)
        plt.title('{} Stock'.format(stock), color='w')
        plt.ylabel('RSI')

        self.set_spines(ax_rsi)
        return ax_rsi

    def graph_candlesticks(self, stock, quotes, dates, close_price, ax_rsi):
        mov_avg_20 = indicators.moving_average(close_price, window=20)
        mov_avg_60 = indicators.moving_average(close_price, window=60)
        mov_avg_100 = indicators.moving_average(close_price, window=100)
        ax_can_sticks = plt.subplot2grid(shape=(7, 1), loc=(
            1, 0), rowspan=4, sharex=ax_rsi, colspan=1)
        ax_can_sticks.set_facecolor('#07000d')
        ylim_low = min(close_price[-300:-1])*0.8
        ylim_high = max(close_price[-300:-1])*1.1
        plt.ylim(ylim_low, ylim_high)
        ax_can_sticks.yaxis.label.set_color('w')
        self.set_ticks(ax_can_sticks)
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
            self.logger.log_error(
                'Not enough stock data to plot 100MA for stock {}'.format(stock))
        plt.setp(ax_can_sticks .get_xticklabels(), visible=False, size=8)
        plt.gca().yaxis.set_major_locator(m_ticker.MaxNLocator(prune='upper'))
        plt.xlabel('Date')
        plt.ylabel('Price and Volume')
        plt.grid()
        ax_can_sticks.xaxis_date()
        ax_can_sticks.xaxis.set_major_formatter(
            m_dates.DateFormatter("%Y-%m-%d"))

        maLeg = plt.legend(loc=9, ncol=2, borderaxespad=0,
                           fancybox=True, prop={'size': 7})
        maLeg.get_frame().set_alpha(0.4)
        return ax_can_sticks

    def graph_volume(self, volume, dates, ax_can_sticks):
        ax_vol = ax_can_sticks.twinx()
        ax_vol.fill_between(dates, 0, volume,
                            facecolor='#00ffe8', alpha=0.5)
        ax_vol.axes.yaxis.set_ticklabels([])
        ax_vol.set_ylim(0, 2*volume.max())

        self.set_ticks(ax_vol)
        self.set_spines(ax_vol)

    def graph_ppo(self, dates, ax_can_sticks, close_price):
        ax_ppo = plt.subplot2grid(shape=(7, 1), loc=(
            5, 0), sharex=ax_can_sticks, rowspan=1, colspan=1)
        nema = 9

        emaslow, emafast, macd = indicators.macd_calc(close_price)
        ema9 = indicators.exp_moving_average(macd, nema)
        ppo = (emafast - emaslow)/emaslow*100
        ppo_ema9 = (ema9)/emaslow*100

        ax_ppo.plot(dates, ppo, color='#4ee6fd', linewidth=2)
        ax_ppo.plot(dates, ppo_ema9, color='#e1edf9', linewidth=1)
        ax_ppo.text(0.015, 0.95, 'PPO (12,26,9)', va='top',
                    color='w', transform=ax_ppo.transAxes)
        ax_ppo.fill_between(dates, ppo-ppo_ema9, 0, alpha=0.5,
                            facecolor='#00ffe8', edgecolor='#00ffe8')
        plt.gca().yaxis.set_major_locator(m_ticker.MaxNLocator(prune='upper'))
        ax_ppo.set_facecolor('#07000d')

        ax_ppo.yaxis.set_major_locator(
            m_ticker.MaxNLocator(nbins=5, prune='upper'))

        self.set_ticks(ax_ppo)
        self.set_spines(ax_ppo)

    def graph_obv(self, dates, ax_can_sticks, existingData):
        ax_obv = plt.subplot2grid(shape=(7, 1), loc=(
            6, 0), sharex=ax_can_sticks, rowspan=1, colspan=1)
        obv = indicators.on_balance_volume(existingData)

        ax_obv.plot(dates[-len(obv):], obv['obv'],
                    color='#4ee6fd', linewidth=1.5)
        ax_obv.plot(dates[-len(obv):], obv['obv_ema21'],
                    color='white', linewidth=1)
        ax_obv.text(0.015, 0.95, 'OBV (21)', va='top',
                    color='w', transform=ax_obv.transAxes)
        plt.gca().yaxis.set_major_locator(m_ticker.MaxNLocator(prune='upper'))
        ax_obv.set_facecolor('#07000d')
        ylim_low = min(obv['obv'].iloc[-300:-1])
        ylim_high = max(obv['obv'].iloc[-300:-1])
        plt.ylim(ylim_low, ylim_high)

        ax_obv.yaxis.set_major_locator(
            m_ticker.MaxNLocator(nbins=5, prune='upper'))

        self.set_ticks(ax_obv)
        self.set_spines(ax_obv)

    def graph_candlestick_volume_show(self, stock, existingData):
        dates_string = existingData.iloc[:, 0]
        dates = [datetime.strptime(d, '%Y-%m-%d') for d in dates_string]
        existingData['Date'] = m_dates.date2num(dates)

        close_price = existingData.loc[:, "Close"]
        volume = existingData.loc[:, "Volume"]
        quotes = [tuple(x) for x in existingData[[
            'Date', 'Open', 'High', 'Low', 'Close']].values]

        # Expands plottet window, weird
        _, _ = plt.subplots(facecolor='#07000d')
        # RSI
        ax_rsi = self.graph_rsi(stock, dates, close_price)
        # Candlesticks
        ax_can_sticks = self.graph_candlesticks(
            stock, quotes, dates, close_price, ax_rsi)
        # Volume
        self.graph_volume(volume, dates, ax_can_sticks)
        # PPO
        self.graph_ppo(dates, ax_can_sticks, close_price)
        # OBV
        self.graph_obv(dates, ax_can_sticks, existingData)

        plt.subplots_adjust(hspace=0.0, bottom=0.1,
                            top=0.94, right=0.96, left=0.06)
        plt.xlim(left=self.start_lim, right=self.end_lim)

        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.show()

    def graph_data_show(self, stocks):
        for stock in stocks:
            try:
                print('Stock {}'.format(stock))
                file_name = os.getcwd() + self.stock.stock_folder + stock + '.csv'
                existingData = pd.read_csv(file_name)
            except Exception as e:
                self.logger.log_error(
                    'Could not read stock file {} with error {}'.format(stock, e))

            self.graph_candlestick_volume_show(stock, existingData)

        stock_data = pd.DataFrame([[0, 0, 0, 0, 0, 0, 0]], columns=[
            'Stock', 'Price', 'RSI', 'MACD', 'abs(MACD - EMA9)', 'MACD norm', 'RSI mean change'])

    def plot_macd_change(self, num_stocks):
        pd.options.display.float_format = '{:.5f}'.format
        [os.remove(file) for file in os.listdir(
            os.getcwd() + self.stock.stock_folder) if file.endswith('_macd_change.png')]
        try:
            file_name = os.getcwd() + self.stock.stock_folder + 'stock_data_test.csv'
            stock_data = pd.read_csv(file_name)
        except Exception as e:
            self.logger.log_error(
                'Could not read stock file with error {}'.format(e))

        stock_data_macd_ema9 = stock_data.sort_values(
            by='MACD norm', ascending=False)
        print('Generating plot for following stocks sorted after MACD norm')
        print(stock_data_macd_ema9[['Stock', 'MACD norm']].iloc[0:num_stocks])
        self.graph_data_show(
            stock_data_macd_ema9['Stock'].iloc[0:num_stocks])

    def plot_RSI_change(self, num_stocks):
        pd.options.display.float_format = '{:.5f}'.format
        [os.remove(file) for file in os.listdir(
            os.getcwd() + self.stock.stock_folder) if file.endswith('_macd_change.png')]
        try:
            file_name = os.getcwd() + self.stock.stock_folder + 'stock_data_test.csv'
            stock_data = pd.read_csv(file_name)
        except Exception as e:
            self.logger.log_error(
                'Could not read stock file with error {}'.format(e))

        stock_data_macd_ema9 = stock_data.sort_values(
            by='RSI mean change', ascending=False)
        print('Generating plot for the following stocks sorted after RSI')
        print(stock_data_macd_ema9[[
            'Stock', 'RSI mean change']].iloc[0:num_stocks])
        self.graph_data_show(
            stock_data_macd_ema9['Stock'].iloc[0:num_stocks])


class UserInput():
    num_stock_to_show = 25

    def __init__(self):
        self.plotter = Plotter()
        self.stock = Stock()

    def user_input(self):
        alternative = input(
            "Valg 1-5: \n 1: MACD norm filter \n 2: RSI change filter \n 3: Pull new stock data \n 4: Plot stocks \n 5: Exit \n >> ")
        while int(alternative) < 1 or int(alternative) > 5:
            print("Wrong input, try again: ")
            alternative = input(">> ")
        if int(alternative) == 1:
            self.plotter.plot_macd_change(
                self.num_stock_to_show)
        elif int(alternative) == 2:
            self.plotter.plot_RSI_change(
                self.num_stock_to_show)
        elif int(alternative) == 3:
            start_time = time.process_time()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                executor.map(self.stock.pull_and_save_stocks,
                             self.stock.obx_stocks)
            intermediate_time = round(time.process_time() - start_time, 1)
            intermedate_start_time = time.process_time()
            self.stock.browse_stocks(self.stock.obx_stocks)
            int_time = round(time.process_time() - intermedate_start_time, 1)
            execution_time = round(time.process_time() - start_time, 1)
            print(f"Pulling and saving took {intermediate_time} seconds")
            print(f"Store and browse took {int_time} seconds")
            print(f"Total execution time: {execution_time} seconds")
        elif int(alternative) == 4:
            stocks_own = ['KOA.OL', 'SHLF.OL', 'ENTRA.OL', 'EQNR.OL', 'PEN.OL', 'DNB.OL',
                          'NHY.OL', 'PHO.OL', 'FRO.OL', 'HUNT.OL', 'AKERBP.OL', 'AKSO.OL',
                          'B2H.OL', 'ODL.OL', 'KID.OL', 'KAHOOT-ME.OL']
            stocks_watch = []

            self.plotter.graph_data_show(stocks_own)
            self.plotter.graph_data_show(stocks_watch)
        elif int(alternative) == 5:
            exit()


def main():
    user = UserInput()
    user.user_input()


if __name__ == '__main__':
    main()
