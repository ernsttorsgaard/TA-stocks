import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from mpl_finance import candlestick_ohlc
import math
import os
from tqdm import tqdm
from indicators import *
import concurrent.futures
import time

stocksToPull = (['VOW.OL', 'FIVEPG.OL', 'ASC.OL', 'AFG.OL', 'AKER.OL', 'AKERBP.OL', 'AKSO.OL', 'ARCHER.OL', 'ARCUS.OL',
                 'ASETEK.OL', 'ATEA.OL',  'AUSS.OL', 'AVANCE.OL', 'AWDR.OL', 'AXA.OL', 'B2H.OL', 'BAKKA.OL', 'BGBIO.OL',
                 'BIOTEC.OL', 'BON.OL', 'BDRILL.OL', 'BRG.OL', 'BOUVET.OL', 'BWLPG.OL', 'BWO.OL', 'COV.OL', 'CRAYON.OL', 'DNB.OL',
                 'DNO.OL', 'DOF.OL', 'EAM.OL', 'EIOF.OL', 'EMGS.OL', 'ELE.OL', 'ELK.OL', 'ENTRA.OL', 'EQNR.OL', 'EPR.OL', 'EVRY.OL', 'FJORD.OL',
                 'FKRAFT.OL', 'FLNG.OL', 'FRO.OL', 'FUNCOM.OL', 'GJF.OL', 'GOGL.OL', 'GOD.OL', 'GSF.OL', 'HYARD.OL', 'HELG.OL', 'HEX.OL', 'HIDDN.OL',
                 'HBC.OL', 'HUNT.OL', 'IDEX.OL', 'IOX.OL', 'ITE.OL', 'JAEREN.OL', 'KID.OL', 'KIT.OL', 'KOMP.OL', 'KOA.OL', 'KOG.OL',
                 'KVAER.OL', 'LSG.OL', 'MSEIS.OL', 'MEDI.OL',  'MOWI.OL', 'MPCC.OL', 'MULTI.OL', 'NAPA.OL', 'NAVA.OL', 'NEL.OL', 'NEXT.OL', 'NORBIT.OL',
                 'NOM.OL', 'NANO.OL', 'NOD.OL', 'NHY.OL', 'NORTH.OL', 'NODL.OL', 'NRS.OL', 'NAS.OL', 'NPRO.OL', 'NRC.OL', 'OCY.OL', 'OTS.OL', 'ODL.OL',
                 'ODFB.OL', 'OET.OL', 'OLT.OL', 'ORK.OL', 'OTELLO.OL', 'PARB.OL', 'PCIB.OL', 'PEN.OL', 'PGS.OL', 'PHLY.OL', 'PHO.OL', 'PLCS.OL',
                 'PLT.OL', 'PRS.OL', 'PROTCT.OL', 'QEC.OL', 'RAKP.OL', 'REC.OL', 'SDSD.OL', 'SALM.OL', 'SALMON.OL', 'SADG.OL', 'SAS-NOK.OL',
                 'SBANK.OL', 'SSHIP.OL', 'SSO.OL', 'SCHA.OL', 'SCHB.OL', 'SBX.OL', 'SDRL.OL', 'SSG.OL', 'SBO.OL', 'SHLF.OL', 'SKUE.OL', 'SOLON.OL',
                 'SOFF.OL', 'SBVG.OL', 'NONG.OL', 'MING.OL', 'SRBANK.OL', 'SOAG.OL', 'SPOL.OL', 'MORG.OL', 'SOR.OL', 'SVEG.OL', 'SPOG.OL',
                 'SBLK.OL', 'SNI.OL', 'STB.OL', 'STRONG.OL', 'SUBC.OL', 'TRVX.OL', 'TEL.OL', 'TGS.OL', 'SSC.OL', 'THIN.OL', 'TOM.OL',
                 'TOTG.OL', 'TRE.OL', 'VEI.OL', 'VISTIN.OL', 'WALWIL.OL', 'WWI.OL', 'XXL.OL', 'YAR.OL', 'ZAL.OL'])

# list of stocks
nslow = 26
nfast = 12
nema = 9
n_days = 5
MA1 = 20
MA2 = 60
MA3 = 100
rsi_mean_change_num_days = 5
rsi_mean_change_threshold_lower = 15
rsi_mean_change_threshold_upper = 30
rsi_buy_trigger = 70
folder = '/Stockmarked/'

if not os.path.exists(os.getcwd() + folder):
    os.makedirs(os.getcwd() + folder)


def pullData(stock):
    start = str(datetime.now().year - 1) + '-' + \
        str(datetime.now().month) + '-' + str(datetime.now().day)
    try:
        #print('Currently pulling stock {} at time {} \n'.format(stock, str(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))))
        #print(str(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
        data = web.DataReader(name=stock, data_source='yahoo', start=start)
        data.sort_index(inplace=True)
        #data.index = data.index.to_datetime()

    except Exception as e:
        print('Could not pull stock {}. Error {} \n'.format(stock, e))
        data = pd.DataFrame([0], columns=['empty'])
    return data


def saveToFile(data, stock):
    file_name = os.getcwd() + folder + stock + '.csv'
    data.to_csv(file_name)


def pull_save_stocks(stock):
    print(f'Pulling and saving stock {stock}')
    # for stock in tqdm(stocksTopull):
    data = pullData(stock)
    saveToFile(data, stock)


def graph_candlestick_volume_show(stock, existingData, MA1, MA2, start_lim, end_lim):

    dates_string = existingData.iloc[:, 0]
    dates = [datetime.strptime(d, '%Y-%m-%d') for d in dates_string]
    xs = matplotlib.dates.date2num(dates)

    openp = existingData.iloc[:, 1]
    highp = existingData.iloc[:, 2]
    lowp = existingData.iloc[:, 3]
    closep = existingData.iloc[:, 4]
    volume = existingData.iloc[:, 5]

    hfmt = matplotlib.dates.DateFormatter('%d-%m-%Y')

    existingData['Date'] = xs
    quotes = [tuple(x) for x in existingData[[
        'Date', 'Open', 'High', 'Low', 'Close']].values]
    volume = existingData.iloc[:, 5]

    Av1 = moving_average(closep, MA1)
    Av2 = moving_average(closep, MA2)
    Av3 = moving_average(closep, MA3)

    SP = len(dates[MA3 - 1:])

    label1 = str(MA1) + ' SMA'
    label2 = str(MA2) + ' SMA'
    label3 = str(MA3) + ' SMA'
    rsiCol = '#c1f9f7'
    posCol = '#386d13'
    negCol = '#8f2020'

    fig, ax = plt.subplots(facecolor='#07000d')

    ax0 = plt.subplot2grid((7, 4), (0, 0), rowspan=1, colspan=4)

    rsi = rsi_func(closep)
    ax0.plot(dates[-SP:], rsi[-SP:], rsiCol, linewidth=1)
    ax0.axhline(70, color=negCol, linewidth=0.5)
    ax0.axhline(30, color=posCol, linewidth=0.5)
    ax0.axhline(50, color='white', linewidth=0.5, linestyle=':')
    ax0.fill_between(dates[-SP:], rsi[-SP:], 70, where=(rsi[-SP:]
                                                        >= 70), facecolor=negCol, edgecolor=negCol)
    ax0.fill_between(dates[-SP:], rsi[-SP:], 30, where=(rsi[-SP:]
                                                        <= 30), facecolor=posCol, edgecolor=posCol)
    ax0.set_yticks([30, 50, 70])
    ax0.set_facecolor('#07000d')
    ax0.yaxis.label.set_color('w')
    ax0.spines['bottom'].set_color('#5998ff')
    ax0.spines['top'].set_color('#5998ff')
    ax0.spines['left'].set_color('#5998ff')
    ax0.spines['right'].set_color('#5998ff')
    ax0.text(0.015, 0.95, 'RSI (14)', va='top',
             color='w', transform=ax0.transAxes)
    ax0.tick_params(axis='y', colors='w')
    ax0.tick_params(axis='x', colors='w')
    plt.setp(ax0.get_xticklabels(), visible=False, size=8)
    plt.title('{} Stock'.format(stock), color='w')
    plt.ylabel('RSI')

    start_lim = datetime.strptime(start_lim, '%Y-%m-%d')
    start_lim = matplotlib.dates.date2num(start_lim)
    end_lim = datetime.strptime(end_lim, '%Y-%m-%d')
    end_lim = matplotlib.dates.date2num(end_lim)
    ax = plt.subplot2grid((7, 4), (1, 0), rowspan=4, sharex=ax0, colspan=4)
    ax.set_facecolor('#07000d')
    plt.xlim(left=start_lim, right=end_lim)
    ylim_low = min(closep[-300:-1])*0.8
    ylim_high = max(closep[-300:-1])*1.1
    plt.ylim(ylim_low, ylim_high)
    ax.yaxis.label.set_color('w')
    ax.spines['bottom'].set_color('#5998ff')
    ax.spines['top'].set_color('#5998ff')
    ax.spines['left'].set_color('#5998ff')
    ax.spines['right'].set_color('#5998ff')
    ax.tick_params(axis='y', colors='w')
    ax.tick_params(axis='x', colors='w')
    candlestick_ohlc(ax, quotes, width=0.75,
                     colorup='#53C156', colordown='#ff1717')
    ax.plot(dates[-len(Av1):], Av1, '#e1edf9', label=label1, linewidth=1)
    ax.plot(dates[-len(Av2):], Av2, '#4ee6fd', label=label2, linewidth=1)
    try:
        ax.plot(dates[-len(Av3):], Av3, 'red', label=label3, linewidth=1)
    except Exception as e:
        print('Not enough stock data to plot 100MA for stock {}'.format(stock))
    plt.setp(ax .get_xticklabels(), visible=False, size=8)
    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
    plt.xlabel('Date')
    plt.ylabel('Price and Volume')
    plt.grid()
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    maLeg = plt.legend(loc=9, ncol=2, borderaxespad=0,
                       fancybox=True, prop={'size': 7})
    maLeg.get_frame().set_alpha(0.4)

    volumeMin = 0
    ax1v = ax.twinx()
    ax1v.fill_between(dates, volumeMin, volume, facecolor='#00ffe8', alpha=0.5)
    ax1v.axes.yaxis.set_ticklabels([])
    ax1v.set_ylim(0, 2*volume.max())
    ax1v.spines['bottom'].set_color('#5998ff')
    ax1v.spines['top'].set_color('#5998ff')
    ax1v.spines['left'].set_color('#5998ff')
    ax1v.spines['right'].set_color('#5998ff')
    ax1v.tick_params(axis='x', colors='w')
    ax1v.tick_params(axis='y', colors='w')

    ax2 = plt.subplot2grid((7, 4), (5, 0), sharex=ax, rowspan=1, colspan=4)
    fill_col = '#00ffe8'
    nslow = 26
    nfast = 12
    nema = 9

    emaslow, emafast, macd = macd_calc(closep)
    ema9 = exp_moving_average(macd, nema)
    ppo = (emafast - emaslow)/emaslow*100
    ppo_ema9 = (ema9)/emaslow*100

    ax2.plot(dates, ppo, color='#4ee6fd', linewidth=2)
    ax2.plot(dates, ppo_ema9, color='#e1edf9', linewidth=1)
    ax2.text(0.015, 0.95, 'PPO (12,26,9)', va='top',
             color='w', transform=ax2.transAxes)
    ax2.fill_between(dates, ppo-ppo_ema9, 0, alpha=0.5,
                     facecolor=fill_col, edgecolor=fill_col)
    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
    ax2.set_facecolor('#07000d')
    plt.xlim(left=start_lim, right=end_lim)
    ax2.spines['bottom'].set_color('#5998ff')
    ax2.spines['top'].set_color('#5998ff')
    ax2.spines['left'].set_color('#5998ff')
    ax2.spines['right'].set_color('#5998ff')
    ax2.tick_params(axis='x', colors='w')
    ax2.tick_params(axis='y', colors='w')
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune='upper'))

    ax3 = plt.subplot2grid((7, 4), (6, 0), sharex=ax, rowspan=1, colspan=4)
    fill_col = '#00ffe8'
    nslow = 26
    nfast = 12
    nema = 9

    obv = on_balance_volume(existingData)

    ax3.plot(dates[-len(obv):], obv['obv'], color='#4ee6fd', linewidth=1.5)
    ax3.plot(dates[-len(obv):], obv['obv_ema21'], color='white', linewidth=1)
    ax3.text(0.015, 0.95, 'OBV (21)', va='top',
             color='w', transform=ax3.transAxes)
    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
    ax3.set_facecolor('#07000d')
    plt.xlim(left=start_lim, right=end_lim)
    ylim_low = min(obv['obv'].iloc[-300:-1])
    ylim_high = max(obv['obv'].iloc[-300:-1])
    plt.ylim(ylim_low, ylim_high)
    ax3.spines['bottom'].set_color('#5998ff')
    ax3.spines['top'].set_color('#5998ff')
    ax3.spines['left'].set_color('#5998ff')
    ax3.spines['right'].set_color('#5998ff')
    ax3.tick_params(axis='x', colors='w')
    ax3.tick_params(axis='y', colors='w')
    ax3.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune='upper'))
    #plt.ylabel('MACD', color='w')
    #plt.ylabel('MACD', color='w')

    plt.subplots_adjust(hspace=0.0, bottom=0.1,
                        top=0.94, right=0.96, left=0.06)

    plt.show()


def graph_data_norsk_show(stock, existingData, MA1, MA2, start_lim, end_lim):

    dates_string = existingData.iloc[:, 0]
    dates = [datetime.strptime(d, '%Y-%m-%d') for d in dates_string]
    xs = matplotlib.dates.date2num(dates)

    closep = existingData.iloc[:, 1]

    hfmt = matplotlib.dates.DateFormatter('%d-%m-%Y')

    existingData['Date'] = xs

    Av1 = moving_average(closep, MA1)
    Av2 = moving_average(closep, MA2)

    SP = len(dates[MA2 - 1:])

    label1 = str(MA1) + ' SMA'
    label2 = str(MA2) + ' SMA'
    rsiCol = '#c1f9f7'
    posCol = '#386d13'
    negCol = '#8f2020'

    fig, ax = plt.subplots(facecolor='#07000d')

    ax0 = plt.subplot2grid((7, 4), (0, 0), rowspan=1, colspan=4)

    rsi = rsi_func(closep)
    ax0.plot(dates, rsi, rsiCol, linewidth=1)
    ax0.axhline(70, color=negCol, linewidth=0.5)
    ax0.axhline(30, color=posCol, linewidth=0.5)
    ax0.axhline(50, color='white', linewidth=0.5, linestyle=':')
    ax0.fill_between(dates[-SP:], rsi[-SP:], 70, where=(rsi[-SP:]
                                                        >= 70), facecolor=negCol, edgecolor=negCol)
    ax0.fill_between(dates[-SP:], rsi[-SP:], 30, where=(rsi[-SP:]
                                                        <= 30), facecolor=posCol, edgecolor=posCol)
    ax0.set_yticks([30, 50, 70])
    ax0.set_facecolor('#07000d')
    ax0.yaxis.label.set_color('w')
    ax0.spines['bottom'].set_color('#5998ff')
    ax0.spines['top'].set_color('#5998ff')
    ax0.spines['left'].set_color('#5998ff')
    ax0.spines['right'].set_color('#5998ff')
    ax0.text(0.015, 0.95, 'RSI (14)', va='top',
             color='w', transform=ax0.transAxes)
    ax0.tick_params(axis='y', colors='w')
    ax0.tick_params(axis='x', colors='w')
    plt.setp(ax0.get_xticklabels(), visible=False, size=8)
    plt.title('{} Stock'.format(stock), color='w')
    plt.ylabel('RSI')

    start_lim = datetime.strptime(start_lim, '%Y-%m-%d')
    start_lim = matplotlib.dates.date2num(start_lim)
    end_lim = datetime.strptime(end_lim, '%Y-%m-%d')
    end_lim = matplotlib.dates.date2num(end_lim)
    ylim_low = min(closep[-400:-1])*1.1
    ylim_high = max(closep[-400:-1])*1.1
    ax = plt.subplot2grid((7, 4), (1, 0), rowspan=4, sharex=ax0, colspan=4)
    ax.set_facecolor('#07000d')
    plt.xlim(left=start_lim, right=end_lim)
    plt.ylim(ylim_low, ylim_high)
    ax.yaxis.label.set_color('w')
    ax.spines['bottom'].set_color('#5998ff')
    ax.spines['top'].set_color('#5998ff')
    ax.spines['left'].set_color('#5998ff')
    ax.spines['right'].set_color('#5998ff')
    ax.tick_params(axis='y', colors='w')
    ax.tick_params(axis='x', colors='w')
    ax.plot(dates[-SP:], Av1[-SP:], '#e1edf9', label=label1, linewidth=1)
    ax.plot(dates[-SP:], Av2[-SP:], '#4ee6fd', label=label2, linewidth=1)
    ax.plot(dates, closep, '#5998ff', label='_nolegend_', linewidth=1)
    print(ax.get_legend())
    plt.setp(ax .get_xticklabels(), visible=False, size=8)
    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
    plt.xlabel('Date')
    plt.ylabel('Price and Volume')
    plt.grid()

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    maLeg = plt.legend(loc=9, ncol=2, borderaxespad=0,
                       fancybox=True, prop={'size': 7})
    maLeg.get_frame().set_alpha(0.4)

    ax2 = plt.subplot2grid((7, 4), (5, 0), sharex=ax, rowspan=1, colspan=4)
    fill_col = '#00ffe8'
    nslow = 26
    nfast = 12
    nema = 9

    emaslow, emafast, macd = macd_calc(closep)
    ema9 = exp_moving_average(macd, nema)

    ax2.plot(dates, macd, color='#4ee6fd', linewidth=2)
    ax2.plot(dates, ema9, color='#e1edf9', linewidth=1)
    ax2.text(0.015, 0.95, 'MACD (12,26,9)', va='top',
             color='w', transform=ax2.transAxes)
    ax2.fill_between(dates, macd-ema9, 0, alpha=0.5,
                     facecolor=fill_col, edgecolor=fill_col)
    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
    ax2.set_facecolor('#07000d')
    plt.xlim(left=start_lim, right=end_lim)
    ax2.spines['bottom'].set_color('#5998ff')
    ax2.spines['top'].set_color('#5998ff')
    ax2.spines['left'].set_color('#5998ff')
    ax2.spines['right'].set_color('#5998ff')
    ax2.tick_params(axis='x', colors='w')
    ax2.tick_params(axis='y', colors='w')
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune='upper'))

    ax3 = plt.subplot2grid((7, 4), (6, 0), sharex=ax, rowspan=1, colspan=4)
    fill_col = '#00ffe8'
    nslow = 26
    nfast = 12
    nema = 9

    emaslow, emaslow, macd = macd_calc(closep)
    ema9 = exp_moving_average(macd, nema)
    ppo = (emafast - emaslow)/emaslow*100

    ax3.plot(dates, ppo, color='#4ee6fd', linewidth=2)
    ax3.plot(dates, ema9, color='#e1edf9', linewidth=1)
    ax3.text(0.015, 0.95, 'MACD (12,26,9)', va='top',
             color='w', transform=ax2.transAxes)
    ax3.fill_between(dates, ppo-ema9, 0, alpha=0.5,
                     facecolor=fill_col, edgecolor=fill_col)
    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
    ax3.set_facecolor('#07000d')
    plt.xlim(left=start_lim, right=end_lim)
    ax3.spines['bottom'].set_color('#5998ff')
    ax3.spines['top'].set_color('#5998ff')
    ax3.spines['left'].set_color('#5998ff')
    ax3.spines['right'].set_color('#5998ff')
    ax3.tick_params(axis='x', colors='w')
    ax3.tick_params(axis='y', colors='w')
    ax3.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune='upper'))

    #plt.ylabel('MACD', color='w')

    #plt.subplots_adjust(hspace=0.0, bottom=0.1, top=0.94, right=0.96, left=0.06)

    plt.show()


def graph_data_show(stock, MA1, MA2, start_lim, end_lim):
    try:
        file_name = os.getcwd() + folder + stock + '.csv'
        existingData = pd.read_csv(file_name)
    except Exception as e:
        print('Could not read stock file {} with error {}'.format(stock, e))

    if existingData.shape[0] > 4:
        graph_candlestick_volume_show(
            stock, existingData, MA1, MA2, start_lim, end_lim)
    else:
        graph_data_norsk_show(stock, existingData, MA1,
                              MA2, start_lim, end_lim)


def browse_stocks(stocks):
    stock_data = pd.DataFrame([[0, 0, 0, 0, 0, 0, 0]], columns=[
                              'Stock', 'Price', 'RSI', 'MACD', 'abs(MACD - EMA9)', 'MACD norm', 'RSI mean change'])
    #print(f'Calculating stats for stock {stock}')
    for stock in tqdm(stocks):
        try:
            file_name = os.getcwd() + folder + stock + '.csv'
            existingData = pd.read_csv(file_name)
            #print('Calculating stats for stock: {}'.format(stock))
            for i in range(len(existingData.iloc[:, 1])):

                try:
                    rsi = rsi_func(existingData.iloc[1:i, 1])

                    emaslow, emaslow, macd = macd_calc(
                        existingData.iloc[1:i, 1])
                    min_macd = min(macd)
                    max_macd = max(macd)
                    if macd[-1] >= 0:
                        macd_norm = macd[-1] / min_macd
                    elif macd[-1] < 0:
                        macd_norm = (macd[-1] / min_macd)
                    ema9 = exp_moving_average(macd, nema)
                    rsi_mean_change = np.sum(np.diff(rsi[-n_days:-1]))/n_days
                    rsi_mean_change = rsi[-1] - rsi[-2]
                    rsi_mean_change = '{:.6f}'.format(rsi_mean_change)
                    macd_mean_change = np.sum(
                        np.diff(macd[-n_days:-1]))/(n_day)
                    macd_mean_change = '{:.6f}'.format(macd_mean_change)
                except Exception as e:
                    pass
            price = existingData.iloc[-1, 1]
            temp_pd = pd.DataFrame([[stock, price, rsi[-1], macd[-1], abs(macd[-1]-ema9[-1]), macd_norm, rsi_mean_change]],
                                   columns=['Stock', 'Price', 'RSI', 'MACD', 'abs(MACD - EMA9)', 'MACD norm', 'RSI mean change'])
            stock_data = stock_data.append(temp_pd)

        except Exception as e:
            print('Could not read stock file {} with error {}'.format(stock, e))

    return stock_data


def browse_and_store_stats(stocks):
    stock_data = browse_stocks(stocks)
    #stock_data = stock_data.sort_values(by='RSI mean change', ascending = True)
    stock_data.to_csv(os.getcwd() +
                      folder + 'stock_data_test' + '.csv')


def pull_stored_stock_data():
    pd.options.display.float_format = '{:.5f}'.format
    try:
        file_name = os.getcwd() + \
            '/Stockmarked/stock_data_test.csv'
        stock_data = pd.read_csv(file_name)
    except Exception as e:
        print('Could not read stock file {} with error {}'.format(stock, e))

    #stock_data[['RSI', 'MACD', 'EMA9', 'RSI mean change', 'MACD/EMA9']] = stock_data[['RSI', 'MACD', 'EMA9', 'RSI mean change', 'MACD/EMA9']].apply(lambda x: '%.5f' % x)
    print(stock_data['Price'])


def plot_and_show_selected_stocks(stocks, MA1, MA2, start_lim, end_lim):

    for stock in stocks:
        try:
            print('Stock {}'.format(stock))
            graph_data_show(stock, MA1, MA2, start_lim, end_lim)

        except Exception as e:
            print('Could not read stock file {} with error {}'.format(stock, e))


def plot_macd_change(MA1, MA2, start_lim, end_lim, num_stocks):
    pd.options.display.float_format = '{:.5f}'.format
    [os.remove(file) for file in os.listdir(
        os.getcwd() + folder) if file.endswith('_macd_change.png')]
    try:
        file_name = os.getcwd() + \
            '/Stockmarked/stock_data_test.csv'
        stock_data = pd.read_csv(file_name)
    except Exception as e:
        print('Could not read stock file {} with error {}'.format(stock, e))

    stock_data_macd_ema9 = stock_data.sort_values(
        by='MACD norm', ascending=False)
    print(stock_data_macd_ema9[['Stock', 'MACD norm']].iloc[0:num_stocks])
    for stock in stock_data_macd_ema9['Stock'].iloc[0:num_stocks]:
        # print(stock)
        print('Generating plot for stock {} sorted after MACD norm'.format(stock))
        #graph_data_show(stock, MA1, MA2, start_lim, end_lim)
        try:
            graph_data_show(stock, MA1, MA2, start_lim, end_lim)

            #fig.savefig('{}_macd.png'.format(stock), bbox_inches = "tight")
        except:
            pass


def plot_RSI_change(MA1, MA2, start_lim, end_lim, num_stocks):
    pd.options.display.float_format = '{:.5f}'.format
    [os.remove(file) for file in os.listdir(
        os.getcwd() + folder) if file.endswith('_macd_change.png')]
    try:
        file_name = os.getcwd() + \
            '/Stockmarked/stock_data_test.csv'
        stock_data = pd.read_csv(file_name)
    except Exception as e:
        print('Could not read stock file {} with error {}'.format(stock, e))

    stock_data_macd_ema9 = stock_data.sort_values(
        by='RSI mean change', ascending=False)
    print(stock_data_macd_ema9[[
          'Stock', 'RSI mean change']].iloc[0:num_stocks])
    for stock in stock_data_macd_ema9['Stock'].iloc[0:num_stocks]:
        # print(stock)
        print('Generating plot for stock {} sorted after RSI'.format(stock))
        try:
            graph_data_show(stock, MA1, MA2, start_lim, end_lim)

            #fig.savefig('{}_macd.png'.format(stock), bbox_inches = "tight")
        except:
            pass


def main():

    start_lim = str(datetime.now().year - 1) + '-' + \
        str(datetime.now().month) + '-' + str(datetime.now().day)
    end_lim = str(datetime.now().strftime('%Y-%m-%d'))
    num_stock_to_show = 25

    alternative = input(
        "Valg 1-5: \n 1: MACD norm filter \n 2: RSI change filter \n 3: Pull new stock data \n 4: Plot stocks \n 5: Exit \n >> ")
    while int(alternative) < 1 or int(alternative) > 5:
        print("Wrong input, try again: ")
        alternative = input(">> ")
    if int(alternative) == 1:
        plot_macd_change(MA1, MA2, start_lim, end_lim, num_stock_to_show)
    elif int(alternative) == 2:
        plot_RSI_change(MA1, MA2, start_lim, end_lim, num_stock_to_show)
    elif int(alternative) == 3:
        start_time = time.clock()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(pull_save_stocks, stocksToPull)
            browse_and_store_stats(stocksToPull)
        print(f"Execution took: ")
        print(time.clock() - start_time)
    elif int(alternative) == 4:
        stocks_own = ['SHLF.OL', 'ENTRA.OL', 'EQNR.OL', 'PEN.OL', 'DNB.OL',
                      'NHY.OL', 'PHO.OL', 'FRO.OL', 'HUNT.OL', 'AKERBP.OL', 'AKSO.OL', 'B2H.OL', 'ODL.OL', 'KID.OL', 'KAHOOT-ME.OL']
        stocks_watch = []

        plot_and_show_selected_stocks(stocks_own, MA1, MA2, start_lim, end_lim)
        plot_and_show_selected_stocks(
            stocks_watch, MA1, MA2, start_lim, end_lim)
    elif int(alternative) == 5:
        exit()


main()
