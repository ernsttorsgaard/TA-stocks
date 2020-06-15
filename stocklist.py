import requests
import json


def get_stock_dict():
    try:
        stock_req = requests.get(
            "https://www.oslobors.no/ob/servlets/components?type=table&generators%5B0%5D%5Bsource%5D=feed.ob.quotes.EQUITIES%2BPCC&generators%5B1%5D%5Bsource%5D=feed.merk.quotes.EQUITIES%2BPCC&filter=&view=DELAYED&columns=PERIOD%2C", timeout=2)
        stock_dict = stock_req.json()
        return stock_dict

    except Exception as e:
        print(f"Requesting stocks failed with an {e}")


def extract_stocks():
    stock_list = []
    stock_dict = get_stock_dict()
    vals = stock_dict['rows']

    for elem in vals:
        _, stock_raw = elem['key'].split("__")
        if stock_raw.endswith("_OSE") or stock_raw.endswith("_OAX"):
            stock = stock_raw[:-len("_OSE")] + ".OL"
        else:
            stock = stock_raw[:-len("_MERK")] + ".OL"

        if "_" in stock:
            stock = stock.replace("_", "-")

        stock_list.append(stock)

    print("Update suceeded!")
    return stock_list


def dump_to_file():
    stocks = extract_stocks()
    with open("Stockmarked/stocks.txt", 'w+') as f:
        json.dump(stocks, f)


def get_stocks_from_file():
    with open("Stockmarked/stocks.txt", 'r') as stock_file:
        with open("Stockmarked/unwanted_stocks.txt", 'r') as unwanted_stock_file:
            stock_full = json.load(stock_file)
            stock_unwanted = json.load(unwanted_stock_file)
            return [stock for stock in stock_full if stock not in stock_unwanted]
