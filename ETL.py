
# Extract, transform, load.



import yfinance as yf
import pandas as pd


# -------EXTRACT----

# Dow 30 composition in 2003
dow_2003 = [
    "MMM", "AA", "MO", "AXP", "T",
    "BA", "CAT", "C", "KO", "DD", 
    "EK", "XOM", "GE", "GM", "HPQ", 
    "HD", "HON", "INTC", "IBM", "IP", 
    "JNJ", "JPM", "MCD", "MRK", "MSFT", 
    "PG", "SBC", "UTX", "WMT", "DIS"
]

def get_dow_forward_history():
    # Historical changes from 2003 to present
    forward_steps = {
        2004: {"remove": ["T", "EK", "IP"], "add": ["AIG", "PFE", "VZ"]},
        2005: {"remove": ["SBC"], "add": ["T"]},
        2008: {"remove": ["AIG", "MO", "HON"], "add": ["KFT", "CVX", "BAC"]},
        2009: {"remove": ["GM", "C"], "add": ["TRV", "CSCO"]},
        2012: {"remove": ["KFT"], "add": ["UNH"]},
        2013: {"remove": ["AA", "BAC", "HPQ"], "add": ["GS", "NKE", "V"]},
        2015: {"remove": ["T"], "add": ["AAPL"]},
        2017: {"remove": ["DD"], "add": ["DWDP"]},
        2018: {"remove": ["GE"], "add": ["WBA"]},
        2019: {"remove": ["DWDP"], "add": ["DOW"]},
        2020: {"remove": ["XOM", "PFE", "UTX"], "add": ["AMGN", "HON", "CRM"]},
        2024: {"remove": ["INTC", "DOW", "WBA"], "add": ["NVDA", "SHW", "AMZN"]},
    }

    history = {}
    temp_list = list(dow_2003)
    history[2003] = sorted(list(temp_list))
    
    for year in range(2004, 2027):
        if year in forward_steps:
            for ticker in forward_steps[year]["remove"]:
                if ticker in temp_list:
                    temp_list.remove(ticker)
            for ticker in forward_steps[year]["add"]:
                temp_list.append(ticker)
        
        history[year] = sorted(list(temp_list))
    
    return history


# Generate data
dow_history = get_dow_forward_history()






# TRANSFORM





#  LOAD
