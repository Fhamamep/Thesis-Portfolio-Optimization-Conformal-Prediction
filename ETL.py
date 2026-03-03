
# Extract, transform, load.



import yfinance as yf
import pandas as pd


# EXTRACT

def get_dow_reverse_history():
    # Final List (Current State for 2026/2025)
    # This reflects the November 8, 2024 changes (NVDA & SHW added)
    current_dow = [
        "MMM", "AXP", "AMGN", "AMZN", "AAPL", "BA", "CAT", "CVX", "CSCO", "KO",
        "DIS", "GS", "HD", "HON", "IBM", "JNJ", "JPM", "MCD", "MRK", "MSFT",
        "NKE", "NVDA", "PG", "CRM", "SHW", "TRV", "UNH", "VZ", "V", "WMT"
    ]

    history = {}
    
    # We work backward from 2026
    # To go back in time, 'reverse_add' means we remove it, 
    # and 'reverse_remove' means we put the old company back.
    backwards_steps = {
        2024: {"reverse_add": ["NVDA", "SHW", "AMZN"], "reverse_remove": ["INTC", "DOW", "WBA"]},
        2020: {"reverse_add": ["AMGN", "HON", "CRM"], "reverse_remove": ["XOM", "PFE", "RTX"]},
        2019: {"reverse_add": ["DOW"], "reverse_remove": ["DowDuPont"]},
        2018: {"reverse_add": ["WBA"], "reverse_remove": ["GE"]},
        2017: {"reverse_add": ["DowDuPont"], "reverse_remove": ["DD"]},
        2015: {"reverse_add": ["AAPL"], "reverse_remove": ["T"]},
        2013: {"reverse_add": ["GS", "NKE", "V"], "reverse_remove": ["AA", "BAC", "HPQ"]},
        2012: {"reverse_add": ["UNH"], "reverse_remove": ["KFT"]},
        2009: {"reverse_add": ["TRV", "CSCO"], "reverse_remove": ["GM", "C"]},
        2008: {"reverse_add": ["KFT", "CVX", "BAC"], "reverse_remove": ["AIG", "MO", "HON"]},
        2004: {"reverse_add": ["AIG", "PFE", "VZ"], "reverse_remove": ["T", "EK", "IP"]},
    }

    temp_list = list(current_dow)
    
    for year in range(2026, 1999, -1):
        if year in backwards_steps:
            # Reversing the change: remove what was added, re-add what was removed
            for ticker in backwards_steps[year]["reverse_add"]:
                if ticker in temp_list: temp_list.remove(ticker)
            for ticker in backwards_steps[year]["reverse_remove"]:
                temp_list.append(ticker)
        
        history[year] = sorted(list(temp_list))
        
    return history

# Generate data
dow_history = get_dow_reverse_history()


#  for example, accessing the year 2005
dow_history_2005= dow_history[2005]
print(f"Dow 30 in 2005: {dow_history_2005}")



# Download DOW data for 10 years
tickers= dow_history_2005
dow = yf.download(tickers, period="10y")

print(f"DOW Data from 2005 tickets:{dow.head()}")



# TRANSFORM





#  LOAD
