### Portfolio Analysis Application

**Disclaimer**: 
This app is only for learning to build and deploy a Streamlit App. Any information or analysis in this app is not an investment advice and should not be relied upon in any investment decisions. Please be also noted that past performance information is given for illustrative purposes only, and it is not an indication of future performance. In addition, many factors (e.g., transaction fees, capital gain tax, investment risks, etc.) have not been taken into account in the portfolio analysis and it will, therefore, not be applicable to an individual's objectives or financial situation.

This is an interactive Streamlit App built in Python with the stock data deriving from Yahoo Finance.

This app can be used for analysis of a selected equity portfolio over the past 5 years. The code of this Streamlit app can be found in the following [port_app.py](https://github.com/DoThNg/portfolio_analysis/blob/main/port_app.py)

![portfolio_app](https://github.com/DoThNg/portfolio_analysis/blob/main/portfolio_app.png)

**Note**: In the sidebar, please drag and drop the csv file with the following format:

        | Ticker          | Allocation    |
        | --------------- | ------------- |
        | CBA.AX          |     0.5       |
        | TLS.AX          |     0.5       |
        
In column **Ticker**, please add stock exchange (e.g., **.AX** for Australian Stock Exchange) after the stock ticker. For example: Commonwealth Bank (CBA.AX), or Telstra (TLS.AX)
            
For column **Allocation**, the sum of allocations is **1.0**

For furthere reference, please review this following [stocks.csv](https://github.com/DoThNg/portfolio_analysis/blob/main/stocks.csv)

The S&P/ASX 200 (^AXJO) is set as the benchmark for portfolio analysis. However, the benchmark can be changed in the sidebar to any others. 
            
