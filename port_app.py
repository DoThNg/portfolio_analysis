import yfinance as yf
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from scipy.optimize import minimize
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="Portfolio Construction", initial_sidebar_state = "expanded", page_icon=":sparkles:")

st.title("Equity Portfolio Construction :telescope:")
st.write("""
        This is an interactive dashboard app built in Python with the stock data deriving from Yahoo Finance.
         
        In general, this app can be used for the following analysis:
         
         1. The quantitative and fundamental of a stock
         2. The analysis of a select portfolio

        **Note**: The sidebar on the left of the dashboard app is for the user inputs
         
        **Disclaimer:**
         This app is only for learning to build and deploy a Streamlit App. Any information or analysis in this app is not an investment advice and should not be relied upon in any investment decisions. Please be also noted that past performance information is given for illustrative purposes only, and it is not an indication of future performance. In addition, many factors (e.g., transaction fees, capital gain tax, investment risks, etc.) have not been taken into account in the stock or portfolio analysis and it will, therefore, not be applicable to an individual's objectives or financial situation.
        """)

st.divider()

with st.sidebar:
    st.write("Please be noted to add stock exchange after the stock ticker. For example: Commonwealth Bank (CBA.AX), or Telstra (TLS.AX)")
    ticker = st.sidebar.text_input("Stock Ticker - Yahoo Finance :coffee:", "TLS.AX")
    benchmark_ticker = st.sidebar.text_input("Benchmark Ticker - Yahoo Finance :green_book:", "^AXJO")
    rf = st.sidebar.text_input("Risk Free Rate :chart_with_upwards_trend:", 0.02)
    option = st.selectbox("Would you like the portfolio to be equally weighted? Only applicable when the csv file containing portfolio is uploaded. :clipboard:",
                        ("Yes", "No"))
    port = st.file_uploader("Please choose a csv file containing a list of stock tickers and allocation percentage :file_folder:")
    start_date = st.sidebar.date_input("Start Date :date:", datetime.today() - timedelta(days=90))
    end_date = st.sidebar.date_input("End Date :date:", datetime.today())

template = dict(
                layout=go.Layout(title_font=dict(family="Rockwell", size=28))
            )

st.header("1. The quantitative and fundamental of a stock")

col1, col2 = st.columns(2)

if ticker:
    try:
        with col1:
            st.subheader("Stock price performances :signal_strength:")
            stock_price_data = yf.download(ticker, period = '5y')
            benchmark_price_data = yf.download(benchmark_ticker, period = '5y')

            fig = go.Figure()
            fig.add_trace(go.Scatter(x = stock_price_data.index, y = stock_price_data['Adj Close'], name = "Stock prices", yaxis='y'))
            fig.add_trace(go.Scatter(x = benchmark_price_data.index, y = benchmark_price_data['Adj Close'], name = "Benchmark", yaxis='y2'))

            # Create axis objects
            fig.update_layout(xaxis=dict(domain=[0.1, 0.99]),
                            
                yaxis=dict(
                    title="Stock prices",
                    titlefont=dict(color="#1f77b4"),
                    tickfont=dict(color="#1f77b4")),
                                              
                yaxis2=dict(title="Benchmark",overlaying="y",
                            side="right",position=0.98))
            
            # title
            fig.update_layout(
                title_text="Stock prices: {} vs. Benchmark: {}".format(ticker, benchmark_ticker),
                width=800,
                template = template
            )

            st.plotly_chart(fig)

            fundamentals_IC, fundamentals_BS, fundamentals_CF = st.tabs(["Income Statement", "Balance Sheet", "Cashflow Statement"])
            
            comp_data = yf.Ticker(ticker = ticker)

            with fundamentals_IC:
                st.header("Income Statement")
                st.write(comp_data.financials)

            with fundamentals_BS:
                st.header("Balance Sheet")
                st.write(comp_data.balancesheet)

            with fundamentals_CF:
                st.header("Cash Flow")
                st.write(comp_data.cashflow)

    except Exception as error:
        st.text("Error found: {}".format(error))

    with col2:
        st.subheader("Fundamental Metrics :books:")

        comp = yf.Ticker(ticker)
        start_price_date = datetime.today() - timedelta(days=7)
        start_price_date = start_price_date.strftime('%Y-%m-%d')
        end_price_date = datetime.today().strftime('%Y-%m-%d')

        avg_price = yf.download(ticker, period = '5y')['Adj Close'].mean()

        cf = comp.cashflow
        bs = comp.balancesheet
        ic = comp.financials

        try:
            if ".AX" in ticker:
                operating_cf = cf.loc["Cash Flowsfromusedin Operating Activities Direct"]
                investing_cf = cf.loc["Cash Flow From Continuing Investing Activities"]

                for ops_cf_idx in range(len(operating_cf)):
                    if np.isnan(operating_cf[ops_cf_idx]):
                        operating_cf[ops_cf_idx] = 0
                    if np.isnan(investing_cf[ops_cf_idx]):
                        investing_cf[ops_cf_idx] = 0
                ops_cf = operating_cf + investing_cf

            else:
                operating_cf = cf.loc["Cash Flow From Continuing Operating Activities"]
                investing_cf = cf.loc["Cash Flow From Continuing Investing Activities"]

                for ops_cf_idx in range(len(operating_cf)):
                    if np.isnan(operating_cf[ops_cf_idx]):
                        operating_cf[ops_cf_idx] = 0
                    if np.isnan(investing_cf[ops_cf_idx]):
                        investing_cf[ops_cf_idx] = 0
                ops_cf = operating_cf + investing_cf
                
            # cashflow per share
            cf_per_share = ops_cf / ic.loc["Diluted Average Shares"]

            price_to_cf_list = avg_price/cf_per_share 

            idx = 0
            for n_yr in price_to_cf_list.index:
                fin_yr = datetime.strftime(n_yr, '%Y')
                curr_yr = datetime.today().strftime('%Y')
                if fin_yr == curr_yr:
                    price_to_cf = price_to_cf_list.iloc[idx]
                elif fin_yr == str(int(curr_yr) - 1):
                    price_to_cf = price_to_cf_list.iloc[idx]
                idx = idx + 1

            st.write("1. Price to Cashflow")
            st.write("Note: Cashflow = Ops CF - CAPEX (or Investing CF)")
            st.write("P/CF where CF is in the most recent annual report")
            st.write(round(price_to_cf, 2))

            st.write("P/CF where CF is the average of historical CFs")
            st.write(round(avg_price/(cf_per_share).mean(), 2)) 

        except:
            st.write("1. Price to Cashflow")
            st.write("Note: Cashflow = Ops CF - CAPEX (or Investing CF)")
            st.write("There is no data on P/CF for {}".format(ticker))
                    
        try:
            EPS = ic.loc["Diluted EPS"]
            PE_ratio = avg_price/EPS.iloc[0]

            st.write("2. Current PE Ratio")
            st.write("Note: diluted shares are consider to compute PE")
            st.write(round(PE_ratio, 2))
        except:       
            st.write("2. Current PE Ratio")
            st.write("Note: diluted shares are consider to compute PE")
            st.write("There is no data on PE ratio for {}".format(ticker))               


        debt_ratio, return_on_equity, return_on_assets = st.tabs(["3. Net Debt to Equity", "4. ROE Ratio", "5. ROA Ratio"])

        # Net Debt / Equity
        with debt_ratio:
            try:
                st.write("3. Net Debt to Equity") 
                try:
                    long_term_debt = bs.loc["Long Term Debt And Capital Lease Obligation"].fillna(0)
                except:
                    long_term_debt = 0

                try:
                    cur_debt_cap_lease = bs.loc["Current Debt And Capital Lease Obligation"].fillna(0)
                except:
                    cur_debt_cap_lease = 0

                try:
                    other_cur_bor = bs.loc["Other Current Borrowings"].fillna(0)
                except:
                    other_cur_bor = 0

                try:
                    non_cur_pen = bs.loc["Non Current Pension And Other Postretirement Benefit Plans"].fillna(0)
                except:
                    non_cur_pen = 0

                try:
                    other_post_ret = bs.loc["Pensionand Other Post Retirement Benefit Plans Current"].fillna(0)
                except:
                    other_post_ret = 0

                comp_net_debt = long_term_debt + cur_debt_cap_lease + other_cur_bor + non_cur_pen + other_post_ret - bs.loc["Cash And Cash Equivalents"]
                historical_debt_ratio = pd.DataFrame(comp_net_debt / bs.loc["Stockholders Equity"])

                historical_debt_ratio.columns = ["Net Debt Ratio"]
                historical_debt_ratio.index.name = "Date"
                debt_ratio_chart = px.line(historical_debt_ratio, x = historical_debt_ratio.index, y = historical_debt_ratio["Net Debt Ratio"], title = "Net Debt to Equity Ratio")
                st.plotly_chart(debt_ratio_chart)

            except:
                st.write("3. Net Debt to Equity")
                st.write("There is no data on Net Debt-to-Equity for {}".format(ticker))   

        
        with return_on_equity:
            try:
                st.write("4. ROE Ratio")
                roe = pd.DataFrame((ic.loc["Net Income Common Stockholders"] / bs.loc["Stockholders Equity"]) * 100)
                roe.columns = ["ROE, %"]
                roe.index.name = "Date"
                roe_chart = px.line(roe, x = roe.index, y = roe["ROE, %"], title = "ROE Ratio, %")
                st.plotly_chart(roe_chart)

            except:       
                st.write("4. ROE Ratio")
                st.write("There is no data on ROE for {}".format(ticker))   

        with return_on_assets:
            try:
                st.write("5. ROA Ratio")
                roa = pd.DataFrame((ic.loc["Net Income Common Stockholders"] / bs.loc["Total Assets"]) * 100)
                roa.columns = ["ROA, %"]
                roa.index.name = "Date"
                roa_chart = px.line(roa, x = roa.index, y = roa["ROA, %"], title = "ROA Ratio, %")
                st.plotly_chart(roa_chart)

            except:       
                st.write("5. ROA Ratio")
                st.write("There is no data on ROA for {}".format(ticker))  
        
        try:
            st.write("6. Cashflow Overview")
            if ".AX" in ticker:
                fig_ax = go.Figure()
                fig_ax.add_trace(go.Scatter(x = cf.loc["Cash Flowsfromusedin Operating Activities Direct"].index, y = cf.loc["Cash Flowsfromusedin Operating Activities Direct"], name = "Ops Cashflow", yaxis='y'))
                fig_ax.add_trace(go.Scatter(x = cf.loc["Cash Flow From Continuing Investing Activities"].index, y = cf.loc["Cash Flow From Continuing Investing Activities"] * -1, name = "Investing Cashflow", yaxis='y2'))

                # Create axis objects
                fig_ax.update_layout(xaxis=dict(domain=[0.1, 0.99]),
            
                    yaxis=dict(
                        title="Operating CF",
                        titlefont=dict(color="#1f77b4"),
                        tickfont=dict(color="#1f77b4")),
                                  
                    yaxis2=dict(title="Investing CF",overlaying="y",
                                side="right",position=0.98))
                
                # title
                fig_ax.update_layout(
                    title_text="Cashflow Overview: {}".format(ticker),
                    width=800,
                    template = template
                )
                st.plotly_chart(fig_ax)

            else:
                fig_non_ax = go.Figure()
                fig_non_ax.add_trace(go.Scatter(x = cf.loc["Cash Flow From Continuing Operating Activities"].index, y = cf.loc["Cash Flow From Continuing Operating Activities"], name = "Ops Cashflow", yaxis='y'))
                fig_non_ax.add_trace(go.Scatter(x = cf.loc["Cash Flow From Continuing Investing Activities"].index, y = cf.loc["Cash Flow From Continuing Investing Activities"] * -1, name = "Investing Cashflow", yaxis='y2'))

                # Create axis objects
                fig_non_ax.update_layout(xaxis=dict(domain=[0.1, 0.99]),          
                    yaxis=dict(
                        title="Operating CF",
                        titlefont=dict(color="#1f77b4"),
                        tickfont=dict(color="#1f77b4")),
                                
                    yaxis2=dict(title="Investing CF",overlaying="y",
                                side="right",position=0.98))
                
                # title
                fig_non_ax.update_layout(
                    title_text="Cashflow Overview: {}".format(ticker),
                    width=800,
                    template = template
                )

                st.plotly_chart(fig_non_ax)
        except:
                st.write("6. Cashflow Overview")
                st.write("There is no historical ata on Cashflow for {}".format(ticker))                  

st.divider()
st.header("2. The portfolio analysis")
st.write("""
            
        ###### Note: In the sidebar, please drag and drop the csv file with the following format:
        
        **Example**: 

        | Ticker          | Allocation    |
        | --------------- | ------------- |
        | CBA.AX          |     0.5       |
        | TLS.AX          |     0.5       |
        
        In column **Ticker**, please add stock exchange (e.g., **.AX** for Australian Stock Exchange) after the stock ticker. For example: Commonwealth Bank (CBA.AX), or Telstra (TLS.AX)
            
        For column **Allocation**, the sum of allocations is **1.0**
         
        The S&P/ASX 200 (^AXJO) is set as the benchmark for portfolio analysis. However, the benchmark can be changed in the sidebar to any others. 
            
        """)
st.write("---")

col3, col4 = st.columns([0.3, 0.7])

if port is not None:
    try:
        with col3:
            df = pd.read_csv(port)
            
            st.subheader("Overview of portfolio selected")
            st.dataframe(df)

        with col4:
            st.subheader("Portfolio performances vs. Benchmark: {}".format(benchmark_ticker))
            port_list = []

            for idx in range(len(df["Ticker"])):
                ticker = df["Ticker"].iloc[idx]

                stock_data = yf.download(ticker, period = '5y')
                stock_price = stock_data['Adj Close']
                port_list.append(stock_price)

            port = pd.concat(port_list, axis=1)
            port.columns = list(df["Ticker"]) 

            return_col = []

            for idx in range(len(df["Ticker"])):
                port['R_' + str(df["Ticker"].iloc[idx])] = port[df["Ticker"].iloc[idx]].pct_change()
                return_col.append('R_' + str(df["Ticker"].iloc[idx]))

            if option == "No":
                allocation_pct = df["Allocation"].iloc[idx]
                WeightedReturns = port[return_col].mul(allocation_pct, axis=1)
                port["Portfolio"] = WeightedReturns.sum(axis=1)
                port["CumulativeReturns"]  = (1+port["Portfolio"]).cumprod()
            else:
                allocation_pct = 1 / len(df["Ticker"])
                WeightedReturns = port[return_col].mul(allocation_pct, axis=1)
                port["Portfolio"] = WeightedReturns.sum(axis=1)
                port["CumulativeReturns"] = (1+port["Portfolio"]).cumprod()

            benchmark_price_data["R_Benchmark"] = benchmark_price_data['Adj Close'].pct_change()
            benchmark_price_data["CumulativeReturns"] = (1+benchmark_price_data["R_Benchmark"]).cumprod()

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x = port.index, y = port["CumulativeReturns"], name = "Portfolio", yaxis='y'))
            fig2.add_trace(go.Scatter(x = benchmark_price_data.index, y = benchmark_price_data['CumulativeReturns'], name = "Benchmark", yaxis='y2'))

            # Create axis objects
            fig2.update_layout(xaxis=dict(domain=[0.1, 0.99]),
                #create 1st y axis              
                yaxis=dict(
                    title="Portoflio",
                    titlefont=dict(color="#1f77b4"),
                    tickfont=dict(color="#1f77b4")),
                            
                #create 2nd y axis       
                yaxis2=dict(title="Benchmark", overlaying="y",
                            side="right", position=0.98))
            
            # Layout
            fig2.update_layout(
                title_text="Portfolio and Benchmark performances",
                width = 1200, height = 500,
                template = template
            )

            st.plotly_chart(fig2)

    except Exception as error:
        st.text("Error found: {}".format(error)) 

    # Calculate CAGR
    star_port_value = port["CumulativeReturns"][1]
    end__port_value = port["CumulativeReturns"][len(port["CumulativeReturns"]) - 1]

    star_bm_value = benchmark_price_data["CumulativeReturns"][1]
    end__bm_value = benchmark_price_data["CumulativeReturns"][len(benchmark_price_data["CumulativeReturns"]) - 1]

    cagr_portfolio = round(((end__port_value/star_port_value)**(1/((end_date-start_date).days/365)) - 1) * 100, 2)
    cagr_benchmark = round(((end__bm_value/star_bm_value)**(1/((end_date-start_date).days/365)) - 1) * 100, 2)

    # The tracking error
    tracking_err = round((np.std(port["Portfolio"] - benchmark_price_data["R_Benchmark"]) * np.sqrt(252) * 100), 2)

    # Maximum Drawdown (MDD) = (Trough Value – Peak Value) ÷ Peak Value
    mdd = round(((port["CumulativeReturns"].max() - port["CumulativeReturns"].min()) / port["CumulativeReturns"].max()) * 100, 2)

    # Calculate Sharpe Ratio
    rp = round(port["Portfolio"].mean()*252, 3)
    sharpe_ratio = round((rp - float(rf)) / (port["Portfolio"].std()*np.sqrt(252)), 3)
    
    # Calculate Sortino Ratio
    downside_returns =  port["Portfolio"][port["Portfolio"] < float(rp)]
    down_stdev = np.sqrt(sum((downside_returns - float(rp))**2)/len(port["Portfolio"])) * np.sqrt(252)

    sortino_ratio = round((rp - float(rf)) / down_stdev, 2)
    
    st.subheader("Portfolio Performance Summary")
    st.dataframe(pd.DataFrame({"Portfolio Return, %" : cagr_portfolio,
                  "Benchmark Return, %" : cagr_benchmark,
                  "Tracking Error, %" : tracking_err,
                  "Max Drawdown, %" : mdd,
                  "Sharpe Ratio" : sharpe_ratio,
                  "Sortino Ratio" : sortino_ratio
                  }, index=["Metrics"]))
    
    # Define function to compute tracking error:
    def opt_tracking_error(weights):
        opt_result =  round((np.std(port[return_col].mul(weights, axis=1).sum(axis=1) - benchmark_price_data["R_Benchmark"]) * np.sqrt(252) * 100), 2)
        return opt_result
    
    # Weight constraint: sum of weights is equal to 1
    constraint = {"type": "eq", "fun": lambda x: np.sum(x) - 1}

    optimizer = minimize(opt_tracking_error, 
                        x0=[0.1 for i in range(len(df["Ticker"]))], 
                        method = "SLSQP", 
                        bounds = [(0.05, 0.3) for i in range(len(df["Ticker"]))], 
                        constraints = constraint)
    
    st.write("Optimization result for minimum tracking error of portfolio with weight boundary within 0.05 and 0.3 is as follows:")
    st.dataframe({"Ticker" : list(df["Ticker"]), "Weights": [round(opt_w, 2) for opt_w in optimizer.x]})
