import yfinance as yf
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from scipy.optimize import minimize
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

st.set_page_config(layout="wide", page_title="Portfolio Construction", initial_sidebar_state = "expanded", page_icon=":sparkles:")

st.title("Equity Portfolio Construction :telescope:")
st.write("""
        This is an interactive dashboard app built in Python with the stock data deriving from Yahoo Finance.
         
        This app can be used for analysis of a selected equity portfolio over the past 5 years.

        **Note**: The sidebar on the left of the dashboard app is for the user inputs
         
        **Disclaimer:**
         This app is only for learning to build and deploy a Streamlit App. Any information or analysis in this app is not an investment advice and should not be relied upon in any investment decisions. Please be also noted that past performance information is given for illustrative purposes only, and it is not an indication of future performance. In addition, many factors (e.g., transaction fees, capital gain tax, investment risks, etc.) have not been taken into account in the portfolio analysis and it will, therefore, not be applicable to an individual's objectives or financial situation.
        """)

st.divider()

with st.sidebar:
    benchmark_ticker = st.sidebar.text_input("Benchmark Ticker - Yahoo Finance :green_book:", "^AXJO")
    rf = st.sidebar.text_input("Risk Free Rate :chart_with_upwards_trend:", 0.02)
    option = st.selectbox("Would you like the portfolio to be equally weighted? Only applicable when the csv file containing portfolio is uploaded. :clipboard:",
                        ("Yes", "No"))
    port = st.file_uploader("Please choose a csv file containing a list of stock tickers and allocation percentage :file_folder:")
                
template = dict(
                layout=go.Layout(title_font=dict(family="Rockwell", size=28))
            )

st.header("The portfolio analysis")
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

benchmark_price_data = yf.download(benchmark_ticker, period = "5y")

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

                stock_data = yf.download(ticker, period = "5y")
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
                yaxis=dict(
                    title="Portoflio",
                    titlefont=dict(color="#1f77b4"),
                    tickfont=dict(color="#1f77b4")),
                     
                yaxis2=dict(title="Benchmark", overlaying="y",
                            side="right", position=0.98))
            
            # Layout
            fig2.update_layout(
                title_text="Portfolio and Benchmark performances from {} to {}".format((datetime.today() - relativedelta(years=5)).strftime('%Y-%m-%d'), datetime.today().strftime('%Y-%m-%d')),
                width = 1200, height = 500,
                template = template
            )

            st.plotly_chart(fig2)

    except Exception as error:
        st.text("Error found: {}".format(error)) 

    # Calculate CAGR
    star_port_value = port["CumulativeReturns"][1]
    end_port_value = port["CumulativeReturns"][len(port["CumulativeReturns"]) - 1]

    star_bm_value = benchmark_price_data["CumulativeReturns"][1]
    end_bm_value = benchmark_price_data["CumulativeReturns"][len(benchmark_price_data["CumulativeReturns"]) - 1]

    cagr_portfolio = round(((end_port_value/star_port_value)**(1/5 - 1) * 100, 2)
    cagr_benchmark = round(((end_bm_value/star_bm_value)**(1/5 - 1) * 100, 2)

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
