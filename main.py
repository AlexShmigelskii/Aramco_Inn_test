import streamlit as st
import pandas as pd
import numpy as np
import warnings

from etna.analysis import plot_residuals, prediction_actual_scatter_plot, plot_backtest
from etna.datasets.tsdataset import TSDataset
from etna.pipeline import Pipeline
from etna.transforms import LagTransform, LogTransform, DifferencingTransform
from etna.models import CatBoostPerSegmentModel
from etna.metrics import MAE, MSE, SMAPE


def main():

    def greeting():
        # Page configuration
        st.set_page_config(
            page_title='Simple Prediction App',
            layout='wide',
            initial_sidebar_state='expanded'
        )
        st.set_option('deprecation.showPyplotGlobalUse', False)
        warnings.filterwarnings(action="ignore", message="Torchmetrics v0.9")

        # Title of the App
        st.title('Simple Prediction App')

        st.text("""
            Welcome to time series forecasting application! We're using a dataset of monthly Australian wine sales, which provides a detailed record of sales over time.
        
        In the sidebar, you can set the "HORIZON" and "lags" for forecasting. The "HORIZON" defines the forecast length in months, while "lags" set the number of prior months used for prediction.
        
        You can also apply various data transformations to improve model performance. Please, adjust the parameters in the sidebar to start your analysis.
        
        Enjoy the application and find it useful in your forecasting tasks!
        """)

    def load_data():
        # Loading dataset
        original_df = pd.read_csv('https://raw.githubusercontent.com/demidovakatya/mashinnoye-obucheniye/master/5-data'
                                  '-analysis-applications/1_1_time_series/monthly-australian-wine-sales.csv')
        # original_df = pd.read_csv("data/monthly-australian-wine-sales.csv")

        original_df["timestamp"] = pd.to_datetime(original_df["month"])
        original_df["target"] = original_df["sales"]
        original_df.drop(columns=["month", "sales"], inplace=True)
        original_df["segment"] = "main"

        df = TSDataset.to_dataset(original_df)
        ts = TSDataset(df, freq="MS")

        st.subheader("Here you can observe the dataset")

        st.write(ts.describe())
        st.write(original_df)
        return original_df, ts

    def sidebar():
        # Input widjects
        st.sidebar.header('Input features')
        st.sidebar.text('you can select the parameters \n - the graph data will change')

        # Creating checkbox for transforms
        transforms = []
        HORIZON = 8

        if st.sidebar.checkbox(
                'LogTransform',
                help='Transforms data by applying a logarithm'
                ):
            """LogTransform checkbox"""
            log = LogTransform(in_column="target")
            transforms.append(log)

        if st.sidebar.checkbox(
                'DifferencingTransform',
                help='Transforms data by subtracting the current value from the previous one. This is typically done to'
                     ' remove a time dependency in the data, making it stationary.'
                ):
            """DifferencingTransform checkbox"""
            period = st.sidebar.slider('choose period for differencing', 1, 24, value=8)
            dif = DifferencingTransform(in_column="target", period=period)
            transforms.append(dif)

        if st.sidebar.checkbox(
                'Select Horizon',
                help='forecast length in months'
                ):
            """Selecting the horizon of prediction"""
            HORIZON = st.sidebar.slider('horizon of prediction', 1, 24, value=8, help='num of months to predict')

        start_lag, end_lag = st.sidebar.select_slider(
            'lags',
            options=list(range(1, 25, 1)),
            value=(1, 24),
            help='the number of prior time units used for predicting each subsequent value')
        lags = LagTransform(in_column="target", lags=list(range(start_lag, end_lag, 1)))
        transforms.append(lags)

        return HORIZON, transforms

    def data_split(horizon):
        # Data split
        last_month = pd.to_datetime(original_df['timestamp'].max())
        train_end = (last_month - pd.DateOffset(months=horizon)).strftime('%Y-%m-%d')
        test_start = (last_month - pd.DateOffset(months=horizon - 1)).strftime('%Y-%m-%d')

        train_ts, test_ts = ts.train_test_split(
            train_start="1980-01-01",
            train_end=train_end,
            test_start=test_start,
            test_end=last_month.strftime('%Y-%m-%d'),
        )
        return train_ts, test_ts

    def plot():
        st.pyplot(plot_residuals(forecast_df=forecast_df, ts=ts))

        st.pyplot(prediction_actual_scatter_plot(forecast_df=forecast_df, ts=ts))

        st.subheader('To visualize the train part, you can specify the "history length" parameter.')
        history_len = st.slider('history length', 1, 136, value=8)
        st.pyplot(plot_backtest(forecast_df=forecast_df, ts=ts, history_len=history_len))


    greeting()
    original_df, ts = load_data()
    HORIZON, transforms = sidebar()
    train_ts, test_ts = data_split(HORIZON)


    # Model
    model = Pipeline(
        model=CatBoostPerSegmentModel(),
        transforms=transforms,
        horizon=HORIZON,
    )

    model.fit(train_ts)
    forecast_ts = model.forecast()

    st.subheader("METRICS")

    st.latex(r"""\text{MAE}(y, \hat{y}) = \frac{ \sum_{i=0}^{N - 1} |y_i - \hat{y}_i| }{N}""")  # MAE
    st.latex(r"""\text{MSE}(y, \hat{y}) = \frac{\sum_{i=0}^{N - 1} (y_i - \hat{y}_i)^2}{N}""")  # MSE
    st.latex(r"""\text{SMAPE}(y, \hat{y}) = \frac{100\%}{N} \sum_{i=0}^{N - 1} \frac{ 2*|y_i -
     \hat{y}_i|}{|y| + |\hat{y}|}""")  # SMAPE

    metrics_df, forecast_df, fold_info_df = model.backtest(ts=ts, metrics=[MAE(), MSE(), SMAPE()])
    st.write(metrics_df.head())
    st.write(fold_info_df.head())


    plot()


    # Merging Data
    d1 = pd.DataFrame({
        'forecast': map(int, forecast_ts.df['main']['target']),
        'test': test_ts.df['main']['target']
    })
    d2 = pd.DataFrame({'train': train_ts.df['main']['target']})

    d = d2.merge(d1, on='timestamp', how='outer')

    st.line_chart(d, height=800)


if __name__ == "__main__":
    main()
