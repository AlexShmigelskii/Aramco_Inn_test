import streamlit as st
import pandas as pd
import warnings

from etna.analysis import plot_residuals, prediction_actual_scatter_plot, plot_backtest
from etna.datasets.tsdataset import TSDataset
from etna.pipeline import Pipeline
from etna.transforms import LagTransform, LogTransform, DifferencingTransform
from etna.models import CatBoostPerSegmentModel
from etna.metrics import MAE, MSE, SMAPE
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


def main():

    # Functions for calculating metrics
    def calculate_smape(forecast_ts, test_ts):
        return 1 / len(test_ts) * np.sum(
            2 * np.abs(forecast_ts - test_ts) / (np.abs(test_ts) + np.abs(forecast_ts)) * 100)

    def calculate_mae(forecast_ts, test_ts):
        return mean_absolute_error(test_ts, forecast_ts)

    def calculate_mse(forecast_ts, test_ts):
        return mean_squared_error(test_ts, forecast_ts)

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

        st.write('''Welcome to time series forecasting application! We're using a dataset of monthly Australian wine 
        sales, which provides a detailed record of sales over time.''')
        
        st.write('''In the sidebar, you can set the "HORIZON" and "lags" for forecasting. The "HORIZON" defines the 
        forecast length in months, while "lags" set the number of prior months used for prediction.''')
        
        st.write('''You can also apply various data transformations to improve model performance. Please, adjust the 
            parameters in the sidebar to start your analysis.''')
        
        st.write('''Enjoy the application and find it useful in your forecasting tasks!''')

    @st.cache_data
    def load_data():
        # Loading dataset

        original_df = pd.read_csv("data/monthly-australian-wine-sales.csv")

        st.subheader("Here you can observe the dataset")

        with st.expander("See dataset", ):

            st.table(original_df)

        original_df["timestamp"] = pd.to_datetime(original_df["month"])
        original_df["target"] = original_df["sales"]
        original_df.drop(columns=["month", "sales"], inplace=True)
        original_df["segment"] = "main"

        df = TSDataset.to_dataset(original_df)
        ts = TSDataset(df, freq="MS")

        return original_df, ts

    def sidebar():
        # Input widjects
        st.sidebar.header('Input features')
        st.sidebar.write('You can select the parameters - the graph data will be changed automatically')

        # Creating checkbox for transforms
        transforms = []

        if st.sidebar.checkbox(
                'LogTransform',
                help='Transforms data by applying a logarithm'
        ):
            # LogTransform checkbox
            log = LogTransform(in_column="target")
            transforms.append(log)

        if st.sidebar.checkbox(
                'DifferencingTransform',
                help='Transforms data by subtracting the current value from the previous one. This is typically done to'
                     ' remove a time dependency in the data, making it stationary.'
        ):
            # DifferencingTransform checkbox
            period = st.sidebar.slider('choose period for differencing', 1, 24, value=8)
            dif = DifferencingTransform(in_column="target", period=period)
            transforms.append(dif)

        HORIZON = 8
        if st.sidebar.checkbox(
                'Select Horizon',
                help='Forecast length in months'
        ):
            # Selecting the horizon of prediction
            HORIZON = st.sidebar.slider('horizon of prediction', 1, 24, value=8, help='num of months to predict')

        # Unresolved problem of duplicate lags when selecting other checkboxes
        start_lag, end_lag = st.sidebar.select_slider(
            'Lags',
            options=list(range(1, 25, 1)),
            value=(1, 24),
            help='The number of prior time units used for predicting each subsequent value')
        lags = LagTransform(in_column="target", lags=list(range(start_lag, end_lag, 1)))

        transforms.append(lags)

        return HORIZON, transforms

    def data_split(horizon):
        # Data split
        last_month = pd.to_datetime(original_df['timestamp'].max())
        train_end = (last_month - pd.DateOffset(months=horizon)).strftime('%Y-%m-%d')
        test_start = (last_month - pd.DateOffset(months=horizon - 1)).strftime('%Y-%m-%d')

        train, test = ts.train_test_split(
            train_start="1980-01-01",
            train_end=train_end,
            test_start=test_start,
            test_end=last_month.strftime('%Y-%m-%d'),
        )
        return train, test

    def plot():
        st.pyplot(plot_residuals(forecast_df=forecast_df, ts=ts))
        with st.expander("See explanation"):
            st.write("""This is a Residuals vs Predicted Values plot. The residuals represent the difference between 
            the actual and predicted values. Ideally, we would like the residuals to be randomly scattered around the 
            centerline of zero, with no obvious patterns. This would indicate that the model's predictions are 
            correct on average irrespective of the input. If there are any obvious patterns in this plot, 
            it means our model might be missing some explanatory variable or there may be an interaction between 
            variables that the model isn't capturing.""")

        st.pyplot(prediction_actual_scatter_plot(forecast_df=forecast_df, ts=ts))
        with st.expander("See explanation"):
            st.write("""This is a Quantile-Quantile (QQ) plot of the residuals. It's a graphical tool to help us assess if 
            the residuals follow a Normal distribution. The points represent our data and are plotted against a 
            theoretical Normal distribution. If the points fall along the dashed 'Identity' line, it means that the 
            residuals are Normally distributed. The 'Best Fit' line represents the best approximation to our data's 
            distribution. The closer the 'Best Fit' line is to the 'Identity' line, the more normally distributed our 
            residuals are. The coefficient of determination, R2, indicates how well the model fits the data. A value 
            of 1.0 indicates a perfect fit.""")

    def load_model(horizon):
        # Model
        m = Pipeline(
            model=CatBoostPerSegmentModel(),
            transforms=transforms,
            horizon=horizon,
        )
        return m

    def merge_data():
        # Merging Data
        d1 = pd.DataFrame({
            'forecast': map(int, forecast_ts.df['main']['target']),
            'test': test_ts.df['main']['target']
        })
        d2 = pd.DataFrame({'train': train_ts.df['main']['target']})

        d = d2.merge(d1, on='timestamp', how='outer')

        return d

    greeting()
    original_df, ts = load_data()
    HORIZON, transforms = sidebar()
    train_ts, test_ts = data_split(HORIZON)

    model = load_model(HORIZON)

    model.fit(train_ts)
    forecast_ts = model.forecast()

    smape = calculate_smape(forecast_ts.df['main']['target'], test_ts.df['main']['target'])
    mae = calculate_mae(forecast_ts.df['main']['target'], test_ts.df['main']['target'])
    mse = calculate_mse(forecast_ts.df['main']['target'], test_ts.df['main']['target'])

    data = {'SMAPE': [smape], 'MAE': [mae], 'MSE': [mse]}
    df = pd.DataFrame(data)

    data_for_chart = merge_data()

    st.line_chart(data_for_chart)

    st.table(df)

    # Backtest
    metrics_df, forecast_df, fold_info_df = model.backtest(ts=ts, metrics=[MAE(), MSE(), SMAPE()])

    st.subheader('To visualize the train part, you can specify the "history length" parameter.')
    history_len = st.slider('history length (in months)', 1, 136, value=8)
    st.pyplot(plot_backtest(forecast_df=forecast_df, ts=ts, history_len=history_len))

    st.subheader("METRICS")

    st.table(metrics_df.head())
    st.table(fold_info_df.head())

    # Metrics formulas
    with st.expander("See formulas"):
        st.latex(r"""\text{MAE}(y, \hat{y}) = \frac{ \sum_{i=0}^{N - 1} |y_i - \hat{y}_i| }{N}""")  # MAE
        st.latex(r"""\text{MSE}(y, \hat{y}) = \frac{\sum_{i=0}^{N - 1} (y_i - \hat{y}_i)^2}{N}""")  # MSE
        st.latex(r"""\text{SMAPE}(y, \hat{y}) = \frac{100\%}{N} \sum_{i=0}^{N - 1} \frac{ 2*|y_i -
             \hat{y}_i|}{|y| + |\hat{y}|}""")  # SMAPE

    plot()


if __name__ == "__main__":
    main()

