import numpy
import pandas as pd
import pandas_datareader as pdr
from pandas import MultiIndex
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.random import seed
from numpy.random import normal
import datetime
from scipy.stats import norm
import multiprocessing
from typing import TypeVar

DAYS_IN_YEAR = 365

NUM_OF_SIMULATIONS = 2000

sns.set_style('whitegrid')

avg = 1
std_dev = .1
num_reps = 500
num_simulations = 1000
num_years = 30
num_periods = num_years * 12
zipcode = '75074'
home_type = "Single Family Residential"
house_current_value = 350000


df_sp500 = pd.DataFrame()

class Brown:

    log_returns = []
    avg_period_return = 0.0
    variance = 0.0
    drift = 0.0
    std_dev = 0.0

    def __init__(self, data, forecast_periods):
        '''

        :param data: Dataframe of past time series data
        :param column_name: name of the column that forecast data will be based upon (monthly sales amount, daily profits, etc)
        '''
        self.df = data
        self.pct_change = self.df.pct_change()
        self.log_returns = np.log(self.pct_change+1)
        self.avg_period_return = self.log_returns.mean()
        self.variance = self.log_returns.var()
        self.drift = self.avg_period_return - (0.5 * self.variance)
        self.std_dev = self.log_returns.std()
        self.forecast_periods = forecast_periods
        self.past_periods = self.df.size -1
        self.forecast_data = np.zeros(forecast_periods)

    def calc_next_period(self, last_value):

        random_value = self.std_dev * norm.ppf(np.random.rand())
        price = last_value * np.exp(self.drift + random_value)

        return price

    def calc_all_futures(self, last_value):
        '''
        Formula based upon Investopedia guidance
        https://www.investopedia.com/terms/m/montecarlosimulation.asp
        :return:
        '''

        count = 0

        for y in range(self.forecast_periods):
            if y == self.forecast_periods:
                break

            random_value = self.std_dev * norm.ppf(np.random.rand())

            if y == 0:
                price = last_value * np.exp(self.drift + random_value)
            else:
                price = self.forecast_data[count] * np.exp(self.drift + random_value)
                count += 1

            self.forecast_data[count] = price



    def graph_futures(self):

        futures_data = pd.DataFrame(index=range(self.forecast_periods), data=self.forecast_data)
        futures_pct_change = futures_data.pct_change()
        futures_pct_change += 1

        futures_pct_change_graph = sns.relplot(data=futures_pct_change, kind="line")
        plt.show()

        futures_graph = sns.relplot(data=futures_data, kind="line")
        plt.show()

def graph_futures(forecast_periods, forecast_data,title):

    futures_data = pd.DataFrame(index=range(forecast_periods), data=forecast_data)
    futures_pct_change = futures_data.pct_change()
    futures_pct_change += 1

    #futures_pct_change_graph = sns.relplot(data=futures_pct_change, kind="line")
    #plt.show()

    futures_graph = sns.relplot(data=futures_data, kind="line").set(title=title)
    plt.show()


def check_for_housing_data():
    print()

def get_housing_market() -> pd.DataFrame:
    '''
    Fetches housing market data for the provided zipcode from redfin and sorts by period
    :return:
    '''
    url = "https://redfin-public-data.s3.us-west-2.amazonaws.com/redfin_market_tracker/zip_code_market_tracker.tsv000.gz"
    today = datetime.date.today()
    start = today - datetime.timedelta(days=num_years * 365)
    df = pd.read_csv("C:\\Users\\justi\\Downloads\\zip_code_market_tracker.tsv000.gz", compression='gzip', sep='\t', on_bad_lines='skip')
    df_zip_filter = df.loc[df['region']== ("Zip Code: " + zipcode)]
    df_copy = pd.DataFrame.copy(df_zip_filter)
    df_Housing_Market = df_copy.loc[df_copy["property_type"] == home_type]

    df_Housing_Market = df_Housing_Market.sort_values(by=["period_begin"])
    df_Housing_Market = df_Housing_Market.reset_index()

    return df_Housing_Market


def calc_housing_variables(df_Housing_Market:pd.DataFrame):

    price_list = []

    housing_pct_change = df_Housing_Market["median_sale_price"].pct_change()
    housing_pct_change = 1 + housing_pct_change
    log_returns = np.log(housing_pct_change)
    avg_monthly_return = log_returns.mean()
    variance = log_returns.var()
    drift = avg_monthly_return - (0.5 * variance)
    monthly_std = log_returns.std()

    last_price = df_Housing_Market['median_sale_price'][df_Housing_Market['median_sale_price'].size-1]

    count = 0

    for y in range(num_periods):
        if y == num_periods:
            break

        random_value = monthly_std * norm.ppf(np.random.rand())
        exp_value = np.exp(drift + random_value)

        if y == 0:
            price = last_price * exp_value
        else:
            price = price_list[count] * exp_value
            count += 1

        price_list.append(price)


    housing_futures = pd.DataFrame(index=range(num_periods), data=price_list)
    housing_future_pct_change = housing_futures.pct_change()
    housing_future_pct_change += 1
    sns.relplot(data=housing_future_pct_change, kind="line")
    plt.show()
    sns.relplot(data=housing_futures,kind="line")
    plt.show()


def calc_my_house_futures():

    current_value = 350000



def calc_brown_data(pct_change_data):
    print()

def get_sp500():

    price_list = []
    today = datetime.date.today()
    start = today - datetime.timedelta(days=num_years * DAYS_IN_YEAR)
    sp500 = pdr.get_data_yahoo('^GSPC',start,interval='m')
    return sp500

def calc_my_sp500_futures(brown : Brown):
    sp500_futures = pd.DataFrame(index=range(num_periods), data=price_list)
    sp500_futures_pct_change = sp500_futures.pct_change()
    sp500_futures_pct_change += 1
    sns.relplot(data=sp500_futures,kind="line")
    plt.show()


def calculate_return(arr):
    """
    Calculates the total return percentage
    :param arr:
    :return return_rate:
    """
    tot_return = arr[num_periods-1] - arr[0]
    return_rate = (tot_return / arr[0]) * 100.0
    return_rate = return_rate / num_years
    return return_rate

def calculate_percentiles(matrix, name):

    percentile_95 = np.percentile(matrix, 95, axis=1)
    print("Avg Annual Return rate for 95 percentile : " + name + " " + str(calculate_return(percentile_95)))

    percentile_50 = np.percentile(matrix, 50, axis=1)
    print("Avg Annual Return rate for 50 percentile : " + name + " " + str(calculate_return(percentile_50)))

    percentile_20 = np.percentile(matrix, 20, axis=1)
    print("Avg Annual Return rate for 20 percentile : " + name + " " + str(calculate_return(percentile_20)))

    percentile_5 = np.percentile(matrix, 5, axis=1)
    print("Avg Annual Return rate for 5 percentile : " + name + " " + str(calculate_return(percentile_5)))

    new_df = pd.DataFrame()
    new_df.insert(0, "5% Success", percentile_95)

    new_df.insert(1, "50% Success", percentile_50)
    new_df.insert(2, "80% Success", percentile_20)
    new_df.insert(3, "95% Success", percentile_5)

    return new_df

def monte_carlo_sim_housing(df, return_dict):

    house_sim_matrix = np.zeros((num_periods, NUM_OF_SIMULATIONS))
    print("In housing monte carlo")
    for x in range(NUM_OF_SIMULATIONS):

        housing_brown = Brown(data=df['median_sale_price'].astype(int), forecast_periods=num_periods)
        housing_brown.calc_all_futures(house_current_value)
        house_sim_matrix[:, x] = housing_brown.forecast_data

    return_dict["housing"] = calculate_percentiles(house_sim_matrix, "housing")
    #avg_pct_change = (return_dict["housing"]["5% Success"][num_periods] - return_dict["housing"]["5% Success"][0]) / num_years
    #print("Housing avg annual pct change for 5% success" + str(avg_pct_change))

    #avg_pct_change = (return_dict["housing"]["50% Success"][num_periods] - return_dict["housing"]["50% Success"][0]) / num_years
    #print("Housing avg annual pct change for 50% success" + str(avg_pct_change))

    #avg_pct_change = (return_dict["housing"]["80% Success"][num_periods] - return_dict["housing"]["80% Success"][0]) / num_years
    #print("Housing avg annual pct change for 80% success" + str(avg_pct_change))

    #avg_pct_change = (return_dict["housing"]["95% Success"][num_periods] - return_dict["housing"]["95% Success"][0]) / num_years
    #print("Housing avg annual pct change for 95% success" + str(avg_pct_change))


def monte_carlo_sim_sp500(df, return_dict):


    sp500_sim_matrix = np.zeros((num_periods, NUM_OF_SIMULATIONS))
    print("In sp500 monte carlo")
    for x in range(NUM_OF_SIMULATIONS):

        sp500_brown = Brown(data=df['Adj Close'].astype(int), forecast_periods=num_periods)
        sp500_brown.calc_all_futures(200000)
        sp500_sim_matrix[:, x] = sp500_brown.forecast_data

    return_dict["sp500"] = calculate_percentiles(sp500_sim_matrix, "sp500")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    df_dict = {}
    brown_dict = {}



    matrix = pd.DataFrame()

    df_sp500 = get_sp500()
    df_housing = get_housing_market()

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    p1 = multiprocessing.Process(target=monte_carlo_sim_sp500, args=(df_sp500, return_dict))
    p2 = multiprocessing.Process(target=monte_carlo_sim_housing, args=(df_housing,return_dict))
    jobs.append(p1)
    jobs.append(p2)
    p1.start()
    p2.start()

    p1.join()
    p2.join()

    print("P1 is alive?: " + str(p1.is_alive()))
    print("P2 is alive?: " + str(p2.is_alive()))

    print(return_dict)



    #graph_futures(num_periods, return_dict["sp500"], "SP500 percentiles")
    #graph_futures(num_periods, return_dict["housing"], "Housing Percentiles")

    multi = MultiIndex.from_frame(df_dict)

    #df_housing_market = get_housing_market()
    #housing_brown = Brown(data = df_housing_market, column_name='median_sale_price', forecast_periods=num_periods)
    #housing_brown.calc_futures(housing_brown.df['median_sale_price'][housing_brown.df['median_sale_price'].size-1])
    #housing_brown.graph_futures()


    print("test")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
