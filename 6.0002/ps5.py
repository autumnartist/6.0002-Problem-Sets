# -*- coding: utf-8 -*-
# Problem Set 5: Modeling Temperature Change
# Name: Autumn Artist
# Collaborators: Office Hours
# Time: 10

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import re
from sklearn.cluster import KMeans
import collections

# cities in our weather data
CITIES = [
    'BOSTON',
    'SEATTLE',
    'SAN DIEGO',
    'PHOENIX',
    'LAS VEGAS',
    'CHARLOTTE',
    'DALLAS',
    'BALTIMORE',
    'LOS ANGELES',
    'MIAMI',
    'NEW ORLEANS',
    'ALBUQUERQUE',
    'PORTLAND',
    'SAN FRANCISCO',
    'TAMPA',
    'NEW YORK',
    'DETROIT',
    'ST LOUIS',
    'CHICAGO'
]

TRAIN_INTERVAL = range(1961, 2000)
TEST_INTERVAL = range(2000, 2017)

##########################
#    Begin helper code   #
##########################

def standard_error_over_slope(x, y, estimated, model):
    """
    For a linear regression model, calculate the ratio of the standard error of
    this fitted curve's slope to the slope. The larger the absolute value of
    this ratio is, the more likely we have the upward/downward trend in this
    fitted curve by chance.

    Args:
        x: a 1-d numpy array with length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d numpy array of values estimated by a linear
            regression model
        model: a numpy array storing the coefficients of a linear regression
            model

    Returns:
        a float for the ratio of standard error of slope to slope
    """
    assert len(y) == len(estimated)
    assert len(x) == len(estimated)
    EE = ((estimated - y)**2).sum()
    var_x = ((x - x.mean())**2).sum()
    SE = np.sqrt(EE/(len(x)-2)/var_x)
    return SE/model[0]

# KMeans class not required until Problem 7
class KMeansClustering(KMeans):

    def __init__(self, data, k):
        super().__init__(n_clusters=k, random_state=0)
        self.fit(data)
        self.labels = self.predict(data)

    def get_centroids(self):
        'return np array of shape (n_clusters, n_features) representing the cluster centers'
        return self.cluster_centers_

    def get_labels(self):
        'Predict the closest cluster each sample in data belongs to. returns an np array of shape (samples,)'
        return self.labels

    def total_inertia(self):
        'returns the total inertia of all clusters, rounded to 4 decimal points'
        return round(self.inertia_, 4)



class Dataset(object):
    """
    The collection of temperature records loaded from given csv file
    """
    def __init__(self, filename):
        """
        Initialize a Dataset instance, which stores the temperature records
        loaded from a given csv file specified by filename.

        Args:
            filename: name of the csv file (str)
        """
        self.rawdata = {}

        f = open(filename, 'r')
        header = f.readline().strip().split(',')
        for line in f:
            items = line.strip().split(',')

            date = re.match('(\d\d\d\d)(\d\d)(\d\d)', items[header.index('DATE')])
            year = int(date.group(1))
            month = int(date.group(2))
            day = int(date.group(3))

            city = items[header.index('CITY')]
            temperature = float(items[header.index('TEMP')])
            if city not in self.rawdata:
                self.rawdata[city] = {}
            if year not in self.rawdata[city]:
                self.rawdata[city][year] = {}
            if month not in self.rawdata[city][year]:
                self.rawdata[city][year][month] = {}
            self.rawdata[city][year][month][day] = temperature

        f.close()

    def get_daily_temps(self, city, year):
        """
        Get the daily temperatures for the given year and city.

        Args:
            city: city name (str)
            year: the year to get the data for (int)

        Returns:
            a 1-d numpy array of daily temperatures for the specified year and
            city
        """
        temperatures = []
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        for month in range(1, 13):
            for day in range(1, 32):
                if day in self.rawdata[city][year][month]:
                    temperatures.append(self.rawdata[city][year][month][day])
        return np.array(temperatures)

    def get_temp_on_date(self, city, month, day, year):
        """
        Get the temperature for the given city at the specified date.

        Args:
            city: city name (str)
            month: the month to get the data for (int, where January = 1,
                December = 12)
            day: the day to get the data for (int, where 1st day of month = 1)
            year: the year to get the data for (int)

        Returns:
            a float of the daily temperature for the specified date and city
        """
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year {} is not available".format(year)
        assert month in self.rawdata[city][year], "provided month is not available"
        assert day in self.rawdata[city][year][month], "provided day is not available"
        return self.rawdata[city][year][month][day]

##########################
#    End helper code     #
##########################

    def calculate_annual_temp_averages(self, cities, years):
        """
        For each year in the given range of years, computes the average of the
        annual temperatures in the given cities.

        Args:
            cities: a list of the names of cities to include in the average
                annual temperature calculation
            years: a list of years to evaluate the average annual temperatures at

        Returns:
            a 1-d numpy array of floats with length = len(years). Each element in
            this array corresponds to the average annual temperature over the given
            cities for a given year.
        """
        
        # NOTE: TO BE IMPLEMENTED IN PART 4B OF THE PSET
        info = Dataset("data.csv")
        avg = []
        for year in years:
            #find average temp in cities
            #find average temp in EACH city
            temps = []
            for city in cities:
                sum_temps = 0
                for i in info.get_daily_temps(city, year):
                    sum_temps += i
                #Taking into account leap years --> 366 days
                if year%4 == 0:
                    avg_temp = sum_temps/366.0
                else:
                    avg_temp = sum_temps/365.0
                temps.append(avg_temp)
            #find average overall
            sum_temps = 0
            for i in temps:
                sum_temps += i
            avg_temp = sum_temps/len(temps)
            #add to array
            avg.append(avg_temp)
        return np.array(avg)


def linear_regression(x, y):
    """
    Calculates a linear regression model for the set of data points.

    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points

    Returns:
        (m, b): A tuple containing the slope and y-intercept of the regression line,
                both of which are floats.
    """
    #Starting variables needed
    avg_x = 0.0
    avg_y = 0.0
    m = 0.0
    m_num = 0.0
    m_denom = 0.0
    b = 0.0
    #finding average
    for i in range(x.size):
        avg_x += x[i]
        avg_y += y[i]
    #average x and y
    avg_x /= x.size
    avg_y /= y.size
    
    #Finding m
    for i in range(x.size):
        m_num += ((x[i]-avg_x)*(y[i]-avg_y))
        m_denom += ((x[i]-avg_x)**2)
    
    m = m_num/m_denom
    b = avg_y-(m*avg_x)
    return (m, b)

def squared_error(x, y, m, b):
    '''
    Calculates the squared error of the linear regression model given the set
    of data points.

    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points
        m: The slope of the regression line
        b: The y-intercept of the regression line


    Returns:
        a float for the total squared error of the regression evaluated on the
        data set
    '''
    #find predicted values of y
    predicted_y = []
    for x in x:
        pred_y = float(m*x+b)
        predicted_y.append(pred_y)
    #findinf the squared error
    y_sum = 0.0
    for i in range(len(predicted_y)):
        y_sum += ((predicted_y[i]-y[i])**2)
    
    return y_sum
        

def generate_polynomial_models(x, y, degrees):
    """
    Generates a list of polynomial regression models with degrees specified by
    degrees for the given set of data points

    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points
        degrees: a list of integers that correspond to the degree of each polynomial
            model that will be fit to the data

    Returns:
        a list of numpy arrays, where each array is a 1-d numpy array of coefficients
        that minimizes the squared error of the fitting polynomial
    """
    coeffs = []
    for degree in degrees:
        coeffs.append(np.array(np.polyfit(x, y, degree)))
    
    return coeffs

def evaluate_models(x, y, models, display_graphs=False):
    """
    For each regression model, compute the R-squared value for this model and
    if display_graphs is True, plot the data along with the best fit curve.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (i.e. the model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        Degree of your regression model,
        R-squared of your model evaluated on the given data points,
        and standard error/slope (if this model is linear).

    R-squared and standard error/slope should be rounded to 4 decimal places.

    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a numpy array storing the coefficients of
            a polynomial
        display_graphs: A boolean whose value specifies if the graphs should be
            displayed

    Returns:
        A list holding the R-squared value for each model
    """
    #list of predicted y values
    pred_y = []
    y_vals = []
    #iterate through models
    for model in models:
        #get a polynomial
        eq = np.poly1d(model)
        #find predicted y's based on the polynomial and x vals
        eq_y = []
        for i in x:
            eq_y.append(eq(i))
            #y values for display_graph, red solid line
            y_vals.append(eq(i))
        eq_y = np.array(eq_y)
        
        pred_y.append(r2_score(y, eq_y))
        
        

    #plotting the data
    if display_graphs:
        plt.scatter(x, y, color='blue', s=5)
        plt.plot(x, eq_y, color = 'red')
        plt.title("Effect of degree of regression model on evaluated r-squared values of given data")
        plt.xlabel("Degree")
        plt.ylabel("R-squared Value")
        plt.show()
    return pred_y
    

def get_max_trend(x, y, length, positive_slope):
    """
    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points
        length: the length of the interval
        positive_slope: a boolean whose value specifies whether to look for
            an interval with the most extreme positive slope (True) or the most
            extreme negative slope (False)

    Returns:
        a tuple of the form (i, j, m) such that the application of linear (deg=1)
        regression to the data in x[i:j], y[i:j] produces the most extreme
        slope m, with the sign specified by positive_slope and j-i = length.

        In the case of a tie, it returns the first interval. For example,
        if the intervals (2,5) and (8,11) both have slope 3.1, (2,5,3.1) should be returned.

        If no intervals matching the length and sign specified by positive_slope
        exist in the dataset then return None
    """
    #initial range of x and y
    start_x = []
    start_y = []
    for i in range(length):
        start_x.append(x[i])
        start_y.append(y[i])
    start_x = np.array(start_x)
    start_y = np.array(start_y)
    #initial slope given from initial x and y
    slope = generate_polynomial_models(start_x, start_y, [1])[0][0]
    start = 0
    #+1 because it needs to include the last elements
    for i in range(x.size-length+1):
        x1 = []
        y1 = []
        for j in range(i, i+length):
            x1.append(x[j])
            y1.append(y[j])
        x1 = np.array(x1)
        y1 = np.array(y1)
        #slope to compare
        slope1 = generate_polynomial_models(x1, y1, [1])[0][0]
        epsilon = abs(slope1-slope)
        #Takes into account rounding errors
        if epsilon > 1e-8:
            if positive_slope:
                if slope < slope1:
                    slope = slope1
                    start = i
            #finding minimum slope
            else:
                if slope > slope1:
                    slope = slope1
                    start = i
    if slope >=0 and positive_slope==False:
        return None
    if slope <0 and positive_slope:
        return None
    
    return (start, start+length, slope)


def get_all_max_trends(x, y):
    """
    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points

    Returns:
        a list of tuples of the form (i,j,m) such that the application of linear
        regression to the data in x[i:j], y[i:j] produces the most extreme
        positive OR negative slope m, and j-i=length.

        The returned list should have len(x) - 1 tuples, with each tuple representing the
        most extreme slope and associated interval for all interval lengths 2 through len(x).
        If there is no positive or negative slope in a given interval length L (m=0 for all
        intervals of length L), the tuple should be of the form (0,L,None).

        The returned list should be ordered by increasing interval length. For example, the first
        tuple should be for interval length 2, the second should be for interval length 3, and so on.

        If len(x) < 2, return an empty list
    """
    #if the len of x is less than 2, return empty list
    if len(x) < 2:
        return []
    trends = []
    #run through each possible length
    for length in range(2, len(x)+1):
        pos_max = get_max_trend(x, y, length, True)
        neg_max = get_max_trend(x, y, length, False)
        
        #both are none
        if pos_max == None and neg_max == None: 
            trends.append((0, length, None))
        #one is none but the other is insignificant
        elif pos_max == None:
            if abs(neg_max[2]) < 1e-8:
                trends.append((0, length, None))
            else:
                trends.append(neg_max)
        elif neg_max == None:
            if pos_max[2] < 1e-8:
                trends.append((0, length, None))
            else:
                trends.append(pos_max)
        #epsilon being 1e-8 - insignificant if less than epsilon
        elif abs(pos_max[2] - abs(neg_max[2])) > 1e-8:
            if abs(pos_max[2]) > abs(neg_max[2]):
                trends.append(pos_max)
            if abs(pos_max[2]) < abs(neg_max[2]):
                trends.append(neg_max)
        else:
            #if both are equal want the first one
            #one with the lowest starting value
            trends.append(min(pos_max, neg_max))
    return trends



def calculate_rmse(y, estimated):
    """
    Calculate the root mean square error term.

    Args:
        y: a 1-d numpy array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d numpy array of values estimated by the regression
            model

    Returns:
        a float for the root mean square error term
    """
    rmse = 0
    for i in range(y.size):
        rmse += ((y[i] - estimated[i])**2)
    
    rmse /= y.size
    return rmse**(0.5)


def evaluate_rmse(x, y, models, display_graphs=False):
    """
    For each regression model, compute the RMSE for this model and if
    display_graphs is True, plot the test data along with the model's estimation.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        RMSE of your model evaluated on the given data points.

    RMSE should be rounded to 4 decimal places.

    Args:
        x: a 1-d numpy array with length N, representing the x-coordinates of
            the N test data sample points
        y: a 1-d numpy array with length N, representing the y-coordinates of
            the N test data sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a numpy array storing the coefficients of
            a polynomial.
        display_graphs: A boolean whose value specifies if the graphs should be
            displayed

    Returns:
        A list holding the RMSE value for each model
    """
    #list of predicted y values
    rmse = []
    y_vals = []
    #iterate through models
    for model in models:
        #get a polynomial
        eq = np.poly1d(model)
        #find predicted y's based on the polynomial and x vals
        eq_y = []
        for i in x:
            eq_y.append(eq(i))
            #y values for display_graph, red solid line
            y_vals.append(eq(i))
        rmse.append(calculate_rmse(y, eq_y))
    eq_y = np.array(eq_y)
    #plotting the data
    if display_graphs:
        plt.scatter(x, y, color='blue', s=5)
        plt.plot(x,eq_y, color = 'red')
        plt.title("Effect of degree of regression model on RMSE values")
        plt.xlabel("Degree")
        plt.ylabel("RMSE")
        plt.show()
    return rmse

def cluster_cities(cities, years, data, n_clusters):
    '''
    Clusters cities into n_clusters clusters using their average daily temperatures
    across all years in years. Generates a line plot with the average daily temperatures
    for each city. Each cluster of cities should have a different color for its
    respective plots.

    Args:
        cities: a list of the names of cities to include in the average
                daily temperature calculations
        years: a list of years to include in the average daily
                temperature calculations
        data: a Dataset instance
        n_clusters: an int representing the number of clusters to use for k-means

    Note that this part has no test cases, but you will be expected to show and explain
    your plots during your checkoff
    '''
    ##PART 1
    #convert daily temperature data for each city into a numpy array feature vector
    info = Dataset(data)
    city_temps = np.empty(shape=(len(cities), 365), dtype=float)
    #for year in range(1961, 2017):
    for j in range(len(cities)):
        temps = []
        for year in years:
            #creates a list of np arrays
            temps.append(info.get_daily_temps(cities[j], year))
        #Finding average daily temp for EACH year
        #print(temps)
        s = 0 #sum of temps
        day = 0 #keeps track of days
        #iterate through the days
        for day in range(365):
            #iterate through the years on a specific day
            for year in range(len(years)):
                #add the temp
                s+= temps[year][day]
            #avg temp
            s/=len(years)   
            #add temp to specific index
            city_temps[j, day] = s
                
                
    ##PART 2
    kMeans = KMeansClustering(city_temps, n_clusters)
    #creating the labels for the legend
    
    #PLOTS
    plt.xlim(1, 365)
    #plt.ylim(0, 50)
    plt.xlabel("Days of the Year")
    plt.ylabel("Degrees (Celcius)")
    plt.title("Clusters of Daily average Temperatures")
    
    #plot each kluster different color  
    colors = ['red', 'green', 'blue', 'orange']
    for i in range(len(city_temps)):
        color = colors[kMeans.get_labels()[i]]
        plt.plot(range(1,366), city_temps[i], color = color, label = f'Cluster {kMeans.get_labels()[i]}')
    
    #making legend unique
    handles, label = plt.gca().get_legend_handles_labels()
    #Creating a dictionary based on the handles and labels
    labels = dict(zip(label, handles))
    ordered_label = collections.OrderedDict(sorted(labels.items()))
    plt.legend(ordered_label.values(), ordered_label.keys(), loc='upper right', fontsize = 7)
    
    return city_temps


if __name__ == '__main__':
    #pass
    ##################################################################################
    # Problem 4A: DAILY TEMPERATURE
    print("4A")
    temp_info = Dataset('data.csv')
    #x values
    years = np.array(range(1961, 2017))
    
    #y values
    temps = []
    for year in years:
        temps.append(temp_info.get_temp_on_date("BOSTON", 12, 1, year))
    temps = np.array(temps) 
    poly = generate_polynomial_models(years, temps, [1])
    #graphs the data
    pred_ys = evaluate_models(years, temps, poly, True)
    #prints list of r squared values
    print(pred_ys)
    print()
        

    ##################################################################################
    # Problem 4B: ANNUAL TEMPERATURE
    print("4B")
    new_temps = temp_info.calculate_annual_temp_averages(["BOSTON"], years)
    poly2 = generate_polynomial_models(years, new_temps, [1])
    new_pred_ys = evaluate_models(years, new_temps, poly2, True)
    print(new_pred_ys)
    print()
    

    ##################################################################################
    # Problem 5B: INCREASING TRENDS
    print("5B")
    #x years 
    #y annual sea rise in Seattle
    seattle_sea_rise = temp_info.calculate_annual_temp_averages(["SEATTLE"], years)
    #length = 30
    slope = get_max_trend(years, seattle_sea_rise, 30, True)[2]
    increase_trend_y = evaluate_models(years, seattle_sea_rise, [np.array(slope)], True)
    print(increase_trend_y)
    print()
     
    ##################################################################################
    # Problem 5C: DECREASING TRENDS
    print("5C")
    slope1 = get_max_trend(years, seattle_sea_rise, 15, False)[2]
    decreasing_trend_y = evaluate_models(years, seattle_sea_rise, [np.array(slope1)], True)
    print(decreasing_trend_y)
    print()

    ##################################################################################
    # Problem 5D: ALL EXTREME TRENDS
    # Your code should pass test_get_max_trend. No written answer for this part, but
    # be prepared to explain in checkoff what the max trend represents.
    
    
    ##################################################################################
    # Problem 6B: PREDICTING 
    print("6B(i)")
    temp_info = Dataset('data.csv')
    #(i) Creating the model
    train_years = [] 
    for i in range(1961, 2000): 
        train_years.append(i)
    train_years = np.array(train_years)
    #Calculates annual temperature for each city for each training year - array
    TRAIN_INTERVAL = temp_info.calculate_annual_temp_averages(CITIES, train_years)
    #finds polynomial 
    training_polys = generate_polynomial_models(train_years, TRAIN_INTERVAL, [2, 10])
    training_y = evaluate_models(train_years, TRAIN_INTERVAL, training_polys, True)
    print(training_y)
    print()
    
    #(ii) Test the model  
    print("6B(ii)")      
    #Actual numbers to compare
    test_years = []
    for i in range(2000, 2017): 
        test_years.append(i)
    test_years = np.array(test_years)
    #Calculates annual temperature for each city for each testing year - array
    TEST_INTERVAL = temp_info.calculate_annual_temp_averages(CITIES, test_years)
    #evaulate predictions made by TEST_INTERVAL
    test_y = evaluate_models(test_years, TEST_INTERVAL, training_polys, True)
    print(test_y)
    #checks the prediction vs the actual answer
    
    estimated_y = evaluate_rmse(test_years, TEST_INTERVAL, training_polys, True)
    print(estimated_y)
    print()
    
    
    ##################################################################################
    # Problem 7: KMEANS CLUSTERING (Checkoff Question Only)
    print("7")
    years = range(1961, 2017)
    clusters = cluster_cities(CITIES, years, 'data.csv', 4)
    print(clusters)

    ####################################################################################
    
