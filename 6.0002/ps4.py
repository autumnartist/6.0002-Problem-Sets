# Problem Set 4
# Name: Autumn Artist
# Collabortors: 
# Time: 10 hours

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.interpolate import interp1d

#####################
# Begin helper code #
#####################

def calculate_std(upper, mean):
    """
	Calculate standard deviation based on the upper 95th percentile

	Args:
		upper: a 1-d numpy array with length N, representing the 95th percentile
            values from N data points
		mean: a 1-d numpy array with length N, representing the mean values from
            the corresponding N data points

	Returns:
		a 1-d numpy array of length N, with the standard deviation corresponding
        to each value in upper and mean
	"""
    return (upper-mean)/st.norm.ppf(.95)

def interp(target_year, input_years, years_data):
    """
	Interpolates data for a given year, based on the data for the years around it

	Args:
		target_year: an integer representing the year which you want the predicted
            sea level rise for
		input_years: a 1-d numpy array that contains the years for which there is data
		    (can be thought of as the "x-coordinates" of data points)
        years_data: a 1-d numpy array representing the current data values
            for the points which you want to interpolate, eg. the SLR mean per year data points
            (can be thought of as the "y-coordinates" of data points)

	Returns:
		the interpolated predicted value for the target year
	"""
    return np.interp(target_year, input_years, years_data, right=-99)

def load_slc_data():
    """
	Loads data from sea_level_change.csv and puts it into a numpy array

	Returns:
		a length 3 tuple of 1-d numpy arrays:
		    1. an array of years as ints
		    2. an array of 2.5th percentile sea level rises (as floats) for the years from the first array
		    3. an array of 97.5th percentile of sea level rises (as floats) for the years from the first array
        eg.
            (
                [2020, 2030, ..., 2100],
                [3.9, 4.1, ..., 5.4],
                [4.4, 4.8, ..., 10]
            )
            can be interpreted as:
                for the year 2020, the 2.5th percentile SLR is 3.9ft, and the 97.5th percentile would be 4.4ft.
	"""
    df = pd.read_csv('sea_level_change.csv')
    df.columns = ['Year','Lower','Upper']
    return (df.Year.to_numpy(),df.Lower.to_numpy(),df.Upper.to_numpy())

###################
# End helper code #
###################


##########
# Part 1 #
##########

def predicted_sea_level_rise(show_plot=False):
    """
	Creates a numpy array from the data in sea_level_change.csv where each row
    contains a year, the mean sea level rise for that year, the 2.5th percentile
    sea level rise for that year, the 97.5th percentile sea level rise for that
    year, and the standard deviation of the sea level rise for that year. If
    the year is between 2020 and 2100 and not included in the data, the values
    for that year should be interpolated. If show_plot, displays a plot with
    mean and the 95%, assuming sea level rise follows a linear trend.

	Args:
		show_plot: displays desired plot if true

	Returns:
		a 2-d numpy array with each row containing a year in order from 2020-2100
        inclusive, mean, the 2.5th percentile, 97.5th percentile, and standard
        deviation of the sea level rise for the given year
	"""
    sea_level = []
    #tuple: year, 25% and 97.5%
    info = load_slc_data()
    #info from the data
    info_years = info[0]
    p25 = info[1]
    p97 = info[2]
    means = []
    for i in range(len(p25)):
        means.append((p25[i]+p97[i])/2.0)
    
    #list of years to compare
    years = []
    for i in range(2020, 2101):
        years.append(i)
    #mean given by the average limits
    index = 0
    for year in years:
        if index < len(info_years) and year in info_years:
            percent_25 = p25[index]
            percent_97 = p97[index]
        else:
            percent_25 = interp(year, info_years, p25)
            percent_97 = interp(year, info_years, p97)
        ##how to find the actual mean
        
        mean = interp(year, info_years, means)
        #mean = (percent_25+percent_97)/2.0
        SD = calculate_std(percent_97, mean)
        sea_level.append([year, mean, percent_25, percent_97, SD])
        index += 1
        
    #GRAPHING?
    if show_plot:
        #y = mean
        #x = year
        
        mean_axis = []
        years = []
        axis_25 = []
        axis_97 = []
        for i in sea_level:
            mean_axis.append(i[1])
            years.append(i[0])
            axis_25.append(i[2])
            axis_97.append(i[3])
                
        
        plt.ylim(0, 10)
        plt.xlim(2020, 2100)
        plt.xlabel("Year")
        plt.ylabel("Projected annual mean water level (ft)")
        plt.plot(years, mean_axis, color='green', label = 'Mean')
        plt.plot(years, axis_25, color='orange', linestyle='dashed', label = 'Lower')
        plt.plot(years, axis_97, color='blue', linestyle='dashed', label = 'Upper')
        plt.legend()
        
    #return array of arrays - [year in order from 2020-2100, mean, 2.5%, 97.5% SD]
    return np.array(sea_level)


def simulate_year(data, year, num):
    """
	Simulates the sea level rise for a particular year based on that year's
    mean and standard deviation, assuming a normal distribution.

	Args:
		data: a 2-d numpy array with each row containing a year in order from 2020-2100
            inclusive, mean, the 2.5th percentile, 97.5th percentile, and standard
            deviation of the sea level rise for the given year
		year: the year to simulate sea level rise for
        num: the number of samples you want from this year

	Returns:
		a 1-d numpy array of length num, that contains num simulated values for
        sea level rise during the year specified
	"""
    info = []
    for i in data:
        if i[0] == year:
            info = i
    predictions = []
    for i in range(num):
        predictions.append(np.random.normal(info[1], info[4]))
    return np.array(predictions)

#GO OVER
def plot_mc_simulation(data):
    """
	Runs and plots a Monte Carlo simulation, based on the values in data and
    assuming a normal distribution. Five hundred samples should be generated
    for each year.

	Args:
		data: a 2-d numpy array with each row containing a year in order from 2020-2100
            inclusive, mean, the 2.5th percentile, 97.5th percentile, and standard
            deviation of the sea level rise for the given year
	"""
    #x_axis
    year_info = []
    mean_axis = []
    axis_25 = []
    axis_97 = []
    for i in data:
        year_info.append(i[0])
        mean_axis.append(i[1])
        axis_25.append(i[2])
        axis_97.append(i[3])
    
    
    #y-axis
    water_change = []
    years = []
    
    for year in year_info:
          #list of values
        nums = simulate_year(data, year, 500)
        for i in nums:
            years.append(year)
            water_change.append(i)
    
    plt.ylim(0,14)
    plt.xlim(2020,2100)
    plt.xlabel("Year")
    plt.ylabel("Relative water level change (ft)")
    plt.plot(year_info, mean_axis, color='green', label = 'Mean')
    plt.plot(year_info, axis_25, color='orange', linestyle='dashed', label = 'Lower')
    plt.plot(year_info, axis_97, color='blue', linestyle='dashed', label = 'Upper')
    plt.scatter(years, water_change, color='grey', s=5)
    plt.legend()

##########
# Part 2 #
##########

def water_level_est(data):
    """
	Simulates the water level for all years in the range 2020 to 2100, inclusive.

	Args:
		data: a 2-d numpy array with each row containing a year in order from 2020-2100
            inclusive, mean, the 2.5th percentile, 97.5th percentile, and standard
            deviation of the sea level rise for the given year

	Returns:
		a list of simulated water levels for each year, in the order in which
        they would occur temporally
	"""
    water_levels = []
    for i in data:
        water_levels.append(simulate_year(data, i[0], 1))
    return water_levels


def repair_only(water_level_list, water_level_loss_no_prevention, house_value=400000):
    """
	Simulates the water level for all years in the range 2020 to 2100, inclusive,
    and calculates damage costs in 1000s resulting from a particular water level
    for each year dependent on a repair only strategy, where you would only pay
    to repair damage that already happened.

    The specific damage cost can be calculated using the numpy array
    water_level_loss_no_prevention, where each water level corresponds to the
    percent of property that is damaged.

    The repair only strategy is as follows:
        1) If the water level is less than or equal to 5ft, the cost is 0.
        2) If the water level is between 5ft and 10ft, the cost is the
           house_value times the percentage of property damage for that water
           level. If the water level is not an integer value, the percentage
           should be interpolated.
        3) If the water level is at least 10ft, the cost is the entire value of
           the house.

	Args:
		water_level_list: list of simulated water levels for 2020-2100
        water_level_loss_no_prevention: a 2-d numpy array where the first column is
            the SLR levels and the second column is the corresponding property damage expected
            from that water level with no flood prevention (as an integer percentage)
        house_value: the value of the property we are estimating cost for

	Returns:
		an list of damage costs in 1000s, in the order in which the costs would
        be incurred temporally
	"""
    costs = []
    #damage if the water level is an int
    damage = {5:0, 6:0.10, 7:0.25, 8:0.45, 9:0.75, 10:1}
    #water level loss no prevention lists
    slr = []
    pd = []
    for i in water_level_loss_no_prevention:
        slr.append(i[0])
        pd.append(i[1])
   
    #i is water level
    for i in water_level_list:
        #if less than 5, no cost
        if i <= 5:
            cost = 0
            #if less than 10 (exclusive)
        elif i < 10:
            if i in damage.keys():
                percent_damage = damage[i]
            else:
                percent_damage = interp1d(slr, pd)(i)
            cost = house_value*percent_damage
        else:
            cost = house_value
        costs.append(cost/100000)
    return costs


def wait_a_bit(water_level_list, water_level_loss_no_prevention, water_level_loss_with_prevention, house_value=400000, cost_threshold=100000):
    """
	Simulates the water level for all years in the range 2020 to 2100, inclusive,
    and calculates damage costs in 1000s resulting from a particular water level
    for each year dependent on a wait a bit to repair strategy, where you start
    flood prevention measures after having a year with an excessive amount of
    damage cost.

    The specific damage cost can be calculated using the numpy array
    water_level_loss_no_prevention and water_level_loss_with_prevention, where
    each water level corresponds to the percent of property that is damaged.
    You should be using water_level_loss_no_prevention when no flood prevention
    measures are in place, and water_level_loss_with_prevention when there are
    flood prevention measures in place.

    Flood prevention measures are put into place if you have any year with a
    damage cost above the cost_threshold.

    The wait a bit to repair only strategy is as follows:
        1) If the water level is less than or equal to 5ft, the cost is 0.
        2) If the water level is between 5ft and 10ft, the cost is the
           house_value times the percentage of property damage for that water
           level, which is affected by the implementation of flood prevention
           measures. If the water level is not an integer value, the percentage
           should be interpolated.
        3) If the water level is at least 10ft, the cost is the entire value of
           the house.

	Args:
		water_level_list: list of simulated water levels for 2020-2100
        water_level_loss_no_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with no flood prevention
        water_level_loss_with_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with flood prevention
        house_value: the value of the property we are estimating cost for
        cost_threshold: the amount of cost incurred before flood prevention
            measures are put into place

	Returns:
		an list of damage costs in 1000s, in the order in which the costs would
        be incurred temporally
	"""
    costs = repair_only(water_level_list, water_level_loss_no_prevention, house_value)
    cost = 0
    #water level loss with prevention lists
    slr = []
    pd = []
    for j in water_level_loss_with_prevention:
        slr.append(j[0])
        pd.append(j[1])
    #finding when we should take prevention measures
    year = 0
    while costs[year]*1000 <= cost_threshold:
        year += 1
 
    damage = {5:0, 6:0.05, 7:0.15, 8:0.30, 9:0.70, 10:1} 
    
    #beginning at the year after the one where we find out we need prevention
    #start implementing prevention
    for i in range((year+1), len(costs)):
        if water_level_list[i] <=5:
            cost = 0
        elif water_level_list[i] < 10:
            if i in damage.keys():
                percent_damage = damage[i]
            else:
                percent_damage = interp1d(slr, pd)(water_level_list[i])
            cost = house_value*percent_damage
        else:
            cost = house_value
        costs[i] = cost/100000 
    return costs



def prepare_immediately(water_level_list, water_level_loss_with_prevention, house_value=400000):
    """
	Simulates the water level for all years in the range 2020 to 2100, inclusive,
    and calculates damage costs in 1000s resulting from a particular water level
    for each year dependent on a prepare immediately strategy, where you start
    flood prevention measures immediately.

    The specific damage cost can be calculated using the numpy array
    water_level_loss_with_prevention, where each water level corresponds to the
    percent of property that is damaged.

    The prepare immediately strategy is as follows:
        1) If the water level is less than or equal to 5ft, the cost is 0.
        2) If the water level is between 5ft and 10ft, the cost is the
           house_value times the percentage of property damage for that water
           level, which is affected by the implementation of flood prevention
           measures. If the water level is not an integer value, the percentage
           should be interpolated.
        3) If the water level is at least 10ft, the cost is the entire value of
           the house.

	Args:
		water_level_list: list of simulated water levels for 2020-2100
        water_level_loss_with_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with flood prevention
        house_value: the value of the property we are estimating cost for

	Returns:
		an list of damage costs in 1000s, in the order in which the costs would
        be incurred temporally
	"""
    costs = []
    #damage if the water level is an int
    damage = {5:0, 6:0.05, 7:0.15, 8:0.30, 9:0.70, 10:1} 
    #water level loss no prevention lists
    slr = []
    pd = []
    for i in water_level_loss_with_prevention:
        slr.append(i[0])
        pd.append(i[1])
   
    #i is water level
    for i in water_level_list:
        #if less than 5, no cost
        if i <= 5:
            cost = 0
            #if less than 10 (exclusive)
        elif i < 10:
            if i in damage.keys():
                percent_damage = damage[i]
            else:
                percent_damage = interp1d(slr, pd)(i)
            cost = house_value*percent_damage
        else:
            cost = house_value
        costs.append(cost/100000)
    return costs


def plot_strategies(data, water_level_loss_no_prevention, water_level_loss_with_prevention, house_value=400000, cost_threshold=100000):
    """
	Runs and plots a Monte Carlo simulation of all of the different preparation
    strategies, based on the values in data and assuming a normal distribution.
    Five hundred samples should be generated for each year.

	Args:
		data: a 2-d numpy array with each row containing a year in order from 2020-2100
            inclusive, the 5th percentile, 95th percentile, mean, and standard
            deviation of the sea level rise for the given year
        water_level_loss_no_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with no flood prevention
        water_level_loss_with_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with flood prevention
        house_value: the value of the property we are estimating cost for
        cost_threshold: the amount of cost incurred before flood prevention
            measures are put into place
	"""
    #info for predicting
    water_level_list = water_level_est(data)
    repair = repair_only(water_level_list, water_level_loss_no_prevention, house_value)
    wait = wait_a_bit(water_level_list, water_level_loss_no_prevention, water_level_loss_with_prevention, house_value, cost_threshold)
    prep = prepare_immediately(water_level_list, water_level_loss_with_prevention, house_value=400000)
    years_info = []
    for i in data:
        years_info.append(i[0])
    
    #x_axis: years
    years = []
    #y_axis: damage
    repair_damage = []
    wait_damage = []
    prep_damage = []
    for year in years_info:
        nums = simulate_year(repair, year, 500)
        for i in nums:
            years.append(year)
            repair_damage.append(i)
        nums1 = simulate_year(wait, year, 500)
        for i in nums1:
            wait_damage.append(i)
        nums2 = simulate_year(prep, year, 500)
        for i in nums2:
            prep_damage.append(i)
        
    #graphing
    #Repair only
    plt.scatter(years, repair_damage, color = 'green')
    plt.plot(years, repair_damage, color = 'green', label = 'Repair-Only Scenario')
    #Wait a bit
    plt.scatter(years, wait_damage, color = 'blue')
    plt.plot(years, wait_damage, color = 'blue', label = 'Wait-A-Bit Scenario')
    #Prepare Immediately 
    plt.scatter(years, prep_damage, color = 'red')
    plt.plot(years, prep_damage, color = 'red', lebel = 'Prepare-Immediately Scenario')
    plt.legend()


if __name__ == '__main__':
    data = predicted_sea_level_rise()
    
    water_level_loss_no_prevention = np.array([[5, 6, 7, 8, 9, 10], [0, 10, 25, 45, 75, 100]]).T
    water_level_loss_with_prevention = np.array([[5, 6, 7, 8, 9, 10], [0, 5, 15, 30, 70, 100]]).T
    plot_mc_simulation(data)
    plot_strategies(data, water_level_loss_no_prevention, water_level_loss_with_prevention)
    