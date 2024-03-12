#Import necessary modules
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
from statannot import add_stat_annotation
from datetime import datetime

#load dataset
cars = pd.read_csv("swiss_cars.csv", index_col = 0)

#Get information about the datatypes in the dataset
cars.info()



#Numpy arrays are much more versatile then lists 

#Standard python list object 
numbers = [0,0,1,2,3,0,0,4,5,0,6]
print("Before removal",numbers)
#Task 1: remove zeros from the list
while 0 in numbers:
    numbers.remove(0)
        
print("After removal",numbers)


#With numpy you can do boolean indexing which is very useful for large datasets 
numbers = np.array([0,0,1,2,3,0,0,4,5,0,6])
nonzero = numbers > 0
print("Before removal",numbers)
print("Boolean index",nonzero)

numbers = numbers[nonzero]
print("After removal",numbers)



#Exercise 1: Create a list of values that only contain the prices of skoda superb

prices = list(cars.actual_price)
models = list(cars.model)

prices_superb = "???"



#Task 2: convert strings in the data to numbers
#Standard method
numbers = [0,"0",1,2,"3",0,0,"4",5,"0",6]
print("Before removal",numbers)

for i in range(len(numbers)): 
    if type(numbers[i]) == str: 
        numbers[i] = int(numbers[i])

print("After removal",numbers)


#With numpy  Method 1
numbers = np.array([0,"0",1,2,"3",0,0,"4",5,"0",6])
print("Before correction",numbers)

numbers =numbers.astype(int)

print("After correction",numbers)



#Exercise 2: Remove or impute non-numerical values in the superb prices array

prices_superb = np.array(prices)[np.array(models) == "superb"]
prices_superb = np.where(prices_superb == "NO VALUE", 0, prices_superb).astype(int)






#Pandas 
cars.actual_price = np.where(cars.actual_price == "NO VALUE", 0, cars.actual_price).astype(int) 
#Indexes and columns 
cars.index
cars.columns

#Slicing DataFrames with .loc and .iloc
#Examples with .loc
cars.loc[1, "actual_price"] 
#Examples with .iloc, the first entry first column
cars.iloc[0,0]
#Examples with .iloc, the first entry second column
cars.iloc[0,1]
#First ten entries and the second column 
cars.iloc[:10, 1]
#10-20 entries and the 2-3 column 
cars.iloc[10:20, 1:3]




#Task 1: Create a new dataframe for only Ford cars
ford = cars.loc[cars.model == "ford", :]

#Task 2: Create a new dataframe for bmws where the price is bigger than 50kCHF
bmw_20k = cars.loc[(cars.actual_price > 50000) & (cars.type == "bmw"), :]

#Task 3: Change the registration column to numerical format

#Epoch time for today
datetime.timestamp(datetime.strptime("2024-03-12", "%Y-%m-%d"))


#Epoch time zero (seconds since 00:00:00 UTC on 1 January 1970)
datetime.timestamp(datetime.strptime("1970-01-01", "%Y-%m-%d"))

def to_timestamp(date):
    return datetime.timestamp(datetime.strptime(date, "%Y-%m-%d"))



#Change the registration date to timestamps
cars.registration = cars.registration.apply(to_timestamp)

#Exercise 1: convert the seconds to days

cars.registration = cars.registration / (60 * 60 * 24)



#Plotting and statistics, Matplotlib and Seaborn

#plot the distribution of prices

plt.hist(cars.actual_price, bins = 50)
plt.show()
plt.close()

#Scatter plot the actual and predicted price 
plt.scatter(cars.actual_price, cars.predicted_price, s = 1, alpha = 0.05, color = "k")
plt.show()
plt.close()

#Scatter plot price and registration on skoda

golf = cars.loc[cars.model == "golf"]

plt.scatter(golf.actual_price, golf.registration, s = 5, alpha = 0.5, color = "k")
plt.show()
plt.close()

#plot the relationship between price and km driven
plt.scatter(golf.actual_price, golf.km_driven, s = 5, alpha = 0.5, color = "k")
plt.show()
plt.close()


#Saving a plot to a file, you can have many file formats like .png, .pdf, .eps
plt.scatter(golf.actual_price, golf.km_driven, s = 5, alpha = 0.5, color = "k")
plt.tight_layout()
plt.savefig("golf_price_km.png")


#Plot customization options

fig, ax = plt.subplots(1,2, figsize = (12,6))
ax[0].scatter(golf.actual_price, golf.registration, s = 5, alpha = 0.5, c = golf.km_driven, cmap = "hot")
ax[1].scatter(golf.actual_price, golf.km_driven, s = 5, alpha = 0.5, c = golf.registration, cmap = "hot")

ax[0].set_xlabel("Price (EUR)", fontsize = 15)
ax[0].set_ylabel("Registration (timestamp)", fontsize = 15)


ax[1].set_xlabel("Price (EUR)", fontsize = 15)
ax[1].set_ylabel("Km Driven", fontsize = 15)

ax[0].set_xscale("log")
ax[1].set_xscale("log")

ax[0].tick_params(axis = "x", labelsize = 15)
ax[0].tick_params(axis = "y", labelsize = 15)
ax[1].tick_params(axis = "x", labelsize = 15)
ax[1].tick_params(axis = "y", labelsize = 15)

sns.despine()
plt.tight_layout()


#Scipy: Do statistical tests
#Seaborn: similar to matplotlib but more convenient if you are working with pandas dataframes 
#Statannot: add statistics to plot

#Task plot the prices for all skoda models

skoda = cars.loc[cars.type == "skoda"]
order = skoda.groupby("model").actual_price.agg("mean").sort_values().index

fig, ax = plt.subplots()
sns.boxplot(data = skoda, x ="model", y = "actual_price", order = order, ax = ax)

test_results = add_stat_annotation(ax = ax, data = skoda, x = "model", y= "actual_price", test = "Mann-Whitney", box_pairs = [("yeti", "fabia"), ("yeti", "superb"), ("scala", "kamiq")], order = order)




#Final exercise: statistically compare the prediction accuracy for all car models
























