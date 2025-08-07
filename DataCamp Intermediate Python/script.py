Intermediate Python

Week 1: Matplotlib

----------Basic plots with Matplotlib--------
There are many visualization packages in python, but the mother of them all, is matplotlib. 
You will need its subpackage pyplot. By convention, this subpackage is imported as plt, like this

import matplot.pyplot as plt
year = [1950, 1970, 1990, 2010]
pop = [2.519, 3.692, 5.263, 6.972]
plt.plot(year, pop)  #popualte on a line chart and use the 2 lists as arguements , x and y axis respectively
plt.show() #display the plot

import matplot.pyplot as plt
year = [1950, 1970, 1990, 2010]
pop = [2.519, 3.692, 5.263, 6.972]
plt.scatter(year, pop)  #popualte on a scatter chart 
plt.show() #display the plot

---------Line PLot-----------

# Print the last item from year and pop
print(year[-1])
print(pop[-1])

# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Make a line plot: year on the x-axis, pop on the y-axis
plt.plot(year, pop)

# Display the plot with plt.show()
plt.show()


-----------Line Plot (2): Interpretation---------

Have another look at the plot you created in the previous exercise; 
its shown on the right. Based on the plot, in approximately what year will there be more than ten billion human beings on this planet?

2060

-----------Line plot (3)-----------

# Print the last item of gdp_cap and life_exp
print(gdp_cap[-1])
print(life_exp[-1])

# Make a line plot, gdp_cap on the x-axis, life_exp on the y-axis
plt.plot(gdp_cap,life_exp)


# Display the plot
plt.show()

----------Scatter Plot (1)-----------

# Change the line plot below to a scatter plot
plt.scatter(gdp_cap, life_exp)

# Put the x-axis on a logarithmic scale
plt.xscale('log')

# Show plot
plt.show()

----------Scatter Plot (2)-----------

# Import package
import matplotlib.pyplot as plt

# Build Scatter plot
plt.scatter(pop, life_exp)

# Show plot
plt.show()

----------Histogram-----------
The height of the bar corresponds to the number of data points that fall in this bin.

import matplotlib.pyplot as plt
help(plt.hist)

values = [0, 0.6, 1.4, 1.6, 2.2, 2.5, 2.6, 3.2, 3.5, 3.9, 4.2, 6]
plt.hist(values, bins = 3) #values are divided into 3 bins
plt.show()

------------Build a histogram (1)-----------

# Create histogram of life_exp data
plt.hist(life_exp)

# Display histogram
plt.show()

--------Build a histogram (2): bins--------

# Build histogram with 5 bins
plt.hist(life_exp, bins = 5)

# Show and clean up plot
plt.show()
plt.clf()

# Build histogram with 20 bins
plt.hist(life_exp, bins = 20)

# Show and clean up again
plt.show()
plt.clf()

----------Build a histogram (3): compare--------

# Histogram of life_exp, 15 bins
plt.hist(life_exp, bins = 15)

# Show and clear plot
plt.show()
plt.clf()

# Histogram of life_exp1950, 15 bins
plt.hist(life_exp1950, bins = 15)

# Show and clear plot again
plt.show()
plt.clf()

--------Choose the right plot (1)-----------

Youre a professor teaching Data Science with Python, 
and you want to visually assess if the grades on your exam follow a particular distribution. Which plot do you use?

Histogram

--------Choose the right plot (2)-----------

Youre a professor in Data Analytics with Python, and you want to visually assess if longer answers on exam questions 
lead to higher grades. Which plot do you use?

Scatter plot

-----------Customization-------------

import matplotlib.pyplot as plt 
year = [1950, 1951, 1952,....,2100]
pop = [2.358, 2.57, 2.62,....., 10.85]

plt.plot(year, pop)

#Add more data
year = [1800, 1850, 1900]
pop = [1, 1.262, 1.650]

plt.xlabel('year') # axis labels
plt.ylabel('population')
plt.title('World Population Projections') #add title
plt.yticks([0,2,4,6,8,10], #values the y label should have
			['0','2B','4B','6B','8B','10B'])

plt.show()

------------Labels------------
Its time to customize your own plot. This is the fun part, you will see your plot come to life!
Youre going to work on the scatter plot with world development data: GDP per capita on the x-axis (logarithmic scale), life expectancy on the y-axis.

# Basic scatter plot, log scale
plt.scatter(gdp_cap, life_exp)
plt.xscale('log') 

# Strings
xlab = 'GDP per Capita [in USD]'
ylab = 'Life Expectancy [in years]'
title = 'World Development in 2007'

# Add axis labels
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')

# Add title
plt.title('World Development in 2007')

# After customizing, display the plot
plt.show()

----------Ticks----------

# Scatter plot
plt.scatter(gdp_cap, life_exp)

# Previous customizations
plt.xscale('log') 
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')

# Definition of tick_val and tick_lab
tick_val = [1000, 10000, 100000]
tick_lab = ['1k', '10k', '100k']

# Adapt the ticks on the x-axis
plt.xticks(tick_val,tick_lab)

# After customizing, display the plot
plt.show()

--------Sizes-----------

# Import numpy as np
import numpy as np

# Store pop as a numpy array: np_pop
np_pop = np.array(pop)

# Double np_pop
np_pop = np_pop * 2

# Update: set s argument to np_pop
plt.scatter(gdp_cap, life_exp, s = np_pop)

# Previous customizations
plt.xscale('log') 
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')
plt.xticks([1000, 10000, 100000],['1k', '10k', '100k'])

# Display the plot
plt.show()

----------Colors----------
How did we make the list col you ask? The Gapminder data contains a list 
continent with the continent each country belongs to. 
A dictionary is constructed that maps continents onto colors:

dict = {
    'Asia':'red',
    'Europe':'green',
    'Africa':'blue',
    'Americas':'yellow',
    'Oceania':'black'
}

# Specify c and alpha inside plt.scatter()
plt.scatter(c = col, alpha = 0.8, x = gdp_cap, y = life_exp, s = np.array(pop) * 2)

# Previous customizations
plt.xscale('log') 
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')
plt.xticks([1000,10000,100000], ['1k','10k','100k'])

# Show the plot
plt.show()

----------Additional Customisations----------

# Scatter plot
plt.scatter(x = gdp_cap, y = life_exp, s = np.array(pop) * 2, c = col, alpha = 0.8)

# Previous customizations
plt.xscale('log') 
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')
plt.xticks([1000,10000,100000], ['1k','10k','100k'])

# Additional customizations
plt.text(1550, 71, 'India')
plt.text(5700, 80, 'China')

# Add grid() call
plt.grid(True)

# Show the plot
plt.show()





Week 2: Dictionaries and Pandas

----------Dictionaries, Part 1----------

imagine the following: you work for the World Bank and want to keep track of the population in each country.
You can put the populations in a list.

pop = [30.55, 2.77, 29.21]
countries = ['afganistan', 'Albania', 'Algeria'] #identifies which population belongs to which country
ind_alb = countries.index('albania')
ind_alb
1
pop[ind_alb]
2.77

So we built two lists, and used the index to connect corresponding elements in both lists. 
It worked, but its a pretty terrible approach: its not convenient and not intuitive. 
Wouldnt it be easier if we had a way to connect each country directly to its population, without using an index? 
This is where the dictionary comes into play.

# convert popualtion data into a dictionary

pop = [30.55, 2.77, 39.21]
countries = ['afganistan', 'Albania', 'Algeria']
world = {'afganistan':30.55, 'Albania':2.77, 'Algeria':39.21} # To create the dictionary, you need curly brackets. Next, inside the curly brackets, you have a bunch of what are called key:value pairs. In our case, the keys are the country names, and the values are the corresponding populations.
world['Albania']
2.77

-------------Motivation dictionaries-----------

# Definition of countries and capital
countries = ['spain', 'france', 'germany', 'norway']
capitals = ['madrid', 'paris', 'berlin', 'oslo']

# Get index of 'germany': ind_ger
ind_ger = countries.index('germany')

# Use ind_ger to print out capital of Germany
print(capitals[ind_ger])

-----------Create dictionary-------------

# Definition of countries and capital
countries = ['spain', 'france', 'germany', 'norway']
capitals = ['madrid', 'paris', 'berlin', 'oslo']

# From string in countries and capitals, create dictionary europe
europe = { 'spain':'madrid', 'france':'paris', 'germany':'berlin', 'norway':'oslo' }

# Print europe
print(europe)

----------Access dictionary------------

# Definition of dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'berlin', 'norway':'oslo' }

# Print out the keys in europe
print(europe.keys())

# Print out value that belongs to key 'norway'
print(europe['norway'])

----------------Dictionaries, Part 2--------------

world = {'afganistan':30.55, 'Albania':2.77, 'Algeria':39.21}

we created the dictionary "world", which basically is a set of key value pairs. 
You could easily access the population of Albania, by passing the key in square brackets

world['Albania']
2.77

For this lookup to work properly, the keys in a dictionary should be unique. If you try to add another key:value pair to world with the same key, Albania, for example, 
youll see that the resulting world dictionary still contains three pairs.
The last pair that you specified in the curly brackets was kept in the resulting dictionary.

world = {'afganistan':30.55, 'Albania':2.77, 'Algeria':39.21, 'Albania':2.81}
world
{'afganistan':30.55, 'Albania':2.81, 'Algeria':39.21}

Strings, booleans, integers and floats are immutable objects, but the list for example is mutable, because you can change its contents after it's created. 
Thats why this dictionary, that has all immutable objects as keys, is perfectly valid. This one, however, that uses a list as a key, is not valid, so we get an error. 

{0:'hello', True:'dear', 'two':'world'}
{["just", "to", "test"]: "value"}

add more data to a dictorary that already exists

world['sealand'] = 0.000027
world 
{'afganistan':30.55, 'Albania':2.77, 'Algeria':39.21, 'sealand':2.7e-05}

world['sealand'] = 0.000028 #change sealand to ..28
del(world['sealand']) #delete sealand

List vs Dictonary:

List ---- 
Select, update and remove using square []
Indexed by range of numbers
Collection of values 
order matters
select entire subsets


Dictonary ---- 
Select, update and remove using square []
indexed by unique keys
lookup table with unique keys


-------Dictionary Manipulation (1)---------

# Definition of dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'berlin', 'norway':'oslo' }

# Add italy to europe
europe['italy'] = 'rome'

# Print out italy in europe
print('italy' in europe)

# Add poland to europe
europe['poland'] = 'warsaw'

# Print europe
print(europe)

----------Dictionary Manipulation (2)-------------

# Definition of dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'bonn',
          'norway':'oslo', 'italy':'rome', 'poland':'warsaw',
          'australia':'vienna' }

# Update capital of germany
europe['germany'] = 'berlin'

# Remove australia
del(europe['australia'])

# Print europe
print(europe)


-----------Dictionariception-------------

# Dictionary of dictionaries
europe = { 'spain': { 'capital':'madrid', 'population':46.77 },
           'france': { 'capital':'paris', 'population':66.03 },
           'germany': { 'capital':'berlin', 'population':80.62 },
           'norway': { 'capital':'oslo', 'population':5.084 } }


# Print out the capital of France
print(europe['france']['capital'])

# Create sub-dictionary data
data = {'capital':'rome', 'population':59.83}

# Add data to europe under key 'italy'
europe['italy'] = data

# Print europe
print(europe)


----------pandas, part 1------------

tabular structure, that is, in the form of a table like in a spreadsheet

collected information on the so-called BRICS countries, Brazil, Russia, India, China and South Africa. You can again build a table with this data,
Each row is an observation and represents a country

To start working on this data in Python, youll need some kind of rectangular data structure. 
Thats easy, we already know one! The 2D Numpy array, right?

Your datasets will typically comprise different data types, so we need a tool thats better suited for the job.

To easily and efficiently handle this data, theres the Pandas package. 
Pandas is a high level data manipulation tool developed by Wes McKinney, built on the Numpy package.


dict = {
	'country' :['brazil', 'russia', 'inda', 'china', 'south africa'],
	'capital':['brasillia', 'moscow', 'new delhi', 'beijing', 'pretoria'],
	'area':[8.516, 17.10, 3.286, 1.221]
	'population':[200.4, 143.5, 1252, 1357, 52.98]}

keys(column labels)
values ( data, column by column)

import pandas as pd
brics = pd.dataframe(dict)

bric.index = ['br', 'ru', 'in', 'ch', 'sa']

brics.csv #comma seperated values

brics = pd.read_csv('path/to/brics.csv', index column = 0)


------Dictionary to DataFrame (1)---------

# Pre-defined lists
names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
dr =  [True, False, False, False, True, True, True]
cpc = [809, 731, 588, 18, 200, 70, 45]


# Import pandas as pd
import pandas as pd

# Create dictionary my_dict with three key:value pairs: my_dict
my_dict = {
    'country':names,
    'drives_right': dr,
    'cars_per_cap': cpc}

# Build a DataFrame cars from my_dict: cars
cars = pd.DataFrame(my_dict)

# Print cars
print(cars)


--------------Dictionary to DataFrame (2)--------------

import pandas as pd

# Build cars DataFrame
names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
dr =  [True, False, False, False, True, True, True]
cpc = [809, 731, 588, 18, 200, 70, 45]
cars_dict = { 'country':names, 'drives_right':dr, 'cars_per_cap':cpc }
cars = pd.DataFrame(cars_dict)
print(cars)

# Definition of row_labels
row_labels = ['US', 'AUS', 'JPN', 'IN', 'RU', 'MOR', 'EG']

# Specify row labels of cars
cars.index = row_labels

# Print cars again
print(cars)


----------CSV to DataFrame (1)----------

# Import pandas as pd
import pandas as pd

# Import the cars.csv data: cars
cars = pd.read_csv('cars.csv')

# Print out cars
print(cars)

----------CSV to DataFrame (2)------------

# Import pandas as pd
import pandas as pd

# Fix import by including index_col
cars = pd.read_csv('cars.csv', index_col = 0)

# Print out cars
print(cars)

-----------Pandas, Part 2--------------

Suppose that you only want to select the country column from brics. 
How to do this with square brackets? Well, you type brics, and then the column label inside square brackets.

brics['country']
type(brics['country'])

Okay, so were dealing with a Pandas Series here. 
In a simplified sense, you can think of the Series as a 1-dimensional array that can be labeled, just like the DataFrame. 
Otherwise put, if you paste together a bunch of Series, you can create a DataFrame.

pandas.core.series.Series #1d labelled array
pandas.core.frame.DataFrame

brics[['country', 'capital']]
brics[1:4] #indexing

loc(label-based)
iloc(position based)


brics.loc['Ru'] #you put the label of the row of intered inside square brackets after loc
brics.loc[['Ru']] #to get pandas dataframe we use double brackets
brics.loc[['Ru', 'In', 'Ch']] # add more rows but returns all columns as we havent prespecified which columns we want
brics.loc[['Ru', 'In', 'Ch'], ['country', 'capital']] # specify columns you want to output and rows 

brics.loc[:, ['country', 'capital']] #intersection spans all rows but 2 column as we specified -- oc[:, = all rows

brics.iloc[[1]]  # index number so its position based
brics.iloc[[1,2,3]] 
brics.loc[[1,2,3], [0,1]] #specify position of rows and columns you want to retrieve
brics.loc[:, [0,1]]

-------Square Brackets (1)----------

# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Print out country column as Pandas Series
print(cars['country'])

# Print out country column as Pandas DataFrame
print(cars[['country']]) #use double brackets

# Print out DataFrame with country and drives_right columns
print(cars[['country','drives_right']])

-----------Square Brackets (2)-------------

# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Print out first 3 observations
print(cars[0:3])

# Print out fourth, fifth and sixth observation
print(cars[3:6])

-------loc and iloc (1)-------------

# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Print out observation for Japan
print(cars.loc['JPN'])

# Print out observations for Australia and Egypt
print(cars.loc[['AUS', 'EG']])

---------loc and iloc (2)-------------

# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)
cars
# Print out drives_right value of Morocco
print(cars.loc[['MOR'], ['drives_right']])

# Print sub-DataFrame
print(cars.loc[['RU', 'MOR'], 
                ['country', 'drives_right']]) #selected countries russian and morocco for country that drives right


------------loc and iloc (3)-------------

# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)
cars
# Print out drives_right column as Series
print(cars.loc[:, 'drives_right'])

# Print out drives_right column as DataFrame
print(cars.loc[:, ['drives_right']])

# Print out cars_per_cap and drives_right as DataFrame
print(cars.loc[:, ['cars_per_cap', 'drives_right']])





Week 3 - Logic, Control Flow and Filtering

-------Comparison Operators---------
NumPy recap:
import numpy as np
np_height = np.array([1.73, 1.68, 1.71, 1.89, 1.79])
np_weight = np.array([65.4, 69.2, 63.6, 88.4, 68.7])
bmi = np_eight/np_height**2
bmi

bmi > 23 # gives you an array of bolean values to say whether the figures are either true/false

bmi[bmi>23] #give syou the actual value of the bmi's that meet this condition


---------Equality--------

# Comparison of booleans
True == False

# Comparison of integers
-5*15 != 75

# Comparison of strings
"pyscript" == "PyScript"

# Compare a boolean with an integer
# What happens if you compare booleans and integers?
True == 1

this is true as true corresponds to 1 and false corresponds to 0 so saying 1==1 is true


-------Greater and less than-------

# Comparison of integers
x = -3 * 6
x >= -10

# Comparison of strings
y = "test"
'test' <= y

# Comparison of booleans
True > False

# yes as true-1 and false =0


------Compare arrays----------

# Create arrays
import numpy as np
my_house = np.array([18.0, 20.0, 10.75, 9.50])
your_house = np.array([14.0, 24.0, 14.25, 9.0])

# my_house greater than or equal to 18
print(my_house >= 18)

# my_house less than your_house
print(my_house < your_house)


-------Boolean Operators----------

boolean operators:
The three most common ones are and, or, and not.


x = 12
x > 5 and x < 15
# true

y = 5
y < 7 or y > 13
#True

bmi = np.array([21.852, 20.975, 21.75, 24.747, 21.441])
bmi > 21
bmi < 22
bmi > 21 and b,mi < 22

np.logical_and(bmi > 21, bmi < 22)
np.[logical_and(bmi > 21, bmi < 22)]


-----------and, or, not (1)----------------

# Define variables
my_kitchen = 18.0
your_kitchen = 14.0

# my_kitchen bigger than 10 and smaller than 18?
print(my_kitchen > 10 and my_kitchen < 18)

# my_kitchen smaller than 14 or bigger than 17?
print(my_kitchen < 14 or my_kitchen > 17)

# Double my_kitchen smaller than triple your_kitchen?
print(my_kitchen * 2 < your_kitchen * 3)


-----------and, or, not (2)--------------

To see if you completely understood the boolean operators, have a look at the following piece of Python code:
x = 8
y = 9
not(not(x < 3) and not(y > 14 or y > 10))

What will the result be if you execute these three commands in the IPython Shell?

NB: Notice that not has a higher priority than and and or, it is executed first.

False:

Correct! x < 3 is False. y > 14 or y > 10 is False as well. 
If you continue working like this, simplifying from inside outwards, youll end up with False.


------------Boolean operators with Numpy------------------

# Create arrays
import numpy as np
my_house = np.array([18.0, 20.0, 10.75, 9.50])
your_house = np.array([14.0, 24.0, 14.25, 9.0])

# my_house greater than 18.5 or smaller than 10
print(np.logical_or(my_house > 18.5,
                     my_house < 10))

# Both my_house and your_house smaller than 11
print(np.logical_and(my_house < 11,
                     my_house < 10))

-----------if, elif, else--------------------

comparison operators:
<, >, >=, <=, ==, !=
boolean operators:
and, or, not
conditional statements:
if, else, elif


general format.....
if condition:
	expression


control.py
 z = 4

 if z % 2 == 0:  #True that is if the number is even as there will be no remainder if you divde an even number by 2
 		print('z is even')

z = 4

 if z % 2 == 0:  #True that is if the number is even as there will be no remainder if you divde an even number by 2
 		print('checking' + str(z)) 3 another print statement so output will be checking 4 z is even
 		print('z is even')

output will only occur if z is even, to change this we can put an else statements


z = 5 # this time z = 5

 if z % 2 == 0:  #True that is if the number is even as there will be no remainder if you divde an even number by 2
 		print('z is even')
else:
		print('z is odd')

more customised behaviour by including elif (else if)

z = 3 # this time z = 3

 if z % 2 == 0:  #True that is if the number is even as there will be no remainder if you divde an even number by 2
 		print('z is divisable by 2')
 elif z % 3 == 0:
 		print('z is divisable by 3')
else:
		print('z is not divisable by 2 or 3')


if z =6 which is both divisable by 2 and 3 the first print out will occur as the statement comes to an end.

z = 6 # this time z = 6

 if z % 2 == 0:  #True that is if the number is even as there will be no remainder if you divde an even number by 2
 		print('z is divisable by 2')
 elif z % 3 == 0:
 		print('z is divisable by 3')
else:
		print('z is not divisable by 2 or 3')


--------------Warmup-----------------

To experiment with if and else a bit, have a look at this code sample:

area = 10.0
if(area < 9) :
    print("small") #does not hold so carries on
elif(area < 12) : #holds so it prints out this associated statement
    print("medium")
else :
    print("large")

What will the output be if you run this piece of code in the IPython Shell?

medium

--------------if-----------------

# Define variables
room = "kit"
area = 14.0

# if statement for room
if room == "kit" :
    print("looking around in the kitchen.")

# if statement for area
# Write another if statement that prints out "big place!" if area is greater than 15

if area > 15:
    print("big place!")

--------------add else-----------------

# Define variables
room = "kit"
area = 14.0

# if-else construct for room
if room == "kit" :
    print("looking around in the kitchen.")
else :
    print("looking around elsewhere.")

# if-else construct for area
if area > 15 :
    print("big place!")
else:
    print("pretty small.")

--------------elif----------------

# Define variables
room = "bed"
area = 14.0

# if-elif-else construct for room
if room == "kit" :
    print("looking around in the kitchen.")
elif room == "bed":
    print("looking around in the bedroom.")
else :
    print("looking around elsewhere.")

# if-elif-else construct for area
if area > 15 :
    print("big place!")
elif area > 10:
    print("medium size, nice!")
else :
    print("pretty small.")

------------Filtering pandas DataFrames-----------

import pandas as pd

lets import the BRICS dataset again from the CSV file; here it is.
brics = pd.read_csv('path/to/brics.csv', index_col = 0)

Suppose you now want to keep the countries, so the observations in this case, for which the area is greater than 8 million square kilometers.

identify only area column can do it in 3 ways 

brics['area']
brics.loc[:,'area']
brics.iloc[:,2]

brics['area'] > 8 # will get a series of booleans true/false
is_huge = brics['area'] > 8

brics[is_huge]


Suppose you only want to keep the observations that have an area between 8 and 10 million square kilometers. 
After importing numpy as np, we can use the logical_and() function to create a Boolean Series.

import numpy as np
np.logical_and(brics['area']>8,brics['area']<10) #gives boolean
brics[np.logical_and(brics['area']>8,brics['area']<10)] #gives values

-----------Driving right (1)------------

# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Extract drives_right column as Series: dr
dr = cars['drives_right']

# Use dr to subset cars: sel
sel = cars[dr]

# Print sel
print(sel)

-------------Driving right (2)-------------

# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Convert code to a one-liner
sel = cars[cars['drives_right']]

# Print sel

print(sel)

----------Cars per capita (1)----------

# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Create car_maniac: observations that have a cars_per_cap over 500
cpc = cars['cars_per_cap'] 
many_cars = cpc > 500
car_maniac = cars[many_cars]

# Print car_maniac
print(car_maniac)


---------Cars per capita (2)----------

# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Import numpy, you'll need this
import numpy as np

# Create medium: observations with cars_per_cap between 100 and 500
cpc = cars['cars_per_cap']
between = np.logical_and(cpc > 100, cpc < 500)
medium = cars[between]


# Print medium
print(medium)





Week 4: Loops

---------while loop----------

The while loop is somewhat similar to an if statement: it executes the code inside if the condition is True. 
However, as opposed to the if statement, the while loop will continue to execute this code over and over again as long as the condition is true.


suppose youre numerically calculating a model based on your data. This typically involves taking the same steps over and over again, 
until the error between your model and your data is below some boundary. When you can reformulate the problem as 
repeating an action until a particular condition is met, a while loop is often the way to go

error start at 50
divide error by 4 every run 
continue until error no longer > 1

error = 50.0
while error > 1
	error = error/4 
	print(error)


-------while: warming up-------

Can you tell how many printouts the following while loop will do?

x = 1
while x < 4 :
    print(x)
    x = x + 1

x = 1, so as long as x < 4 it will run and increments go up by 1
1st run: x=1
2nd run: x=2
3rd run: x=3
4th run: x=4 --- stops, 3 runs

----------Basic while loop-----------

# Initialize offset
offset = 8

# Code the while loop
while offset != 0:
    print("correcting...")
    offset = offset - 1
    print(offset)


-------Add conditionals-----------

# Initialize offset
offset = -6

# Code the while loop
while offset != 0 :
    print("correcting...")
    if offset > 0:
      offset = offset - 1
    else : 
      offset = offset + 1    
    print(offset)


---------for_loop-----------

for each var, a variable, in seq, a sequence, execute the expressions. 

for var in seq:
	expression 

Remember about the fam list, containing the heights of your family? Here it is again, 
in the family (dot) py script. Suppose that instead of a single printout of the entire list, 
like this, we want to print out each element in the list separately.

family.py
fam = [1.73, 1.68, 1.71, 1.89]
print(fam)


fam = [1.73, 1.68, 1.71, 1.89]
print(fam[0])
print(fam[1])
print(fam[2])
print(fam[3])

You could do this by doing 4 print calls with the correct subsetting operations. 
Instead of this repetitive and tedious approach, you can use a for loop.


fam = [1.73, 1.68, 1.71, 1.89]
for height in fam:
	print(height)

we end up with 4 seperate print outs 

In this solution, you dont have access to the index of the elements youre iterating over.
Say that, together with printing out the height, you also want to display the index in the list, 
so that the printouts are converted to this. How should the for loop be built in this case?


To achieve this, you can use enumerate(). Lets update the for loop definition like this. 
Now, enumerate(fam) produces two values on each iteration: the index of the value, and the value itself.

fam = [1.73, 1.68, 1.71, 1.89]
for index, height in enumerate(fam):
	print('index ' + str(index) + ': ' + str(height))

index 0: 1.73
index 1: 1.68 
...etc

looping over string

for c in 'family:'
	print(c.capitalize())

 F
 A
 M
 I
 L
 Y

---------------loop over a_list----------

# areas list
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Code the for loop
for i in areas:
    print(i)


-----------Indexes and values (1)-------------

# areas list
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Change for loop to use enumerate() and update print()
for x, a in enumerate(areas) :
    print('room ' + str(x) + ': ' + str(a))

---------Indexes and values (2)---------------

# areas list
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Code the for loop
#Adapt the print() function in the for loop so that the first printout becomes "room 1: 11.25", the second one "room 2: 18.0" and so on.

for index, area in enumerate(areas) :
    print("room " + str(index + 1) + ": " + str(area))

-------------Loop over list of lists---------------

# house list of lists
house = [["hallway", 11.25], 
         ["kitchen", 18.0], 
         ["living room", 20.0], 
         ["bedroom", 10.75], 
         ["bathroom", 9.50]]
         
# Build a for loop from scratch
# Write a for loop that goes through each sublist of house and prints out the x is y sqm, where x is the name of the room and y is the area of the room.

for x in house:
    print('the ' + x[0]  + ' is ' +  str(x[1]) + ' sqm')


--------Loop Data Structures Part 1-------------

dictionary.py

world = { 'afganistan':30.55, 'albania':2.77, 'algeria':39.21}

#generate a key and value in each iteration.

for key, value in world.itens():
	print(key + '--' + str(value))

algeria -- 39.21
afganistan -- 30.55
albania -- 2.77

print out the order in which theyre iterated over is not fixed, at least in Python 3.5.

np.loop.py

import numpy as np
np_height = np.array([1.73, 1.68, 1.71, 1.89, 1.79])
np_height = np.array([65.4, 59.2, 63.6, 88.4, 68.7])
bmi = np_weight / np_height ** 2
for val in bmi:
	print(val)

#this will print everything out fine

2d numpy array

import numpy as np
np_height = np.array([1.73, 1.68, 1.71, 1.89, 1.79])
np_height = np.array([65.4, 59.2, 63.6, 88.4, 68.7])
meas = np.array([np_height, np_weight]) #combine np_height and np_weight arrays
for val in meas:
	print(val)

If we want to print out each element in this 2D array separately, the same basic for loop won't do the trick though. 
The 2D array is actually built up from an array of 1D arrays. The for loop simply prints out an entire array on each iteration.

To get every element of an array, you can use a Numpy function called nditer().


import numpy as np
np_height = np.array([1.73, 1.68, 1.71, 1.89, 1.79])
np_height = np.array([65.4, 59.2, 63.6, 88.4, 68.7])
meas = np.array([np_height, np_weight]) #combine np_height and np_weight arrays
for val in np.nditer(meas):
	print(val)

# you get all of the individual height first then all of the indiividal weights 

Recap
for dictionary -- use for key, val in my_dict.items():
for numpy arrays -- use for val in np.nditer(my array):


-------Loop over dictionary -------------

# Definition of dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'berlin',
          'norway':'oslo', 'italy':'rome', 'poland':'warsaw', 'austria':'vienna' }
          
# Iterate over europe
for key, value in europe.items():
    print('the capital of ' + key + ' is ' + value)

----------Loop over Numpy array-----------
#Write a for loop that iterates over all elements in np_height and prints out "x inches" for each element, where x is the value in the array.
#Write a for loop that visits every element of the np_baseball array and prints it out.

# Import numpy as np
import numpy as np

# For loop over np_height
for x in np_height :
    print(str(x) + " inches")

# For loop over np_baseball
for x in np.nditer(np_baseball) :
    print(x)


----------Loop Data Structures Part 2------------

#There's one killer data structure out there that we haven't covered up to now when it comes to looping: the Pandas DataFrame.

import pandas as pd
brics = pd.read_csv('brics.csv', index_col = 0)

If a Pandas DataFrame were to function the same way as a 2D Numpy array, then maybe a basic for loop like this, to print out each row, could work.

import pandas as pd
brics = pd.read_csv('brics.csv', index_col = 0)
for val in brics :
print(val)

#country
#capital
#area
#population


We simply got the column names. In Pandas, you have to mention explicitly that you want to iterate over the rows.

The iterrows method looks at the data frame, and on each iteration generates two pieces of data: the label of the row and then the actual data in the row as a Pandas Series. 


import pandas as pd
brics = pd.read_csv('brics.csv', index_col = 0)
for lab, row in brics.iterrows() :
	print(lab)
	print(row)


Suppose you only want to print out the capital on each iteration: lets change the print statement as follows

import pandas as pd
brics = pd.read_csv('brics.csv', index_col = 0)
for lab, row in brics.iterrows() :
	print(lab + ': ' + row['capital'])

Lets add a new column to the brics DataFrame, named name_length, containing the number of characters in the countrys name.

import pandas as pd
brics = pd.read_csv('brics.csv', index_col = 0)
for lab, row in brics.iterrows() :
	#creating series on each iteration
	brics.loc[lab, 'name_length'] = len(row['country'])
	print(brics)


theres a new column in there with the length of the country names. Nice, but not especially efficient, because youre creating a Series object on every iteration. 
For this small DataFrame that doesnt matter, but if youre doing funky stuff on a ginormous dataset, this loss in efficiency can become problematic.


A way better approach if you want to calculate an entire DataFrame column by applying a function on a particular column in an element-wise fashion, is apply(). 
In this case, you dont even need a for loop


import pandas as pd
brics = pd.read_csv('brics.csv', index_col = 0)
brics.loc[lab, 'name_length'] = len(row['country'].apply(len))
print(brics)

Basically, youre selecting the country column from the brics DataFrame, and then, on this column, you apply the len function. Apply calls the len function with each country name as input and produces a new array, 
that you can easily store as a new column, "name_length".


---------Loop over DataFrame (1)-----------

# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Iterate over rows of cars
for lab, row in cars.iterrows() :
    print(lab)
    print(row)


----------Loop over DataFrame (2)------------

# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Adapt for loop
for lab, row in cars.iterrows() :
    print(lab + ': ')
    print(row[str("cars_per_cap")])

-----------Add column (1)--------------
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)
print(cars)

# Code for loop that adds COUNTRY column
for lab, row in cars.iterrows() :
    cars.loc[lab, "COUNTRY"] = row["country"].upper()

# Print cars
print(cars)

----------Add column (2)---------------

# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Use .apply(str.upper)
cars["COUNTRY"] = cars["country"].apply(str.upper)

print(cars)

-----------------Random Numbers---------------

You throw a die one hundred times
If it's 1 or 2 you'll go one step down.
If it's 3, 4, or 5, you'll go one step up.
If you throw a 6, youll throw the die again and will walk up the resulting number of steps.


Of course, you can not go lower than step number 0. And also, you admit that youre a bit clumsy and have a chance of 0.1% of falling down the stair
when you make a move. Falling down means that you have to start again from step 0. With all of this in mind, you bet with your friend that youll reach 60 steps high.

What is the chance that you will win this bet?

simulate orocess which is hacker stats

seed was chosen by Python when we called the rand function, but you can also set this manually. Suppose we set it to 123, just a number I chose, like this, and then call the rand function twice. We get two random numbers.

import numpy as np
np.random.seed(123) #starting form a seed
np.random.rand()
np.random.rand() # now we have 2 random numbers

Now, if I set the seed back to 123, and call rand twice more, we get the exact same random numbers. 
np.random.seed(123) #starting form a seed
np.random.rand()
np.random.rand() # now we have 2 random numbers

Suppose we want to simulate a coin toss. First set the seed - again, this could be anything - and then use the randint() function. To have it randomly generate either 0 or 1, 
we pass two arguments: the first argument should be 0, the second one 2, because 2 is not going to be included. If we print out coin, and then run the script, we get a random integer, 0. You can now use this coin to play a game.

game.py
import numpy as np
np.random.seed(123)
coin = np.random.randint(0,2) #randomly generate 0 or 1
print(coin)

game.py
import numpy as np
np.random.seed(123)
coin = np.random.randint(0,2) #randomly generate 0 or 1
print(coin)
if coin ==0:
	print('heads')
else:
	print('tails')

-----------Random float-------------

# Import numpy as np
import numpy as np

# Set the seed
np.random.seed(123)

# Generate and print random float
print(np.random.rand())

---------Roll the dice-----------

# Import numpy and set seed
import numpy as np
np.random.seed(123)

# Use randint() to simulate a dice
print(np.random.randint(1,7))

# Use randint() again
print(np.random.randint(1,7))

---------Determine your next move-----------

# Numpy is imported, seed is set

# Starting step
step = 50

# Roll the dice
dice = np.random.randint(1,7)

# Finish the control construct
if dice <= 2 :
    step = step - 1
elif dice >=3 and dice <=5 :
    step = step + 1
else :
    step = step + np.random.randint(1,7)

# Print out dice and step
print(dice)
print(step)

-----------Random Walk---------------

If you use a dice to determine your next step, you can call this a random step. What if you use a dice 100 times to determine your next step? 
You would have a succession of random steps, or in other words, a random walk.

import numpy as np
np.random.seed(123) #random numnber generator
outcomes = [] # initalise an empty list called outcomes
for x in range(10): #build a for loop that should run 10 times
	coin = np.random.randint(0,2) #inside this for loop we generate a random integer coin that's either 0 or 1 i.e. binary; 0=heads, 1 tails
	if coin == 0:
		outcomes.append('heads') # append the string heads
	else:
		outcomes.append('tails')

if we run this script eventally a list with 10 strings will be printed out which contin heads or tails. it is random but not a random walk as the 
items in the list are not based on the previous ones. Its just a bunch of random steps.r


You could turn this example into a random walk by tracking the -total- number of tails while youre simulating the game.

import numpy as np
np.random.seed(123) #random numnber generator
tails = [0] # initalise a list called tails which already contians the number 0; because at the start you haven't thrown any tails
for x in range(10): #build a for loop that should run 10 times
	coin = np.random.randint(0,2) #If coin is 0, so heads, the number of tails you've thrown shouldn't change. If a 1 is generated, the number of tails should be incremented with 1. 
	tails.append(tails[x] + coin)
print(tails)

This means that you can simply add coin to the previous number of tails, and add this count to the list with append. 
Finally, you again print the list tails. After running this script, a list with 11 elements will be printed out. 
The final element in this list tells you how often tails was thrown.

---------The next step----------

# Numpy is imported, seed is set

# Initialize random_walk
random_walk = [0]

# Complete the ___
for x in range(100) :
    # Set step: last element in random_walk
    step = random_walk[-1]

    # Roll the dice
    dice = np.random.randint(1,7)

    # Determine next step
    if dice <= 2:
        step = step - 1
    elif dice <= 5:
        step = step + 1
    else:
        step = step + np.random.randint(1,7)

    # append next_step to random_walk
    random_walk.append(step)

# Print random_walk
print(random_walk)


-----------How low can you go?------------

# Numpy is imported, seed is set

# Initialize random_walk
random_walk = [0]

for x in range(100) :
    step = random_walk[-1]
    dice = np.random.randint(1,7)

    if dice <= 2:
        # Replace below: use max to make sure step can't go below 0
        step = max(0, step - 1)
    elif dice <= 5:
        step = step + 1
    else:
        step = step + np.random.randint(1,7)

    random_walk.append(step)

print(random_walk)


------------Visualize the walk-----------

# Numpy is imported, seed is set

# Initialization
random_walk = [0]

for x in range(100) :
    step = random_walk[-1]
    dice = np.random.randint(1,7)

    if dice <= 2:
        step = max(0, step - 1)
    elif dice <= 5:
        step = step + 1
    else:
        step = step + np.random.randint(1,7)

    random_walk.append(step)

# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Plot random_walk
plt.plot(random_walk)

# Show the plot
plt.show()


------------Distribution-----------

Lets go back to the initial problem. you throw a die one hundred times. Depending on the result you go some steps up or some steps down. 
This is called a random walk, and you know how to simulate this. But you still have to answer the main question: what is the chance that 
youll reach 60 steps high?

Each random walk will end up on a different step. If you simulate this walk thousands of times, you will end up with thousands of final steps. 
This is actually a distribution of final steps. And once you know the distribution, you can start calculating chances.

Lets go back to the example of the total number of tails after 10 coin tosses.

import numpy as np
np.random.seed(123) #random numnber generator
tails = [0] # initalise a list called tails which already contians the number 0; because at the start you haven't thrown any tails
for x in range(10): #build a for loop that should run 10 times
	coin = np.random.randint(0,2) #If coin is 0, so heads, the number of tails you've thrown shouldn't change. If a 1 is generated, the number of tails should be incremented with 1. 
	tails.append(tails[x] + coin)

The number of tails starts at zero and, ten times, we calculate a random number which is either 0 or 1. We then update the number of times tails has been thrown by appending it to the list.

To find the distribution of this walk, we start by setting a random seed, and then create an empty list named final_tails.
Lets write a for loop that runs 100 times. Inside this for loop, we put the code from before, that gradually builds up the tails list.


import numoy as np
np.random.seed(123)
final_tails = [] # this empty list will create the number of tails you end up with 
for x in range(100):
	tails = [0]
	for x in range(10):
		coin = np.random.randint(0,2)
		tails.append(tails[x] + coin)
	final_tails.append(tails[-1]) #After simulating this single game, we append the last number, so the number of tails after tossing 10 times, to the final_tails list. 
print(final_tails) #If you put a last line in here to print final_tails, outside of the for loops, and run the script, you see that final_tails contains numbers between 0 and 10. 
#Each number is the number of tails that were thrown in a game of 10 tosses. All these values actually represent a distribution, that we can visualize. 



import numoy as np
import matplotlib.pyplot as plt
np.random.seed(123)
final_tails = [] # this empty list will create the number of tails you end up with 
for x in range(1000): #change range to 1000 runs
	tails = [0]
	for x in range(10):
		coin = np.random.randint(0,2)
		tails.append(tails[x] + coin)
	final_tails.append(tails[-1])
plt.hist(final_tails, bins = 10)
plt.show()


If we change the code to do ten thousand simulations,
and run the script once more, the distribution starts to converge to a bell-shape. In fact, it starts to look like the theoretical distribution. 
That means the distribution that you would find by doing analytical pen-and-paper calculations.
Ideally, you want to carry out the experiment zillions of times to get a distribution that is exactly the same as the theoretical distribution. 

import numoy as np
import matplotlib.pyplot as plt
np.random.seed(123)
final_tails = [] # this empty list will create the number of tails you end up with 
for x in range(10000): #change range to 1000 runs
	tails = [0]
	for x in range(10):
		coin = np.random.randint(0,2)
		tails.append(tails[x] + coin)
	final_tails.append(tails[-1])
plt.hist(final_tails, bins = 10)
plt.show()

-----------Simulate multiple walks-------------

# Numpy is imported; seed is set

# Initialize all_walks (don't change this line)
all_walks = []

# Simulate random walk 10 times
for i in range(10) :

    # Code from before
    random_walk = [0]
    for x in range(100) :
        step = random_walk[-1]
        dice = np.random.randint(1,7)

        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1,7)
        random_walk.append(step)

    # Append random_walk to all_walks
    all_walks.append(random_walk)

# Print all_walks
print(all_walks)

----------Visualize all walks----------

# numpy and matplotlib imported, seed set.

# initialize and populate all_walks
all_walks = []
for i in range(10) :
    random_walk = [0]
    for x in range(100) :
        step = random_walk[-1]
        dice = np.random.randint(1,7)
        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1,7)
        random_walk.append(step)
    all_walks.append(random_walk)

# Convert all_walks to Numpy array: np_aw
np_aw = np.array(all_walks)

# Plot np_aw and show
plt.plot(np_aw)
plt.show()

# Clear the figure
plt.clf()

# Transpose np_aw: np_aw_t
np_aw_t = np.transpose(np_aw)

# Plot np_aw_t and show
plt.plot(np_aw_t)
plt.show()


---------Implement clumsiness-----------

# numpy and matplotlib imported, seed set

# Simulate random walk 250 times
all_walks = []
for i in range(250) :
    random_walk = [0]
    for x in range(100) :
        step = random_walk[-1]
        dice = np.random.randint(1,7)
        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1,7)

        # Implement clumsiness
        if np.random.rand() <= 0.001 :
            step = 0

        random_walk.append(step)
    all_walks.append(random_walk)

# Create and plot np_aw_t
np_aw_t = np.transpose(np.array(all_walks))
plt.plot(np_aw_t)
plt.show()


---------Plot the distribution-----------

# numpy and matplotlib imported, seed set

# Simulate random walk 500 times
all_walks = []
for i in range(500) :
    random_walk = [0]
    for x in range(100) :
        step = random_walk[-1]
        dice = np.random.randint(1,7)
        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1,7)
        if np.random.rand() <= 0.001 :
            step = 0
        random_walk.append(step)
    all_walks.append(random_walk)

# Create and plot np_aw_t
np_aw_t = np.transpose(np.array(all_walks))

# Select last row from np_aw_t: ends
ends = np_aw_t[-1,:]

# Plot histogram of ends, display plot
plt.hist(ends)
plt.show()


---------Calculate the odds-----------

The histogram of the previous exercise was created from a Numpy array ends, that contains 500 integers. Each integer represents the end point of a random walk. 
To calculate the chance that this end point is greater than or equal to 60, you can count the number of integers in ends that are greater than or equal to 60 
and divide that number by 500, the total number of simulations.

Well then, whats the estimated chance that youll reach 60 steps high if you play this Empire State Building game? The ends array is everything you need; 
its available in your Python session so you can make calculations in the IPython Shell.

78.4%

---write the python code here!



































