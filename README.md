# Prediction-ml-home-lab

## Description ##

This project showcases an analysis of California housing data using the fetch_california_housing dataset from the scikit-learn library. The project includes exploratory data analysis, correlation visualization, and the creation of a linear regression model to predict the median house value based on features such as number of rooms, building age, and median income. The model was evaluated, and key variables influencing house prices were identified and interpreted.

## Utilities and Libraries Used ##

- **Python 3.13.1**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Sklearn**
- **Seaborn**


## Environments Used ##

- **Visual Studio Code**
- **Google Colab**

## Results walk-through ##

### Correlation Matrix  ###
![Image of correlation matrix](https://i.imgur.com/z7KLZoA.png)



### Median Home Value Distribution ###

![Image of medHomeValueDistrib](https://i.imgur.com/SA07b55.png)



### The impact of variables on home value ###

![Image of regression coefficiens](https://i.imgur.com/2IBpWzC.png)


### Final notes ###

The biggest influence on the price of a house is the average income of residents (MedInc) - the higher the income in the area, the higher the prices of the property,
which is quite intuitive. The features of the house itself also matter, e.g. the number of rooms (AveRooms) - larger and more spacious houses usually cost more. The age of the building (HouseAge) also affects the value,
but its significance may vary depending on the location.

Generally speaking, the most important factors are related to the standard of living and the standard of the property.
