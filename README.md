# Google Trends Quantile Forecaster
This application creates quantile forecasts for the popularity of a series of keywords using Long short-term memory (LSTM), an artificial recurrent neural network (RNN) architecture which is capable of learning long-term dependencies. The user can specify a list of keywords in an input text file, with the following schema:
```
Category:Category1
Keyword:Keyword1
Keyword:Keyword2
...
```
with a maximum of 5 keywords per category. The application then accesses Google Trend historical interest over time using the Pytrends API, and organizes the collected data
into a Pandas DataFrame which is then stored as a CSV file. This file is used as the input dataset to a LSTM model built in Keras after which the forecasts are displayed back to the user.

The front-end of this application uses the Seaborn library along with Matplotlib.
