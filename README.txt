DESCRIPTION

You can use our code or public Tableau website to perform data and visual analytics on the City of Philadelphia data set.

You have three options to view our analysis: 
1. You can access our public Tableau workbook here: https://public.tableau.com/profile/aubry.snow6913#!/vizhome/PhiladelphiaRealEstatePricingModel/POCDashboard
2. You can download the Tableau workbook and data directly from the portal (see above link)
3. You can follow the instructions below to create your own dataset and link to our prepared Tableau workbook.

INSTALLATION

Requirements:
In order for you to replicate our Tableau workbook, you need to get started by downloading a few Python packages, downloading publicly available datasets, run our models on these datasets and then use our pre-packaged Tableau workbook and point to our model outputs.

Sounds like a lot, doesn’t it? Don’t worry! We will get you set up quickly. 

To get started, we need you to set up a Python environment. One advantage of setting up a virtual environment is that you will be able to replicate our analysis consistently and not affect 
other packages that you have installed on your machine. 

The easiest way to set up a virtual environment is via Conda or Anaconda Navigator. If you already know how to set up an environment, you can simply use the provided requirements.txt file to get up and running. 

For the macroeconomic data, please note that we have already provided this information for you. However, you can easily duplicate the process by following these steps:
For the BLS Unemployment and Inflation data, go to https://data.bls.gov/cgi-bin/srgate. 
Then, for each series, enter the series ID (e.g., CUUR0000SA0) to download the most recent data. After clicking on next, we recommend setting the starting year to 1980 (or oldest available data if after 1980) and ending date to be the most recent available date. No other settings need to be changed, download the data as HTML. On the next page, you will see an option to download the dataset as a xlsx file. This is the file we will read into our script.

Series ID / Specific geography
LNS14000000 - (Seas) Unemployment Rate
LAUMT423798000000003 - Philadelphia-Camden-Wilmington, PA-NJ-DE-MD Metropolitan Statistical Area Unemployment statistics
LAUMT114790000000003 - Washington-Arlington-Alexandria, DC-VA-MD-WV Metropolitan Statistical Area Unemployment statistics
CUUR0000SA0 - Consumer Price Index - All Urban Consumers
CUURS12BSA0 - All items in Philadelphia-Camden-Wilmington, PA-NJ-DE-MD, all urban consumers, not seasonally adjusted
CUURS35ASA0 - All items in Washington-Arlington-Alexandria, DC-VA-MD-WV, all urban consumers, not seasonally adjusted

Federal Reserve:
Federal Reserve interest rate data can be downloaded by simply navigating to this location and downloading the data:
Direct link: https://www.federalreserve.gov/datadownload/Download.aspx?rel=H15&series=4d2bcc12cb7f9b598f72a26ab9fdaf1d&filetype=spreadsheetml&label=include&layout=seriescolumn&from=01/01/1919&to=12/31/2030

Please note that minor formatting changes need to be made to make them readable to our script. This is completed by simply changing the names of the columns in row 7 to the following:
1 Month Treasury, 3 Month Treasury, 6 Month Treasury, 1 Year Treasury, 3 Year Treasury, 5  Year Treasury, 7 Year Treasury, 10 Year Treasury, 30 Year Treasury, Federal Funds Rate, and PRIME Rate for the respective columns. 

Once this is down, all rows from 1 through 6 may be deleted. You are now ready to run the scripts.

Lastly, for the market data, please use Yahoo Finance to download monthly data for the Russell 3000 index and NASDAQ.

You can export as CSV from the following links:
1. Russell 3000: https://finance.yahoo.com/quote/%5ERUA/history?period1=631152000&period2=1606694400&interval=1mo&filter=history&frequency=1mo&includeAdjustedClose=true
2. NASDAQ: https://finance.yahoo.com/quote/%5ENDX/history?period1=496972800&period2=1606003200&interval=1mo&filter=history&frequency=1mo&includeAdjustedClose=true

EXECUTION

Navigate to the CODE folder in Bash or Terminal and simply run this one line of code:

user$ conda create -n team66 --file requirements.txt

You may replace “team66” with an environment name of your choice. However, please ensure you are replacing that name elsewhere as part of this code run. We recommend leaving it as “team66” for convenience.

Once the installation is complete, to activate the environment, please run the following command:

user$ conda activate team66

Caution: No API key or registration is required to access this data. However, it is possible in certain situations for the server to reject your request if there are too many requests in a small period of time. In such cases, please wait 30 mins before retrying. The code already provides some wait time to avoid such rejections but server settings change at a moment’s notice. Finally, please note that the script can take up to a few hours to run and will very likely download in excess of a few gigabytes.

Then, you can run our different python scripts in sequence as follows:

user$ python Real_Estate_Data_Download.py
user$ python Random_Forest_PHL.py
user$ python Random_Forest_Fairfax.py
user$ python Multiple_Regression_PHL.py
user$ python Multiple_Regression_Fairfax.py
user$ python GB_SVR_PHL.py
user$ python GB_SVR_Fairfax.py

Once successfully run, the scripts will reproduce the data needed for the Tableau workbook. 

Linking the data to Tableau: This section describes how to link our model outputs to the Tableau workbook.

Follow these steps to link the data to Tableau workbook:
Open Tableau (version 2020.03). 
Import the CSV into Tableau via the import data feature. 

You are now ready to visually explore the data. 
