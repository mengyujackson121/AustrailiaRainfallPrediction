# Project 3 -- Rainfall Prediction Variables
by Mengyu Jackson



## Overview

Australia has a variety of weather stations across the country, which collect data used for rainfall forecasting.
Although basic metrics are recorded at all stations (Max Temperature, Minimum Temperature, Rainfall), other metrics are intermittently or never recorded at certain stations (such as amount of Sunshine, cloudiness, Evaporation, Pressure, etc.)

The Australian government wants to determine whether additional investment in recording these metrics will improve rainfall forecasting in these areas.

Our models (which match state of the art accuracy) are able to predict rainfall at stations which do not collect all metrics as well as stations which do record all metrics.

Spending more money to collect these metrics will not improve rainfall forecasting.


## Business Problem

Australia has a variety of weather stations across the country, which collect data used for rainfall forecasting.
Although basic metrics are recorded at all stations (Max Temperature, Minimum Temperature, Rainfall), other metrics are intermittently or never recorded at certain stations (such as amount of Sunshine, cloudiness, Evaporation, Pressure, etc.)

The Australian government wants to determine whether additional investment in recording these metrics will improve rainfall forecasting in these areas.

Rainfall forecasting is used for a variety of purposes by the public, and different uses are sensitive to different types of errors (false positive vs false negative) in forecasting. We need to evaluate model performance with and without all features on multiple types of errors.

If additional data is recommended, which data is most important?



## Data Understanding

This dataset contains about 10 years of daily weather observations from 49 locations across Australia. Not all sites have an equal amount of data, but all have at least 3 years.
“Core” variables are recorded at all stations
“Optional” variables are intermittently or never recorded at certain stations 
26 locations collect all Core and Optional variables
23 locations never collect at least 1 Optional variable

Data From Kaggle

Observations were drawn from numerous weather stations. The daily observations are available from http://www.bom.gov.au/climate/data.

An example of latest weather observations in Canberra: http://www.bom.gov.au/climate/dwo/IDCJDW2801.latest.shtml

Definitions adapted from http://www.bom.gov.au/climate/dwo/IDCJDW0000.shtml

Data source: http://www.bom.gov.au/climate/dwo/ and http://www.bom.gov.au/climate/data.

Copyright Commonwealth of Australia 2010, Bureau of Meteorology.



### 1st Model : Linear Model 
### 2nd Model :  Ada Boost Model

Interim Result
(Some of) the Optional metrics appear to be important.

Average Precision consistently drops by ~.05 when these columns are removed, for a variety of models
AUC_ROC consistently drops by ~.03 when these columns are removed, for a variety of models
We need to drill down to specific locations to figure out which missing columns are actually problematic...


### 3rd Model : Time Series (Location Specific) Models

#### Note:

The Location Specific Time Series Models have combined metrics better than either the LogisticRegression or AdaBoost models in almost every aspect:

* Negative class Precision and Recall are equal or better
* Positive class Precision is better and recall is close. 
* F1 scores are better
* Macro averages for precision recall and f1 score are all better.
* Weighted averages for precision, recall, and f1-score are equal or better than earlier models.

#### Surprising Result

The Location Specific Time Series models are better for locations which do not record at least one Optional metric:
* Mean, min, 25%, 75% and max avg_precision are all better
* 50% avg_precision is nearly identical
* mean, 25%, 50%, 75%, and max ROC AUC are all better

The minimum ROC AUC is much lower however (.70 vs .79).

Looking at this outlier gives us a key insight:

#### Key Insight
* Newcastle is by far the worst ROC AUC score of any location specific classifier. 
* Albany is the second worst out of all sites missing a column. It would be <25% in either group of locations
* These are the only two sites which do not record the WindGustSpeed Optional metric



# Final Time Series Results

WindGustSpeed is the only Optional metric that seriously improved the Location Specific Time Series Models (or made them worse when it was missing).

Rainfall predictions for locations missing one or more of the other metrics actually did *better* than predictions for locations with full Optional metrics (this is a good subject for future investigation).

We don't want to assume the Australian government will only ever use Location Specific Time Series models, so we'll check our earlier models. We want to see if "Core+WindGustSpeed" models (trained on all Core features + WindGustSpeed) does better than the Core models (trained on Core features with no Optional features) and is close to the "Full" models (trained on all Core + Optional features)


## Conclusions
Recording Wind Gust Speed at all Locations will help improve rainfall forecasting at those locations.
Other Optional features can be helpful for some model types, but the best models do not need them.

## Recommendation
* Newcastle and Albany should record Wind Gust Speed
* Newcastle is the worst-predicted location by the Time Series Model
* Albany is in the bottom 25%


