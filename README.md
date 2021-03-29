# Project 2 -- Bank Helper



## Overview

Our stakeholder is a mortgage lender who would like more accurate appraisals to reduce risk for home loans. We analyzed the King County (CA) House Sales dataset using machine learning to develop a model for predicting the value of a house. The model accurately predicts the value of a house with information available to the bank at time of appraisal, and would be a good tool for making loan decisions. We recommend using this model along with existing appraisers to reduce risk and increase profit margins.


## Business Problem

Our stakeholder is a mortgage lender who wants to increase the accuracy of their appraisals in order to reduce the risk of default, especially loans which have the minimum possible down payment (20%) without Private Mortgage Insurance. These loans are worth 80% of the purchase price of the house. If a borrower defaults immediately, our stakeholder wants confidence they'll be able to re-sell the house and cover the entire loan. At the same time, they do not want artificially low appraisals, as those would drive clients to competing lenders. Specifically, we want to maximize the number of appraisals which are between 80% and 105% of the true value of the house in order to minimize risk while remaining attractive to borrowers.


## Data Understanding

This project uses the King County House Sales dataset, which can be found in `kc_house_data.csv` in the data folder in this repo. The description of the column names can be found in `column_names.md` in the same folder. As with most real world data sets, the column names are not perfectly described, so you'll have to do some research or use your best judgment if you have questions about what the data means.


- Where did the data come from, and how do they relate to the data analysis questions?
    The data come from house sales in King County, and they help us relate all of the features we are interested in (sqft, waterfront, renovated or not)
- What do the data represent? Who is in the sample and what variables are included?
    Only houses that have sold are in the sample, and variables include comparisons to nearby houses (_15 suffixed variables), metrics about the house that was sold and its lot (sqft), whether and when renovations were last done, and the original year it was built.
- What is the target variable?
    The target variable is the sale price of the house. A secondary target could be views; which could be used as a proxy for time-on-market.
- What are the properties of the variables you intend to use?
    Almost all of the variables we intend to use are numeric, except one binary variable (waterfront or not). Some of the variables are cyclic in nature (month), which we hope to capture in our feature selection.


### First Model

After decide use sklean, first thing to try is `LinearRegression()`.


## Modeling

Try different model:

* Ridge(random_state = RANDOM_SEED),
* BayesianRidge(),
* LinearRegression(),
* RandomForestRegressor(random_state = RANDOM_SEED),
* GradientBoostingRegressor(random_state = RANDOM_SEED),
* neural_network.MLPRegressor(solver="lbfgs", random_state = RANDOM_SEED)
* XGBRegressor() 

Use pipline with PCA, PolynomialFeatures or StandardScaler, use GridSearchCV to found the best hyperparameters:



Questions to consider:

We found explainability was good enough with partial_dependence plots, so we did not restrict our analysis to easily explainable models like linear regression. Explainability was less important because getting the right answer on average is the most important thing for making a profit as a mortgage lender.

Some variables had clear nonlinear effects (yr_built, latitude, longitude), which made it hard to get good performance from a linear model. We tried many different regressors built into scikitlearn with default parameters to decide which models were worth tuning. After we found that GradientBoostingRegressor was best, we decided to install and use xgboost (third party library for boosting decision trees) to see if that improved performance.

We decided xgboost was best, so we used GridSearchCV to find good hyperparameters without overfitting. We had to leave it running overnight several days in a row, but the results got us very close to our goal R^2 of .9.


## Conclusions

The model is very good at predicting house prices in 2014-2015. Training on data outside this period will be necessary to help it understand larger trends in housing prices.
Using this model to appraise houses nearly guarantees interest made from loans will cover money lost to bad appraisals + default, even under very adverse assumptions (12% foreclosure rate, 2% APR).
We assumed the market remained stable during foreclosures; we did not analyze the case where a market crash depresses housing values simultaneously with default. That could increase losses significantly in a worst case scenario
This model could definitely generate a profit, but client should work with us to determine expected ROI using more realistic assumptions of default rate and APR to evaluate whether this model is more profitable than their existing process.




# Interpretation of Linear Model

## Summary
My final linear model (Orthogonal Matching Pursuit) is worse than most non-linear models I tried at predicting prices, but easy to interpret and explain.

It only uses 10 features with low collinearity, and the R^2 of .752 is still very close to Ordinary Least Squares using all 26 provided and engineered features (.758). The OLS version is much harder to understand, and also much worse than non-linear models.

The features used and their rounded coefficients are 

```
waterfront:          708,015
grade:               89,385
condition:           32,483
yr_built_below_1978: 2,594
sqft_living:         215
zipcode:            -818
bedrooms:           -34,867
long:               -293,409
lat_below_47.64:    -1,163,944
lat_above_47.64:    -1,605,208
```

Five(!) of the top 10 features are about lot location (waterfront, zipcode, long, lat_below_47.64, lat_above_47.64). Real estate really is about location, location, location. Waterfront is a huge `$700,000` boost in value.

The yr_built is important for homes built before 1978 but did not make the cut for newer homes, and neither did any renovation metrics. The only variables that a homeowner can control that our model selected were bedrooms, condition, grade, and sqft_living. Fewer bedrooms is actually better, but you want more of everything else. Each step in grade is worth `$90,000`, condition is `$30,000`, and each additional sqft is `$215`.

Don't take any of this analysis too seriously however, analysis shows that key assumptions have been violated and Linear Regression is not a great fit for this dataset.




## Multicollinearity of Features 

In the 10 features selected by the OrthogonalMatchingPursuit algorithm https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuit.html, multicollinearity is very low. Only the (grade, sqft_living) features had a correlation barely above .75	(0.762779). This did not seem problematic enough to correct for. The Heatmap shows the same information (only diagonals seem light pink).

Both OMPCV (Orthogonal Matching Pursuit) and RFECV (Recursive Feature Elimination Cross Validation) which I found looking at sklearn API were good at training linear models to give interpretable results without much multicollinearity.

OMPCV gave slightly better results than RFECV, but was also much faster for me to train, which helped as I experimented with feature engineering.

## Regression Analysis
### Summary (OMP Model)
The data seems to violate the assumptions for Linear Regression Models. This is not surprising since the best Linear Model is a lot worse than other models (even before optimization).

### Residual Analysis (OMP Model)
The residual analysis using both the KDE plot and the QQ plot shows the residuals are not normally distributed. The residual distribution is skewed and light tailed.

### Feature Analysis (OMP Model)
Many features are not normally distributed, but transforming them using logtransforms did not improve model performance.

### Jarque-Bera Test (statsmodel OLS Model)
I wasn't sure how to do the Jarque-Bera test directly on an sklearn model, so I trained a new model using statsmodel Ordinary Least Squares using only the 10 features selected by the OMP model. The coefficients are close (e.g., -817.54 vs -793.65 for Zip Code). The statsmodel OLS model has an R^2 that is lower, but that shouldn't change the final analysis. 

Doing the Jarque-Bera test on the OLS model shows results that confirm the residual analysis: Key Assumptions have been violated and Linear Regression is probably not the right choice for this dataset.