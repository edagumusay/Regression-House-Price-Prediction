# House price prediction in King County, USA: A Regression Approach
In this project, we will develop a model that predicts house prices using regression in King County, WA. 

  <img src='https://www.racialequityalliance.org/wp-content/uploads/2016/10/assessors_social-1.jpg'>
<a href='https://www.kaggle.com/code/shiv28/house-price-prediction-in-king-county-usa' target=_blank>
    
    You can find the data here.
</a>

## Files and Directories

### 1. Exploratory Data Analysis (EDA) and Model Development
- **File**: `analysis_and_model.html`
- **Description**: This HTML file contains all the steps performed in the project, including exploratory data analysis (EDA), feature engineering, model training, and evaluation. It provides comprehensive insights into the dataset and the developed regression model.

### 2. Regression Model Implementation
- **File**: `RegressionModel.py`
- **Description**: The `model.py` file includes the implementation of the regression model. It contains the code for data preprocessing, feature engineering, model training, and evaluation. This file specifically focuses on the regression model development.


## Dataset
The data has been obtained from the Kaggle project. This dataset contains house sale prices for King County, which includes Seattle. It includes homes sold between May 2014 and May 2015.

### Data Dictionary:
<ul>
    <li><strong>id:</strong> A notation for a house</li>
    <li><strong>date:</strong> Date house was sold</li>
    <li><strong>price:</strong> Price is prediction target</li>
    <li><strong>bedrooms:</strong> Number of Bedrooms/House</li>
    <li><strong>bathrooms:</strong> Number of bathrooms/bedrooms</li>
    <li><strong>sqft_living:</strong> Square footage of the home</li>
    <li><strong>sqft_lot:</strong> Square footage of the lot</li>
    <li><strong>floors:</strong> Total floors (levels) in house</li>
    <li><strong>waterfront:</strong> House which has a view to a waterfront</li>
    <li><strong>view:</strong> Has been viewed</li>
    <li><strong>condition:</strong> How good the condition is Overall</li>
    <li><strong>grade:</strong> Overall grade given to the housing unit, based on King County grading system</li>
    <li><strong>sqft_above:</strong> Square footage of house apart from basement</li>
    <li><strong>sqft_basement:</strong> Square footage of the basement</li>
    <li><strong>yr_built:</strong> Built Year</li>
    <li><strong>yr_renovated:</strong> Year when house was renovated</li>
    <li><strong>zipcode:</strong> Zip code</li>
    <li><strong>lat:</strong> Latitude coordinate</li>
    <li><strong>long:</strong> Longitude coordinate</li>
    <li><strong>sqft_living15:</strong> Living room area in 2015 (implies-- some renovations) This might or might not have affected the lotsize area</li>
    <li><strong>sqft_lot15:</strong> LotSize area in 2015 (implies-- some renovations)</li>
</ul>

## Data Analysis
- Exploratory Data Analysis (EDA) has been performed to gain insights into the dataset.
- Descriptive statistics, correlation analysis, and visualization techniques have been utilized to understand the relationships between variables and identify potential patterns.
        
## Data Preprocessing
- Outliers have been identified and removed from the dataset.
- Feature engineering techniques have been applied to derive new features and enhance the predictive power of the model.
- Categorical variables have been encoded for model compatibility.
    
## Model Development
- Various regression models such as Linear Regression, KNeighbors Regressor, Gradient Boosting Regressor, Extra Tree Regressor, Decision Tree Regressor, and XGBoost Regressor have been implemented.
- Model performance has been evaluated using metrics such as R-squared (R2), Mean Squared Error (MSE), and Mean Absolute Error (MAE).

## Final Results
- The XGBRegressor yielded the best result with an R-squared score of 0.842439 and an Root Mean Square Error (RMSE) of 84547.143342.
