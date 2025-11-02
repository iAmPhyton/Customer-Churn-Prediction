# Customer-Churn-Prediction
The goal of this project is to predict customer churn using demographic, billing, and service-usage data from a telecommunications company.
I've explored the dataset, built classification models, and extracted insights into what drives customer attrition.

Data Preparation:
Dataset: Telco Customer Churn [https://www.kaggle.com/datasets/blastchar/telco-customer-churn]

Steps performed:
- Handled missing values in the TotalCharges column.
- Encoded categorical variables using label encoding and one-hot encoding.
- Splitted the dataset into training and test sets (80/20).
- 
Model Building::

Models Used:
- Logistic Regression
- Random Forest Classifier [The Random Forest model achieved higher accuracy and F1-score, making it my preferred choice for feature analysis.]

Model Evaluation::
Evaluation metrics included:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix Visualization (Plotly Interactive)

Feature Importance Analysis::
- Using the Random Forest model, I identified the top 15 features contributing most to churn prediction.

Interpretation of Results::

The feature importance scores reflect how much each variable influenced the model’s prediction.
Here’s what they mean:

Feature	Importance	Business Insight
- TotalCharges (0.155):	Customers with high total charges tend to churn more often, possibly due to cumulative billing dissatisfaction.	
- MonthlyCharges (0.142)	High monthly fees are a strong driver of churn — customers may switch to cheaper plans or competitors.	
- Contract (0.129)	Month-to-month customers churn more frequently than those on annual contracts, indicating loyalty is tied to contract duration.	
- Tenure (0.100)	Longer-tenured customers are more stable and less likely to leave — early retention efforts are key.	
- OnlineSecurity (0.076)	Customers without online security add-ons are more likely to churn, suggesting bundled services can improve retention.	
- TechSupport (0.060)	Access to tech support reduces churn — customers value responsive service.	
- InternetService (0.044)	Service type (DSL vs Fiber) also plays a role in satisfaction and retention.	
(Other features such as PaymentMethod, PaperlessBilling, and StreamingTV have smaller but notable effects.)		

What the Numbers Mean:
- Each score represents how much a feature contributed to improving the model’s accuracy — the higher the value, the more influential the feature.
For example, a score of 0.155 for TotalCharges means it accounted for about 15.5% of the model’s predictive power.

Interpretation in Plain Terms:
- “If the model were a judge deciding who will churn, these numbers show which witnesses (features) had the strongest voice in the verdict.”

Business Recommendations::
- Encourage longer contracts: Month-to-month plans have the highest churn rates — offer discounts for 1-year or 2-year commitments.
- Review pricing structure: High MonthlyCharges are linked to churn; consider flexible or loyalty-based pricing.
- Promote add-on services: Features like OnlineSecurity and TechSupport correlate with retention — bundle them strategically.
- Focus retention campaigns on new customers: Customers with low tenure are more likely to churn; early engagement is crucial.

Tools & Libraries::
- Python, Pandas, NumPy, Matplotlib, Seaborn, Plotly
- Scikit-Learn (for ML modeling and evaluation)

Author:
- Chukwuemeka Eugene Obiyo
- Data Scientist, Machine Learning Engineer
