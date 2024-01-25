**Author: Ishaan Chadha**

---
# Introduction

The following dataset was downloaded from [food.com](https://www.food.com) as two datasets: one containing recipes and their specific attributes and another with ratings and their reviews of recipes. We merged these datasets into one to be able to answer the following question: **Is there a relationship between the number of steps in a recipe and the average rating of that recipe?**

This is an important question in the world of cooking. By looking at recipes and their respective average rating, this question will provide some insight as to whether the number of steps in a recipe factors into its rating. This can influence cooks to edit the number of steps in their new recipes in the hopes of a greater rating by reviewers.

After cleaning the data, this dataset has 234,429 rows and 26 columns (234,429 rows and 18 columns prior to cleaning the data). In order to get familiar with the dataset, here are some important columns that will be mentioned:

- `name`: name of the recipe
- `id`: the id of a recipe, where the same recipe has the same value for this column
- `description`: description of the recipe with given `id`. 
- `n_steps`: the number of steps in the recipe
- `user_id`: reviewer of the recipe `name`
- `rating`: the rating given for the recipe by the reviewer, `user_id`
- `average_rating`: the mean rating of a recipe `id`; recipes with the same `id` have the same value for this column

---
# Data Cleaning
<iframe src="assets/table.html" width=800 height=600 frameBorder=0></iframe>



There is a list of data cleaning steps that were followed in order to make the analysis of the relationship between the number of steps in a recipe and its average rating easier and more accurate.

* **Merging Dataframes:**
   The merging of the `recipes_df` and `interactions_df` dataframes allows for a comprehensive analysis by combining recipe information with interaction data. It ensures that the average rating of each recipe can be accurately calculated and associated with the corresponding recipe.

* **Handling Missing Ratings:**
   By replacing zero values in the `rating` column with 'NaN', we distinguish between missing ratings and zero ratings. This step ensures that the calculation of the average rating is not skewed by zero values and allows for a more accurate assessment of the relationship between the number of steps and the average rating.

* **Calculating Average Ratings:**
   The calculation of the average rating for each recipe provides a single value representing the overall rating. This average rating serves as a metric to measure the recipe's quality and popularity, allowing for comparisons and analysis based on the aggregated ratings.

* **Cleaning the name Column:**
   Cleaning the `name` column by removing leading and trailing white spaces has a negligible effect on the analysis of the relationship between the number of steps and the average rating. It ensures data consistency and avoids potential inconsistencies in subsequent analyses.

* **Cleaning the submitted Column:**
   Converting the `submitted` column to a datetime format ensures that temporal information can be properly analyzed.

* **Cleaning the tags Column:**
   Cleaning the `tags` column ensures that the tags are transformed into a suitable format for analysis.

* **Cleaning the nutrition Column:**
   The cleaning of the `nutrition` column and extracting specific nutrition values may be useful for exploring other aspects of the dataset.

* **Cleaning the steps Column:**
   Cleaning the `steps` column by splitting the string into a list of individual steps and removing unnecessary characters allows for a structured analysis of recipe steps. It enables us to make sure that the `n_steps` column has correct data and is in accordance with the the number of steps listed in the `steps` column. This ensures that the data is accurate, which is crucial for assessing the relationship with the average rating.

* **Cleaning the description Column:**
   Cleaning the `description` column by replacing missing values with 'MISSING' ensures that missing descriptions are appropriately identified and handled.

* **Cleaning the ingredients Column:**
    Cleaning the `ingredients` column by splitting the string into a list of individual ingredients and removing unnecessary characters facilitates structured analysis of the ingredients and ensures that the length of the lists in `ingredients` is equal to `n_ingredients`.

* **Cleaning the date Column:**
    Converting the `date` column to a datetime format ensures proper handling of temporal information, which may be relevant for time-related analyses.

* **Cleaning the review Column:**
    Cleaning the `review` column by replacing HTML entities ensures the readability and consistency of the review text.

* **Dropping the recipe_id Column:**
    Dropping the `recipe_id` column after the merge operation removes repition in our dataframe with the column `id`.
    
Overall, by following these steps and cleaning the data, we made the analysis of the relationship between the number of steps in a recipe and its average rating easier and more accurate.

---
# Univariate Analysis #
<iframe src="assets/univariate-plot.html" width=800 height=600 frameBorder=0></iframe>

In this analysis, we examine the box plot representation of the "n_steps" column, which provides insights into the distribution of step lengths in recipes. The box plot reveals that, on average, recipes tend to have around 6 to 13 steps. The box plot's median line represents the middle value of the dataset, showing that recipes typically have a step count of 9. A lower fence of 1 and an upper fence of 23 indicates that the length for almost all recipes lies between 1 and 23 steps. However, outliers are observed with step counts reaching 100, suggesting the presence of a few recipes with significantly more complex procedures.

---
# Bivariate Analysis
<iframe src="assets/bivariate-plot.html" width=800 height=600 frameBorder=0></iframe>

The plot above is a scatter plot of `n_steps` versus `average_rating` (where duplicate `id` rows were dropped as `n_steps` and `average_rating` are the same for the same recipe). From the scatter plot, we can see a general trend where recipes with more steps have a lower proportion of recipes with a low `average_rating` compared to recipes with less steps; this would also convey that recipes with more steps have a higher proportion of recipes with a higher average rating compared to recipes with less steps. This graph, therefore, possibly indicates that--with further analysis needing to prove it--recipes with more steps, on average, are more likely to have a higher `average_rating`.

---
# Data Aggregate
<iframe src="assets/pivot_table_steps.html" width="800" height="200" frameborder="0"></iframe>
The pivot table above looks at the distribution of average ratings (grouped together in intervals of 1 to better visualize any possible trends) for each `n_steps` value in the dataframe. This is interesting as it allows one to potentially point out any possible relationship between the two variables to further explore. 

*Sidenote*: The column for 71 steps is all 0's, which is an outlier as the sum of everyone column should be 1; this is due to the fact that the only recipe with 71 steps had no review, so we replaced the `np.NaN` values with 0's.

---
# NMAR Analysis

The missingness in the `review` column could potentially be classified as NMAR based on the premise that the reviewer might have been too lazy to write out an explanation for their `rating` of the recipe `id`.

**Reasoning:** Depending on how the reviewer felt when filling out their review, they could have been too lazy, too tired, or just accidentally failed to fill out the review column. This would lead to a `np.NaN` value for their review of the recipe `id`. This would indicate that the `review` column is NMAR.

On the other hand, `review` could possibly be proven as MAR if more data is collected. From more data being collected, we could possibly prove that this column is MAR based on the following columns in their respective scenarios:

1. `user_id`: In this scenario, we would get more reviews from the same reviewers. If the same users are neglecting to leave a review when reviewing a recipe, we could label `review` as MAR due to this column being dependent on `user_id` for missingness.

2. `rating`: In this scenario, we would get more reviews in general from any `user_id`, new or already in the dataframe. Depending on someone's emotional state when filling out a review (enraged, neutral, or happy), this could impact whether they leave `review` blank or not. If someone was very dissatisfied with the recipe, they could leave a 1 star `rating` and be done with their review as they could feel that their rating conveys their feelings. If someone was content or neutral with the recipe, they could leave 3 stars and no review as they may not have anything to add beyond the `rating`. And people happy with the recipe could leave 5 stars and no review due to the fact that they feel that the stars convey their satisfaction. Therefore, we could label `review` as MAR due to this column's missingness being dependent on `rating` if a trend between the two columns is present.

3. `id`:  In this scenario, we would get more reviews for the same recipes, `id`, in the dataset. Perhaps more common recipes, like a grilled cheese, may not elicit a `review` due to the fact that the reviewer felt that the recipe is too common and does not evoke a need for a review beyond a rating. Therefore, we could label `review` as MAR due to this column's missingness being dependent on `id`.


By incorporating these additional data and strategies, it may be possible to explore alternative explanations for the missingness in the `review` column and evaluate whether the missingness can be considered MAR rather than NMAR. This deeper understanding can guide appropriate data handling techniques and mitigate potential biases caused by missing values during subsequent analysis.

---
# Missingness Dependency Analysis

We are exploring whether `description` is MCAR or MAR. We used the following permutation tests to help us decide: 

***First Test***

**Null hypothesis:** The `description` column is not dependent on `n_steps` column.

**Alternative hypothesis:** The `description` column is dependent on `n_steps` column, indicating that `description` is MAR.


**Test Statistic:** Absolute difference in mean of `n_steps` between rows with a `description` and rows with a missing `description`.

**alpha = 0.05**

<iframe src="assets/missingnessgraph1.html" width=800 height=600 frameBorder=0></iframe>


**P-value = 0.207**

**Conclusion:** Because p-value = 0.207 > alpha level = 0.05, we fail to reject the null hypothesis that the `description` column is not dependent on `n_steps` column. There is not statistically significant evidence that the `description` column is dependent on `average_rating` column. 

Now, let us look at if the missingness `description` is dependent on `average_rating` column.




***Second Test***

**Null hypothesis:** The `description` column is not dependent on `average_rating` column.

**Alternative hypothesis:** The `description` column is dependent on `average_rating` column, indicating that `description` is MAR.


**Test Statistic:** Absolute difference in mean of `average_rating` between rows with a `description` and rows with a missing `description`.

**alpha = 0.05**

<iframe src="assets/emperical_dist_description_avg_rating.html" width=800 height=600 frameBorder=0></iframe>


**P-value = 0.006**

**Conclusion:** Because p-value = 0.006 < alpha level = 0.05, we reject the null hypothesis that the `description` column is not dependent on `average_rating` column. There is statistically significant evidence that the `description` column is dependent on `average_rating` column. This indicates that that **`description` is MAR** as its missingness is dependent on `average_rating`.

---
# Hypothesis Testing

As stated previously, we chose to explore the relationship between the number of steps in a recipe and the average rating of that recipe. To test this we set up the following permutation test:

**We define short recipes as recipes with less than 9 steps and long recipes as those with 9 or more steps.** This value was chosen carefully as to split the number of recipes in each category evenly (roughly 50% of the data in each category).

**Null hypothesis:** There is no observable relationship between the number of steps in a recipe and the average rating of that recipe.

**Alternative hypothesis:** There is a relationship between the number of steps in a recipe and the average rating of that recipe.




**Test Statistic:** Absolute difference in mean of `average_rating` between short and long recipes.

**alpha = 0.05**

<iframe src="assets/perm-test-steps.html" width=800 height=600 frameBorder=0></iframe>


**P-value = 0.001**

**Conclusion:** Because p-value = 0.001 < alpha level = 0.05, we reject the null hypothesis that there is no observable relationship between the number of steps in a recipe and the average rating of that recipe. There is statistically significant evidence that there is a relationship between the number of steps in a recipe and the average rating of that recipe. Further testing should be done to determine the strength and direction of this relationship.




**Reasoning:** In order to test our question, we needed to run a permutation test as we do not have both the population and a sample from that population. We only have two samples (short and long recipes) and no population. To run the permutation test, we chose to create two labels for the `n_steps` variable. As mentioned before, we chose 9 steps in order to split the number of recipes in each category evenly (roughly 50% of the data in each category). This unbiasedly split up the data for the permutation test. We chose the null and alternative hypotheses above as the null hypothesis is what we are currently operating under when running the test (there is no difference in the means of `average_rating` for different length recipes), whereas the alternative hypothesis is the alternative explanation for the results of the permutation test if the results disprove the null hpypothesis (there is a difference present). Because we are comparing two values (mean `average_rating` for short and long recipes), it is best for our test statistic to look at the absolute value of the difference between these two means. We chose an alpha level of 0.05 as this is a common alpha level for statistical tests. And because our p-value was less than the alpha level, we rejected the null hypothesis and stated that we had statistically significant evidence for the alternative hypothesis.

---
# Framing the Problem

From the Recipes and Ratings dataframes, we decided to investigate the following prediction problem: is there a way to predict the rating of a recipe given by a reviewer based on the other columns in the merged dataframe? In order to predict the response variable that is rating--which is an integer that is 1, 2, 3, 4, or 5--we chose to utilize a classificaton model in the form of a DecisionTreeClassifier with multiclass classification. We chose to try to predict rating in order to gain some insight as to what influences a rating, from the aspects of the recipe to the reviewer. In order to evaluate the model, we will utilize the metric accuracy  overall to see how the model preforms overall and accuracy for each of the response variable's values to see any bias in predictions made. We used this over others as there is no instrinsically worse error to commit (false negative and false positive are both equally bad), which means precision and recall are not suitable; we also want to make it such that the model makes as many accurate predictions as possible, which accuracy measures. Furthermore, since we do not particularly care about precision and recall as much as we care about overall success of the model, F1-score is not all that suitable either, which lead us to choose accuracy as our evaluation metric. And we will also look into the model's accuracy for each of the response variabale's values to better see if our model is biased towards guessing any ratings over others.

At the time of prediction of the rating of a particular recipe and reviewer, we have access to the particular information of the recipe; for the recipe aspect, we can utilize the recipe id, description (both of the overall recipe, steps, and ingredients), time to make, number of steps, and number of ingredients of the recipe. As for the review, we can only access the reviewer identification (`user_id`) at the time of the prediction of rating. This is because reviewers usually register to rate a recipe beforehand or access a website to input their reviews, which would be tied to their `user_id`. All these predictor variables will help us to predict the rating given by the reviewer for that particular recipe.

---
# Baseline Model

In our Baseline Model, we try to predict the ratings that different recipes on [food.com](https://www.food.com) are given.

In our model, we use a pipeline that consists of a preprocessing step followed by a decision tree classifier. The preprocessing step involves transforming the features before feeding them into the decision tree classifier. The preprocessor consists of a ColumnTransformer which applies different features to different columns of the dataset. We Standard Scale the `calories (#)` column and Function Transform the `n_ingredients` column. Since the `n_ingredinets` column is an ordinal column, we can't One-Hot-Encode it. Thus, we use a FunctionTransformer to ordinally transform it. Our FunctionTransformer categorizes the number of ingredients into three categories: 0 for less than 7 ingredients, 1 for 7-10 ingredients, and 2 for more than 10 ingredients. We chose these splits as it evenly splits up recipes into three groups (each with about 33% of the datatset). The rest of the columns in the dataset are dropped and thus aren't factored in while predicting the `rating` column of the dataset.

**Features:**

**1. Quantitative features:** `calories (#)` (one feature). It is a quantitative feature representing the number of calories in a recipe. It is standardized (scaled) using the StandardScaler.

**2. Ordinal feature:** `n_ingredients` (one feature). It categorizes the number of ingredients into three categories: 0 for less than 7 ingredients, 1 for 7-10 ingredients, and 2 for more than 10 ingredients. It is ordinally transformed using the FunctionTransformer.

The model is split randomly into a training model (0.8) and test model (0.2) to make sure that we don't overfit our model.

The performance of the model is evaluated using accuracy, which is calculated by the `score` method of the pipeline on the test set. The accuracy represents the proportion of correctly predicted ratings out of all the predictions made. We get an accuracy of approximately 74% when we run this model on our test dataset.

We believe that our model isn't up to the mark because it is highly biased towards predicting '5' as the rating of the recipes. Our model over predicts 5's and underpredicts all the other ratings (due to the fact that a 5 rating is so common in the dataset). We believe that adding more features to our model that help distinguish between the rating of a recipe as a '5' and all the other ratings would greatly help our model predict other ratings and increase the accuracy of our model.

Additionally, we need to improve upon the accuracy of the overall model. A good baseline to compare our model's accuracy to is the rate of 5's in the dataset due to it being so common; our baseline model has an accuracy of 74%, whereas the number of 5's in the testing dataset is roughly 77%. Therefore, our model is not good right now, but we can greatly improve on the accuracy (and thus the overall effectivenss) of our model by adding more valuable predictor variables.

---
# Final Model

The features we added to the model are:

**1. 'Nutrition Data':** We used a StandardScalar to standardize the nutritional data such as 'calories', 'total fat', 'sodium', and 'sugar' to then use to predict the rating of the recipe. We believe these are important because reviewers who care about nutrition might give recipes with high calories or high fat a lower rating.

**2. 'n_steps':** We used a Binarizer to convert this feature into a binary variable based on a threshold of 8. This feature represents the number of steps required to prepare the recipe. By converting it into a binary variable, the model can capture whether a recipe has a relatively small number of steps or a larger number of steps, which may affect the cooking time or complexity of the recipe. We feel that the number of steps matter because a recipe with too many steps might cause the reviewers to get exhausted and give a bad review.

**3. 'minutes':** We used a Binarizer to convert this feature into a binary variable based on a threshold of 100. This feature represents the total cooking time in minutes. By converting it into a binary variable, the model can differentiate between recipes with shorter cooking times and those with longer cooking times. This can be useful in capturing potential correlations between cooking time and the target variable. We feel that the cooking time matters because a recipe that takes too long to make might overcomplicate a recipe, which leads to a negative impact on rating by the reviewer.

**4. 'user_id':** We used a Binarizer to convert this feature into a binary variable based on a threshold of 85 reviews; this separates reviewers by experience (relevance in the dataframe). We chose 85 as this splits the total reviews handed out into two groups. Thus, this feature represents reviewers by experience. By converting it into a binary variable, the model can differentiate between recipes being reviewed by less experienced reviewers and those with more experience. This can be useful in capturing potential correlations between the reviewer level of experience and the target variable. We feel that the reviewer experience matters because a reviewer with more experience could potentially tend to hand out more generous ratings.

**5. 'n_ingredients':** We applied a custom transformation using a FunctionTransformer. This feature represents the number of ingredients required for the recipe. By transforming this feature into three equal-szie categories, we can extract additional information, such as whether a recipe has a small number of ingredients or a larger number of ingredients. We believe that, like minutes and n_steps, more complex recipes could negatively impact ratings, which is represented by steps with more ingredients. This can be relevant as recipes with a higher number of ingredients might require more effort or be more complex.

**6. Categorical Features:** We used OneHotEncoder to encode the common words that were in the `name` column of the dataset. We found out that there are certain very frequent words in names which could have a high correlation with the ratings column such as sweet treats like 'brownies' and 'cookies' and healthier recipes like 'salad' and 'fruits', among others. We added these common words as columns to our dataframe with '0' representing that this word wasn't present in the recipe name and '1' representing that the word is present. By one-hot encoding these variables, the model can capture the relationships between different words and their impact on the rating of the recipe. This can be revelant as certain kinds of recipes like desserts could have higher/lower ratings that healtheir recipes like salads and those with fruits.

The modeling algorithm we chose for this task is the Decision Tree Classifier. Decision trees are suitable for both classification tasks and handling a mix of continuous and categorical features. They can capture complex interactions and non-linear relationships in the data. 

To select the best hyperparameters, we used the GridSearchCV function, which performs an exhaustive search over the specified parameter values using cross-validation. The model's performance was evaluated based on accuracy, which measures the proportion of correct predictions. The hyperparameters that were tuned using grid search are:

1. `max_depth`: This parameter determines the maximum depth of the decision tree. It controls the complexity of the model and helps prevent overfitting. The values tested are 10, 15, 25, 30, 50. We saw which values of `max_depth` were actually being implemented in the best fit of the model and dependent on that chose the values of 10, 15, 25, 30, 50.

2. `min_samples_split`: This parameter specifies the minimum number of samples required to split an internal node. It affects the decision tree's ability to capture fine-grained patterns in the data. The values tested are 100, 200, 250, 275, 300. We saw which values of `min_samples_split` were actually being implemented in the best fit of the model and dependent on that chose the values of 100, 200, 250, 275, 300.

3. `criterion`: This parameter defines the quality measure used to evaluate the splits in the decision tree. The two options tested are 'gini' and 'entropy'. 'Gini' measures the impurity of a node, while 'entropy' calculates the information gain.

The hyperparameters that gave the best results were:
* `max_depth`: 10
* `min_samples_split`: 275
* `criterion`: 'entropy'

Compared to the baseline model, the final model includes additional features and uses a more advanced modeling algorithm. By incorporating information about the number of steps, cooking time, and number of ingredients, as well as one-hot encoding the categorical features of the names, the model can capture more nuanced patterns in the data. Additionally, the decision tree classifier can handle non-linear relationships and interactions between features. The grid search helped in finding the best hyperparameters, resulting in a model that is fine-tuned for the data. Overall, the final model's performance shows a significant improvement over the baseline model in overall accuracy due to the inclusion of these additional features and the use of a more sophisticated algorithm. However, this comes at the cost of the bias of guessing 5's more often; this is due to the overall frequency of 5's in the dataset, leading to possible trends being overlooked by this fact.

---

# Fairness Analysis


We decided to look into whether our model preform differently based on the number of minutes of a recipe. To further investigate this, we set up the following permuation test:


Group 1: Short recipes: recipes that take 35 minutes or more to make

Group 2: Long recipes: recipes that take less than 35 minutes to make

We chose 35 minutes as the divider between the two groups as this is the value that most evenly splits up the dataset into the groups (roughly 50% of the dataframe `to_work` is in each group).

- Null Hypothesis: Our model is fair. Its accuracy for recipes that take less than 35 minutes and 35 minutes or more are roughly the same, and any differences are due to random chance.

- Alternative Hypothesis: Our model is unfair. Its accuracy for recipes that take less than 35 minutes is not equal to the model's accuracy for recipes that take 35 minutes or more.

Evaluation Metric: Accuracy

Test Statistic: abs((Accuracy of recipes that take less than 35 minutes) - (Accuracy of recipes that take 35 minutes or more))

Significance Level: alpha = 0.05

<iframe src="assets/perm_test_recipes7.html" width=800 height=600 frameBorder=0></iframe>
Observed test statistic: 0.0143

p-value = 0.00

Because our p-value = 0.00 is less than the significance level of alpha = 0.05, we can reject the null hypothesis. There is statistically significant evidence that our model could be unfair, specifically that its accuracy for recipes that take less than 35 minutes could be not equal to its accuracy for recipes that take 35 minutes or more. It should be noted that due to not having randomized controlled trials, we cannot make definitive conclusions about the truth or falsehood of either hypothesis.

