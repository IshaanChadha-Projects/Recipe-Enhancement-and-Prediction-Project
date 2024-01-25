**Authors: Joey Kaminsky and Ishaan Chadha**

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
