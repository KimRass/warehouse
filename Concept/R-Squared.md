# R-Squared
- 출처 : https://statisticsbyjim.com/regression/r-squared-invalid-nonlinear-regression/
- ![image.png](/wikis/2670857615939396646/files/2773066282071095432)
- Explained variance + Error variance = Total variance.
- However, this math works out correctly only for linear regression models. In nonlinear regression, these underlying assumptions are incorrect. Explained variance + Error variance DO NOT add up to the total variance! The result is that R-squared isn’t necessarily between 0 and 100%.
- If you use R-squared for nonlinear models, their study indicates you will experience the following problems:
R-squared is consistently high for both excellent and appalling models.
R-squared will not rise for better models all of the time.
If you use R-squared to pick the best model, it leads to the proper model only 28-43% of the time.
# Adjusted R-Squared
