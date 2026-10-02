# GitHub Upload Guide

## Repository name

`food-safety-risk-prediction-ml`

## Recommended repository description

AI/ML prototype for food-safety sample risk assessment and prioritization using Python, Scikit-learn and Flask.

## Upload

1. Create a new GitHub repository.
2. Use the repository name above.
3. Upload all files from this project.
4. Do not upload your `venv` folder.
5. Make sure `README.md` is visible on the repository home page.

## Git commands

```bash
git init
git add .
git commit -m "Initial food safety risk assessment project"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your own repository URL.

## Resume project title

Food Safety Risk Assessment using Machine Learning

## Interview explanation

"I developed an educational ML prototype that uses food-sample parameters to classify samples into lower-risk and at-risk groups for prioritization. I used Pandas for data handling, Scikit-learn pipelines for preprocessing, Logistic Regression and Random Forest for classification, F1-score and other metrics for evaluation, Joblib for model persistence, and Flask for deployment. The current dataset is synthetic, so I present it as a prototype rather than an official regulatory system."
