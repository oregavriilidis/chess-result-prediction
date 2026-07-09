 Chess Insights – Chess Result Prediction

## About the Project

Chess Insights is a machine learning project that predicts the outcome of a chess game based on information available before the game starts.

The user provides:

* White player's rating
* Black player's rating
* Opening played
* Time control
* Expected number of moves

Using these inputs, the model predicts whether the game is more likely to end in a **White win**, **Draw**, or **Black win**.

I created this project to combine two of my interests: **chess** and **data science**. The goal was not only to build a prediction model, but also to go through the complete machine learning workflow, from data generation and preprocessing to training, evaluation, and making predictions from the command line.

## Project Structure


chess-insights/
├── data/
├── images/
├── models/
├── src/
├── tests/
├── requirements.txt
└── README.md


## Features

* Generates a realistic synthetic chess dataset
* Cleans and prepares the data
* Performs exploratory data analysis (EDA)
* Trains a Random Forest classifier
* Predicts chess game results from the command line
* Includes unit tests for the preprocessing pipeline


## Example


python src/predict.py --white 1900 --black 1500 --opening "Ruy Lopez" --time-control Rapid --moves 45


Example output:

```text
Prediction: White Win

Probabilities:
White: 54.7%
Draw: 24.2%
Black: 21.1%
```

## Future Improvements

* Train the model on real Lichess games
* Compare multiple machine learning models
* Build a Streamlit web application
* Add move-by-move analysis using python-chess



Developed as a personal portfolio project to apply machine learning techniques to chess game prediction.

