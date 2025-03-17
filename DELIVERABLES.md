# Project Title: NBA Lineup Prediction for Optimized Team Performance

## Objective
The goal of this project is to design and develop a machine learning model that predicts the optimal fifth player for a home team in an NBA game, given partial lineup data and other game-related features. The model should maximize the home team's overall performance.

## Project Details

### Dataset
You have been provided with a dataset containing lineups of all NBA games from 2007 to 2015. The dataset includes:
- Game-related features
- Player statistics
- Team compositions
- Game outcomes

### Feature Restrictions
Not all features from the dataset are permitted for use in the model. A separate file will be provided that specifies which features are allowed. You must strictly adhere to this file when selecting input features for the model.

### Training and Testing
- **Training Data:** Historical data of NBA games with full lineups and allowed features, including the performance outcomes of teams.
- **Test Data:** Each test sample will be selected from game segments where the home team's performance (outcome=1) is better than the away team's performance. Each test sample will include:
  - The lineup of four players from the home team.
  - The lineup of five players from the away team.
  - Other relevant game-related features.

### NBA Test Data
The NBA test data, necessary for the final evaluation of your project, is now available for download. The file, named `NBA_test.csv`, contains 1,000 test cases. Please note that one home team player has been randomly removed from each case. The names of the removed players are listed in a separate file, `NBA_test_labels.csv`. I will distribute the test data to each team. You are required to test your model against this data and add a slide to your slide deck that reports the results of your testing.

### Task Requirements
- Build a machine learning model using the provided training data.
- Predict the fifth player for the home team based on the test data. The predicted player should optimize the home team’s performance based on historical patterns and features.
- Create one slide that reports the following information:
  - Number of matches per year within the test dataset.
  - Average number of matches across the entire dataset.

### Important Deadlines
- **Submission Deadline:** The deadline to submit your slides is March 19th, 11:59 AM. Please be aware that this deadline is final, and there will be no extensions. Additionally, once submitted, slides cannot be changed.
- **Testing and Verification:** Please ensure that your code is thoroughly tested before submission. We will be running your code independently to verify that the results produced match those reported on your slide.

### Guidelines and Constraints

#### Feature Usage
Only the features specified in the provided file may be used for model training and testing.

#### Evaluation Metric
The model's performance will be evaluated based on:
- The predicted player’s historical contribution to team success in similar scenarios.
- Team performance metrics derived from the predicted lineup.

### Model Explainability
You must include an explanation of how the model makes predictions, particularly the reasoning behind selecting a specific player as the fifth member.

### Data Preprocessing
You are required to perform:
- Cleaning of the dataset (handling missing values, standardizing formats, etc.).
- Feature selection based on the provided allowed feature file.

### Prediction Constraints
- The recommended fifth player must be chosen from the roster of eligible players in the dataset.
- Models must avoid recommending players who are absent due to injury or other constraints (use provided metadata).

## Code and Documentation Repository
All code and the final project report must be maintained in a GitHub repository. The repository must be well-organized with a README file that explains how to run the code and interpret the results.

## Deliverables

### GitHub Repository
- The complete project code.
- The project report (in GitHub and PDF format) documenting the approach, methods, and results.
- A README file detailing:
  - Project objectives.
  - Instructions for setting up and running the code.
  - An overview of the results.

### Model Implementation
- The final machine learning model with properly documented code.
- A clear explanation of the model’s architecture and choice of algorithm(s).

### Prediction Outputs
- Submit predictions for the test data (fifth player recommendations).
- **Format:** Game_ID, Home_Team, Fifth_Player

### Report
- Detailed explanation of the data preprocessing steps, feature selection, and model evaluation.
- Include visualizations and analysis of the results.

### Presentation Details

**Duration:** Each team will have 10 minutes for their presentation, including time for Q&A.

**Content Requirements:** Your presentation must cover the following areas:
- **Methodology:** Outline the methods used in your project.
- **Feature Engineering:** Describe the features engineered for your model.
- **Selected Model:** Discuss why you chose your particular model.
- **Training Process:** Explain how you trained your model.
- **Pre and Post Processing:** Detail any preprocessing or postprocessing steps.
- **Evaluation:** Present how you evaluated your model.
- **Performance:** Include your team’s performance against the provided test data.

**Important Notes:**
- Presentations will take place during regular class hours, and each team will be assigned a specific time slot on their respective days.
- All team members must be present and participate in the presentation. This is a crucial part of your project grade, and attendance is mandatory.
- Please ensure that your presentation is concise, informative, and well-organized.

## Evaluation Criteria
- **Model Accuracy (40%):** Effectiveness of the model in predicting the optimal fifth player.
- **Adherence to Feature Constraints (20%):** Proper use of only the allowed features.
- **Code Quality and GitHub Usage (10%):** Organization, clarity, and thoroughness of the GitHub repository.
- **Presentation (30%):** Quality and thoroughness of the documentation and presentation.

This project integrates machine learning, data analytics, and sports science into a practical application, emphasizing collaboration and clear communication through GitHub.

## Deliverables
1. Enter the GitHub information by March 17. The last time to update your code is March 17, 11:59 pm. 
   [GitHub Information Form](https://docs.google.com/forms/d/e/1FAIpQLSeQzWiWaNdm8Ve90KZosVYVJ6TH8Yz76hWurOeFI46fJ2FDkw/viewform?usp=sharing)

2. Submit your presentation slides in ppt, pptx, or pdf format by March 19th, 11:59 am. During your presentation, you will be asked to walk through your GitHub page too.