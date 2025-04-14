# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix


# ************************* TASK 1: Model Training and Evaluation *************************
print("\n************************* TASK 1: Model Training and Evaluation *************************")

# Load the dataset
nba_stats_path = 'nba_stats.csv'  # Adjust this path to where your dataset is located
nba_stats_df = pd.read_csv(nba_stats_path).fillna(0)

# Encode the 'Pos' (position) column, turning categorical position labels into numbers
encoder = LabelEncoder()
nba_stats_df['Pos'] = encoder.fit_transform(nba_stats_df['Pos'])

# Adding new features for insights
# Points per game
nba_stats_df['PTS_per_game'] = nba_stats_df['PTS'] / nba_stats_df['G']
# Assists per game
nba_stats_df['AST_per_game'] = nba_stats_df['AST'] / nba_stats_df['G']

# Feature selection
features = ['3P%', 'FG%', 'FT%', 'TRB', 'AST', 'STL', 'BLK', 'PTS', 'PTS_per_game', 'AST_per_game']
X = nba_stats_df[features]
y = nba_stats_df['Pos']

# Splitting dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Train the model on the training set
dt_classifier.fit(X_train, y_train)

# Evaluate the model on the training set
y_train_pred = dt_classifier.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
train_conf_matrix = confusion_matrix(y_train, y_train_pred)  # Training set confusion matrix
print(f"Training Set Accuracy: {train_accuracy:.4f}")
print("Training Set Confusion Matrix:")
print(train_conf_matrix)

# Evaluate the model on the validation set
y_val_pred = dt_classifier.predict(X_val)
# Validation accuracy and confusion matrix
val_accuracy = accuracy_score(y_val, y_val_pred)
val_conf_matrix = confusion_matrix(y_val, y_val_pred)
print(f"Validation Set Accuracy: {val_accuracy:.4f}")
print("Validation Set Confusion Matrix:")
print(val_conf_matrix)

# ************************* TASK 2: Apply the Model to New Data *************************
print("\n************************* TASK 2: Apply the Model to New Data *************************")

# Load the dummy test dataset
dummy_test_path = 'dummy_test.csv'  # Adjust the path as needed
dummy_test_df = pd.read_csv(dummy_test_path).fillna(0)

# Apply the same feature engineering to the dummy test dataset
dummy_test_df['PTS_per_game'] = dummy_test_df['PTS'] / dummy_test_df['G']
dummy_test_df['AST_per_game'] = dummy_test_df['AST'] / dummy_test_df['G']
X_dummy_test = dummy_test_df[features]

# Predict on the dummy test set using the trained model
y_dummy_pred = dt_classifier.predict(X_dummy_test)

# Encode the 'Pos' column in the dummy test dataset for accuracy calculation
dummy_test_df['Pos_encoded'] = encoder.transform(dummy_test_df['Pos'])

# Dummy test set accuracy and confusion matrix
dummy_test_accuracy = accuracy_score(dummy_test_df['Pos_encoded'], y_dummy_pred)
dummy_test_conf_matrix = confusion_matrix(dummy_test_df['Pos_encoded'], y_dummy_pred)
print(f"Dummy Test Set Accuracy: {dummy_test_accuracy:.4f}")
print("Dummy Test Set Confusion Matrix:")
print(dummy_test_conf_matrix)

# ************************* TASK 3: 10-Fold Cross-Validation *************************
print("\n************************* TASK 3: 10-Fold Cross-Validation *************************")

# Perform 10-fold cross-validation to assess model generalizability
cv_scores = cross_val_score(dt_classifier, X, y, cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42), scoring='accuracy')

# Print each fold's accuracy
for i, score in enumerate(cv_scores, start=1):
    print(f"Fold {i} Accuracy: {score:.4f}")

# Print the average and standard deviation of the 10-fold cross-validation accuracies
print(f"\n10-Fold CV Average Accuracy: {cv_scores.mean():.4f}")
