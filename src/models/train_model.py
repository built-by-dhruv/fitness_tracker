import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LearningAlgorithms import ClassificationAlgorithms
import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2



df = pd.read_pickle("../../data/processed/03_data_features.pkl")
df_copy = df.copy()
df.drop( ['participant','set' , 'category'] , axis = 1 , inplace = True)

X = df.drop(["label"] , axis = 1) # features
y = df["label"] # target variable

# --------------------------------------------------------------
# Create a training and test set
# --------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(  
    X ,y ,
    stratify=y,
    test_size=0.25, random_state=42)


import seaborn as sns
import matplotlib.pyplot as plt

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 5))

# Plot the 'label' value counts for the entire dataset
sns.barplot(x=df["label"].value_counts().index, 
            y=df["label"].value_counts().values, 
            ax=ax, 
            color="lightpink",
            alpha=0.5,
            # edgecolor="black",
            # linewidth=1.5,
            # hatch="."
            )

# Overlay the value counts for the training data
sns.barplot(x=y_train.value_counts().index, 
            y=y_train.value_counts().values, 
            ax=ax, 
            color="salmon",
            alpha=0.75,
            # edgecolor="black",
            # linewidth=1.5,
            # hatch="x"
            )

# Overlay the value counts for the test data
sns.barplot(x=y_test.value_counts().index, 
            y=y_test.value_counts().values, 
            ax=ax, 
            color="red",
            # alpha=0.5,
            # edgecolor="black",
            # linewidth=1.5,
            # hatch="/"
            )

# Add legend
plt.legend(["Total", "Training", "Test"])

# Show plot
plt.show()



X_train.columns
print(X_train.columns)


# --------------------------------------------------------------
# Split feature subsets
# --------------------------------------------------------------

basic_features = ['acc_x', 'acc_y', 'acc_z', 'gyc_x', 'gyc_y', 'gyc_z']
sqr_features = ['acc_r','gyc_r']
pca_features = ['pca_1', 'pca_2', 'pca_3']
rolling_features = [f for f in X_train.columns if "_temp_" in f]
freq_features = [f for f in X_train.columns if (("_freq_" in f) or ("_pse" in f)) ]
cluster_features = ['cluster']

print("Basic features: ", len(basic_features))
print("Squared features: ", len(sqr_features))
print("PCA features: ", len(pca_features))
print("Rolling features: ", len(rolling_features))
print("Frequency features: ", len(freq_features))
print("Cluster features: ", len(cluster_features))


feature_set_1 = list(set(basic_features))
feature_set_2 = list(set(sqr_features + pca_features )) + feature_set_1
feature_set_3 = list(set(rolling_features )) + feature_set_2
feature_set_4 = list(set(freq_features + cluster_features)) + feature_set_3

# --------------------------------------------------------------
# Perform forward feature selection using simple decision tree
# --------------------------------------------------------------

learner = ClassificationAlgorithms()
selected_features, ordered_features, ordered_scores =learner.forward_selection(10 , X_train, y_train)



sns.lineplot(x=range(1, len(ordered_scores) + 1), y=ordered_scores)

# --------------------------------------------------------------
# Grid search for best hyperparameters and model selection
# --------------------------------------------------------------

possible_feature_sets = [feature_set_1 ,
                         feature_set_2,
                         feature_set_3,
                         feature_set_4 ,
                         selected_features]
possible_feature_names = [
                            "Basic features",
                            "Basic + squared + PCA",
                            "Basic + squared + PCA + rolling",
                            "Basic + squared + PCA + rolling + frequency + cluster",
                            "Forward selection"
                            ]

iterations = 1
score_df = pd.DataFrame()

for i, f in zip(range(len(possible_feature_sets)), possible_feature_names):
    print("Feature set:", i)
    selected_train_X = X_train[possible_feature_sets[i]]
    selected_test_X = X_test[possible_feature_sets[i]]

    # First run non deterministic classifiers to average their score.
    performance_test_nn = 0
    performance_test_rf = 0

    for it in range(0, iterations):
        print("\tTraining neural network,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.feedforward_neural_network(
            selected_train_X,
            y_train,
            selected_test_X,
            gridsearch=False,
        )
        performance_test_nn += accuracy_score(y_test, class_test_y)

        print("\tTraining random forest,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.random_forest(
            selected_train_X, y_train, selected_test_X, gridsearch=True
        )
        performance_test_rf += accuracy_score(y_test, class_test_y)

    performance_test_nn = performance_test_nn / iterations
    performance_test_rf = performance_test_rf / iterations

    # And we run our deterministic classifiers:
    print("\tTraining KNN")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.k_nearest_neighbor(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_knn = accuracy_score(y_test, class_test_y)

    print("\tTraining decision tree")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.decision_tree(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_dt = accuracy_score(y_test, class_test_y)

    print("\tTraining naive bayes")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.naive_bayes(selected_train_X, y_train, selected_test_X)

    performance_test_nb = accuracy_score(y_test, class_test_y)

    # Save results to dataframe
    models = ["NN", "RF", "KNN", "DT", "NB"]
    new_scores = pd.DataFrame(
        {
            "model": models,
            "feature_set": f,
            "accuracy": [
                performance_test_nn,
                performance_test_rf,
                performance_test_knn,
                performance_test_dt,
                performance_test_nb,
            ],
        }
    )
    score_df = pd.concat([score_df, new_scores])


# --------------------------------------------------------------
# Create a grouped bar plot to compare the results
# --------------------------------------------------------------
score_df.sort_values(by="accuracy", ascending=False , inplace=True)
score_df.iloc[0]['feature_set']

plt.figure(figsize=(10, 10))
sns.barplot(x='model', y='accuracy', hue='feature_set', data=score_df )
plt.legend(loc='lower left')

# --------------------------------------------------------------
# Select best model and evaluate results
# --------------------------------------------------------------
best_model_df = score_df.iloc[0]

(
    class_train_y,
    class_test_y ,
    class_train_prob_y,
    class_test_prob_y,
) = learner.random_forest(X_train[feature_set_4], y_train,X_test[feature_set_4],  feature_set_4, gridsearch=True)

rf_accuracy_score = accuracy_score(y_test, class_test_y)

rf_confusion_matrix = confusion_matrix(y_test, class_test_y)

plt.figure(figsize=(20, 10))
sns.heatmap(rf_confusion_matrix, annot=True, fmt='g', cmap='Blues')

# --------------------------------------------------------------
# Select train and test data based on participant
# --------------------------------------------------------------

X  =  df_copy[df_copy['participant'] != 'A'].drop(['participant', 'category' , 'set'] , axis=1)
y   =  df_copy[df_copy['participant'] != 'A']['label']

X_train, X_test, y_train, y_test = train_test_split(  
    X ,y ,
    stratify=y,
    test_size=0.25, random_state=1)


(
    class_train_y,
    class_test_y ,
    class_train_prob_y,
    class_test_prob_y,
) = learner.random_forest(X_train[feature_set_4], y_train,X_test[feature_set_4],  feature_set_4, gridsearch=True)




# --------------------------------------------------------------
# Use best model again and evaluate results
# --------------------------------------------------------------

rf_accuracy_score = accuracy_score(y_test, class_test_y)

rf_confusion_matrix = confusion_matrix(y_test, class_test_y)

plt.figure(figsize=(20, 10))

sns.heatmap(rf_confusion_matrix, annot=True, fmt='g', cmap='Blues')


# --------------------------------------------------------------
# Try a simpler model with the selected features
# --------------------------------------------------------------
