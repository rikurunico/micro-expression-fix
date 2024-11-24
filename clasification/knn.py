import os
import time  # Import time module
import pandas as pd
from sklearn.model_selection import (
    cross_val_score,
    KFold,
    train_test_split,
    GridSearchCV,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class KNNClassifier:
    def __init__(
        self,
        dataset_file,
        label_column,
        feature_column=None,
        except_feature_column=None,
        directory="models",
    ):
        self.dataset_file = dataset_file
        self.feature_column = feature_column
        self.except_feature_column = except_feature_column
        self.label_column = label_column
        self.directory = directory
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.label_encoder = LabelEncoder()

    def load_data(self):
        data = pd.read_csv(self.dataset_file)
        print(f"Loaded dataset from {self.dataset_file}\n")

        if self.feature_column is None and (
            self.except_feature_column is None or self.except_feature_column == [None]
        ):
            raise ValueError(
                "The 'feature_column' and 'except_feature_column' parameters are both empty. One of them must be provided."
            )

        if self.except_feature_column is not None and self.except_feature_column != [
            None
        ]:
            self.X = data.drop(self.except_feature_column, axis=1).values
        elif self.feature_column is not None:
            self.X = data[self.feature_column].values

        self.y = self.label_encoder.fit_transform(data[self.label_column].values)

    def split_data(self, test_size=0.2, random_state=0):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

        # Feature scaling
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def train_model(self, autoParams=False):
        if autoParams:
            param_grid = {
                "n_neighbors": np.arange(3, 21, 2),  # Odd values from 3 to 21
                "metric": ["euclidean", "manhattan", "minkowski"],  # Multiple metrics
            }
    
            # Use KFold with k=10
            cv = KFold(n_splits=10, shuffle=True, random_state=0)
    
            # Store results
            results = []
    
            # Iterate over all parameter combinations
            for n_neighbors in param_grid["n_neighbors"]:
                for metric in param_grid["metric"]:
                    # Create subdirectory for the metric if it doesn't exist
                    metric_directory = os.path.join(self.directory, metric)
                    if not os.path.exists(metric_directory):
                        os.makedirs(metric_directory)

                    start_time = time.time()  # Start the timer
    
                    # Train model with current parameters
                    if metric == "minkowski":
                        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, p=2)
                    else:
                        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
    
                    self.model.fit(self.X_train, self.y_train)
    
                    # Evaluate model
                    predictions = self.model.predict(self.X_test)
                    accuracy = accuracy_score(self.y_test, predictions)
                    precision = precision_score(self.y_test, predictions, average='macro')
                    recall = recall_score(self.y_test, predictions, average='macro')
                    f1 = f1_score(self.y_test, predictions, average='macro')
                    
                    # If it's a binary classification or all classes are present
                    try:
                        auc = roc_auc_score(self.y_test, self.model.predict_proba(self.X_test), multi_class="ovr")
                    except ValueError:
                        auc = "N/A"
    
                    cm = confusion_matrix(self.y_test, predictions)
    
                    # Save model and confusion matrix in the respective metric directory
                    self.save_model(n_neighbors, metric, directory=metric_directory)
                    self.save_confusion_matrix(cm, n_neighbors, metric, directory=metric_directory)
    
                    end_time = time.time()  # Stop the timer
                    elapsed_time = end_time - start_time  # Calculate the elapsed time
                    
                    results.append((n_neighbors, metric, accuracy, precision, recall, f1, auc, elapsed_time))
                    print(
                        f"Trained with n_neighbors={n_neighbors}, metric={metric}, accuracy={accuracy:.4f}, precision={precision:.4f}, recall={recall:.4f}, f1-score={f1:.4f}, AUC={auc}, time={elapsed_time:.2f} seconds"
                    )
    
            # Create dictionary to store evaluation metrics
            metrics_table = {
                "K": [],
                "Metric": [],
                "Accuracy": [],
                "Precision": [],
                "Recall": [],
                "F1-Score": [],
                "AUC": [],
                "Time (seconds)": []  # Add column for elapsed time
            }
    
            # Populate metrics table
            for n_neighbors, metric, accuracy, precision, recall, f1, auc, elapsed_time in results:
                metrics_table["K"].append(n_neighbors)
                metrics_table["Metric"].append(metric)
                metrics_table["Accuracy"].append(accuracy)
                metrics_table["Precision"].append(precision)
                metrics_table["Recall"].append(recall)
                metrics_table["F1-Score"].append(f1)
                metrics_table["AUC"].append(auc)
                metrics_table["Time (seconds)"].append(elapsed_time)  # Add elapsed time to the table
    
            # Convert to DataFrame for better formatting
            metrics_df = pd.DataFrame(metrics_table)
            print("\nMetrics Table:")
            print(metrics_df)

            # Save the metrics table to CSV
            metrics_df.to_csv('metrics_table.csv', index=False)
            print("Metrics table saved to 'metrics_table.csv'")
            
            # Plot the graphs
            self.plot_metrics(metrics_df)

        else:
            print("Manual parameter training is not yet implemented.")

    def plot_metrics(self, metrics_df):
        # Plot Accuracy vs K
        plt.figure(figsize=(10, 6))
        sns.lineplot(x="K", y="Accuracy", data=metrics_df, marker="o", hue="Metric")
        plt.title("Accuracy vs K")
        plt.xlabel("K (Number of Neighbors)")
        plt.ylabel("Accuracy")
        plt.legend(title="Metric")
        plt.grid(True)
        plt.savefig("accuracy_vs_k.png")  # Save the plot
        plt.show()

        # Plot Time vs K
        plt.figure(figsize=(10, 6))
        sns.lineplot(x="K", y="Time (seconds)", data=metrics_df, marker="o", hue="Metric")
        plt.title("Time vs K")
        plt.xlabel("K (Number of Neighbors)")
        plt.ylabel("Time (seconds)")
        plt.legend(title="Metric")
        plt.grid(True)
        plt.savefig("time_vs_k.png")  # Save the plot
        plt.show()

    def save_model(self, n_neighbors, metric, directory):
        output_model_path = directory
        if not os.path.exists(output_model_path):
            os.makedirs(output_model_path)

        # Save the KNN model
        model_filename = f"knn_model_n_neighbors_{n_neighbors}_metric_{metric}.joblib"
        joblib.dump(self.model, os.path.join(output_model_path, model_filename))
        print(f"Model saved to {os.path.join(output_model_path, model_filename)}")

        # Save the label encoder
        label_encoder_filename = (
            f"label_encoder_n_neighbors_{n_neighbors}_metric_{metric}.joblib"
        )
        joblib.dump(
            self.label_encoder, os.path.join(output_model_path, label_encoder_filename)
        )
        print(
            f"Label encoder saved to {os.path.join(output_model_path, label_encoder_filename)}"
        )

    def save_confusion_matrix(self, cm, n_neighbors, metric, directory="confusion_matrix"):
        plt.figure(figsize=(10, 7))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_,
        )
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.title(f"Confusion Matrix (n_neighbors={n_neighbors}, metric={metric})")
        plt.savefig(
            f"{directory}/confusion_matrix_n_neighbors_{n_neighbors}_metric_{metric}.png"
        )
        plt.close()  # Close the plot to avoid displaying it immediately


# Example usage:
# knn_classifier = KNNClassifier(dataset_file='data.csv', label_column='target', feature_column=['feature1', 'feature2'])
# knn_classifier.load_data()
# knn_classifier.split_data()
# knn_classifier.train_model(autoParams=True)
# knn_classifier.evaluate_model()
# knn_classifier.save_model()
