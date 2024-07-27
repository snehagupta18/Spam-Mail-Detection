This project constructs a machine learning-based system to classify emails as spam or legitimate (ham) with high accuracy.

**Project Goal:**

* Develop a robust spam filter to automatically categorize emails, minimizing the influx of unwanted messages.

**Data and Preprocessing:**

* Utilizes the `mail_data.csv` dataset containing email messages and their corresponding classifications (spam or ham).
* Handles missing values by replacing them with empty strings (`''`).
* Transforms categorical labels (`spam` and `ham`) into numerical values (0 for spam, 1 for ham) for model compatibility.
* Separates email messages (`Message`) and their categories (`Category`) into separate features (`X`) and target variable (`Y`).

**Model Training and Evaluation:**

* Splits the data into training and testing sets (80%/20% split) using `train_test_split` to ensure model generalizability.
* Employs TF-IDF Vectorizer (`TfidfVectorizer`) to transform textual email content into numerical feature vectors suitable for machine learning algorithms.
* Utilizes Logistic Regression as the classification model to learn the patterns between email features and spam/ham labels.
* Trains the model on the training data (`X_train_features`, `Y_train`).
* Evaluates model performance on both training and testing data using accuracy score.

**Key Findings:**

* The Logistic Regression model achieves an accuracy of 96.70% on the test data, demonstrating its effectiveness in identifying spam emails.

**Future Work:**

* Explore the use of other machine learning algorithms like Naive Bayes or Support Vector Machines (SVM) for potentially improved performance.
* Implement hyperparameter tuning to optimize the chosen model's parameters.
* Consider incorporating techniques like stemming or lemmatization for text pre-processing to further improve feature extraction.
* Develop a user-friendly interface that integrates the model for real-world spam filtering.

**Dependencies:**

* numpy
* pandas
* sklearn (including model_selection, feature_extraction.text, linear_model, metrics)

**Instructions:**

1. Ensure the dependencies are installed.
2. Place the `mail_data.csv` file in the project directory.
3. Run the Python script to execute the model training, evaluation, and spam classification.

**Note:**

This project provides a foundation for building a spam detection system using machine learning. The accuracy can be further improved with additional techniques and a larger training dataset.
