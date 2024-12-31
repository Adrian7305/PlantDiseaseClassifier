# PlantDiseaseClassifier

Problem Statement
Agricultural losses due to plant diseases are significant. Traditional disease detection methods are prone to human error and inefficiency.

Objective
To design a machine learning-based system to automate plant disease detection, improving accuracy and reducing time and cost.

Dataset
•Source: Kaggle Plant Disease Dataset.
•Training and Validation Data Paths: 'train' and 'valid'
•Test Data Path: ‘test’
•Classes: 38 plant diseases.


2.Methodology

2.1 Data loading and Preprocessing

The dataset was divided into training, validation, and test sets using the image_dataset_from_directory function. Each image was resized to 128x128 pixels, normalized, and categorized.
Python Code:
training_set = tf.keras.utils.image_dataset_from_directory('train', image_size=(128, 128), batch_size=32, label_mode="categorical")
validation_set = tf.keras.utils.image_dataset_from_directory('valid', image_size=(128, 128), batch_size=32, label_mode="categorical")
test_set = tf.keras.utils.image_dataset_from_directory('test', image_size=(128, 128), batch_size=1, label_mode="categorical", shuffle=False)

2.2 CNN Model Architecture

The CNN model consisted of multiple layers:
1.Convolutional Layers: Extract spatial features with filters of sizes 32 to 512.
2.MaxPooling Layers: Downsample feature maps while retaining key features.
3.Dropout Layers: Mitigate overfitting during training.
4.Flatten Layer: Convert 2D feature maps into 1D vectors.
5.Dense Layers: Fully connected layers with ReLU activation for learning and softmax activation for classification.

Python Code:
cnn = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    # Additional layers...
    tf.keras.layers.Dense(38, activation='softmax')])

2.3 Training Phase

The model was compiled with the Adam optimizer and categorical cross-entropy loss. Training was conducted for 10 epochs, with accuracy and loss tracked for both training and validation datasets.
Python Code:
cnn.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
training_history = cnn.fit(training_set, validation_data=validation_set, epochs=10)

2.4 Testing Phase

The trained model was tested on unseen test data. Predictions were evaluated using a confusion matrix and classification report.
Python Code:
y_pred = cnn.predict(test_set)
predicted_categories = tf.argmax(y_pred, axis=1)
true_categories = tf.concat([y for x, y in test_set], axis=0)
Y_true = tf.argmax(true_categories, axis=1)

3. Results

3.1 Training Results

•Training Accuracy: {train_acc}
•Validation Accuracy: {val_acc}
•Loss Trend: A steady decrease in loss was observed over epochs.

Python Code:
epochs = range(1, 11)
plt.plot(epochs, training_history.history['accuracy'], color='red', label='Training Accuracy')
plt.plot(epochs, training_history.history['val_accuracy'], color='blue', label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

3.2 Testing Results

The model was tested on the test dataset, and predictions were visualized with a confusion matrix and a classification report.
1.Confusion Matrix
2.Classification Report-A detailed report providing precision, recall, F1-score, and support for each class.
3.Prediction Example-Test image with its predicted class.

1.Confusion Matrix:

Python Code:
cm = confusion_matrix(Y_true, predicted_categories)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

2.Classification Report:
Python Code:
print(classification_report(Y_true, predicted_categories, target_names=class_name))

3.Prediction Example:
Python Code:
plt.imshow(test_image)
plt.title(f"Disease Name: {predicted_class}")
plt.show()

4. Evaluation

•Model Accuracy: High accuracy on both training and validation datasets indicates a well-trained model.
Model Accuracy (Training   Dataset: 97.601)
Model Accuracy (Validation Dataset: 94.297)
Confusion Matrix Analysis: Provided insights into misclassifications.
Classification Metrics: Precision and recall were consistently high across classes, validating the model's robustness.

5. Conclusion

The CNN model demonstrated high performance in plant disease classification across 38 classes, achieving reliable results on the test dataset. This project showcases the potential of deep learning in agricultural disease detection.

Limitations
•Dataset dependency: Model performance is limited by the quality and diversity of training data.

•Applicability to real-world scenarios may require additional preprocessing steps.


6. Reference
1.Kaggle Plant Disease Dataset
2.TensorFlow Documentation
3.Scikit-learn Documentation
4.https://ijcrt.org/papers/IJCRT1801646.p
