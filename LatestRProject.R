# Load necessary libraries
if (!require("e1071")) install.packages("e1071")        # SVM
if (!require("caret")) install.packages("caret")        # Evaluation and cross-validation
if (!require("randomForest")) install.packages("randomForest")  # Random Forest
if (!require("nnet")) install.packages("nnet")          # Multinomial Logistic Regression
if (!require("caTools")) install.packages("caTools")    # Train-test split
if (!require("ggplot2")) install.packages("ggplot2")    # Visualisation
if (!require("dplyr")) install.packages("dplyr")        # Data manipulation

library(e1071)
library(caret)
library(randomForest)
library(nnet)
library(caTools)
library(ggplot2)
library(dplyr)

# Load the dataset
dataset <- read.csv("user_behavior_dataset.csv")

# Remove User.ID if present
dataset <- dataset %>% select(-User.ID)

# Convert target variable to factor
dataset$User.Behavior.Class <- as.factor(dataset$User.Behavior.Class)

# Convert categorical features to factors
dataset$Gender <- as.factor(dataset$Gender)
dataset$Device.Model <- as.factor(dataset$Device.Model)
dataset$Operating.System <- as.factor(dataset$Operating.System)

# EDA Section ------------------------------------------------------------
# 1. Summary of the dataset
print("Dataset Summary:")
print(summary(dataset))

# 2. Check for missing values
missing_values <- sapply(dataset, function(x) sum(is.na(x)))
print("Missing Values in Each Column:")
print(missing_values)

# 3. Distribution of the target variable
ggplot(dataset, aes(x = User.Behavior.Class)) +
  geom_bar(fill = "orange", color = "black") +
  theme_minimal() +
  labs(title = "Distribution of User Behavior Class", x = "Behavior Class", y = "Count")

# 4. Distribution of numeric variables
# Corrected Code: Distribution of Numeric Variables
library(dplyr)
library(ggplot2)
library(tidyr)

dataset %>%
  select_if(is.numeric) %>%
  pivot_longer(cols = everything(), names_to = "variable", values_to = "value") %>%
  ggplot(aes(x = value)) +
  geom_histogram(bins = 30, fill = "blue", color = "white") +
  facet_wrap(~ variable, scales = "free") +
  theme_minimal() +
  labs(title = "Distribution of Numeric Variables")

dataset %>%
  select_if(is.numeric) %>%
  pivot_longer(cols = everything(), names_to = "variable", values_to = "value") %>%
  ggplot(aes(x = variable, y = value)) +
  geom_boxplot(outlier.color = "red") +
  theme_minimal() +
  coord_flip() +
  labs(title = "Outliers in Numeric Variables")

# Corrected Code: Outliers in Numeric Variables
dataset %>%
  select_if(is.numeric) %>%  # Use select_if instead of select(where())
  gather(key = "variable", value = "value") %>%
  ggplot(aes(x = variable, y = value)) +
  geom_boxplot(outlier.color = "red") +
  theme_minimal() +
  coord_flip() +
  labs(title = "Outliers in Numeric Variables")


# Preprocessing and Model Training ----------------------------------------
# Set seed for reproducibility
set.seed(42)

# Inject noise into the dataset (simulate real-world imperfections)
# Add random Gaussian noise to numeric columns
numeric_features <- c("App.Usage.Time..min.day.", "Screen.On.Time..hours.day.", 
                      "Battery.Drain..mAh.day.", "Number.of.Apps.Installed", "Data.Usage..MB.day.", "Age")
dataset[, numeric_features] <- dataset[, numeric_features] + rnorm(nrow(dataset), mean = 0, sd = 0.1)

# Introduce some label noise by flipping class labels for 5% of the dataset
label_flip_indices <- sample(1:nrow(dataset), size = 0.05 * nrow(dataset))
dataset$User.Behavior.Class[label_flip_indices] <- sample(levels(dataset$User.Behavior.Class), 
                                                          size = length(label_flip_indices), replace = TRUE)

print(colnames(train_data))

# Save training column structure
saveRDS(colnames(train_data), "train_data_columns_manual.rds")

# Train-test split (80% training, 20% testing)
split <- sample.split(dataset$User.Behavior.Class, SplitRatio = 0.8)
train_data <- subset(dataset, split == TRUE)
test_data <- subset(dataset, split == FALSE)

# Standardise numeric features for SVM
preProc <- preProcess(train_data[, numeric_features], method = c("center", "scale"))
train_data[, numeric_features] <- predict(preProc, train_data[, numeric_features])
test_data[, numeric_features] <- predict(preProc, test_data[, numeric_features])

### Cross-validation setup for all models
train_control <- trainControl(method = "cv", number = 5)  # 5-fold cross-validation

### Multinomial Logistic Regression
model_lr_multiclass <- train(
  User.Behavior.Class ~ ., 
  data = train_data, 
  method = "multinom", 
  trControl = train_control
)
predictions_lr_multiclass <- predict(model_lr_multiclass, newdata = test_data)
confusion_matrix_lr_multiclass <- confusionMatrix(predictions_lr_multiclass, test_data$User.Behavior.Class)
print("Multinomial Logistic Regression Performance:")
print(confusion_matrix_lr_multiclass)

### Support Vector Machine (SVM)
model_svm <- train(
  User.Behavior.Class ~ ., 
  data = train_data, 
  method = "svmLinear", 
  tuneGrid = expand.grid(C = seq(0.1, 1, by = 0.1)),  # Tune the cost parameter
  trControl = train_control
)
predictions_svm <- predict(model_svm, newdata = test_data)
confusion_matrix_svm <- confusionMatrix(predictions_svm, test_data$User.Behavior.Class)
print("SVM Performance:")
print(confusion_matrix_svm)

### Random Forest
model_rf <- train(
  User.Behavior.Class ~ ., 
  data = train_data, 
  method = "rf", 
  tuneGrid = expand.grid(mtry = seq(2, 6, by = 1)),  # Test mtry values
  trControl = train_control,
  ntree = 100  # Limit number of trees to prevent overfitting
)
print(paste("Best mtry for Random Forest:", model_rf$bestTune$mtry))
predictions_rf <- predict(model_rf, newdata = test_data)
confusion_matrix_rf <- confusionMatrix(predictions_rf, test_data$User.Behavior.Class)
print("Random Forest Performance:")
print(confusion_matrix_rf)

# Display Random Forest Variable Importance
print("Random Forest Variable Importance:")
print(varImp(model_rf))


# Load necessary library
if (!require("ggplot2")) install.packages("ggplot2")
library(ggplot2)

# Create the model_accuracies data frame
model_accuracies <- data.frame(
  Model = c("Logistic Regression", "SVM", "Random Forest"),
  Accuracy = c(90.71, 95.71, 95.71)  # Accuracy values in percentage
)

# Plot bar chart
ggplot(data = model_accuracies, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", width = 0.6) + 
  theme_minimal() +
  labs(
    title = "Overall Accuracy of Models",
    x = "Model",
    y = "Accuracy (%)"
  ) +
  scale_y_continuous(limits = c(0, 100), breaks = seq(0, 100, 10)) +
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
    axis.title.x = element_text(size = 12, face = "bold"),
    axis.title.y = element_text(size = 12, face = "bold"),
    axis.text.x = element_text(size = 10),
    axis.text.y = element_text(size = 10),
    legend.position = "none"
  )

# Save the Logistic Regression model
saveRDS(model_lr_multiclass, file = "model_lr_multiclass.rds")

# Save the SVM model
saveRDS(model_svm, file = "model_svm.rds")

# Save the Random Forest model
saveRDS(model_rf, file = "model_rf.rds")

# Save the column structure
saveRDS(colnames(train_data), "train_data_columns_manual.rds")


