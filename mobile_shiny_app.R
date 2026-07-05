# Mobile Device Usage & User Behavior Prediction
# Shiny Application
# Author: Tochukwu Aroh

library(shiny)
library(shinydashboard)
library(ggplot2)
library(dplyr)
library(randomForest)
library(caret)

# ── Load dataset and train model on startup ───────────────────────────────────
dataset <- read.csv("user_behavior_dataset.csv")
dataset <- dataset %>% select(-User.ID)
dataset$User.Behavior.Class <- as.factor(dataset$User.Behavior.Class)
dataset$Gender              <- as.factor(dataset$Gender)
dataset$Device.Model        <- as.factor(dataset$Device.Model)
dataset$Operating.System    <- as.factor(dataset$Operating.System)

# Behavior class labels
class_labels <- c(
  "1" = "Light User",
  "2" = "Moderate User", 
  "3" = "Heavy User",
  "4" = "Very Heavy User",
  "5" = "Extreme User"
)

class_colors <- c(
  "1" = "#02C39A",
  "2" = "#028090",
  "3" = "#F57F17",
  "4" = "#E65100",
  "5" = "#C62828"
)

class_descriptions <- c(
  "1" = "Minimal daily usage. Uses phone primarily for calls and basic tasks. Low app engagement and data consumption.",
  "2" = "Occasional usage throughout the day. Moderate app usage and screen time. Balanced digital lifestyle.",
  "3" = "Regular and frequent phone usage. High app engagement and considerable screen time. Active digital user.",
  "4" = "Intensive daily usage. Very high screen time and data consumption. Phone is central to daily activities.",
  "5" = "Extremely high usage. Maximum screen time, data, and app engagement. Phone dependency is significant."
)

# Numeric features for scaling
numeric_features <- c("App.Usage.Time..min.day.", "Screen.On.Time..hours.day.",
                      "Battery.Drain..mAh.day.", "Number.of.Apps.Installed",
                      "Data.Usage..MB.day.", "Age")

# Train Random Forest model
set.seed(42)
split       <- sample(1:nrow(dataset), size = 0.8 * nrow(dataset))
train_data  <- dataset[split, ]
test_data   <- dataset[-split, ]

preProc     <- preProcess(train_data[, numeric_features], method = c("center", "scale"))
train_scaled <- train_data
test_scaled  <- test_data
train_scaled[, numeric_features] <- predict(preProc, train_data[, numeric_features])
test_scaled[, numeric_features]  <- predict(preProc, test_data[, numeric_features])

rf_model <- randomForest(User.Behavior.Class ~ ., data = train_scaled, ntree = 100, mtry = 3)

# Model accuracy on test set
test_preds <- predict(rf_model, test_scaled)
accuracy   <- round(mean(test_preds == test_scaled$User.Behavior.Class) * 100, 1)

# ── UI ────────────────────────────────────────────────────────────────────────
ui <- dashboardPage(
  skin = "blue",

  dashboardHeader(
    title = span(icon("mobile-alt"), " Mobile Behavior Intelligence"),
    titleWidth = 320
  ),

  dashboardSidebar(
    width = 320,
    sidebarMenu(
      menuItem("Predict Behavior",   tabName = "predict",  icon = icon("brain")),
      menuItem("Data Overview",      tabName = "overview", icon = icon("chart-bar")),
      menuItem("Model Performance",  tabName = "model",    icon = icon("trophy")),
      menuItem("About",              tabName = "about",    icon = icon("info-circle"))
    )
  ),

  dashboardBody(
    tags$head(tags$style(HTML("
      .content-wrapper { background-color: #F7F9FA; }
      .box { border-radius: 8px; }
      .prediction-box { 
        padding: 20px; border-radius: 10px; text-align: center;
        margin: 10px 0; color: white; font-weight: bold;
      }
      .info-text { color: #475569; font-size: 14px; }
    "))),

    tabItems(

      # ── Tab 1: Predict ─────────────────────────────────────────────────────
      tabItem(tabName = "predict",
        fluidRow(
          box(title = "Enter User Profile", status = "primary", solidHeader = TRUE,
              width = 5,
              sliderInput("app_usage",    "App Usage Time (min/day)",  0, 600, 200, step = 10),
              sliderInput("screen_time",  "Screen On Time (hours/day)", 0, 12,  4,   step = 0.5),
              sliderInput("battery",      "Battery Drain (mAh/day)",   300, 3000, 1000, step = 50),
              sliderInput("num_apps",     "Number of Apps Installed",  10, 100, 40,  step = 5),
              sliderInput("data_usage",   "Data Usage (MB/day)",       100, 3000, 500, step = 50),
              sliderInput("age",          "Age",                       18, 70, 30),
              selectInput("gender",       "Gender",        choices = c("Male", "Female")),
              selectInput("os",           "Operating System",
                          choices = c("Android", "iOS")),
              selectInput("device",       "Device Model",
                          choices = c("Samsung Galaxy S21", "iPhone 12",
                                      "Google Pixel 5", "OnePlus 9",
                                      "Xiaomi Mi 11")),
              actionButton("predict_btn", "Predict Behavior Class",
                           class = "btn-primary btn-lg btn-block",
                           icon = icon("magic"))
          ),

          column(7,
            uiOutput("prediction_result"),
            br(),
            box(title = "Feature Importance", status = "info",
                solidHeader = TRUE, width = 12,
                plotOutput("importance_plot", height = "300px"))
          )
        )
      ),

      # ── Tab 2: Data Overview ───────────────────────────────────────────────
      tabItem(tabName = "overview",
        fluidRow(
          valueBox(nrow(dataset), "Total Records",     icon = icon("users"),    color = "blue"),
          valueBox(ncol(dataset), "Features",          icon = icon("columns"),  color = "teal"),
          valueBox(5,             "Behavior Classes",  icon = icon("layer-group"), color = "green"),
          valueBox(paste0(accuracy, "%"), "Model Accuracy", icon = icon("check-circle"), color = "orange")
        ),
        fluidRow(
          box(title = "Distribution of User Behavior Classes", status = "primary",
              solidHeader = TRUE, width = 6,
              plotOutput("class_dist_plot", height = "300px")),
          box(title = "Screen Time vs App Usage by Class", status = "info",
              solidHeader = TRUE, width = 6,
              plotOutput("scatter_plot", height = "300px"))
        ),
        fluidRow(
          box(title = "Numeric Feature Distributions", status = "warning",
              solidHeader = TRUE, width = 12,
              plotOutput("feature_dist_plot", height = "350px"))
        )
      ),

      # ── Tab 3: Model Performance ───────────────────────────────────────────
      tabItem(tabName = "model",
        fluidRow(
          box(title = "Model Comparison", status = "success", solidHeader = TRUE,
              width = 6,
              plotOutput("model_comparison_plot", height = "300px")),
          box(title = "Random Forest Confusion Matrix", status = "primary",
              solidHeader = TRUE, width = 6,
              plotOutput("confusion_matrix_plot", height = "300px"))
        ),
        fluidRow(
          box(title = "Model Details", status = "info", solidHeader = TRUE,
              width = 12,
              tableOutput("model_metrics_table"))
        )
      ),

      # ── Tab 4: About ──────────────────────────────────────────────────────
      tabItem(tabName = "about",
        fluidRow(
          box(title = "About This Application", status = "primary",
              solidHeader = TRUE, width = 12,
              h4("Mobile Device Usage and User Behavior Analysis"),
              p("This application classifies mobile device users into 5 behavioral categories
                based on their usage patterns. It is built using machine learning models
                trained on 700 mobile device user records."),
              h4("Behavior Classes"),
              tags$table(class = "table table-bordered",
                tags$tr(tags$th("Class"), tags$th("Label"), tags$th("Description")),
                tags$tr(tags$td("1"), tags$td("Light User"),      tags$td("Minimal daily usage")),
                tags$tr(tags$td("2"), tags$td("Moderate User"),   tags$td("Occasional usage throughout the day")),
                tags$tr(tags$td("3"), tags$td("Heavy User"),      tags$td("Regular and frequent phone usage")),
                tags$tr(tags$td("4"), tags$td("Very Heavy User"), tags$td("Intensive daily usage")),
                tags$tr(tags$td("5"), tags$td("Extreme User"),    tags$td("Extremely high usage and dependency"))
              ),
              h4("Models Used"),
              tags$ul(
                tags$li("Logistic Regression — 90.71% accuracy"),
                tags$li("Support Vector Machine (SVM) — 95.71% accuracy"),
                tags$li("Random Forest — 95.71% accuracy (selected for deployment)")
              ),
              h4("Tech Stack"),
              p("R | randomForest | caret | ggplot2 | Shiny | shinydashboard"),
              h4("Author"),
              p("Tochukwu Aroh — Data Scientist & ML Engineer")
          )
        )
      )
    )
  )
)

# ── SERVER ────────────────────────────────────────────────────────────────────
server <- function(input, output, session) {

  # Prediction
  prediction <- eventReactive(input$predict_btn, {
    new_data <- data.frame(
      App.Usage.Time..min.day.  = input$app_usage,
      Screen.On.Time..hours.day. = input$screen_time,
      Battery.Drain..mAh.day.   = input$battery,
      Number.of.Apps.Installed  = input$num_apps,
      Data.Usage..MB.day.       = input$data_usage,
      Age                       = input$age,
      Gender                    = factor(input$gender, levels = levels(dataset$Gender)),
      Device.Model              = factor(input$device,  levels = levels(dataset$Device.Model)),
      Operating.System          = factor(input$os,      levels = levels(dataset$Operating.System))
    )
    new_data[, numeric_features] <- predict(preProc, new_data[, numeric_features])
    predict(rf_model, new_data)
  })

  output$prediction_result <- renderUI({
    req(input$predict_btn)
    pred  <- as.character(prediction())
    label <- class_labels[pred]
    color <- class_colors[pred]
    desc  <- class_descriptions[pred]

    tagList(
      div(class = "prediction-box",
          style = paste0("background-color:", color),
          h2(icon("mobile-alt"), paste("Predicted Class:", pred)),
          h3(label),
          p(desc)
      ),
      box(title = "What This Means", status = "warning", solidHeader = TRUE,
          width = 12,
          p(class = "info-text", desc),
          p(class = "info-text",
            strong("App Usage: "), input$app_usage, " min/day | ",
            strong("Screen Time: "), input$screen_time, " hrs/day | ",
            strong("Data Usage: "), input$data_usage, " MB/day")
      )
    )
  })

  # Feature importance plot
  output$importance_plot <- renderPlot({
    imp <- importance(rf_model)
    imp_df <- data.frame(
      Feature    = rownames(imp),
      Importance = imp[, 1]
    ) %>% arrange(desc(Importance)) %>% head(8)

    imp_df$Feature <- gsub("\\.\\..*", "", imp_df$Feature)
    imp_df$Feature <- gsub("\\.", " ", imp_df$Feature)

    ggplot(imp_df, aes(x = reorder(Feature, Importance), y = Importance)) +
      geom_col(fill = "#028090", alpha = 0.85) +
      coord_flip() +
      theme_minimal(base_size = 12) +
      labs(title = "Top Feature Importance (Random Forest)",
           x = "", y = "Mean Decrease Gini") +
      theme(plot.title = element_text(face = "bold"))
  })

  # Class distribution
  output$class_dist_plot <- renderPlot({
    ggplot(dataset, aes(x = User.Behavior.Class, fill = User.Behavior.Class)) +
      geom_bar(show.legend = FALSE) +
      scale_fill_manual(values = unname(class_colors)) +
      theme_minimal(base_size = 12) +
      labs(title = "Distribution of Behavior Classes",
           x = "Behavior Class", y = "Count") +
      theme(plot.title = element_text(face = "bold"))
  })

  # Scatter plot
  output$scatter_plot <- renderPlot({
    ggplot(dataset, aes(x = App.Usage.Time..min.day.,
                        y = Screen.On.Time..hours.day.,
                        color = User.Behavior.Class)) +
      geom_point(alpha = 0.6, size = 2) +
      scale_color_manual(values = unname(class_colors),
                         labels = unname(class_labels),
                         name = "Behavior Class") +
      theme_minimal(base_size = 12) +
      labs(title = "App Usage vs Screen Time",
           x = "App Usage Time (min/day)",
           y = "Screen On Time (hrs/day)") +
      theme(plot.title = element_text(face = "bold"))
  })

  # Feature distributions
  output$feature_dist_plot <- renderPlot({
    dataset %>%
      select_if(is.numeric) %>%
      tidyr::pivot_longer(cols = everything(),
                          names_to = "variable", values_to = "value") %>%
      mutate(variable = gsub("\\.\\..*", "", variable),
             variable = gsub("\\.", " ", variable)) %>%
      ggplot(aes(x = value)) +
      geom_histogram(bins = 25, fill = "#028090", color = "white", alpha = 0.8) +
      facet_wrap(~ variable, scales = "free", ncol = 3) +
      theme_minimal(base_size = 11) +
      labs(title = "Distribution of Numeric Features") +
      theme(plot.title = element_text(face = "bold"))
  })

  # Model comparison
  output$model_comparison_plot <- renderPlot({
    model_df <- data.frame(
      Model    = c("Logistic Regression", "SVM", "Random Forest"),
      Accuracy = c(90.71, 95.71, 95.71)
    )
    ggplot(model_df, aes(x = reorder(Model, Accuracy), y = Accuracy, fill = Model)) +
      geom_col(show.legend = FALSE, width = 0.6) +
      scale_fill_manual(values = c("#028090", "#02C39A", "#0B2545")) +
      geom_text(aes(label = paste0(Accuracy, "%")),
                hjust = -0.1, fontface = "bold", size = 4) +
      coord_flip() +
      theme_minimal(base_size = 12) +
      scale_y_continuous(limits = c(0, 105)) +
      labs(title = "Model Accuracy Comparison",
           x = "", y = "Accuracy (%)") +
      theme(plot.title = element_text(face = "bold"))
  })

  # Confusion matrix
  output$confusion_matrix_plot <- renderPlot({
    preds <- predict(rf_model, test_scaled)
    cm    <- as.data.frame(table(Predicted = preds,
                                  Actual    = test_scaled$User.Behavior.Class))
    ggplot(cm, aes(x = Actual, y = Predicted, fill = Freq)) +
      geom_tile(color = "white") +
      geom_text(aes(label = Freq), fontface = "bold", size = 5) +
      scale_fill_gradient(low = "#E8F4F8", high = "#028090") +
      theme_minimal(base_size = 12) +
      labs(title = "Confusion Matrix — Random Forest",
           x = "Actual Class", y = "Predicted Class") +
      theme(plot.title = element_text(face = "bold"))
  })

  # Model metrics table
  output$model_metrics_table <- renderTable({
    data.frame(
      Model     = c("Logistic Regression", "SVM", "Random Forest"),
      Accuracy  = c("90.71%", "95.71%", "95.71%"),
      Recall    = c("90.71%", "95.71%", "95.71%"),
      Precision = c("91.20%", "96.10%", "96.00%"),
      F1_Score  = c("90.80%", "95.80%", "95.80%"),
      Selected  = c("No", "No", "Yes — Deployed")
    )
  }, striped = TRUE, bordered = TRUE, hover = TRUE)
}

# ── Run ───────────────────────────────────────────────────────────────────────
shinyApp(ui = ui, server = server)
