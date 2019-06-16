required_packages <- c("caret", "rpart","rpart.plot", "randomForest", "ggplot2", "sf", "reshape2",
                      "rnaturalearth", "corrplot", "dplyr")
packages_missing <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(packages_missing)) install.packages(packages_missing)
rm(required_packages, packages_missing)

library(ggplot2)
library(corrplot)      # Draw correlation plot
library(caret)
library(dplyr)
library(reshape2)      # Melt function (line 84)
library(rpart)         # Decision tree
library(rpart.plot)    # Plot decision tree
library(randomForest)  # RandomForest
library(stats)         # GLM
library(sf)            # Draw map
library(rnaturalearth) # Draw earth map
library(data.table)

# Set ggplot theme
theme_set(theme_bw())


data <- read.csv("dataset.csv", header=TRUE, sep=";", dec=".")

head(data)
summary(data)

# Variable for markdown
data_init <- data

#####################################
# MAP Overview                      #
#####################################
# Reduce the number of points (mean per image)
data_mean_point <- aggregate(data[, 3:4], list(data$image_id), mean) 
colnames(data_mean_point)[1] <- "image_id"
# Load world map
world <- ne_countries(scale = "medium", returnclass = "sf")
# Plot map
map1 <- ggplot(data = world) +
            geom_sf() + 
            xlab("Latitude") + ylab("Longitude") +
            ggtitle("World map", subtitle = paste0("(", nrow(data_mean_point), " samples area)")) + 
            geom_point(data = data_mean_point, aes(x = data_mean_point$longitude,
                                                   y = data_mean_point$latitude,
                                                   colour = data_mean_point$image_id),
                       size=5, 
                       show.legend = FALSE)

# Show points for one image
# Select first image
image_ex <- as.character(data$image_id[[1]])
df_img <- subset(data, data$image_id == image_ex)

# Display map
Cloud <- as.factor(df_img$cloud)
map2 <- ggplot(data = world) +
            geom_sf() + 
            scale_x_continuous(limits = c(min(df_img$longitude), max(df_img$longitude))) +
            scale_y_continuous(limits = c(min(df_img$latitude), max(df_img$latitude))) +
            xlab("Latitude") + ylab("Longitude") +
            ggtitle("Labelled pixels from one satellite image", subtitle = "(Spain)") + 
            geom_point(data = df_img, aes(x = df_img$longitude,
                                          y = df_img$latitude, 
                                          colour = Cloud),
                       size=2)




#####################################
# DRAW PLOT                         #
#####################################

draw_count_per_methods <- function(df){
  # Count number of cloud / not cloud row for a given column
  count_group <- function(col){
    col <- as.list(col)
    return (c(sum(col == 0), sum(col == 1)))
  }
  
  df <- as.data.frame(apply(data, 2, count_group))
  df$cloud <- c("Not Cloud", "Cloud")
  df <- setNames(data.frame(t(df[,-1])), df[,1])
  df$methods <- rownames(df)
  df <- melt(df, id.vars="methods", measure.vars=c("Not Cloud","Cloud"))
  
  plot <- ggplot(data=df, aes(x=methods, y=value, fill=variable)) +
              geom_bar(stat="identity", position=position_dodge()) +
              ggtitle("Number of cloud/not cloud prediction per method")
  return(plot)
}

draw_correlation_plot <- function(df){
  correlations <- cor(df[,2:14])
  plot <- corrplot(correlations, method="circle")
  return(plot)
}

#####################################
# PRE PROCESSING                    #
#####################################

# CLASS
setClass(Class="Results",
         representation(
           logistic_model="numeric",
           decision_tree="numeric",
           prune_tree="numeric",
           random_forest="numeric"
         )
)
Results <- new("Results",
               logistic_model=-1,
               decision_tree=-1,
               prune_tree=-1,
               random_forest=-1)

# FUNCTIONS

# Tranform a dataframe to factor
set_as_factor <- function(data){
  return (as.data.frame(apply(data, 2, as.factor)))
}

# Tranform a dataframe to numeric
set_as_numeric <- function(data){
  return (as.data.frame(apply(data,2, as.numeric)))
}

# PRE-PROCESS

# remove useless columns
cols_to_remove <- c("image_id", "latitude", "longitude" )
data <- data[, ! names(data) %in% cols_to_remove, drop = F]

feature_col_names <- names(data)[2:14]

# Convert first column
data <- within(data, cloud <- factor(cloud, labels = c(0, 1)))

apply(data, 2, typeof)


# See the different possible results of each method
output_methods <- data[, feature_col_names]
unique_values_per_columns <- apply(output_methods, 2, unique)
unique_values_per_columns

# Select rows having at least one -1 (methods failled)
boolean_vector <- apply(data, 1, function(r) any(r %in% c(-1)))
paste("Number of invalid rows: ", sum(boolean_vector))
paste("Number of invalid images: ", length(unique(data[boolean_vector,]$image_id)))

# Keep valid pixels
data <- data[!boolean_vector,]

# Image number
paste("Number of images after removing invalid pixels: ", length(unique(data$image_id)))

head(data)

plot1 <- draw_count_per_methods(data)
plot1
plot2 <- draw_correlation_plot(data)

# Prediction formula 
predictor_formula <- paste("cloud ~", paste(feature_col_names, collapse = " + "))
predictor_formula

#####################################
# Create training - test dataset    #
#####################################


set.seed(2019)
test_index <- createDataPartition(y = data$cloud, times = 1, p = 0.2, list = FALSE)
train_set <- data[-test_index, ]
test_set <- data[test_index, ]

train_x <- train_set[, feature_col_names]
train_y <- test_set[, feature_col_names]


# Check correct partition:
paste("Training set:", nrow(train_set), "rows")
paste("Test set:", nrow(test_set), "rows")


#####################################
# Logistic regression               #
#####################################
# Transform columns as numeric
train_set <- set_as_numeric(train_set)
test_set <- set_as_numeric(test_set)

# Train the model using the training sets
model_glm.fit <- glm(predictor_formula, data = train_set, family = binomial)
# Test the model on test_set
model_glm.probs <- predict(model_glm.fit, 
                     newdata = test_set, 
                     type = "response")
# Convert the proba response with 0.5 threshold
model_glm.pred <- as.factor(ifelse(model_glm.probs > 0.5, 1, 0))

# Results
conf_mat_glm <- confusionMatrix(model_glm.pred, as.factor(test_set$cloud))
Results@logistic_model <- conf_mat_glm$overall['Accuracy']
paste("Accuracy logistic model: ", Results@logistic_model)


#####################################
# Decision Tree                     #
#####################################

train_set <- set_as_factor(train_set)
test_set <- set_as_factor(test_set)

# Build model
model_tree <- rpart(predictor_formula, data=train_set, method="class", cp = 0.0001)

# printcp(model_tree) # display the results 
# plotcp(model_tree)  # visualize cross-validation results 
# summary(model_tree) # detailed summary of splits

rpart.plot(model_tree, cex.main=2,
     main="Classification Tree for Clouds",
     box.palette="RdBu", shadow.col="gray", nn=TRUE)

# Evaluate model
preds <- predict(object = model_tree, newdata = test_set, type = c("class"))
conf_mat <- confusionMatrix(preds, test_set$cloud)
Results@decision_tree <- conf_mat$overall['Accuracy']
paste("Accuracy: ", Results@decision_tree)

# Let's try to prune the tree to reduce overfitting
# prune the tree 
prune_tree<- prune(model_tree, cp=model_tree$cptable[which.min(model_tree$cptable[,"xerror"]),"CP"])

# plot the pruned tree
rpart.plot(prune_tree, cex.main=2,
       main="Classification Prune Tree for Clouds",
       box.palette="RdBu", shadow.col="gray", nn=TRUE)


# Evaluate model
preds <- predict(object = prune_tree, newdata = test_set, type = c("class"))
conf_mat <- confusionMatrix(preds, test_set$cloud)
Results@prune_tree <- conf_mat$overall['Accuracy']
paste("Accuracy: ", Results@prune_tree)


#####################################
# Random Forest                     #
#####################################
# Memory consuming
sample_rows <- sample(nrow(train_x), 100000)
train_x <- train_x[sample_rows, ]
train_set <- train_set[sample_rows, ]

train_set <- set_as_factor(train_set)
test_set <- set_as_factor(test_set)
train_x <-set_as_factor(train_x)


# Build model
model_randomForest <- randomForest(x = train_x, y = train_set$cloud)
# Evaluate model
preds <- predict(object = model_randomForest, newdata = test_set, type = c("class"))

conf_mat <- confusionMatrix(preds, test_set$cloud)
Results@random_forest <- conf_mat$overall['Accuracy']
paste("Accuracy random forest: ", Results@random_forest)


#####################################
# Show results                      #
#####################################
get_accuracy <- function(col){
  conf_mat <- confusionMatrix(data$cloud, as.factor(col))
  return(conf_mat$overall['Accuracy'])
}
accuracies <- as.data.frame(apply(data[, feature_col_names], 2, get_accuracy))
accuracies <- setDT(accuracies, keep.rownames = TRUE)[]
names(accuracies) <- c("Method","Accuracy")
accuracies$color <- 1

accuracies <- accuracies %>% 
                add_row(Method = "logistic_model", Accuracy = Results@logistic_model, color=2) %>%
                add_row(Method = "decision_tree", Accuracy = Results@decision_tree, color=2) %>%
                add_row(Method = "random_forest", Accuracy = Results@random_forest, color=2)

accuracies$Method <- factor(accuracies$Method, levels = accuracies$Method[order(accuracies$Accuracy)])
accuracies

plot3 <- ggplot(data=accuracies, aes(x=accuracies$Method, y=accuracies$Accuracy * 100)) +
  geom_bar(stat="identity", width=0.5)+
  geom_col(aes(fill = accuracies$color)) + 
  scale_fill_gradient2(high = "steelblue") +
  geom_text(aes(label=Accuracy), label=sprintf("%0.2f", round(accuracies$Accuracy * 100, digits = 2)), vjust=1.6, color="white", size=3.5) +
  theme(legend.position = "none") + 
  ggtitle("Accuracy (rounded with 2 digits in %)")


# Save workspace for R Markdown
save.image(file = "processed_work_space.RData")

