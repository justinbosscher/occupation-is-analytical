# Justin Bosscher
# Categorizing O*Net Occupations as Analytical or Not Analytical



########
########  Setup Workspace  #####################################################
########

# Working Directory
wd <- paste("~/Dev/DS_Projects/is_analytical")
setwd(wd)

# Load libraries
library(pastecs)    # For descriptive stats
library(ggplot2)    # For plotting
library(dplyr)      # For data manipulation
library(caret)      # For varImp, confusionMatrix
library(gridExtra)  # For printing/arranging multiple plots
library(stargazer)  # For saving model summaries as jpeg / html files
library(tidyr)      # To reshape the data to different format
library(MASS)       # For linear/quadratic discriminant analysis
library(klaR)       # For simple LDA plot
library(scales)     # For tick label precision


########
########  Load Data  ###########################################################
########

work_styles_df <- read.csv("assets/Work_Styles.csv")
abilities_df <- read.csv("assets/Abilities.csv")
cognitive_abilities_df <- read.csv("assets/Cognitive_Abilities.csv")

#==============================================================================#

# Read in total_df only after Data Engineering has been completed
total_df <- read.csv("assets/total_df.csv", row.names=NULL, header=T)

#==============================================================================#


########
########  Data Engineering  ####################################################
########

# Take a peek
head(work_styles_df)
head(abilities_df)
head(cognitive_abilities_df)

# Create df containing values measuring importance of analytical thinking
# work style and SOCs
analytical <- subset(work_styles_df, Element.Name=="Analytical Thinking")
analytical <- analytical[ , c("O.NET.SOC.Code", "Data.Value")]
n_distinct(abilities_df$O.NET.SOC.Code)       # prints 968
n_distinct(analytical$O.NET.SOC.Code)         # prints 967

# Filter out the SOC that is not in work_styles_df
abilities_df <- abilities_df %>%
    filter(O.NET.SOC.Code %in% work_styles_df$O.NET.SOC.Code)

# Create df containing cognitive abilities and SOC's
cognitive_abilities_df[, 1] <- sapply(cognitive_abilities_df[, 1], as.character)
abilities_df[, 1] <- sapply(abilities_df[, 1], as.character)

cog_abilities <- abilities_df %>%
    filter(Element.Name %in% cognitive_abilities_df$Cognitive.Abilities)

# Subset cog_abilities to include only level data
cog_abilities <- subset(cog_abilities, Scale.ID=="LV")

# Drop unnecessary columns from cog_abilities
cog_abilities <- cog_abilities[ , c("O.NET.SOC.Code", "Element.Name", "Data.Value")]

# Reshape cog_abilities before merge with analytical
# Note: The period separating words in column headers is replaced with a space
cog_abilities <- spread(cog_abilities, key=Element.Name, value=Data.Value)

# Merge analytical with cog_abilities
total_df <- merge(analytical, cog_abilities, by=c("O.NET.SOC.Code"))

# Rename Data.Value to Analytical.IM
names(total_df)[2] <- "Analytical.IM"

# Check for missing values
table(is.na(total_df))

# Use mean for cutoff value
cutoff_median <- median(total_df$Analytical.IM)   # 3.854767
cutoff_mean <- mean(total_df$Analytical.IM)       # 3.88

# is_analytical()
# Function applies cutoff for categorizing the response variable
is_analytical <- function(x){
    if(x < cutoff_mean){
      return(0)
    }else{
      return(1)
    }
} # End is_analytical()

# Categorize response variable
total_df$y <- sapply(total_df$Analytical.IM, function(x) is_analytical(x))

# Convert y to factor
total_df$y <- as.factor(total_df$y)

# Descriptive stats on this set of abilities
summary(total_df$Analytical.IM < cutoff_mean)   # 501 F; 466 T

# Visually inspect distribution of data
analytical.im.histogram <-
    ggplot(total_df, aes(x=Analytical.IM)) + 
    geom_histogram(binwidth=.2, color="black", fill="white") +
    geom_vline(aes(xintercept=mean(Analytical.IM, na.rm=T)),
               color="red", linetype="dashed", size=1)

analytical.im.histogram


# Table of number of 0's and 1's
table(total_df$y)     # 466 0's & 501 1's

# Replace the space with a period in column headers
names(total_df) <- make.names(names(total_df), unique=T)

# Save total_df to disk
write.csv(total_df, file="assets/total_df.csv", row.names=F)


########
########  Data Exploration & Feature Engineering  ##############################
########

###############################################
#                                             #
# **** START HERE AFTER LOADING total_df **** #
#                                             #
###############################################

# If necessary, you can load total_df here
total_df <- read.csv("assets/total_df.csv", row.names=NULL, header=T)

summ <- stat.desc(total_df)
summ                           # Variables have different scales

########
########  Functions & Global Variables  ########################################
########

# check_accuracy()
# Function takes either the discretized results or the probabilities as model
# outputs; in the case of the latter, it uses a cutoff of 0.50 to determine
# whether the occupation is analytical or not
check_accuracy <- function(df){
    count<- 0
    i <- 0
    if(class(df$yhat) == 'factor'){
      for(i in 1:length(df$y)){
        if(df[i, "yhat"] == df[i, "y"]){
          count <- count + 1
        }
      }
    }else if(class(df$yhat) == 'numeric'){
      for(i in 1:length(df$y)){
        if(df[i, "yhat"] < 0.5 & df[i, "y"] == 0){
          count <- count + 1
        }else if(df[i, "yhat"] >= 0.5 & df[i, "y"] == 1){
          count <- count + 1
        }
      }
    }
    return(count / length(df$y))
}# End check_accuracy()

# discretize_output()
# Function returns a numerical value, 0 or 1, where 0 is not analytical and 1 is
# analytical with a cutoff of 0.50; not analytical < 0.50 >= is analytical
discretize_output <- function(val, cutoff){
    if(val < cutoff){
      return(as.factor(0))}
      return(as.factor(1))
} # End discretize_output()

# calc_f1()
# Function returns the f1 stat for a model; takes recall and precision inputs
calc_f1 <- function(precision, recall){
  return(2 * (precision * recall) / (precision + recall))
} #End calc_f


########
########  Test / Train Splits  #################################################
########

# Set seed for train/test splits
set.seed(987654)

# 70 / 30 test / train split
# For L/QDA, n >= 5k
dt <- sample(nrow(total_df), nrow(total_df) * 0.7)
train <- total_df[dt,]
test <- total_df[-dt,]

nrow(train)
nrow(test)

# Split training data into 3 sets for 3 models
# Logit model
dt_log <- sample(nrow(train), nrow(train) * 0.333)
logit_train <- train[dt_log,]
train <- train[-dt_log,]

# LDA model
dt_lda <- sample(nrow(train), nrow(train) * 0.5)
lda_train <- train[dt_lda,]

# QDA model
qda_train <- train[-dt_lda,]

nrow(logit_train)
nrow(lda_train)
nrow(qda_train)


########
########  Logistic Regression  #################################################
########

# Fit logistic regression model
logit_model <- 
    glm(y ~ Category.Flexibility + Deductive.Reasoning + Flexibility.of.Closure +
            Fluency.of.Ideas + Inductive.Reasoning + Information.Ordering +
            Mathematical.Reasoning + Memorization + Number.Facility + Oral.Comprehension +
            Oral.Expression + Originality + Perceptual.Speed + Problem.Sensitivity +
            Selective.Attention + Spatial.Orientation + Speed.of.Closure + Time.Sharing +
            Visualization + Written.Comprehension + Written.Expression,
            family="binomial", data=logit_train)

# Turn off scientific notation
options(scipen=999)

# Save output
logit_train$yhat <- fitted(logit_model)
  
# Discretize output: not analytical < 0.50 >= analytical
logit_train$class <- sapply(logit_train$yhat, function(x) discretize_output(x, 0.5))

# Check accuracy
logit_train_score <- check_accuracy(logit_train)
logit_train_score                                # Prints 0.8844444

#Confusion Matrix
# Convert y values to factor
logit_train$y <- sapply(logit_train$y, function(x) as.factor(x))
confusionMatrix(logit_train$class, logit_train$y)

# Create confusion matrix
logit_cm <- table(logit_train$y, logit_train$class)
logit_cm

# Calculate true positive, false positive, and false negative
logit_tp <- logit_cm[2, 2]
logit_fp <- logit_cm[1, 2]
logit_fn <- logit_cm[2, 1]

# Calculate precision
logit_precision <- logit_tp / (logit_tp + logit_fp)
logit_precision                                    # Prints 0.8813559

# Calculate recall
logit_recall <- logit_tp / (logit_tp + logit_fn)
logit_recall                                       # Prints 0.8965517

# Calculate F1
logit_f1 <- calc_f1(logit_precision, logit_recall)
logit_f1                                           # Prints 0.8888889

# Take a look at model performance measures
logit_summary <- summary(logit_model)            # Null deviance: 311.70
logit_summary                                    # Residual deviance: 130.13
                                                 # AIC: 174.13

# Most / least influential variables
logit_influence = varImp(logit_model)       # Fluency.of.Ideas is most at 2.81
logit_influence                             # Memorization is least at 0.006

# Plots
plot(logit_model)

# Plot predicted probabilities against occupations
logit_gg <- ggplot(logit_train,
                   aes(x=Category.Flexibility + Deductive.Reasoning +
                         Flexibility.of.Closure + Fluency.of.Ideas + 
                         Inductive.Reasoning + Information.Ordering +
                         Mathematical.Reasoning + Memorization + 
                         Number.Facility + Oral.Comprehension +
                         Oral.Expression + Originality + 
                         Perceptual.Speed + Problem.Sensitivity +
                         Selective.Attention + Spatial.Orientation + 
                         Speed.of.Closure + Time.Sharing +
                         Visualization + Written.Comprehension + 
                         Written.Expression,
                       y=yhat)) + 
                   geom_point(alpha=.5) +
                   stat_smooth(method="glm", se=FALSE, fullrange=TRUE,
                               method.args=list(family=binomial)) + 
                   ggtitle("Logistic Regression Training Probabilities") +
                   xlab("Occupation") +
                   ylab("Probability Occupation Is Analytical")

logit_gg


########
########  Linear Discriminant Analysis  ########################################
########

# Fit LDA model
lda_model <- lda(y ~ Category.Flexibility + Deductive.Reasoning +
                     Flexibility.of.Closure + Fluency.of.Ideas + 
                     Inductive.Reasoning + Information.Ordering +
                     Mathematical.Reasoning + Memorization + 
                     Number.Facility + Oral.Comprehension +
                     Oral.Expression + Originality + 
                     Perceptual.Speed + Problem.Sensitivity +
                     Selective.Attention + Spatial.Orientation + 
                     Speed.of.Closure + Time.Sharing +
                     Visualization + Written.Comprehension + 
                     Written.Expression,
                 data=lda_train)

# Training data predictions
predmodel_train_lda <- predict(lda_model, data=lda_train)

# Save output
lda_train$yhat <- predmodel_train_lda$class

# Check accuracy
lda_train_score <- check_accuracy(lda_train)   
lda_train_score                                  # Prints 0.8577778

# Create confusion matrix
lda_cm <- table(lda_train$y, lda_train$yhat)
lda_cm

# Calculate true positive, false positive, and false negative
lda_tp <- lda_cm[2, 2]
lda_fp <- lda_cm[1, 2]
lda_fn <- lda_cm[2, 1]

# Calculate precision
lda_precision <- lda_tp / (lda_tp + lda_fp)
lda_precision                                    # Prints 0.8582677

# Calculate recall
lda_recall <- lda_tp / (lda_tp + lda_fn)
lda_recall                                       # Prints 0.8861789

# Calculate F1
lda_f1 <- calc_f1(lda_precision, lda_recall)
lda_f1                                           # Prints 0.872

lda_model


########
########  Quadratic Discriminant Analysis  #####################################
########

# Fit QDA model
qda_model <- qda(y ~ Category.Flexibility + Deductive.Reasoning +
                   Flexibility.of.Closure + Fluency.of.Ideas + 
                   Inductive.Reasoning + Information.Ordering +
                   Mathematical.Reasoning + Memorization + 
                   Number.Facility + Oral.Comprehension +
                   Oral.Expression + Originality + 
                   Perceptual.Speed + Problem.Sensitivity +
                   Selective.Attention + Spatial.Orientation + 
                   Speed.of.Closure + Time.Sharing +
                   Visualization + Written.Comprehension + 
                   Written.Expression,
                 data=qda_train)
    
# Training data predictions
predmodel.train.qda <- predict(qda_model, newdata=qda_train)

# Save labels
qda_train$yhat <- predmodel.train.qda$class

# Check accuracy
qda_train_score <- check_accuracy(qda_train)   
qda_train_score                                   #Prints 0.8893805

# Create confusion matrix
qda_cm <- table(qda_train$y, qda_train$yhat)
qda_cm

# Calculate true positive, false positive, and false negative
qda_tp <- qda_cm[2, 2]
qda_fp <- qda_cm[1, 2]
qda_fn <- qda_cm[2, 1]

# Calculate precision
qda_precision <- qda_tp / (qda_tp + qda_fp)
qda_precision                                    # Prints 0.85

# Calculate recall
qda_recall <- qda_tp / (qda_tp + qda_fn)
qda_recall                                       # Prints 0.9357798

# Calculate F1
qda_f1 <- calc_f1(qda_precision, qda_recall)
qda_f1                                           # Prints 0.8908297

qda_model


########
########  Compare Training Results & Run Chosen Model on Test Data  ############
########

# Save training scores
train_scores <- c(logit_train_score, lda_train_score, qda_train_score)
models <- c("Logit", "LDA", "QDA")
train_scores_ma <- cbind(models, train_scores)
train_scores_df <- data.frame(train_scores_ma)
colnames(train_scores_df) <- c("Model", "Accuracy")

# Combine precision, recall, f1 scores
precisions <- c(logit_precision, lda_precision, qda_precision)
recalls <- c(logit_recall, lda_recall, qda_recall)
f1s <- c(logit_f1, lda_f1, qda_f1)
train_scores_df$Precision <- precisions
train_scores_df$Recall <- recalls
train_scores_df$F1 <- f1s

train_scores_df

# Run Logistic Regression on the test data set
predmodel.test.logit <- predict(logit_model, newdata=test, type="response")

# Save labels
test$yhat <- predmodel.test.logit

# Discretize output: not analytical < 0.50 >= analytical
test$class <- sapply(test$yhat, function(x) discretize_output(x, 0.5))

# Print output
table(test$class)

# Check accuracy
logit_test_score <- check_accuracy(test)                 # Prints 0.8247423
logit_test_score

# Create confusion matrix
test_cm <- table(test$y, test$class)
test_cm

# Calculate true positive, false positive, and false negative
test_tp <- test_cm[1, 2]
test_fp <- test_cm[1, 1]
test_fn <- test_cm[2, 2]

# Calculate precision
test_precision <- test_tp / (test_tp + test_fp)
test_precision                                    # Prints 0.8015267

# Calculate recall
test_recall <- test_tp / (test_tp + test_fn)
test_recall                                       # Prints 0.8076923

# Calculate F1
test_f1 <- calc_f1(test_precision, test_recall)
test_f1                                           # Prints 0.8045977

# TODO: Print the most, least, average, and median analytical occupations





