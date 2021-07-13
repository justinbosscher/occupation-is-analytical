# Justin Bosscher
# Categorizing O*Net Occupations as Analytical or Not Analytical



########
########  Setup Workspace  #####################################################
########

# Working Directory
wd <- paste("~/Dev/DS_Projects/is_analytical")
setwd(wd)

# Load libraries
#library(pastecs)    # For descriptive stats
library(ggplot2)    # For plotting
library(dplyr)      # For data manipulation
library(gridExtra)  # For printing/arranging multiple plots
#library(stargazer)  # For saving model summaries as jpeg / html files
library(tidyr)      # To reshape the data to different format
library(MASS)       # For linear/quadratic discriminant analysis
#library(klaR)       # For simple LDA plot
#library(scales)     # For tick label precision


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
# and SOCs; Analytical Thinking will be our target vector
analytical_df <- subset(work_styles_df, Element.Name=="Analytical Thinking")
analytical_df <- analytical_df[ , c("O.NET.SOC.Code", "Title", "Data.Value")]

# TODO: Run models w/o filtering out these SOC codes
# Filter out the SOCs that are not in work_styles_df
abilities_df <- as.data.frame(abilities_df) %>%
    filter(O.NET.SOC.Code %in% work_styles_df$O.NET.SOC.Code)

# Create df's containing cognitive abilities and SOC's
cognitive_abilities_df[, 1] <- sapply(cognitive_abilities_df[, 1], as.character)
abilities_df[, 1] <- sapply(abilities_df[, 1], as.character)

# Filter out all non-cognitive abilities of which there are 21
cog_abilities <- abilities_df %>%
    filter(Element.Name %in% cognitive_abilities_df$Cognitive.Abilities)

# Subset cog_abilities to include only importance data
cog_abilities <- subset(cog_abilities, Scale.ID=="IM")

# Drop unnecessary columns from cog_abilities
cog_abilities <-
    cog_abilities[ , c("O.NET.SOC.Code", "Title", "Element.Name", "Data.Value")]

# Pivot cog_abilities so that cog_abilities are columns
cog_abilities_pivot <- cog_abilities %>%
    pivot_wider(names_from=Element.Name, values_from=Data.Value)

# Merge analytical with cog_abilities
total_df <- merge(analytical_df, cog_abilities_pivot, by=c("O.NET.SOC.Code"))

# Drop extra Title column, Title.y
drop <- "Title.y"
total_df <- total_df[ , !names(total_df) %in% drop]

# Rename Data.Value to Analytical.IM; once discretized, this will be the target
names(total_df)[2] <- "Occupation"
names(total_df)[3] <- "Analytical.IM"

# Check for missing values
sum(is.na(total_df))

# Replace the space with a period in column headers
names(total_df) <- make.names(names(total_df), unique=T)

# Save total_df to disk
write.csv(total_df, file="assets/total_df.csv", row.names=F)


########
########  Data Exploration, Feature Engineering, Train/Test Splits  ############
########

########################################################
#                                                      #
# **** START HERE IF DATA ENGINEERING IS COMPLETE **** #
#                                                      #
########################################################

# If necessary, you can load total_df here
total_df <- read.csv("assets/total_df.csv", row.names=NULL, header=T)


########
########  Functions  ###########################################################
########

# is_analytical()
# Function applies cutoff for categorizing the response variable
is_analytical <- function(x, cutoff){
    if(x < cutoff){
      return(0)
    }else{
      return(1)
    }
} # End is_analytical()

# check_accuracy()
check_accuracy <- function(actual_col, predicted_col){
    count <- 0
    i <- 0
    for(i in 1:length(actual_col)){
      if(actual_col[i] == predicted_col[i]){
        count <- count + 1
      }
    }
    return(count / length(actual_col))
} # End check_accuracy()

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
} # End calc_f1()


########
########  Test / Train Splits  #################################################
########

# Set seed for train/test splits
set.seed(987654)# is_analytical()
# Function applies cutoff for categorizing the response variable

# 70 / 30 test / train split

# train / test
dt <- sample(nrow(total_df), nrow(total_df) * 0.7)
train <- total_df[dt,]
test <- total_df[-dt,]

nrow(train)   #676
nrow(test)    #291

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

# Check number of rows
nrow(logit_train)    #225
nrow(lda_train)      #225
nrow(qda_train)      #226


########
########  Explore logit_train Data  ############################################
########

logit_summ <- summary(logit_train)
logit_summ

# Check distribution of target data
logit_train_y_hist <-
  ggplot(logit_train,
          aes(x=Analytical.IM), title="Distribution") +
          theme_classic() +
          geom_histogram(color="black", fill="#737CA1", bins=70) +
          labs(title="Distribution of Analytical Importance",
               subtitle="Logistic Regression Training Data") +
          labs(x="Analytical Importance", y="Count") +
          geom_vline(aes(xintercept=mean(Analytical.IM), color="Mean"),
                         linetype="dashed", size=0.75) +
          geom_vline(aes(xintercept=median(Analytical.IM), color="Median"),
                         linetype="dotdash", size=0.75) +
          scale_color_manual(name="Statistics",
                             values=c(Mean="#6CC417", Median="#F88017"))

logit_train_y_hist

# Discretize logit model target data
# Use median to classify
logit_train_y_cutoff <- median(logit_train$Analytical.IM)

logit_train$y <- sapply(logit_train$Analytical.IM, 
                        function(x) is_analytical(x, logit_train_y_cutoff))

table(logit_train$y)        # 0: 110, 1: 115
                            # fairly balanced; sample size is large enough


########
########  Logistic Regression  #################################################
########

# Fit logistic regression model
logit_model <- glm(y ~ Category.Flexibility + Deductive.Reasoning +
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
                   family="binomial",
                   data=logit_train)

# Turn off scientific notation
options(scipen=999)

# Save output
logit_train$yhat <- fitted(logit_model)

# Plot distribution of the output
logit_train_yhat_hist <-
          ggplot(logit_train, aes(x=yhat), title="Distribution") +
          theme_classic() +
          geom_histogram(color="black", fill="#737CA1", bins=70) +
          labs(title="Distribution of Predicted Probabilities",
               subtitle="Logistic Regression Training Data") +
          labs(x="Occupation", y="Count") +
          geom_vline(aes(xintercept=mean(yhat),
                 color="Mean"),
             linetype="dashed", size=0.75) +
         geom_vline(aes(xintercept=median(yhat),
                 color="Median"), 
             linetype="dotdash", size=0.75) +
          scale_color_manual(name="Statistics",
                     values=c(Mean="#6CC417", Median="#F88017"))

logit_train_yhat_hist

# Discretize output for class
# Use not analytical < 0.5 >= analytical
logit_train_class_cutoff <- 0.5
logit_train$class <- as.integer(sapply(logit_train$yhat, function(x)
                                  is_analytical(x, logit_train_class_cutoff)))

# Check accuracy
logit_train_score <- check_accuracy(logit_train$y, logit_train$class) 
logit_train_score        # 0.8622222

# Create confusion matrix
logit_cm <- table(logit_train$y, logit_train$class)
logit_cm

# Calculate true positive, false positive, and false negative
logit_tp <- logit_cm[2, 2]
logit_fp <- logit_cm[1, 2]
logit_fn <- logit_cm[2, 1]

# Calculate precision
logit_precision <- logit_tp / (logit_tp + logit_fp)
logit_precision                                    # Prints 0.8559322

# Calculate recall
logit_recall <- logit_tp / (logit_tp + logit_fn)
logit_recall                                       # Prints 0.8782609

# Calculate F1
logit_f1 <- calc_f1(logit_precision, logit_recall)
logit_f1                                           # Prints 0.8669528

# Take a look at model performance measures
logit_summary <- summary(logit_model)
logit_summary        # AIC: 195.38

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
              ggtitle("Logistic Regression Training Probabilities",
                      subtitle="All Features") +
              xlab("Occupation") +
              ylab("Probability Occupation Is Analytical")

png("plots/logit_gg.png")
logit_gg
# Close device
dev.off()

# Backwards Stepwise Model
# Remove those independent variables that are not statistically significant
# Fit logistic regression model without Oral Expression 0.970660
logit_model_1 <- glm(y ~ Category.Flexibility + Deductive.Reasoning +
                         Flexibility.of.Closure + Fluency.of.Ideas + 
                         Inductive.Reasoning + Information.Ordering +
                         Mathematical.Reasoning + Memorization + 
                         Number.Facility + Oral.Comprehension +
                         Originality + Perceptual.Speed +
                         Problem.Sensitivity + Selective.Attention +
                         Spatial.Orientation + Speed.of.Closure +
                         Time.Sharing + Visualization +
                         Written.Comprehension + Written.Expression,
                     family="binomial",
                     data=logit_train)

logit_1_summary <- summary(logit_model_1)
logit_1_summary      # AIC: 193.38

# Fit logistic regression model without Information Ordering 0.973220
logit_model_2 <- glm(y ~ Category.Flexibility + Deductive.Reasoning +
                         Flexibility.of.Closure + Fluency.of.Ideas + 
                         Inductive.Reasoning + Mathematical.Reasoning +
                         Memorization + Number.Facility +
                         Oral.Comprehension + Originality +
                         Perceptual.Speed + Problem.Sensitivity + 
                         Selective.Attention + Spatial.Orientation + 
                         Speed.of.Closure + Time.Sharing + 
                         Visualization + Written.Comprehension +
                         Written.Expression,
                     family="binomial",
                     data=logit_train)

logit_2_summary <- summary(logit_model_2)
logit_2_summary      # AIC: 191.38

# Fit logistic regression model without Written Comprehension 0.953401
logit_model_3 <- glm(y ~ Category.Flexibility + Deductive.Reasoning +
                         Flexibility.of.Closure + Fluency.of.Ideas + 
                         Inductive.Reasoning + Mathematical.Reasoning +
                         Memorization + Number.Facility +
                         Oral.Comprehension + Originality +
                         Perceptual.Speed + Problem.Sensitivity + 
                         Selective.Attention + Spatial.Orientation + 
                         Speed.of.Closure + Time.Sharing + 
                         Visualization + Written.Expression,
                     family="binomial",
                     data=logit_train)

logit_3_summary <- summary(logit_model_3)
logit_3_summary      # AIC: 189.39

# Fit logistic regression model without Time Sharing 0.905075
logit_model_4 <- glm(y ~ Category.Flexibility + Deductive.Reasoning +
                         Flexibility.of.Closure + Fluency.of.Ideas + 
                         Inductive.Reasoning + Mathematical.Reasoning +
                         Memorization + Number.Facility +
                         Oral.Comprehension + Originality +
                         Perceptual.Speed + Problem.Sensitivity + 
                         Selective.Attention + Spatial.Orientation + 
                         Speed.of.Closure + Visualization +
                         Written.Expression,
                     family="binomial",
                     data=logit_train)

logit_4_summary <- summary(logit_model_4)
logit_4_summary     # AIC: 187.4

# Fit logistic regression model without Oral Comprehension 0.914421
logit_model_5 <- glm(y ~ Category.Flexibility + Deductive.Reasoning +
                         Flexibility.of.Closure + Fluency.of.Ideas + 
                         Inductive.Reasoning + Mathematical.Reasoning +
                         Memorization + Number.Facility +
                         Originality + Perceptual.Speed + 
                         Problem.Sensitivity + Selective.Attention + 
                         Spatial.Orientation + Speed.of.Closure +
                         Visualization + Written.Expression,
                     family="binomial",
                     data=logit_train)

logit_5_summary <- summary(logit_model_5)
logit_5_summary     # AIC: 185.41

# Fit logistic regression model without Selective Attention 0.8588
logit_model_6 <- glm(y ~ Category.Flexibility + Deductive.Reasoning +
                         Flexibility.of.Closure + Fluency.of.Ideas + 
                         Inductive.Reasoning + Mathematical.Reasoning +
                         Memorization + Number.Facility +
                         Originality + Perceptual.Speed + 
                         Problem.Sensitivity + Spatial.Orientation + 
                         Speed.of.Closure + Visualization + 
                         Written.Expression,
                     family="binomial",
                     data=logit_train)

logit_6_summary <- summary(logit_model_6)
logit_6_summary   # AIC: 183.45

# Fit logistic regression model without Perceptual Speed: 0.7813
logit_model_7 <- glm(y ~ Category.Flexibility + Deductive.Reasoning +
                         Flexibility.of.Closure + Fluency.of.Ideas + 
                         Inductive.Reasoning + Mathematical.Reasoning +
                         Memorization + Number.Facility +
                         Originality + Problem.Sensitivity + 
                         Spatial.Orientation + Speed.of.Closure +
                         Visualization + Written.Expression,
                     family="binomial",
                     data=logit_train)

logit_7_summary <- summary(logit_model_7)
logit_7_summary     # AIC: 181.52

# Fit logistic regression model without Number Facility: 0.7807
logit_model_8 <- glm(y ~ Category.Flexibility + Deductive.Reasoning +
                         Flexibility.of.Closure + Fluency.of.Ideas + 
                         Inductive.Reasoning + Mathematical.Reasoning +
                         Memorization + Originality +
                         Problem.Sensitivity + Spatial.Orientation +
                         Speed.of.Closure + Visualization +
                         Written.Expression,
                     family="binomial",
                     data=logit_train)

logit_8_summary <- summary(logit_model_8)
logit_8_summary     # AIC: 179.6

# Fit logistic regression model without Deductive Reasoning: 0.73569
logit_model_9 <- glm(y ~ Category.Flexibility + Flexibility.of.Closure + 
                         Fluency.of.Ideas + Inductive.Reasoning + 
                         Mathematical.Reasoning + Memorization +
                         Originality + Problem.Sensitivity + 
                         Spatial.Orientation + Speed.of.Closure +
                         Visualization + Written.Expression,
                     family="binomial",
                     data=logit_train)

logit_9_summary <- summary(logit_model_9)
logit_9_summary      # AIC: 177.71

# Fit logistic regression model without Memorization: 0.59519
logit_model_10 <- glm(y ~ Category.Flexibility + Flexibility.of.Closure +
                          Fluency.of.Ideas + Inductive.Reasoning +
                          Mathematical.Reasoning + Originality + 
                          Problem.Sensitivity + Spatial.Orientation + 
                          Speed.of.Closure + Visualization + 
                          Written.Expression,
                     family="binomial",
                     data=logit_train)

logit_10_summary <- summary(logit_model_10)
logit_10_summary     # AIC: 176

# Logit Model 11 ###############################################################
# Fit logistic regression model without Category Flexibility: 0.48813
logit_model_11 <- glm(y ~ Flexibility.of.Closure + Fluency.of.Ideas + 
                          Inductive.Reasoning + Mathematical.Reasoning + 
                          Originality + Problem.Sensitivity + 
                          Spatial.Orientation + Speed.of.Closure + 
                          Visualization + Written.Expression,
                      family="binomial",
                      data=logit_train)

logit_11_summary <- summary(logit_model_11)
logit_11_summary    # AIC: 174.47

# Save output
logit_train$yhat_11 <- fitted(logit_model_11)

# Plot distribution of the output
logit_train_yhat_11_hist <-
                ggplot(logit_train, aes(x=yhat_11), title="Distribution") +
                  theme_classic() +
                  geom_histogram(color="black", fill="#737CA1", bins=70) +
                  labs(title="Distribution of Predicted Probabilities",
                       subtitle="Logistic Regression Training Data") +
                  labs(x="Occupation", y="Count") +
                  geom_vline(aes(xintercept=mean(yhat_11),
                                 color="Mean"),
                                 linetype="dashed", size=0.75) +
                  geom_vline(aes(xintercept=median(yhat_11),
                                 color="Median"), 
                                 linetype="dotdash", size=0.75) +
                  scale_color_manual(name="Statistics",
                                     values=c(Mean="#6CC417", Median="#F88017"))

logit_train_yhat_11_hist

# Discretize output for class
# Use not analytical < 0.5 >= analytical
logit_train_class_cutoff_11 <- 0.5
logit_train$class_11 <- as.integer(sapply(logit_train$yhat_11, function(x)
  is_analytical(x, logit_train_class_cutoff_11)))

# Check accuracy
logit_train_score_11 <- check_accuracy(logit_train$y, logit_train$class_11)
logit_train_score_11                                  # Prints 0.8666667

# Create confusion matrix
logit_cm_11 <- table(logit_train$y, logit_train$class_11)
logit_cm_11

# Calculate true positive, false positive, and false negative
logit_tp_11 <- logit_cm_11[2, 2]
logit_fp_11 <- logit_cm_11[1, 2]
logit_fn_11 <- logit_cm_11[2, 1]

# Calculate precision
logit_precision_11 <- logit_tp_11 / (logit_tp_11 + logit_fp_11)
logit_precision_11                                    # Prints 0.8632479

# Calculate recall
logit_recall_11 <- logit_tp_11 / (logit_tp_11 + logit_fn_11)
logit_recall_11                                       # Prints 0.8782609

# Calculate F1
logit_f1_11 <- calc_f1(logit_precision_11, logit_recall_11)
logit_f1_11                                           # Prints 0.8706897

# Fit logistic regression model without Problem Sensitivity: 0.3754
logit_model_12 <- glm(y ~ Flexibility.of.Closure + Fluency.of.Ideas + 
                          Mathematical.Reasoning + Originality + 
                          Problem.Sensitivity + Spatial.Orientation + 
                          Speed.of.Closure + Visualization + 
                          Written.Expression,
                      family="binomial",
                      data=logit_train)

logit_12_summary <- summary(logit_model_12)
logit_12_summary       # AIC: 181.54

# Fit logistic regression model without Fluency of Ideas: 0.558886
logit_model_13 <- glm(y ~ Flexibility.of.Closure + Mathematical.Reasoning + 
                          Originality + Problem.Sensitivity + 
                          Spatial.Orientation + Speed.of.Closure + 
                          Visualization + Written.Expression,
                      family="binomial",
                      data=logit_train)

logit_13_summary <- summary(logit_model_13)
logit_13_summary       # AIC: 179.89

# Fit logistic regression model without Originality: 0.811040
logit_model_14 <- glm(y ~ Flexibility.of.Closure + Mathematical.Reasoning + 
                          Problem.Sensitivity + Spatial.Orientation + 
                          Speed.of.Closure + Visualization + 
                          Written.Expression,
                      family="binomial",
                      data=logit_train)

logit_14_summary <- summary(logit_model_14)
logit_14_summary      # AIC: 177.94

# Fit logistic regression model without Speed of Closure: 0.33815
logit_model_15 <- glm(y ~ Flexibility.of.Closure + Mathematical.Reasoning + 
                          Problem.Sensitivity + Spatial.Orientation + 
                          Visualization + Written.Expression,
                      family="binomial",
                      data=logit_train)

logit_15_summary <- summary(logit_model_15)
logit_15_summary      # AIC: 176.88

# Logit Model 16 ###############################################################
# Fit logistic regression model without Speed of Closure: 0.33815
logit_model_16 <- glm(y ~ Mathematical.Reasoning + Problem.Sensitivity + 
                          Spatial.Orientation + Visualization + 
                          Written.Expression,
                      family="binomial",
                      data=logit_train)

logit_16_summary <- summary(logit_model_16)# AIC: 177.16
logit_16_summary      # All features are significant at or greater than 95%
                      # AIC: 177.16

# Save output
logit_train$yhat_16 <- fitted(logit_model_16)

# Plot distribution of the output
logit_train_yhat_16_hist <-
                  ggplot(logit_train, aes(x=yhat_16), title="Distribution") +
                  theme_classic() +
                  geom_histogram(color="black", fill="#737CA1", bins=70) +
                  labs(title="Distribution of Predicted Probabilities",
                       subtitle="Logistic Regression Training Data") +
                  labs(x="Occupation", y="Count") +
                  geom_vline(aes(xintercept=mean(yhat_16),
                                 color="Mean"),
                             linetype="dashed", size=0.75) +
                  geom_vline(aes(xintercept=median(yhat_16),
                                 color="Median"), 
                             linetype="dotdash", size=0.75) +
                  scale_color_manual(name="Statistics",
                                     values=c(Mean="#6CC417", Median="#F88017"))

logit_train_yhat_16_hist

# Discretize output for class
# Use not analytical < 0.5 >= analytical
logit_train_class_cutoff_16 <- 0.5
logit_train$class_16 <- as.integer(sapply(logit_train$yhat_16, function(x)
                                   is_analytical(x, logit_train_class_cutoff_16)))

# Check accuracy
logit_train_score_16 <- check_accuracy(logit_train$y, logit_train$class_16)
logit_train_score_16                                  # Prints 0.8311111

# Create confusion matrix
logit_cm_16 <- table(logit_train$y, logit_train$class_16)
logit_cm_16

# Calculate true positive, false positive, and false negative
logit_tp_16 <- logit_cm_16[2, 2]
logit_fp_16 <- logit_cm_16[1, 2]
logit_fn_16 <- logit_cm_16[2, 1]

# Calculate precision
logit_precision_16 <- logit_tp_16 / (logit_tp_16 + logit_fp_16)
logit_precision_16                                    # Prints 0.8181818

# Calculate recall
logit_recall_16 <- logit_tp_16 / (logit_tp_16 + logit_fn_16)
logit_recall_16                                       # Prints 0.8608696

# Calculate F1
logit_f1_16 <- calc_f1(logit_precision_16, logit_recall_16)
logit_f1_16                                           # Prints 0.8389831

# Plot predicted probabilities against occupations
logit_16_gg <- ggplot(logit_train,
                      aes(x=Mathematical.Reasoning + Problem.Sensitivity + 
                            Spatial.Orientation + Visualization + 
                            Written.Expression,
                          y=yhat)) + 
                      geom_point(alpha=.5) +
                      stat_smooth(method="glm", se=FALSE, fullrange=TRUE,
                                  method.args=list(family=binomial)) + 
                      ggtitle("Logistic Regression Training Probabilities: Model 16") +
                      xlab("Occupation") +
                      ylab("Probability Occupation Is Analytical")

# Save plot
png("plots/logit_16_gg.png")
logit_16_gg
dev.off()

# Logit Model 17 ###############################################################
# Fit logistic regression model without Visualization: 0.01784
logit_model_17 <- glm(y ~ Mathematical.Reasoning + Problem.Sensitivity + 
                          Spatial.Orientation + Written.Expression,
                      family="binomial",
                      data=logit_train)

logit_17_summary <- summary(logit_model_17)
logit_17_summary      # All features are significant at or greater than 95%
                      # AIC: 181.01

# Save output
logit_train$yhat_17 <- fitted(logit_model_17)

# Plot distribution of the output
logit_train_yhat_17_hist <-
                  ggplot(logit_train, aes(x=yhat_17), title="Distribution") +
                  theme_classic() +
                  geom_histogram(color="black", fill="#737CA1", bins=70) +
                  labs(title="Distribution of Predicted Probabilities",
                       subtitle="Logistic Regression Training Data") +
                  labs(x="Occupation", y="Count") +
                  geom_vline(aes(xintercept=mean(yhat_17),
                                 color="Mean"),
                             linetype="dashed", size=0.75) +
                  geom_vline(aes(xintercept=median(yhat_17),
                                 color="Median"), 
                             linetype="dotdash", size=0.75) +
                  scale_color_manual(name="Statistics",
                                     values=c(Mean="#6CC417", Median="#F88017"))

logit_train_yhat_17_hist

# Discretize output for class
# Use not analytical < 0.5 >= analytical
logit_train_class_cutoff_17 <- 0.5
logit_train$class_17 <- as.integer(sapply(logit_train$yhat_17, function(x)
                                   is_analytical(x, logit_train_class_cutoff_17)))

# Check accuracy
logit_train_score_17 <- check_accuracy(logit_train$y, logit_train$class_17)
logit_train_score_17                                  # Prints 0.8311111

# Create confusion matrix
logit_cm_17 <- table(logit_train$y, logit_train$class_17)
logit_cm_17

# Calculate true positive, false positive, and false negative
logit_tp_17 <- logit_cm_17[2, 2]
logit_fp_17 <- logit_cm_17[1, 2]
logit_fn_17 <- logit_cm_17[2, 1]

# Calculate precision
logit_precision_17 <- logit_tp_17 / (logit_tp_17 + logit_fp_17)
logit_precision_17                                    # Prints 0.8181818

# Calculate recall
logit_recall_17 <- logit_tp_17 / (logit_tp_17 + logit_fn_17)
logit_recall_17                                       # Prints 0.8608696

# Calculate F1
logit_f1_17 <- calc_f1(logit_precision_17, logit_recall_17)
logit_f1_17                                           # Prints 0.8389831

# Plot predicted probabilities against occupations
logit_17_gg <- ggplot(logit_train,
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
                      ggtitle("Logistic Regression Training Probabilities: Model 17") +
                      xlab("Occupation") +
                      ylab("Probability Occupation Is Analytical")

# Save plot
png("plots/logit_17_gg.png")
logit_17_gg
dev.off()

# Logit Model 18 ###############################################################
# Fit logistic regression model without Spatial Orientation: 0.041985
logit_model_18 <- glm(y ~ Mathematical.Reasoning + Problem.Sensitivity + 
                          Written.Expression,
                      family="binomial",
                      data=logit_train)

logit_18_summary <- summary(logit_model_18)
logit_18_summary      # All features are significant at or greater than 99%
                      # AIC: 183.29

# Save output
logit_train$yhat_18 <- fitted(logit_model_18)

# Plot distribution of the output
logit_train_yhat_18_hist <-
              ggplot(logit_train, aes(x=yhat_18), title="Distribution") +
                theme_classic() +
                geom_histogram(color="black", fill="#737CA1", bins=70) +
                labs(title="Distribution of Predicted Probabilities",
                     subtitle="Logistic Regression Training Data") +
                labs(x="Occupation", y="Count") +
                geom_vline(aes(xintercept=mean(yhat_18),
                               color="Mean"),
                               linetype="dashed", size=0.75) +
                geom_vline(aes(xintercept=median(yhat_16),
                               color="Median"), 
                               linetype="dotdash", size=0.75) +
                scale_color_manual(name="Statistics",
                                   values=c(Mean="#6CC417", Median="#F88017"))

logit_train_yhat_18_hist

# Discretize output for class
# Use not analytical < 0.5 >= analytical
logit_train_class_cutoff_18 <- 0.5
logit_train$class_18 <- as.integer(sapply(logit_train$yhat_18, function(x)
                                   is_analytical(x, logit_train_class_cutoff_18)))

# Check accuracy
logit_train_score_18 <- check_accuracy(logit_train$y, logit_train$class_18)
logit_train_score_18                                  # Prints 0.84

# Create confusion matrix
logit_cm_18 <- table(logit_train$y, logit_train$class_18)
logit_cm_18

# Calculate true positive, false positive, and false negative
logit_tp_18 <- logit_cm_18[2, 2]
logit_fp_18 <- logit_cm_18[1, 2]
logit_fn_18 <- logit_cm_18[2, 1]

# Calculate precision
logit_precision_18 <- logit_tp_18 / (logit_tp_18 + logit_fp_18)
logit_precision_18                                    # Prints 0.8264463

# Calculate recall
logit_recall_18 <- logit_tp_18 / (logit_tp_18 + logit_fn_18)
logit_recall_18                                       # Prints 0.8695652

# Calculate F1
logit_f1_18 <- calc_f1(logit_precision_16, logit_recall_18)
logit_f1_18                                           # Prints 0.8430913

# Compare logistic regression models ###########################################

# Save model performance metrics for top 3 models as df
logit_model_names <- c("Logit 16", "Logit 17", "Logit 18")

logit_aics <- c(logit_16_summary$aic,
                logit_17_summary$aic,
                logit_18_summary$aic)

logit_accuracies <- c(logit_train_score_16,
                      logit_train_score_17,
                      logit_train_score_18)

logit_f1s <- c(logit_f1_16,
               logit_f1_17,
               logit_f1_18)

logit_precisions <- c(logit_precision_16,
                      logit_precision_17,
                      logit_precision_18)

logit_recalls <- c(logit_recall_16,
                   logit_recall_17,
                   logit_recall_18)

metrics_df <- data.frame(logit_model_names,
                         logit_accuracies,
                         logit_aics,
                         logit_f1s,
                         logit_precisions,
                         logit_recalls)

metrics_col_names <- c("Model", "Accuracy", "AIC", "F1", "Precision", "Recall")
colnames(metrics_df) <- metrics_col_names

save(metrics_df, file="logit_metrics.png")
metrics_df

# Save model metrics as png
png("plots/metrics.png", height=40*nrow(metrics_df),
                         width=80*ncol(metrics_df))
grid.table(metrics_df)
dev.off()


########
########  Linear Discriminant Analysis  ########################################
########

# LDA assumes Gaussian class-conditional distributions and variance-covariance
# homogeneity of classes

# Decision boundary is set of points for which the probability of being
# analytical is equal to the probability of not being analytical (log-odds are
# zero)

# Discretize lda model target data
# Use median to classify
lda_train_y_cutoff <- median(logit_train$Analytical.IM)

lda_train$y <- sapply(lda_train$Analytical.IM, 
                        function(x) is_analytical(x, lda_train_y_cutoff))

table(lda_train$y)        # 0: 100, 1: 125

# TODO: DOES LDA NEED TO BE BALANCED ??
# fairly balanced; sample size is large enough

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
lda_train$class <- predmodel_train_lda$class

# Check accuracy
lda_train_score <- check_accuracy(lda_train, lda_train$class)   
lda_train_score                                  # Prints 0.88444444

# Create confusion matrix
lda_cm <- table(lda_train$y, lda_train$class)
lda_cm

# Save true positive, false positive, and false negative
lda_tp <- lda_cm[2, 2]              #     0   1
lda_fp <- lda_cm[1, 2]              # 0  87  13
lda_fn <- lda_cm[2, 1]              # 1  13 112                              
                                  

# Calculate precision
lda_precision <- lda_tp / (lda_tp + lda_fp)
lda_precision                                    # Prints 0.896

# Calculate recall
lda_recall <- lda_tp / (lda_tp + lda_fn)
lda_recall                                       # Prints 0.896

# Calculate F1
lda_f1 <- calc_f1(lda_precision, lda_recall)
lda_f1                                           # Prints 0.896

lda_model

# Add LDA metrics to metrics df
LDA_metrics <- c("LDA", lda_train_score, "n/a", lda_f1, lda_precision, lda_recall)
metrics_df <- rbind(LDA_metrics)


#lda_plot <- as(cbind(lda_train$y, lda_model$LD1)
ggplot(lda_plot, aes(LD1))


########
########  Quadratic Discriminant Analysis  #####################################
########

# Discretize lda model target data
# Use median to classify
qda_train_y_cutoff <- median(qda_train$Analytical.IM)

qda_train$y <- sapply(qda_train$Analytical.IM, 
                      function(x) is_analytical(x, qda_train_y_cutoff))

table(qda_train$y)        # 0: 111, 1: 115

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
qda_train$class <- predmodel.train.qda$class

# Check accuracy
qda_train_score <- check_accuracy(qda_train)   
qda_train_score                                   #Prints 0.8938053

# Create confusion matrix
qda_cm <- table(qda_train$y, qda_train$class)
qda_cm

# Calculate true positive, false positive, and false negative
qda_tp <- qda_cm[2, 2]              #    0   1
qda_fp <- qda_cm[1, 2]              # 0  96  15
qda_fn <- qda_cm[2, 1]              # 1  9   106

# Calculate precision
qda_precision <- qda_tp / (qda_tp + qda_fp)
qda_precision                                    # Prints 0.8760331

# Calculate recall
qda_recall <- qda_tp / (qda_tp + qda_fn)
qda_recall                                       # Prints 0.9217391

# Calculate F1
qda_f1 <- calc_f1(qda_precision, qda_recall)
qda_f1                                           # Prints 0.898051

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
#   Model          Accuracy Precision    Recall        F1
# 1 Logit 0.862222222222222 0.8559322 0.8782609 0.8669528
# 2   LDA 0.884444444444444 0.8960000 0.8960000 0.8960000
# 3   QDA 0.893805309734513 0.8760331 0.9217391 0.8983051

# Discretize lda model target data
# Use median as cutoff
test_y_cutoff <- median(test$Analytical.IM)

test$y <- sapply(test$Analytical.IM, 
                      function(x) is_analytical(x, test_y_cutoff))

# Run QDA on the test data set
predmodel.test.lda <- predict(lda_model, newdata=test)

# Save labels
test$class <- predmodel.test.lda$class

# Print output               #  0   1 
table(test$class)            # 120 171

# Check accuracy
lda_test_score <- check_accuracy(test)         # Prints 0.8178694
lda_test_score

# Create confusion matrix
test_cm <- table(test$y, test$class)
test_cm

# Calculate true positive, false positive, and false negative
test_tp <- test_cm[2, 2]      #     0   1
test_fp <- test_cm[1, 2]      # 0 106  39
test_fn <- test_cm[2, 1]      # 1  14 132

# Calculate precision
test_precision <- test_tp / (test_tp + test_fp)
test_precision                                    # Prints 0.7719298

# Calculate recall
test_recall <- test_tp / (test_tp + test_fn)
test_recall                                       # Prints 0.9041096

# Calculate F1
test_f1 <- calc_f1(test_precision, test_recall)
test_f1                                           # Prints 0.8328076

# TODO: Print the most, least, average, and median analytical occupations

# Run QDA on the test data set
predmodel.test.logint <- predict(logit_model, newdata=test)

# Save labels
test$class <- predmodel.test.logit$class

# Print output               #  0   1 
table(test$class)            # 120 171

# Check accuracy
logit_test_score <- check_accuracy(test)         # Prints 0.8178694
logit_test_score

# Create confusion matrix
test_cm <- table(test$y, test$class)
test_cm

# Calculate true positive, false positive, and false negative
test_tp <- test_cm[2, 2]      #     0   1
test_fp <- test_cm[1, 2]      # 0 106  39
test_fn <- test_cm[2, 1]      # 1  14 132

# Calculate precision
test_precision <- test_tp / (test_tp + test_fp)
test_precision                                    # Prints 0.7719298

# Calculate recall
test_recall <- test_tp / (test_tp + test_fn)
test_recall                                       # Prints 0.9041096

# Calculate F1
test_f1 <- calc_f1(test_precision, test_recall)
test_f1                       



