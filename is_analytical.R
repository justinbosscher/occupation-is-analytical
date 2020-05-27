# Justin Bosscher
# Categorizing O*Net Occupations as Analytical or Not Analytical



########
########  Setup workspace  #####################################################
########

# Working Directory
wd <- paste("/home/optimus/Dev/DS_Projects/is_analytical", sep="")

setwd(wd)

# Load libraries
library(ggplot2)    # For plotting
library(dplyr)      # For data manipulation
library(gridExtra)  # For printing/arranging multiple plots
library(stargazer)  # For saving model summaries as jpeg/html files
library(tidyr)      # To reshape the data to different format
library(caret)      # For train/test split
library(MASS)       # For linear/quadratic discriminant analysis


########
########  Load Data  ###########################################################
########

work_styles_df <- read.csv("assets/Work_Styles.csv")
abilities_df <- read.csv("assets/Abilities.csv")
cognitive_abilities_df <- read.csv("assets/Cognitive_Abilities.csv")

#=============================================================================+#

# Read in total_df only after Data & Exploration has been completed
total_df <- read.csv("assets/total_df.csv", row.names=NULL, header=T)

#==============================================================================#


########
########  Data Exploration & Preparation  ######################################
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

# Drop unecessary columns from cog_abilities
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
    ggplot(total_df, aes(x=Analytical.IM)) + geom_histogram(binwidth=.2, color="black", fill="white") +
          geom_vline(aes(xintercept=mean(Analytical.IM, na.rm=T)), color="red", linetype="dashed", size=1)

analytical.im.histogram

# Table of number of 0's and 1's
table(total_df$y)     # 466 0's & 501 1's

# Replace the space with a period in column headers
names(total_df) <- make.names(names(total_df), unique=T)

# Save total_df for posterity
write.csv(total_df, file="assets/total_df.csv", row.names=F)

x <- total_df[ , 4:(ncol(total_df) - 1)]
y <- total_df$y

########
########  Functions & Global Variables  ########################################
########

###############################################
#                                             #
# **** START HERE AFTER LOADING total_df **** #
#                                             #
###############################################


# check_accuracy()
# Function takes either the discretized results or the probalities as model
# outputs; in the case of the latter, it uses a cutoff of 0.50 to determine
# whether the occupation is analytical or not

# TODO: This should return/print a confusion matrix!!
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

# Set seed for train/test splits
set.seed(987654)


########
########  Logistic Regression  #################################################
########

# Create training set for logistic regression
logit_sample <- createDataPartition(y=total_df$Analytical.IM, p=0.12, list=F)
logit_train <- total_df[logit_sample, ]

# Remove logit_sample from total_df and save as total to keep total_df unchanged
total <- total_df[-logit_sample, ]

# Fit logistic regression model
logit_model <- 
  glm(y ~ Category.Flexibility + Deductive.Reasoning + Flexibility.of.Closure +
          Fluency.of.Ideas + Inductive.Reasoning + Information.Ordering +
          Mathematical.Reasoning + Memorization + Number.Facility + Oral.Comprehension +
          Oral.Expression + Originality + Perceptual.Speed + Problem.Sensitivity +
          Selective.Attention + Spatial.Orientation + Speed.of.Closure + Time.Sharing +
          Visualization + Written.Comprehension + Written.Expression,
          data=logit_train, family=binomial("logit"))

# Take a look at model performance measures
logit_summary <- summary(logit_model)
print(logit_summary)
anova(logit_model, test="Chisq")
confint(logit_model)
plot(logit_model)

ggplot(logit_train, aes(x=Analytical, y=Predicted_Probability)) + 
  geom_ribbon(aes(ymin=LL, ymax=UL, fill=rank), alpha=0.2) + 
  geom_line(aes(colour=rank), size=1)

# Turn off scientific notation
options(scipen = 999)

# Save and plot output
logit_train$yhat <- fitted(logit_model)

# TODO: plot line of best fit
plot(logit_train$yhat, pch=16)
#lines(logit_train$yhat, col="red", lwd=2)

# Boundary
abline(h=0.5, lty=2)

# Check accuracy
logit_train_score <- check_accuracy(logit_train)
print(logit_train_score)            # Prints 0.9327731


########
########  Linear Discriminant Analysis  ########################################
########

# Check length of logit_sample
length(logit_sample)    # Prints 119

# Create training set for LDA
# Increased the probability to result in the same sample size as previous model
LDA_sample <- createDataPartition(y=total$Analytical.IM, p=(119 / nrow(total)), list=F)
LDA_train <- total[LDA_sample, ]


# Remove LDA_sample from total_df
total <- total[-LDA_sample, ]

# Check descriptive stats
summary(LDA_train)

# Check standard deviation for all x's
# applyabilities_df[, 1] <- sapply(abilities_df[, 1], as.character)
LDA_train_stdev <- sapply(LDA_train[, 1], )

LDA_train_center <- preProcess(LDA_train[ , 4:(ncol(LDA_train) - 1)], method=c("center", "scale"))

# Transform sample data to Gaussian distribution; DO NOT normalize y
LDA_train_center <- scale(LDA_train[ , 4:(ncol(LDA_train) - 1)], center=TRUE, scale=TRUE)

# Convert back to dataframe
# NOTE: This must be done before appending y
LDA_train_center <- as.data.frame(LDA_train_center)

# Append y back to LDA_train_center
LDA_train_center$y <- LDA_train$y

# Check descriptive stats
summary(LDA_train_center)

# Fit LDA model
LDA_model <- 
  lda(y ~ Category.Flexibility + Deductive.Reasoning + Flexibility.of.Closure +
          Fluency.of.Ideas + Inductive.Reasoning + Information.Ordering +
          Mathematical.Reasoning + Memorization + Number.Facility + Oral.Comprehension +
          Oral.Expression + Originality + Perceptual.Speed + Problem.Sensitivity +
          Selective.Attention + Spatial.Orientation + Speed.of.Closure + Time.Sharing +
          Visualization + Written.Comprehension + Written.Expression,
          data=LDA_train_center)

predmodel.train.lda <- predict(LDA_model, data=LDA_train_center)

# Save output
LDA_train_center$yhat <- predmodel.train.lda$class

# Take a look at performance measures
# TODO: Fix this plot
plot(LDA_model)     
table(predmodel.train.lda$class)
print(LDA_model)

# Check accuracy
LDA_train_score <- check_accuracy(LDA_train_center)   
print(LDA_train_score)        # Prints 0.8333333


########
########  Quadratic Discriminant Analysis  #####################################
########


# Create training set for QDA
# Increased the probability to result in the same sample size as previous model
QDA_sample <- createDataPartition(y=total$Analytical.IM, p=(119 / nrow(total)), list=F)
QDA_train <- total[QDA_sample, ]

# Remove LDA_sample from total_df
total <- total[-QDA_sample, ]

# Check descriptive stats
summary(QDA_train)

# Transform sample data to Gaussian distribution; DO NOT normalize y
# QDA_train_center <- scale(QDA_train[ , 4:(ncol(QDA_train) - 1)], center=TRUE, scale=TRUE)
# Append y back to LDA_train_center
# QDA_train_center$y <- QDA_train$y

# Convert back to dataframe
# QDA_train_center <- as.data.frame(LDA_train_center)

# summary(QDA_train_center)

# Fit QDA model
QDA_model <- 
  qda(y ~ Category.Flexibility + Deductive.Reasoning + Flexibility.of.Closure +
          Fluency.of.Ideas + Inductive.Reasoning + Information.Ordering +
          Mathematical.Reasoning + Memorization + Number.Facility + Oral.Comprehension +
          Oral.Expression + Originality + Perceptual.Speed + Problem.Sensitivity +
          Selective.Attention + Spatial.Orientation + Speed.of.Closure + Time.Sharing +
          Visualization + Written.Comprehension + Written.Expression,
          data=QDA_train)

predmodel.train.qda <- predict(QDA_model, newdata=QDA_train)

# Save labels
QDA_train_center$yhat <- predmodel.train.qda$class

# Take a look at performance measures
table(predmodel.train.qda$class)
print(QDA_model)

# Check accuracy
QDA_train_score <- check_accuracy(QDA_train)   
print(QDA_train_score)      # Prints 0.9416667


########
########  Run Chosen Algorithm On Remaining Data  ##############################
########

# TODO: Normalize data  

# Save training scores
train_scores <- c(logit_train_score, LDA_train_score, QDA_train_score)
print(train_scores)   # QDA has highest score by nearly 1 percentage point

# Run QDA on the remaining data
predmodel.test.qda <- predict(QDA_model, data=total)

# Save labels
total$yhat <- predmodel.test.qda$class

# Print output
table(predmodel.test.qda$class)

# Check accuracy
QDA_test_score <- check_accuracy(total)   
print(QDA_test_score)    # Prints 0.9416667

table(Predicted=predmodel.train.qda$class, y=y)


# TODO: Combine test and train data, then plot

# TODO: Print the most/least analytical occupations









