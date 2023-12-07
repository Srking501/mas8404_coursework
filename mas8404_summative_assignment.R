##################################################
## MAS8404 Statistical Learning for Data Science #
## Summative Coursework on Breast Cancer Data    #
## Abdullah Turki H Alshadadi, 190582184         #
##################################################

## Load packages
library(mlbench)
library(tidyverse)
library(leaps)
library(bestglm)
library(glmnet)
library(nclSLR)
library(MASS)
library(gridExtra)

## Load the data
data(BreastCancer)

## Check size
dim(BreastCancer)

## Print first few rows
head(BreastCancer)

###################################
## 0 Data Cleaning and Formatting #
###################################

## Remove NA values
bc = BreastCancer %>%
    na.omit()

## Convert the 9 cytological characteristics into
## numeric data type
bc = bc %>%
    mutate_at(vars(2:10), as.numeric)

## Convert Class into integers:
## 0: benign
## 1: malignant
bc = bc %>%
    mutate(Class = as.integer(Class) - 1)

## Without id
bc_no_id = bc %>%
    dplyr::select(-1)


######################################
## 1 Exploratory Data Analysis (EDA) #
######################################
## Check the distribution of
## 0: benign
## 1: malignant
table(bc$Class)

class_distribution = data.frame(
    benign = bc_no_id %>% dplyr::select(Class) %>% filter(Class == 0) %>% nrow(),
    malignant = bc_no_id %>% dplyr::select(Class) %>% filter(Class == 1) %>% nrow()
)

## Does not include the categorical data id
pairs(bc_no_id[,-ncol(bc_no_id)], # X1 explanatory variables
      col = bc_no_id$Class+1)     # y response variable

cor_bc = cor(bc_no_id)

summary_stats = aggregate(. ~ Class, data = bc_no_id,
                          function(x) c(Minimum = min(x),
                                        Maximum = max(x),
                                        Median = median(x),
                                        Mean = mean(x),
                                        SD = sd(x)))
print(summary_stats)

# summary_stats = summary_stats %>%
#     pivot_longer(col = -Class, names_to = "Variables",
#                              values_to = "v")
#
# colnames(summary_stats) <- c("Class", "Variables", "Min",
#                              "Max", "Median", "Mean", "SD")
# benign_stats = summary_stats %>%
#     filter(Class == "0") %>%
#     mutate(Class = "Benign (0)")
#
# malignant_stats = summary_stats %>%
#     filter(Class == "1") %>%
#     mutate(Class = "Malignant (1)")

data_cols = as.vector(colnames(bc_no_id[,-ncol(bc_no_id)]))

create_summary_stats = function(data_columns) {
    # 0: benign
    ss0 = data.frame(
        Variables = data_columns,
        Min = NA,
        Max = NA,
        Median = NA,
        Mean = NA,
        SD = NA
    )

    # 1: malignant
    ss1 = data.frame(
        Variables = data_columns,
        Min = NA,
        Max = NA,
        Median = NA,
        Mean = NA,
        SD = NA
    )

    # 0: benign
    stat_min_b = c()
    stat_max_b = c()
    stat_median_b = c()
    stat_mean_b = c()
    stat_sd_b = c()

    # 1: malignant
    stat_min_m = c()
    stat_max_m = c()
    stat_median_m = c()
    stat_mean_m = c()
    stat_sd_m = c()
    for (column in data_columns) {
        # 0: benign
        benign_stats = bc_no_id %>% filter(Class == 0) %>% pull(column)

        stat_min_b = c(stat_min_b, min(benign_stats))
        stat_max_b = c(stat_max_b, max(benign_stats))
        stat_median_b = c(stat_median_b, median(benign_stats))
        stat_mean_b = c(stat_mean_b, mean(benign_stats))
        stat_sd_b = c(stat_sd_b, sd(benign_stats))

        # 1: malignant
        malignant_stats = bc_no_id %>% filter(Class == 1) %>% pull(column)

        stat_min_m = c(stat_min_m, min(malignant_stats))
        stat_max_m = c(stat_max_m, max(malignant_stats))
        stat_median_m = c(stat_median_m, median(malignant_stats))
        stat_mean_m = c(stat_mean_m, mean(malignant_stats))
        stat_sd_m = c(stat_sd_m, sd(malignant_stats))
    }

    # 0: benign
    ss0 = ss0 %>% mutate(
                Min = stat_min_b,
                Max = stat_max_b,
                Median = stat_median_b,
                Mean = round(stat_mean_b, 2),
                SD = round(stat_sd_b, 2))

    # 1: malignant
    ss1 = ss1 %>% mutate(
                Min = stat_min_m,
                Max = stat_max_m,
                Median = stat_median_m,
                Mean = round(stat_mean_m, 2),
                SD = round(stat_sd_m, 2))

    return(list(benign = ss0,
                malignant = ss1))
}

summary_stats = create_summary_stats(data_cols)

################################
## 2 Cross-validation function #
################################
# 2.0 Initial setups
#####
## Store n and p
n = nrow(bc_no_id); p = ncol(bc_no_id) - 1

## Set the seed to make the analysis reproducible
set.seed(1)

## 10-fold cross validation
nfolds = 10
## Sample fold-assignment index
fold_index = sample(nfolds, n, replace=TRUE)
## Print first few fold-assignments
head(fold_index)

# 2.1 Cross-validation function
#####
#
#' Cross-validation Function (using K-fold)
#'
#' Given a dataset of `X1` explanatory variables and `y` response variable,
#' provide a cross-validation using k-fold on the chosen `model_fit` function
#' and return the average `training_error`, `testing_error`; and total
#' `train_confusion` and `test_confusion` matrices.
#'
#' `model_name` options:
#' BIC   = Logistic Regression with BIC subset selection
#' LASSO = Logistic Regression with LASSO regularisation penalty
#' LDA   = Linear Discriminant Analysis
#' QDA   = Quadratic Discriminant Analysis
#'
#' @param X1 The explanatory variables
#' @param y The response variable
#' @param fold_ind The sampled fold indices
#' @param model_name The model that is going to be cross-validated ("BIC", "LASSO", "LDA", "QDA")
#'
#' @return A list of average `training_error`, `testing_error`; and overall
#' `train_confusion` and `test_confusion`.
#'
#'
cv_function = function(X1, y, fold_ind, model_name) {
    Xy = data.frame(X1, y=y)
    nfolds = max(fold_ind)
    if(!all.equal(sort(unique(fold_ind)), 1:nfolds)) stop("Invalid fold partition.")

    train_confusion_matrices = list()  # Store train confusion matrices for each fold
    test_confusion_matrices = list()  # Store test confusion matrices for each fold

    models_per_fold = list()
    other = list()

    for(fold in 1:nfolds) {
        ##
        ## Logistic Regression with BIC subset selection
        ##
        if (model_name == "BIC") {
            # Reset the dataframe for subset selection (due to previous
            # loop updating the subset selection)
            Xy = data.frame(X1, y=y)

            # Perform subset selection with bestglm using BIC on training dataset
            bss_model_bic = bestglm(Xy[fold_ind!=fold,][,-ncol(Xy)], # X1 explanatory variables
                                    y[fold_ind!=fold],               # y response variable
                                    family = gaussian,
                                    IC = "BIC")

            # Other details of model
            other[[fold]] = bss_model_bic

            ## Identify best-fitting models
            (model_best_BIC = bss_model_bic$ModelReport$Bestk)

            ##
            ## TRAIN PLOT
            ##
            # par = par(mfrow=c(5,5))
            # print("IM HERE")
            # plot(0:p, bss_model_bic$Subsets$BIC, xlab="Number of predictors", ylab="BIC", type="b")
            # points(model_best_BIC, bss_model_bic$Subsets$BIC[model_best_BIC+1], col="red", pch=16)

            # Get the selected subset of predictors
            bss = bss_model_bic$Subsets[model_best_BIC+1,]
            bss = bss %>% dplyr::select(-`(Intercept)`, -logLikelihood, -BIC)

            # print("bss")
            # print(bss)

            # Get the column names of the best subset selection
            col_bss = as.vector(colnames(bss)[apply(bss, 2, any)])

            # print("colnames")
            # print(col_bss)
            # print("-----------------------")

            # Make the Xy dataframe only contain the best subset from BIC
            Xy = Xy %>% dplyr::select(col_bss, y)

            # Train model with training dataset
            model_fit = glm(y ~ ., data = Xy[fold_ind!=fold,], family = "binomial")

            # Extracting model
            models_per_fold[[fold]] = model_fit

            # Find phat_train and phat_test for the model_fit
            phat_train = predict(model_fit, Xy[fold_ind!=fold,], type = "response")
            phat_test = predict(model_fit, Xy[fold_ind==fold,], type = "response")
        }

        ##
        ## Logistic Regression with LASSO regularisation penalty
        ##
        else if (model_name == "LASSO") {
            ## Choose grid of values for the tuning parameter
            grid = 10^seq(4, -4, length.out=100)

            # Extract the best optimal tuning parameter
            model_cv_fit = cv.glmnet(as.matrix(Xy[fold_ind!=fold,][,-ncol(Xy)]), # X1 explanatory variables
                               y[fold_ind!=fold],                                # y response variable
                               family="binomial",
                               alpha=1, standardize=FALSE, lambda=grid,
                               type.measure="class")

            lambda_lasso_min_train = model_cv_fit$lambda.min

            # Other details of model
            other[[fold]] = lambda_lasso_min_train

            # Train model with training dataset and optimal tuning parameter
            model_fit = glmnet(as.matrix(Xy[fold_ind!=fold,][,-ncol(Xy)]), # X1 explanatory variables
                                  y[fold_ind!=fold],                       # y response variable
                                  family="binomial",
                                  alpha=1, standardize=FALSE, lambda=lambda_lasso_min_train)

            # Extracting model
            models_per_fold[[fold]] = model_fit

            # Find phat_train and phat_test for the model_fit
            phat_train = predict(model_fit, as.matrix(Xy[fold_ind!=fold,][,-ncol(Xy)]), # X1 explanatory variables
                                 s=lambda_lasso_min_train, # Optimal Tuning Parameter
                                 type = "response")
            phat_test = predict(model_fit, as.matrix(Xy[fold_ind==fold,][,-ncol(Xy)]), # X1 explanatory variables
                                s=lambda_lasso_min_train, # Optimal Tuning Parameter
                                type = "response")
        }

        ##
        ## Linear Discriminant Analysis
        ##
        else if (model_name == "LDA") {
            lda_details = linDA(bc_no_id[,-ncol(bc_no_id)], # X1 explanatory variables
                                bc_no_id$Class)             # y response variable

            # Other details of model
            other[[fold]] = lda_details

            # Train model with training dataset
            model_fit = lda(y~., data=Xy[fold_ind!=fold,])

            # Extracting model
            models_per_fold[[fold]] = model_fit

            # Find da_train and da_test for the model_fit (Discriminant Analysis)
            da_train = predict(model_fit, Xy[fold_ind!=fold,], type = "response")
            da_test = predict(model_fit, Xy[fold_ind==fold,], type = "response")
        }

        ##
        ## Quadratic Discriminant Analysis
        ##
        else if (model_name == "QDA") {
            qda_details = quaDA(bc_no_id[,-ncol(bc_no_id)], # X1 explanatory variables
                                bc_no_id$Class,             # y response variable
                                functions=TRUE)

            # Other details of model
            other[[fold]] = qda_details

            # Train model with training dataset
            model_fit = qda(y~., data=Xy[fold_ind!=fold,])

            # Extracting model
            models_per_fold[[fold]] = model_fit

            # Find da_train and da_test for the model_fit (Discriminant Analysis)
            da_train = predict(model_fit, Xy[fold_ind!=fold,], type = "response")
            da_test = predict(model_fit, Xy[fold_ind==fold,], type = "response")
        }

        else {
            stop("Invalid input of `model_name`, please check function comments.")
        }

        # Observations
        yobs_train = y[fold_ind!=fold]
        yobs_test = y[fold_ind==fold]

        # Logistic Regression training and error rates
        if (model_name == "BIC" | model_name == "LASSO") {
            ##
            ## TRAINING & TESTING ERROR RATES
            ##
            # Training error calculation
            yhat_train = ifelse(phat_train > 0.5, 1, 0)
            # Train Confusion matrix calculation
            train_confusion_matrices[[fold]] = table(Observed = yobs_train, Predicted = yhat_train)

            # Testing error calculation
            yhat_test = ifelse(phat_test > 0.5, 1, 0)
            # Test Confusion matrix calculation
            test_confusion_matrices[[fold]] = table(Observed = yobs_test, Predicted = yhat_test)
        }

        # Discriminant Analysis training and error rates
        else if (model_name == "LDA" | model_name == "QDA") {
            ##
            ## TRAINING & TESTING ERROR RATES
            ##
            # Training error calculation
            yhat_train = da_train$class
            # Train Confusion matrix calculation
            train_confusion_matrices[[fold]] = table(Observed = yobs_train, Predicted = yhat_train)

            # Testing error calculation
            yhat_test = da_test$class
            # Test Confusion matrix calculation
            test_confusion_matrices[[fold]] = table(Observed = yobs_test, Predicted = yhat_test)
        }

        else {
            stop("Invalid input of `model_name`, please check function comments.")
        }
    }

    # Combine confusion matrices from all folds to get an overall confusion matrix
    overall_train_confusion_matrix = Reduce('+', train_confusion_matrices)
    overall_test_confusion_matrix = Reduce('+', test_confusion_matrices)

    ## Normalises a vector
    normalise = function(x) {
        return(x / sum(x))
    }
    ## Normalise each row of the confusion matrix:
    overall_train_confusion_matrix = t(apply(overall_train_confusion_matrix, 1, normalise))
    overall_test_confusion_matrix = t(apply(overall_test_confusion_matrix, 1, normalise))

    avg_training_error = 1 - (sum(diag(overall_train_confusion_matrix)) / sum(overall_train_confusion_matrix))
    avg_testing_error = 1 - (sum(diag(overall_test_confusion_matrix)) / sum(overall_test_confusion_matrix))

    return(list(training_error  = avg_training_error,
                testing_error   = avg_testing_error,
                train_confusion = overall_train_confusion_matrix,
                test_confusion  = overall_test_confusion_matrix,
                models          = models_per_fold,
                other           = other))
}


##########################
## 3 Logistic Regression #
##########################
# 3.1 Subset selection
#####
## Apply best subset selection
bss_fit_BIC = bestglm(bc_no_id, family=binomial, IC="BIC")

## Identify best-fitting models
(best_BIC = bss_fit_BIC$ModelReport$Bestk)

## Reset multipanel back to 1x1
par(mfrow=c(1,1))

plot(0:p, bss_fit_BIC$Subsets$BIC, xlab="Number of predictors", ylab="BIC", type="b")
points(best_BIC, bss_fit_BIC$Subsets$BIC[best_BIC+1], col="red", pch=16)

# Get the selected subset of predictors
bss = bss_fit_BIC$Subsets[best_BIC+1,]
print(bss)
bss = bss %>% dplyr::select(-`Intercept`, -logLikelihood, -BIC)

# Get the column names of the best subset selection
col_bss = as.vector(colnames(bss)[apply(bss, 2, any)])

print(col_bss)
print(bc_no_id %>% dplyr::select(col_bss, Class) %>% head())

## Obtain regression coefficients for this model
bss_fit = glm(Class ~ ., data=bc_no_id %>% dplyr::select(col_bss, Class), family="binomial")
summary(bss_fit)

# Coefficients of BIC in Logsitic Regression
coef(bss_fit)


# 3.2 Regression methods
#####
## Choose grid of values for the tuning parameter
grid = 10^seq(4, -4, length.out=100)
## Fit a model with LASSO penalty for each value of the tuning parameter
lasso_fit = glmnet(as.matrix(bc_no_id[,-ncol(bc_no_id)]), # X1 explanatory variables
                   bc_no_id$Class,  # y response variable
                   family="binomial",
                   alpha=1, standardize=FALSE, lambda=grid)

# ## Reset multipanel back to 1x1
# par(mfrow=c(1,1))

## Examine the effect of the tuning parameter on the parameter estimates
plot(lasso_fit, xvar="lambda", col=rainbow(p), label=TRUE)

lasso_cv_fit =
    cv.glmnet(as.matrix(bc_no_id[,-ncol(bc_no_id)]), # X1 explanatory variables
                         bc_no_id$Class,  # y response variable
                         family="binomial",
                         alpha=1, standardize=FALSE, lambda=grid,
                         type.measure="class",
                         nfolds = nfolds) # To make sure it is 10-folds!

plot(lasso_cv_fit)

## Identify the optimal value for the tuning parameter
(lambda_lasso_min = lasso_cv_fit$lambda.min)

which_lambda_lasso = which(lasso_cv_fit$lambda == lambda_lasso_min)
## Find the parameter estimates associated with optimal value of the tuning parameter
coef(lasso_fit, s=lambda_lasso_min)


############################
## 4 Discriminant Analysis #
############################
# 4.0 Initial Details
#####
# plot(0:10, bc_no_id[,-ncol(bc_no_id)], col=bc_no_id$Class,
#      pch=bc_no_id$Class, xlab="GPA", ylab="GMAT",
#      ylim=c(200, 800), xlim=c(2,4))
# legend("topleft", c("admit","borderline","not admit"), col=2:4, pch=1:3)

# 4.1 Linear Discriminant Analysis (LDA)
#####
lda_info = linDA(bc_no_id[,-ncol(bc_no_id)], # X1 explanatory variables
                 bc_no_id$Class)             # y response variable

lda_data = cbind(as.data.frame(lda_info$scores),
                 as.data.frame(lda_info$classification))

lda_data = lda_data %>%
    dplyr::rename("lda1" = `0`,
                  "lda2" = `1`,
                  "Class" = `lda_info$classification`)

lda_plot = ggplot(lda_data, aes(x = lda1, y = lda2, color = Class)) +
    geom_point() +
    labs(title = "LDA Plot", x = "LD1", y = "LD2") +
    scale_color_manual(values = c("lightblue", "orange")) +
    theme_minimal()

lda_fit = lda(Class~., data=bc_no_id)
print(lda_fit)
summary(lda_fit)
coef(lda_fit)

lda_values = predict(lda_fit)
ldahist(lda_values$x[,1], g = bc_no_id$Class)

# 4.2 Quadratic Discriminant Analysis (QDA)
#####
qda_info = quaDA(bc_no_id[,-ncol(bc_no_id)], # X1 explanatory variables
                 bc_no_id$Class,             # y response variable
                 functions=TRUE)


qda_fit = qda(Class~., data=bc_no_id)
print(qda_fit)
summary(qda_fit)
coef(qda_fit)


###########################
## 5 Determine Best Model #
###########################
# 5.1 Cross validate the different models
#####
bic_model = cv_function(bc_no_id[,-ncol(bc_no_id)], # X1 explanatory variables
                        bc_no_id$Class,             # y response variable
                        fold_index,
                        model_name = "BIC")

lasso_model = cv_function(bc_no_id[,-ncol(bc_no_id)], # X1 explanatory variables
                          bc_no_id$Class,             # y response variable
                          fold_index,
                          model_name = "LASSO")

lda_model = cv_function(bc_no_id[,-ncol(bc_no_id)], # X1 explanatory variables
                        bc_no_id$Class,             # y response variable
                        fold_index,
                        model_name = "LDA")

qda_model = cv_function(bc_no_id[,-ncol(bc_no_id)], # X1 explanatory variables
                        bc_no_id$Class,             # y response variable
                        fold_index,
                        model_name = "QDA")

# 5.2 Plot
#####
train_accuracy = 1 - c(bic_model$training_error, lasso_model$training_error,
                       lda_model$training_error)
test_accuracy  = 1 - c(bic_model$testing_error, lasso_model$testing_error,
                       lda_model$testing_error)

cv_data = data.frame(
    models = c("Log Reg with BIC", "Log Reg with LASSO", "LDA"),
    train_accuracy_rate = round(100 * train_accuracy, 2),
    train_error_rate = round(100 * c(bic_model$training_error, lasso_model$training_error,
                         lda_model$training_error), 2),
    test_accuracy_rate = round(100 * test_accuracy,2),
    test_error_rate = round(100* c(bic_model$testing_error, lasso_model$testing_error,
                        lda_model$testing_error), 2)
)

cv_data

# index = 1
# lda_plots = list()
# for (lda_details in lda_model$other) {
#     lda_data = cbind(as.data.frame(lda_details$scores),
#                      as.data.frame(lda_details$classification))
#
#     print(head(lda_data))
#     print(paste("index:", index))
#
#     lda_data = lda_data %>%
#         dplyr::rename("lda1" = `0`,
#                       "lda2" = `1`,
#                       "Class" = `lda_details$classification`)
#
#     lda_plot_per_fold = ggplot(lda_data, aes(x = lda1, y = lda2, color = Class)) +
#         geom_point() +
#         labs(title = paste("LDA Plot", index), x = "LD1", y = "LD2") +
#         scale_color_manual(values = c("lightblue", "orange")) +
#         theme_minimal()
#
#     lda_plots[[index]] = lda_plot_per_fold
#
#     index = index + 1
# }
#
# print(lda_plots)
#
# grid.arrange(grobs = lda_plots, ncol = 5)
