# Clear plots
if(!is.null(dev.list())) dev.off()
# Clear console
cat("\014") 
# Clean workspace
rm(list=ls())

library(data.table)
library(caret)
library(glmnet)
library(MASS)
library(pROC)


# Read in Data into data table
wdbc2.dt <- as.data.table(read.csv('wdbc2.csv'))

# Set the randomising seed to 1 for consistent results
set.seed(1)

# Create list of indices to partition the data into training and test
partition <- createDataPartition(wdbc2.dt$diagnosis, p=0.7)$Resample1

training.set <- wdbc2.dt[partition]
test.set <- wdbc2.dt[-partition]

# prepare data for glmnet
prepare.glmnet <- function(data, formula=~ .) {
  ## create the design matrix to deal correctly with factor variables,
  ## without losing rows containing NAs
  old.opts <- options(na.action='na.pass')
  x <- model.matrix(formula, data)
  options(old.opts)
  ## remove the intercept column, as glmnet will add one by default
  x <- x[, -match("(Intercept)", colnames(x))]
  return(x)
}

# Split into outcome and predictors
outcome.training.set <- training.set$diagnosis

# Standardise data
standardise.data <- function(x){
  y <- setNames(data.table(matrix(nrow=nrow(x),ncol=ncol(x))), colnames(x))
  for (col in colnames(x)){
   y[[col]] <- c((x[[col]]-mean(x[[col]]))/sd(x[[col]]))
  }
  return(y)
  
  # for (i in 1:ncol(x)){
  #   x[,i] <- (x[,i]-mean(x[,i]))/sd(x[,i])
  # }
  # return(x)
}
vars.training.set <- standardise.data(training.set[,!c('id','diagnosis')])
vars.training.set.glmnet <- prepare.glmnet(training.set[,!c('id')],~.-diagnosis)

# fit.lasso <- glmnet(vars.training.set, outcome.training.set)
# fit.ridge <- glmnet(vars.training.set, outcome.training.set, alpha=0)
# 
# plot(fit.lasso, main='Lasso Trajectories')
# plot(fit.ridge, main='Ridge Trajectories')

# Fit lasso and ridge regression models
cv.fit.lasso <- cv.glmnet(vars.training.set.glmnet, outcome.training.set, family='binomial', type.measure = 'auc', alpha=1)
cv.fit.ridge <- cv.glmnet(vars.training.set.glmnet, outcome.training.set, family='binomial', type.measure = 'auc', alpha=0)

# Plot Models
plot(cv.fit.lasso, main='Lasso Trajectories')
plot(cv.fit.ridge, main='Ridge Trajectories')

# Lambda that maximises the AUC
lasso.lambda.max <- cv.fit.lasso$lambda.min
ridge.lambda.max <- cv.fit.ridge$lambda.min

# Lambda that gives AUC 1 standard error away from max
lasso.lambda.1se <- cv.fit.lasso$lambda.1se
ridge.lambda.1se <- cv.fit.ridge$lambda.1se

# Corresponding value of AUC
lasso.auc.max <- cv.fit.lasso$cvm[cv.fit.lasso$lambda==lasso.lambda.max]
ridge.auc.max <- cv.fit.ridge$cvm[cv.fit.ridge$lambda==ridge.lambda.max]
lasso.auc.1se <- cv.fit.lasso$cvm[cv.fit.lasso$lambda==lasso.lambda.1se]
ridge.auc.1se <- cv.fit.ridge$cvm[cv.fit.ridge$lambda==ridge.lambda.1se]

# Data table comparing different models
models.dt <- data.table(model.name=c('Lasso', 'Ridge'), 
                        lambda.max=signif(c(lasso.lambda.max, ridge.lambda.max),3),
                        lambda.1se=signif(c(lasso.lambda.1se, ridge.lambda.1se),3),
                        auc.max=signif(c(lasso.auc.max, ridge.auc.max),3),
                        auc.1se=signif(c(lasso.auc.1se, ridge.auc.1se),3),
                        Model.Size.aucmax=c(cv.fit.lasso$nzero[cv.fit.lasso$lambda==lasso.lambda.max], cv.fit.ridge$nzero[cv.fit.ridge$lambda==ridge.lambda.max]),
                        Model.Size.auc1se=c(cv.fit.lasso$nzero[cv.fit.lasso$lambda==lasso.lambda.1se], cv.fit.ridge$nzero[cv.fit.ridge$lambda==ridge.lambda.1se])
                        )


# Backward elimination
full.model <- glm(outcome.training.set ~ ., data=vars.training.set, family='binomial')
back.elim <- stepAIC(full.model, direction='back')
back.coef <- back.elim$coefficients[names(back.elim$coefficients)!='(Intercept)']
back.coef <- back.coef[order(-abs(back.coef))]
back.coef <- data.table(variables=names(back.coef), Regression.coefficient=back.coef)

# Stepwise selection
null.model <- glm(outcome.training.set ~ 1, data=vars.training.set, family='binomial')
step.sel <- step(null.model, scope=list(upper=full.model), direction='both')
step.coef <- abs(step.sel$coefficients)[order(abs(step.sel$coefficients))]
step.coef <- step.sel$coefficients[names(step.sel$coefficients)!='(Intercept)']
step.coef <- step.coef[order(-abs(step.coef))]
step.coef <- data.table(variables=names(step.coef), Regression.coefficient=step.coef)

# Compare back and stepwise
goodness.of.fit <- data.table(Model = c('Backward Elimination', 'Step-wise Selection'),
                              AIC = c(AIC(back.elim), AIC(step.sel)))

print(AIC(back.elim)) # This is lower so better model
print(AIC(step.sel))

# Make Predictions on training set
pred.back <- back.elim$fitted.values
back.roc <- roc(outcome.training.set ~ pred.back, plot=FALSE)
back.auc <- back.roc$auc
pred.step <- step.sel$fitted.values
step.roc <- roc(outcome.training.set ~ pred.step, plot=FALSE)
step.auc <- step.roc$auc


# Prep test set
outcome.test.set <- as.numeric(test.set$diagnosis)-1
vars.test.set <- standardise.data(test.set[,!c('id','diagnosis')])
vars.test.set.glmnet <- prepare.glmnet(test.set[,!c('id')],~.-diagnosis)

# Make predictions on test set
test.pred.lasso <- c(predict(cv.fit.lasso, newx=vars.test.set.glmnet,s=lasso.lambda.1se, type='response'))
test.pred.ridge <- c(predict(cv.fit.ridge, newx=vars.test.set.glmnet,s=ridge.lambda.1se, type='response'))
test.pred.back <- unname(predict(back.elim, newdata=vars.test.set,type='response'))
test.pred.step <- unname(predict(step.sel, newdata=vars.test.set,type='response'))

predictions <- data.table(Lasso=test.pred.lasso, ridge=test.pred.ridge, 
                          back=test.pred.back, step=test.pred.step)

# Plot ROC and get AUC
test.roc.lasso <- roc(outcome.test.set~test.pred.lasso, plot=TRUE, col='blue')
test.auc.lasso <- test.roc.lasso$auc
test.roc.ridge <- roc(outcome.test.set~test.pred.ridge, plot=TRUE, col='green',  add=TRUE)
test.auc.ridge <- test.roc.ridge$auc
test.roc.back <- roc(outcome.test.set~test.pred.back, plot=TRUE, col='red',  add=TRUE)
test.auc.back <- test.roc.back$auc
test.roc.step <- roc(outcome.test.set~test.pred.step, plot=TRUE, col='brown',  add=TRUE)
test.auc.step <- test.roc.step$auc

legend('bottomright', c('Lasso regression','Ridge regression','Back elimination','Stepwise selection'), col=c('blue','green', 'red', 'brown'), lty=1)

test.auc.dt <- data.table(Lasso=test.auc.lasso, ridge=test.auc.ridge, 
                         back=test.auc.back, step=test.auc.step)
