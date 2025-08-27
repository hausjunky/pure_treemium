#install.packages("CASdatasets", repos = "http://cas.uqam.ca/pub/", type = "source")

library(CASdatasets)
library(data.table)
library(mltools)
library(xgboost)
options(scipen = 999)

data("pg15training")
data <- data.table(pg15training)
rm(pg15training)

vars <- c("POL", "CAL", "GEN", "TYP", "CAT", "OCC", "AGE", "GRP", "BNS", "TEN",
          "VAL", "MAT", "SUB", "REG", "DEN", "EXP", "CPD", "CBI", "APD", "ABI")

setnames(data, names(data), vars)

data[, TST := fifelse(CAL == 2009, 0, 1)]

dups <- data[, .(count = .N), .(POL, CAL)]
dups <- dups[count > 1]
dups <- merge.data.table(data, dups[, .(POL, CAL)], c("POL", "CAL"))
dups[, MAX := max(APD + ABI), .(POL, CAL)]
dups <- dups[APD + ABI == MAX, -c("MAX")]
data <- data[!dups, on = .(POL, CAL)]
data <- rbind(data, dups)
setorder(data, POL, CAL)
data[, CAL := NULL]
rm(dups)

head <- c("POL", "TST", "EXP", "CPD", "APD", "CBI", "ABI")
vars <- setdiff(names(data), head)
keep <- c(head, vars)
data <- data[, ..keep]

facs <- names(data[, sapply(data, is.factor), with = FALSE])
data <- one_hot(data)
head <- c("POL", "TST", "EXP", "CPD", "APD", "CBI", "ABI")
vars <- setdiff(names(data), head)

cpd <- data[TST == 0, -c("TST", "CPD", "CBI", "ABI")]
apd <- data[TST == 0 & APD > 1, -c("TST", "CPD", "CBI", "ABI")]

cbi <- data[TST == 0, -c("TST", "CPD", "CBI", "APD")]
abi <- data[TST == 0 & ABI > 1, -c("TST", "CPD", "CBI", "APD")]

scr <- data[TST == 1, -c("TST")]

gc(rm(data))

adj <- 1 / (mean(cpd$EXP / 365))

cbi <- xgb.DMatrix(as.matrix(cbi[, ..vars]), label = fifelse(cbi$ABI > 0, 1, 0), weight = cbi$EXP)
cpd <- xgb.DMatrix(as.matrix(cpd[, ..vars]), label = fifelse(cpd$APD > 0, 1, 0), weight = cpd$EXP)

hyp <- list(objective = "binary:logistic",
            tree_method = "hist",
            grow_policy = "lossguide",
            max_depth = 0,
            eta = .01,
            max_leaves = 3,
            subsample = .5,
            colsample_bytree = .5)

set.seed(42)
xcv <- xgb.cv(hyp, cbi, 1e6, 5, early_stopping_rounds = 10)
cbi <- xgb.train(hyp, cbi, xcv$best_iteration)

set.seed(42)
xcv <- xgb.cv(hyp, cpd, 1e6, 5, early_stopping_rounds = 10)
cpd <- xgb.train(hyp, cpd, xcv$best_iteration)

mtx <- xgb.DMatrix(as.matrix(scr[, ..vars]))
scr$IBC <- predict(cbi, mtx)
scr$DPC <- predict(cpd, mtx)

abi <- xgb.DMatrix(as.matrix(abi[, ..vars]), label = log(abi$ABI / abi$EXP), weight = abi$EXP)
apd <- xgb.DMatrix(as.matrix(apd[, ..vars]), label = log(apd$APD / apd$EXP), weight = apd$EXP)

hyp <- list(objective = "reg:squarederror",
            tree_method = "hist",
            grow_policy = "lossguide",
            max_depth = 0,
            eta = .01,
            max_leaves = 3,
            subsample = .5,
            colsample_bytree = .5)

set.seed(42)
xcv <- xgb.cv(hyp, abi, 1e6, 5, early_stopping_rounds = 10)
mod <- xgb.train(hyp, abi, xcv$best_iteration)
bac <- mean(exp(getinfo(abi, "label"))) / mean(exp(predict(mod, abi)))
abi <- mod

set.seed(42)
xcv <- xgb.cv(hyp, apd, 1e6, 5, early_stopping_rounds = 10)
mod <- xgb.train(hyp, apd, xcv$best_iteration)
pac <- mean(exp(getinfo(apd, "label"))) / mean(exp(predict(mod, apd)))
apd <- mod

mtx <- xgb.DMatrix(as.matrix(scr[, ..vars]))
scr$IBA <- exp(predict(abi, mtx))
scr$DPA <- exp(predict(apd, mtx))
scr$PBI <- scr$IBC * scr$IBA * scr$EXP * bac * adj
scr$PPD <- scr$DPC * scr$DPA * scr$EXP * pac * adj

rm(hyp, mod, xcv, facs, head, keep, mtx)
