library(caret)
library(randomForest)
library(e1071)
library(glmnet)
library(MASS)
# library(deepnet)
# library(h2o)
# library(neuralnet)
library(nnet)

breast.cancer = read.csv("~/Documents/MacineLearning/FinalProj/breast_cancer_compelete.csv")
forest = read.csv("~/Documents/MacineLearning/FinalProj/forest_type.csv")
hand.writing = read.csv("~/Documents/MacineLearning/FinalProj/hand_writing_digits.csv")

#### Function to divide dataset ####
div.dat = function(dat, p) {
  class.dat = table(dat$class)
  ind.training = numeric(0)
  for(i in 1:length(class.dat)) {
    class.ind = which(dat$class==names(class.dat)[i])
    ind.training = c(ind.training, sample(class.ind, ceiling(p*length(class.ind))))
  }
  training = dat[ind.training, ]
  test = dat[-ind.training, ]
  list(training=training, test=test)
}

#### Optimize alpha/lambda in elastic net ####
elastic.opt = function(X, y) {
  alpha.list = 0:10/10
  min.dev = Inf
  for(alpha in alpha.list) {
    glmnet.mod = cv.glmnet(X, y, family="multinomial", alpha=alpha)
    if(min.dev > min(glmnet.mod$cvm)) {
      min.dev = min(glmnet.mod$cvm)
      best.mod = glmnet.mod
    }
  }
  best.mod
}

p.training = (5:1)/10

rf.mis.class = matrix(NA, 3, length(p.training))
lda.mis.class = matrix(NA, 3, length(p.training))
nn.mis.class = matrix(NA, 3, length(p.training))
el.mis.class = matrix(NA, 3, length(p.training))

set.seed(4.0)

for(i in 1:length(p.training)) {
  
  print(paste("i=", i, sep=""))
  
  div.bc = div.dat(breast.cancer[, -1], p.training[i])
  bc.training = div.bc$training
  bc.test = div.bc$test
  
  div.fo = div.dat(forest, p.training[i])
  fo.training = div.fo$training
  fo.test = div.fo$test
  
  div.hw = div.dat(hand.writing, p.training[i])
  hw.training = div.hw$training
  hw.test = div.hw$test
  
  ######## randomForest ########
  bc.rf = randomForest(factor(class)~., bc.training, ntree=10000)
  bc.rf.pred = predict(bc.rf, bc.test)
  print(table(bc.rf.pred, bc.test$class))
  
  fo.rf = randomForest(factor(class)~., fo.training, ntree=10000)
  fo.rf.pred = predict(fo.rf, fo.test)
  print(table(fo.rf.pred, fo.test$class))
  
  hw.rf = randomForest(factor(class)~., hw.training, ntree=10000)
  hw.rf.pred = predict(hw.rf, hw.test)
  print(table(hw.rf.pred, hw.test$class))
  
  rf.mis.class[, i] = c(mean(bc.rf.pred!=bc.test$class), 
                        mean(fo.rf.pred!=fo.test$class), 
                        mean(hw.rf.pred!=hw.test$class))
  
  print("randomForest done")
  
  ######## LDA ########
  try(bc.lda <- lda(factor(class)~., bc.training))
  try(bc.lda.pred <- predict(bc.lda, bc.test))
  try(print(table(bc.lda.pred$class, bc.test$class)))
  
  try(fo.lda <- lda(factor(class)~., fo.training))
  try(fo.lda.pred <- predict(fo.lda, fo.test))
  try(print(table(fo.lda.pred$class, fo.test$class)))
  
  try(hw.lda <- lda(factor(class)~., hw.training[, -c(1, 17, 25, 29, 32, 40, 57)]))
  try(hw.lda.pred <- predict(hw.lda, hw.test))
  try(print(table(hw.lda.pred$class, hw.test$class)))
  
  try(lda.mis.class[, i] <- c(mean(bc.lda.pred$class!=bc.test$class), 
                              mean(fo.lda.pred$class!=fo.test$class), 
                              mean(hw.lda.pred$class!=hw.test$class)))
  
  print("LDA done")
  
  ######## QDA ########
  # bc.qda = qda(factor(class)~., bc.training)
  # bc.qda.pred = predict(bc.qda, bc.test)
  # table(bc.qda.pred$class, bc.test$class)
  # 
  # fo.qda = qda(factor(class)~., fo.training)
  # fo.qda.pred = predict(fo.qda, fo.test)
  # table(fo.qda.pred$class, fo.test$class)
  # 
  # hw.qda = qda(factor(class)~., hw.training[, -c(1, 40, 57)])
  # hw.qda.pred = predict(hw.qda, hw.test)
  # table(hw.qda.pred$class, hw.test$class)
  
  ######## Neural Network ########
  
  message = capture.output(bc.nn <- train(factor(class) ~ ., bc.training, method='nnet', 
                                          tuneGrid=expand.grid(.size=1:10*2-1,.decay=c(0,0.1,0.001)))) 
  bc.nn.pred = predict(bc.nn, bc.test)
  print(table(bc.nn.pred, bc.test$class))
  
  message = capture.output(fo.nn <- train(factor(class) ~ ., fo.training, method='nnet', 
                                          tuneGrid=expand.grid(.size=1:10*2-1,.decay=c(0,0.1,0.001)))) 
  fo.nn.pred = predict(fo.nn, fo.test)
  print(table(fo.nn.pred, fo.test$class))
  
  message = capture.output(hw.nn <- train(factor(class) ~ ., hw.training, method='nnet', 
                                          tuneGrid=expand.grid(.size=1:10,.decay=c(0,0.1,0.001)))) 
  hw.nn.pred = predict(hw.nn, hw.test)
  print(table(hw.nn.pred, hw.test$class))
  
  nn.mis.class[, i] = c(mean(bc.nn.pred!=bc.test$class), 
                        mean(fo.nn.pred!=fo.test$class), 
                        mean(hw.nn.pred!=hw.test$class))
  
  print("Neural network done")
  
  ######## Elastic net ########
  bc.el = elastic.opt(as.matrix(bc.training[, -10]), factor(bc.training$class))
  bc.el.pred = predict(bc.el, as.matrix(bc.test[, -10]), s="lambda.min")
  print(table(apply(bc.el.pred, 1, which.max), bc.test$class))
  
  fo.el = elastic.opt(as.matrix(fo.training[, -28]), factor(fo.training$class))
  fo.el.pred = predict(fo.el, as.matrix(fo.test[, -28]), s="lambda.min")
  print(table(apply(fo.el.pred, 1, which.max), fo.test$class))
  
  hw.el = elastic.opt(as.matrix(hw.training[, -65]), factor(hw.training$class))
  hw.el.pred = predict(hw.el, as.matrix(hw.test[, -65]), s="lambda.min")
  print(table(apply(hw.el.pred, 1, which.max), hw.test$class))
  
  el.mis.class[, i] = c(mean(apply(bc.el.pred, 1, which.max)!=as.numeric(bc.test$class)), 
                        mean(apply(fo.el.pred, 1, which.max)!=as.numeric(fo.test$class)), 
                        mean(apply(hw.el.pred, 1, which.max)-1!=hw.test$class))
  
}

bc.mis.class = rbind(rf.mis.class[1, ], 
                     lda.mis.class[1, ], 
                     el.mis.class[1, ], 
                     nn.mis.class[1, ])
fo.mis.class = rbind(rf.mis.class[2, ], 
                     lda.mis.class[2, ], 
                     el.mis.class[2, ], 
                     nn.mis.class[2, ])
hw.mis.class = rbind(rf.mis.class[3, ], 
                     lda.mis.class[3, ], 
                     el.mis.class[3, ], 
                     nn.mis.class[3, ])
rownames(bc.mis.class) = c("Random Forest", "LDA", "Elastic Net", "Neural Network")
colnames(bc.mis.class) = c("10% Train", "20% Train", "30% Train", "40% Train", "50% Train")
rownames(fo.mis.class) = c("Random Forest", "LDA", "Elastic Net", "Neural Network")
colnames(fo.mis.class) = c("10% Train", "20% Train", "30% Train", "40% Train", "50% Train")
rownames(hw.mis.class) = c("Random Forest", "LDA", "Elastic Net", "Neural Network")
colnames(hw.mis.class) = c("10% Train", "20% Train", "30% Train", "40% Train", "50% Train")

write.csv(bc.mis.class, "~/Documents/MacineLearning/FinalProj/breast_cancer_mis.csv")
write.csv(fo.mis.class, "~/Documents/MacineLearning/FinalProj/forest_mis.csv")
write.csv(hw.mis.class, "~/Documents/MacineLearning/FinalProj/handwriting_mis.csv")





