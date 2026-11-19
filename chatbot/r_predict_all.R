#!/usr/bin/env Rscript
# All 56 final caret models belonging to one PLM.
options(stringsAsFactors = FALSE, warn = 1)
args <- commandArgs(trailingOnly = TRUE)
arg <- function(flag) { n <- match(flag, args); if (is.na(n) || n >= length(args)) stop('Missing ', flag); args[[n+1L]] }
root <- arg('--model-root')
inp <- arg('--embedding-dir')
out <- arg('--out')
plm_key <- arg('--plm-key')
embedding_dim <- as.integer(arg('--embedding-dim'))
if (!requireNamespace('caret', quietly=TRUE)) stop('R package caret is not installed.')
if (!embedding_dim %in% c(1024L,1152L,1280L)) stop('Invalid embedding dimension')
cases <- c('case1_all_proteins','case2_target_weight16','case3_target_weight32','case4_target_only')
models <- c('Logistic_regression_GLMNET','C5_0_tree','Random_Forest_Ranger','SVM_linear','SVM_radial',
            'Gradient_Boosting_GBM','Random_Forest','LDA_PCA','PLS','Neural_network_nnet',
            'MARS_earth','Naive_Bayes','kNN','Decision_tree_RPART')
rows <- vector('list',56)
k <- 0L
for (case in cases) {
  file <- file.path(inp,paste0(case,'.csv'))
  if (!file.exists(file)) stop('Embedding CSV not found: ',file)
  dat <- read.csv(file,check.names=FALSE)
  if (!'assembly' %in% names(dat) || nrow(dat)!=1L) stop('Expected exactly one strain and an assembly column in ',file)
  for (name in models) {
    pth <- file.path(root,case,paste0(name,'.rds'))
    if (!file.exists(pth)) stop('Model file missing: ',pth)
    obj <- readRDS(pth)
    if (!identical(obj$PLM_Key,plm_key) || !identical(obj$Case,case) || !identical(obj$model_name,name)) {
      stop('Model metadata mismatch: ',pth)
    }
    cols <- obj$feature_columns
    if (length(cols)!=embedding_dim || length(unique(cols))!=embedding_dim ||
        !setequal(cols,sprintf('emb_%04d',seq_len(embedding_dim)))) {
      stop('Unexpected model feature names: ',pth)
    }
    if (!all(cols %in% names(dat))) stop('Missing embedding features for ',name)
    x <- dat[,cols,drop=FALSE]
    for (feature in cols) x[[feature]] <- suppressWarnings(as.numeric(x[[feature]]))
    if (!all(is.finite(as.matrix(x)))) stop('Non-finite features for ',case)
    prob <- predict(obj$fit,newdata=x,type='prob')
    if (!'positive' %in% colnames(prob)) stop('Model did not return positive probability: ',name)
    pr <- as.numeric(prob[['positive']])
    if (length(pr)!=1L || !is.finite(pr) || pr<0 || pr>1) stop('Invalid prediction for ',name)
    threshold <- as.numeric(obj$probability_threshold)
    if (length(threshold)!=1L || !is.finite(threshold)) stop('Invalid model threshold: ',name)
    k <- k+1L
    rows[[k]] <- data.frame(assembly=dat$assembly[[1]], PLM=plm_key, Case=case, Model=name,
          probability_positive=pr, predicted=ifelse(pr>=threshold,'positive','negative'),
          probability_threshold=threshold, stringsAsFactors=FALSE)
  }
}
if (k != 56L) stop('Expected 56 predictions, got ',k)
res <- do.call(rbind,rows)
dir.create(dirname(out),recursive=TRUE,showWarnings=FALSE)
write.csv(res,out,row.names=FALSE)
cat(sprintf('Completed %d predictions for %s\n',nrow(res),plm_key))
