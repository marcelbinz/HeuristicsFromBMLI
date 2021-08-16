library(lme4)
require(foreign)
require(MASS)

df <- read.csv("data/exp1.csv")
db <- read.csv("data/betas1.csv")
colnames(db)[colnames(db) == 'X'] <- 'participant'
colnames(db)[colnames(db) == 'X0'] <- 'beta'

total_time = numeric(max(df$participant) + 1)
for (i in 0:29) {
  total_time <- total_time + (df[df$task == i & df$step == 9, ]$time - df[df$task == i & df$step == 0, ]$time)
}
db$time <- total_time

cor.test(db$time, db$beta, method='kendall')
