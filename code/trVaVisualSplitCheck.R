inputFile <- 'data/rawCardsSummary.csv'
testProb <- 0.2

set.seed(3456)

inputDf <- read.csv(inputFile, colClasses = list(
  cardType='factor',
  species='factor',
  sex='factor'
  ))

inputDf[inputDf$photoCount > 4,]$photoCount <- ">= 5"
inputDf$photoCount <- as.factor(inputDf$photoCount)

testIndicator <- as.logical(rbinom(nrow(inputDf),1, testProb))

inputDf$dataset <- 'train'
inputDf[testIndicator,]$dataset <- 'test'
inputDf$dataset <- as.factor(inputDf$dataset)

require(ggplot2)

genPlot <- function() {
  require("ggpubr")
  # cardType species sex photoCount
  p1 <- ggplot(inputDf) + geom_bar(aes(x=dataset, fill=photoCount))
  p2 <- ggplot(inputDf) + geom_bar(aes(x=dataset, fill=photoCount), position="fill") +
    labs(y="Portion")
  
  ggarrange(p1, p2,
            #labels = c("Counts", "Portion"),
            nrow = 2)
}
genPlot()