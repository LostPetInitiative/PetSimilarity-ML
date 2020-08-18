require(ggplot2)

args <- commandArgs(trailingOnly = TRUE)
if(length(args)>0){
  imagesTablePath <- args[1]
  outputDir <- args[2]
} else {
  imagesTablePath <- 'data/imagesSummary.csv'
  outputDir <- 'data/imageSummaryAnalysis/'
}

if (!(dir.exists(outputDir))) {
  dir.create(outputDir)
}

df1 <- read.csv(imagesTablePath, colClasses = c('character','factor','character','factor','logical','logical'))

df2 <- df1[(df1$dublicate == F) & (df1$currupted == F),]

uniqueDogsImgDf <- df2[df2$pet=="dog", c("petId","imageFile","type")]
uniqueDogsImgOutFile <- file.path(outputDir,"uniqueDogsImages.csv")
write.csv(uniqueDogsImgDf, file = uniqueDogsImgOutFile, row.names = F)

uniqueCatsImgDf <- df2[df2$pet=="cat", c("petId","imageFile","type")]
uniqueCatsImgOutFile <- file.path(outputDir,"uniqueCatsImages.csv")
write.csv(uniqueCatsImgDf, file = uniqueCatsImgOutFile, row.names = F)


plot1 <-
  ggplot(df2)+
  geom_bar(aes(x=type,fill=pet),position = "dodge") +
  ggtitle("How many images of pets were posted on pet911.ru till August 2020") +
  xlab("Card type") +
  ylab("Images posted")
stats1File <- file.path(outputDir,"lost_found_stats.png")
ggsave(stats1File, plot = plot1)

df2$imagesCount <- 1

df3 <- aggregate(imagesCount ~ petId + pet + type, data=df2, sum)
df3$imagesCount <- as.factor(df3$imagesCount)

plot2 <- ggplot(df3) +
  geom_bar(aes(x=imagesCount,fill=pet),position = "dodge") +
  scale_y_log10() +
  #scale_x_continuous(breaks = seq(0,20,5),minor_breaks = seq(1,20),limits=c(0,20)) +
  #xlim(c(1,20)) +
  ylab("Number of cards") +
  xlab("Number of images per pet card") +
  ggtitle("How many photos are posted within a card")
stats2File <- file.path(outputDir,"lost_found_stats2.png")
ggsave(stats2File, plot = plot2)

plot2
