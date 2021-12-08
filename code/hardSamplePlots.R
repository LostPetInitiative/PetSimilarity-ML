require(ggplot2)

meanAbsDiff <- function(df1) {
  df1$meanAbsDiff <- (df1$isup_abs_diff.fold1 + df1$isup_abs_diff.fold3)*0.5
  return(df1)
}

df_32c <- meanAbsDiff(df_32c)
df_32c$trRunNum <- 1
df_35c <- meanAbsDiff(df_35c)
df_35c$trRunNum <- 2
df_37c <- meanAbsDiff(df_37c)
df_37c$trRunNum <- 3

gathred <- rbind(df_32c, df_35c, df_37c)
gathred$trRunNum <- factor(gathred$trRunNum)

plot1 <- function(df1,roundNum) {
  samplesCount <- nrow(df1)
  
  ggplot(df1) +
    theme_bw() +
    labs(
      x="Absolute difference between predicted and ground truth ISUP grade",
      y="Sample count",
      title = paste0("Iteration ",roundNum),
      caption = paste0(samplesCount," samples are discarded in total"),
      subtitle="Histogram of high (>=2.5) absolute ISUP grade prediction error") +
    #geom_histogram(aes(x = meanAbsDiff, fill=trRunNum),
    #               bins = 15) +
    geom_histogram(aes(x = meanAbsDiff), bins = 6*5) +
    guides(fill=guide_legend(title="Training round")) +
    xlim(c(2.5,5.0)) +
    ylim(c(0,80.0))
}

plot1(df_32c,1)