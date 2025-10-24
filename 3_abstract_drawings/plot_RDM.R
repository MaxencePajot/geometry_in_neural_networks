if (!require("pacman")) install.packages("pacman")
pacman::p_load(ggplot2, grid, cowplot,     # Plot manipulation & theme
               jsonlite,                   # Data parsing
               tidyverse, broom, magrittr, # Pipelineing, %>%, tidying, etc.
               stringi,                    # UUID generation
               afex,                       # Sane defaults for `aov` through `aov_ez`
               ggrain,
               ggsignif,
               rsvg,
               ggimage,
               patchwork,
               MetBrewer,
               smacof,
               png,
               broom.mixed,
               ggtext,
               pbapply,
               parallel,
               optimx,
               car,
               ggpubr,
               corrplot,
               reshape2,
               install = T)

theme_set(theme_cowplot() +
          theme(text = element_text(family = "Times", size=9),
                axis.text = element_text(size=9),
                panel.grid.major.x = element_blank() ,
                panel.grid.major.y = element_line( linewidth=.1, color="black")))

img_size <- 1600


dissimilarity_matrix <- as.matrix(read.csv(paste0('3_abstract_drawings/RDMs/theoretical'), row.names = 1))
selected_matrix <- dissimilarity_matrix[1:72, 1:72]
# Extract upper triangle of the dissimilarity matrix
upper_triangle <- upper.tri(selected_matrix)
# print(upper_triangle)
# Set lower triangle values to NA to only show upper triangle
selected_matrix[is.na(selected_matrix)] <- 0
selected_matrix[!upper_triangle] <- NA

melted_matrix <- melt(selected_matrix)
plot<-
  ggplot(data = melted_matrix, aes(x=Var1, y=Var2,fill=value)) +
  geom_tile() +
  coord_fixed(expand=F) +
  scale_fill_gradientn(na.value = 'transparent', name="Dissimilarity\nscore", colors=met.brewer("Hiroshige", type="continuous")) +
  theme(panel.grid.major.y = element_blank()) +
  ylab("") + xlab("") +

  theme(
    axis.text.x = element_blank(),  # Remove x-axis labels
    axis.text.y = element_blank(),  # Remove y-axis labels
    axis.ticks = element_blank()    # Optionally remove axis ticks
  ) +
  theme(axis.line = element_blank())+

  theme(legend.position="none") +
  ggtitle("")
png(file=paste0('3_abstract_drawings/plots/RDMs/theoretical.png'), width=img_size, height=img_size)
print(plot)
dev.off()




clip <- '3_abstract_drawings/RDMs/vit_giant_patch14_clip_224.laion2b/layer_36'
dino <- '3_abstract_drawings/RDMs/dinov3_base/layer_13'
models <- c(clip,dino)
model_names <- c('clip', 'dino')

for (i in c(1, 2)) {
  model <- models[i]
  dissimilarity_matrix <- as.matrix(read.csv(paste0(model), row.names = 1))
  selected_matrix <- dissimilarity_matrix[1:72, 1:72]
  # Extract upper triangle of the dissimilarity matrix
  upper_triangle <- upper.tri(selected_matrix)
  # print(upper_triangle)
  # Set lower triangle values to NA to only show upper triangle
  selected_matrix[is.na(selected_matrix)] <- 0
  selected_matrix[!upper_triangle] <- NA

  melted_matrix <- melt(selected_matrix)
  plot<-
    ggplot(data = melted_matrix, aes(x=Var1, y=Var2,fill=value)) +
    geom_tile() +
    coord_fixed(expand=F) +
    scale_fill_gradientn(na.value = 'transparent', name="Dissimilarity\nscore", colors=met.brewer("Hiroshige", type="continuous")) +
    theme(panel.grid.major.y = element_blank()) +
    ylab("") + xlab("") +
 
    theme(
      axis.text.x = element_blank(),  # Remove x-axis labels
      axis.text.y = element_blank(),  # Remove y-axis labels
      axis.ticks = element_blank()    # Optionally remove axis ticks
    ) +
    theme(axis.line = element_blank())+

    theme(legend.position="none") +
    ggtitle("")

  png(file=paste0('3_abstract_drawings/plots/RDMs/',model_names[i],'.png'), width=img_size, height=img_size)
  print(plot)
  dev.off()
}







