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
levelsInOrder <- c("square", "rectangle", "parallelogram", "losange", "isoTrapezoid", "kite", "rightKite", "rustedHinge", "hinge", "trapezoid", "random")
imgsInOrder <-
  lapply(levelsInOrder,
         function(x) {
          width <- img_size / 40
           paste0("<img src='/utils/imgs_labels/",x,".png' width= '", width, "' />")
         })


clip = '/1_quadrilaterals_RSA/RDMs/vit_giant_patch14_clip_224.laion2b/layer_28'
dino = '/1_quadrilaterals/RDMs/dinov3_base/layer_13'


dissimilarity_matrix <- as.matrix(read.csv(paste0('', dino), row.names = 1))
# Extract upper triangle of the dissimilarity matrix
upper_triangle <- upper.tri(dissimilarity_matrix)

# Set lower triangle values to NA to only show upper triangle
dissimilarity_matrix[is.na(dissimilarity_matrix)] <- 0
dissimilarity_matrix[!upper_triangle] <- NA

melted_matrix <- melt(dissimilarity_matrix)
plot<-
  ggplot(data = melted_matrix, aes(x=Var1, y=Var2,fill=value)) +
  geom_tile() +
  coord_fixed(expand=F) +
  scale_fill_gradientn(na.value = 'transparent', name="Dissimilarity\nscore", colors=met.brewer("Hiroshige", type="continuous")) +
  theme(panel.grid.major.y = element_blank()) +
  ylab("") + xlab("") +
  scale_x_discrete(labels=imgsInOrder) +
  scale_y_discrete(labels=imgsInOrder) +

  theme(axis.text.x = element_markdown(color = "black", size = 11)) +
  theme(axis.text.y = element_markdown(color = "black", size = 11)) +

  theme(legend.position="none") +
  ggtitle("")

ggsave("plots/RDM_dinov3_base.svg", plot, device='svg',width = 8, height = 8, units = "in")




dissimilarity_matrix <- as.matrix(read.csv(paste0('', dino), row.names = 1))
# Extract upper triangle of the dissimilarity matrix
upper_triangle <- upper.tri(dissimilarity_matrix)

# Set lower triangle values to NA to only show upper triangle
dissimilarity_matrix[is.na(dissimilarity_matrix)] <- 0
dissimilarity_matrix[!upper_triangle] <- NA

melted_matrix <- melt(dissimilarity_matrix)
plot<-
  ggplot(data = melted_matrix, aes(x=Var1, y=Var2,fill=value)) +
  geom_tile() +
  coord_fixed(expand=F) +
  scale_fill_gradientn(na.value = 'transparent', name="Dissimilarity\nscore", colors=met.brewer("Hiroshige", type="continuous")) +
  theme(panel.grid.major.y = element_blank()) +
  ylab("") + xlab("") +
  scale_x_discrete(labels=imgsInOrder) +
  scale_y_discrete(labels=imgsInOrder) +

  theme(axis.text.x = element_markdown(color = "black", size = 11)) +
  theme(axis.text.y = element_markdown(color = "black", size = 11)) +

  theme(legend.position="none") +
  ggtitle("")
ggsave("plots/RDM_clip.svg", plot, device='svg',width = 8, height = 8, units = "in")


dissimilarity_matrix <- as.matrix(read.csv('RDMs/behavioral', row.names = 1))
  # Extract upper triangle of the dissimilarity matrix
  upper_triangle <- upper.tri(dissimilarity_matrix)

  # Set lower triangle values to NA to only show upper triangle
  dissimilarity_matrix[is.na(dissimilarity_matrix)] <- 0
  dissimilarity_matrix[!upper_triangle] <- NA

  melted_matrix <- melt(dissimilarity_matrix)
  plot<-
    ggplot(data = melted_matrix, aes(x=Var1, y=Var2,fill=value)) +
    geom_tile() +
    coord_fixed(expand=F) +
    scale_fill_gradientn(na.value = 'transparent', name="Dissimilarity\nscore", colors=met.brewer("Hiroshige", type="continuous")) +
    theme(panel.grid.major.y = element_blank()) +
    ylab("") + xlab("") +
    scale_x_discrete(labels=imgsInOrder) +
    scale_y_discrete(labels=imgsInOrder) +
    theme(legend.position="none") +

    theme(axis.text.x = element_markdown(color = "black", size = 11)) +
    theme(axis.text.y = element_markdown(color = "black", size = 11)) +

    ggtitle("")
  ggsave("plots/RDM_behavioral.svg", plot, device='svg',width = 8, height = 8, units = "in")
