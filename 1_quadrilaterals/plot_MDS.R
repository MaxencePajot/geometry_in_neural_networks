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


clip = '/1_quadrilaterals/RDMs/vit_giant_patch14_clip_224.laion2b/layer_28'
dino = '/1_quadrilaterals/RDMs/dinov3_base/layer_13'

img_size <- 1600
levelsInOrder <- c("square", "rectangle", "parallelogram", "losange", "isoTrapezoid", "kite", "rightKite", "rustedHinge", "hinge", "trapezoid", "random")
imgsInOrder <-
  lapply(levelsInOrder,
         function(x) {
          width <- img_size / 12
           paste0("<img src='/utils/imgs_labels/",x,".png' width= '", width, "' />")
         })


dissimilarity_matrix <- as.matrix(read.csv(paste0('', dino), row.names = 1))
# Perform MDS on the RDM
mds_result <- mds(dissimilarity_matrix, type = "ordinal")  # You can change 'type' if needed

# Extract the MDS configuration (coordinates)
conf <- mds_result$conf
conf[, 1] <- conf[, 1] * (-1)  # Reflect the y-axis
conf[, 2] <- conf[, 2]   # Reflect the y-axis

# Create a data frame for the MDS coordinates and corresponding images
data_plot <- data.frame(
x = conf[, 1],  # First dimension of MDS configuration
y = conf[, 2],  # Second dimension of MDS configuration
shape = factor(levelsInOrder, levels = levelsInOrder)  # Assuming 'levelsInOrder' is your object labels
)

# Plot using ggplot with images
plot<-
    ggplot(data_plot, aes(x = x, y = y, image = paste0("/utils/imgs_labels/", shape, "_2.png"))) +
        geom_hline(yintercept = 0, linewidth = 3) +  # Add horizontal line at y = 0
        geom_vline(xintercept = 0, linewidth = 3) +  # Add vertical line at x = 0
        geom_image(size = 0.15, by = "height", asp = 3.25) +  # Adjust size and aspect ratio
    scale_x_continuous(limits = c(-1.5,1.5), breaks = seq(-2, 2, 0.5)) +  # Control x-axis limits and breaks
    scale_y_continuous(limits = c(-1, 1), breaks = seq(-2, 2, 0.5)) +  # Control y-axis limits and breaks
    coord_fixed(ratio = 2) +  
    theme_minimal() +  # Minimal theme for the plot
    theme(
        legend.position = "none",  # No legend
        plot.background = element_blank(),  # Remove background
        panel.grid.major = element_blank(),  # Remove major gridlines
        panel.grid.minor = element_blank(),  # Remove minor gridlines
        panel.border = element_blank(),  # Remove panel border
        axis.title.x = element_blank(),  # No axis labels
        axis.title.y = element_blank()
    )
ggsave("plots/MDS_dinov3_base.svg", plot, device='svg',width = 8, height = 8, units = "in")



dissimilarity_matrix <- as.matrix(read.csv(paste0('', clip), row.names = 1))
# Perform MDS on the RDM
mds_result <- mds(dissimilarity_matrix, type = "ordinal")  # You can change 'type' if needed

# Extract the MDS configuration (coordinates)
conf <- mds_result$conf

# Create a data frame for the MDS coordinates and corresponding images
data_plot <- data.frame(
x = conf[, 1],  # First dimension of MDS configuration
y = conf[, 2],  # Second dimension of MDS configuration
shape = factor(levelsInOrder, levels = levelsInOrder)  # Assuming 'levelsInOrder' is your object labels
)

# Plot using ggplot with images
plot<-
    ggplot(data_plot, aes(x = x, y = y, image = paste0("/utils/imgs_labels/", shape, "_2.png"))) +
        geom_hline(yintercept = 0, linewidth = 3) +  # Add horizontal line at y = 0
        geom_vline(xintercept = 0, linewidth = 3) +  # Add vertical line at x = 0
        geom_image(size = 0.15, by = "height", asp = 3.25) +  # Adjust size and aspect ratio
    scale_x_continuous(limits = c(-1.5,1.5), breaks = seq(-2, 2, 0.5)) +  # Control x-axis limits and breaks
    scale_y_continuous(limits = c(-1, 1), breaks = seq(-2, 2, 0.5)) +  # Control y-axis limits and breaks
    coord_fixed(ratio = 1) +  # Ensures 1:1 aspect ratio between x and y axes
    theme_minimal() +  # Minimal theme for the plot
    theme(
        legend.position = "none",  # No legend
        plot.background = element_blank(),  # Remove background
        panel.grid.major = element_blank(),  # Remove major gridlines
        panel.grid.minor = element_blank(),  # Remove minor gridlines
        panel.border = element_blank(),  # Remove panel border
        axis.title.x = element_blank(),  # No axis labels
        axis.title.y = element_blank()
    )
# png(file=paste0('plots/MDS/',model,'.png'), width=img_size, height=img_size)
ggsave("plots/MDS_clip_laion.svg", plot, device='svg',width = 8, height =8, units = "in")


dissimilarity_matrix <- as.matrix(read.csv('RDMs/behavioral', row.names = 1))
# Perform MDS on the RDM
mds_result <- mds(dissimilarity_matrix, type = "ordinal")  # You can change 'type' if needed

# Extract the MDS configuration (coordinates)
conf <- mds_result$conf
conf[, 1] <- conf[, 1] * (-1)
# Create a data frame for the MDS coordinates and corresponding images
data_plot <- data.frame(
x = conf[, 1],  # First dimension of MDS configuration
y = conf[, 2],  # Second dimension of MDS configuration
shape = factor(levelsInOrder, levels = levelsInOrder)  # Assuming 'levelsInOrder' is your object labels
)

# Plot using ggplot with images
plot<-
    ggplot(data_plot, aes(x = x, y = y, image = paste0("/utils/imgs_labels/", shape, "_2.png"))) +
    geom_hline(yintercept = 0, linewidth = 3) +  # Add horizontal line at y = 0
    geom_vline(xintercept = 0, linewidth = 3) +  # Add vertical line at x = 0
    geom_image(size = 0.15, by = "height", asp = 3.25) +  # Adjust size and aspect ratio

    scale_x_continuous(limits = c(-1.5,1.5), breaks = seq(-2, 2, 0.5)) +  # Control x-axis limits and breaks
    scale_y_continuous(limits = c(-1, 1), breaks = seq(-2, 2, 0.5)) +  # Control y-axis limits and breaks
    coord_fixed(ratio = 2) +  # Ensures 1:1 aspect ratio between x and y axes
    theme_minimal() +  # Minimal theme for the plot
    theme(
        legend.position = "none",  # No legend
        plot.background = element_blank(),  # Remove background
        panel.grid.major = element_blank(),  # Remove major gridlines
        panel.grid.minor = element_blank(),  # Remove minor gridlines
        panel.border = element_blank(),  # Remove panel border
        axis.title.x = element_blank(),  # No axis labels
        axis.title.y = element_blank()
    )
ggsave("plots/MDS_behavioral.svg", plot, device='svg',width = 8, height = 8, units = "in")

