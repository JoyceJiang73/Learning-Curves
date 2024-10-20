library(ggplot2)
library(dplyr)
library(gridExtra)
library(viridis)
library(RColorBrewer)
library(lme4)
library(writexl)
library(broom)

#
# Run fuse_data.R first
#

#
# Ribbon plots used in paper
#

nepochs=100
doRibbon = function(data, ylim=c(0,1), lab='', palette='') {
  data$cl <- factor(data$cl, levels = c(setdiff(unique(data$cl), 'null'), 'null'))
  data_summary <- data %>%
    group_by(epoch, cl) %>%
    summarize(mean_dv = mean(dv),
              se_dv = sd(dv)/sqrt(n()))
  
  print(data_summary[data_summary$epoch==1|data_summary$epoch==nepochs,],n=30)
  
  p <- ggplot(data_summary, aes(x = epoch, y = mean_dv, group = cl, fill = cl, color = cl)) +
    geom_ribbon(aes(ymin = mean_dv - se_dv, ymax = mean_dv + se_dv), alpha = 0.2) +
    geom_line() + coord_cartesian(ylim=ylim) +
    labs(x = "Epoch", y = "", title = lab, fill = 'Classification', color = 'Classification') +
    theme_minimal() +
    coord_cartesian(ylim=ylim)
  
  if (palette == 'viridis') {
    viridis_colors <- viridis::viridis(length(unique(data$cl)) - 1, option = "D", direction = 1)
    full_palette <- c(viridis_colors, "black")  # Append "grey" for the 'null' category
    p <- p + scale_fill_manual(values = setNames(full_palette, levels(data$cl))) +
      scale_color_manual(values = setNames(full_palette, levels(data$cl)))
    
    #p <- p + scale_fill_viridis(discrete = TRUE) + scale_color_viridis(discrete = TRUE)
  } else if (palette == 'brewer') {
    #brewer_colors <- brewer.pal(length(unique(data$cl)) - 1, "Set3")
    base_palette <- brewer.pal(9, "Set1")
    brewer_colors <- colorRampPalette(base_palette)(length(unique(data$cl)) - 1)
    full_palette <- c(brewer_colors, "black")
    p <- p + scale_fill_manual(values = full_palette) + scale_color_manual(values = full_palette)
  
    #p <- p + scale_fill_brewer(palette = "Set1") + scale_color_brewer(palette = "Set1")
  } else if (nchar(palette)>0) {
    p <- p + scale_fill_manual(values = full_palette) + scale_color_manual(values = full_palette)
  } 

  return(p)
}

#
# Generate results and plots
#

process_data <- function(nlp_dv, gesture_dv, nlp_null_dv, gesture_null_dv,dv_name, ylim_nlp = c(0, 1), ylim_gesture = c(0, 1)) {
  
  # Set up the dependent variable
  nlp$dv <- nlp_dv
  gesture$dv <- gesture_dv
  nlp_null$dv <- nlp_null_dv
  gesture_null$dv <- gesture_null_dv
  
  nlp_combine = rbind(nlp, nlp_null)
  gesture_combine = rbind(gesture, gesture_null)
  
  # Generate the plots using custom doRibbon function
  p_n <- doRibbon(nlp_combine, ylim = ylim_nlp, paste0(dv_name, " (sentence)"), palette = 'brewer')
  p_g <- doRibbon(gesture_combine, ylim = ylim_gesture, paste0(dv_name, " (gesture)"), palette = 'viridis')
  
  # Combine plots into one
  combined_plot <- grid.arrange(p_n, p_g, ncol = 2)
  
  # Save the plot
  ggsave(filename = paste0("plots/", dv_name, ".png"), plot = combined_plot, width = 12, height = 5, dpi = 500)
  # Fit the models
  lm_nlp_cl <- lm(dv ~ as.factor(cl), data = nlp)
  lm_gesture_cl <- lm(dv ~ as.factor(cl), data = gesture)
  lm_combined_training <- lm(dv ~ as.factor(training), data = rbind(nlp, gesture))
  lm_nlp_null <- lm(dv ~ as.factor(training), data = nlp_combine)
  lm_gesture_null <- lm(dv ~ as.factor(training), data = gesture_combine)
  lm_nlp_epoch_cl <- lm(dv ~ epoch * as.factor(cl), data = nlp)
  lm_gesture_epoch_cl <- lm(dv ~ epoch * as.factor(cl), data = gesture)
  
  # ANOVA for NLP and Gesture datasets
  anova_nlp <- anova(lm(dv ~ epoch, data = nlp), lm(dv ~ epoch * as.factor(cl), data = nlp))
  anova_gesture <- anova(lm(dv ~ epoch, data = gesture), lm(dv ~ epoch * as.factor(cl), data = gesture))
  
  # R-squared differences
  rsq_diff_nlp <- summary(lm(dv ~ epoch * as.factor(cl), data = nlp))$r.squared - summary(lm(dv ~ epoch, data = nlp))$r.squared
  rsq_diff_gesture <- summary(lm(dv ~ epoch * as.factor(cl), data = gesture))$r.squared - summary(lm(dv ~ epoch, data = gesture))$r.squared
  
  # Function to format p-values
  format_p_value <- function(p) {
    if (p < 1e-5) {
      return("<0.00001")
    } else if (p < 1e-4) {
      return("<0.0001")
    } else if (p < 0.001) {
      return("<0.001")
    } else {
      return(as.character(round(p, 4)))
    }
  }
  
  # Extract adjusted R-squared and p-values for the first five models
  model_summaries <- list(
    lm_combined_training = glance(lm_combined_training),
    lm_nlp_null = glance(lm_nlp_null),
    lm_gesture_null = glance(lm_gesture_null),
    lm_nlp_cl = glance(lm_nlp_cl),
    lm_gesture_cl = glance(lm_gesture_cl),
    lm_nlp_epoch_cl = glance(lm_nlp_epoch_cl),
    lm_gesture_epoch_cl = glance(lm_gesture_epoch_cl)
  )
  
  # Create a summary table for adj R-squared and p-values
  summary_table <- bind_rows(
    model_summaries$lm_combined_training %>% mutate(model = "Data Source"),
    model_summaries$lm_nlp_null %>% mutate(model = "NLP Reshuffle"),
    model_summaries$lm_gesture_null %>% mutate(model = "Gesture Reshuffle"),
    model_summaries$lm_nlp_cl %>% mutate(model = "NLP Class"),
    model_summaries$lm_gesture_cl %>% mutate(model = "Gesture Class"),
    model_summaries$lm_nlp_epoch_cl %>% mutate(model = "NLP Epoch*Class"),
    model_summaries$lm_gesture_epoch_cl %>% mutate(model = "Gesture Epoch*Class")
  ) %>%
    select(model, adj.r.squared, p.value) %>%
    mutate(p.value = sapply(p.value, format_p_value))
  
  # Add R-squared differences and ANOVA significance
  anova_table <- data.frame(
    model = c("NLP Epoch*Class - Epoch", "Gesture Epoch*Class - Epoch"),
    adj.r.squared = c(rsq_diff_nlp, rsq_diff_gesture),
    p.value = c(anova_nlp$`Pr(>F)`[2], anova_gesture$`Pr(>F)`[2])
  ) %>%
    mutate(p.value = sapply(p.value, format_p_value))
  
  # Combine the tables
  final_table <- bind_rows(
    summary_table,
    anova_table
  )
  
  # Write the final table to an Excel file
  write_xlsx(final_table, paste0("results/", dv_name, "_summary_results.xlsx"))
  
  return(final_table)
}


load('metricsAll.Rd') # see fuse_data.R

nlp = metricsAll[metricsAll$training=='NLP',]
gesture = metricsAll[metricsAll$training=='gesture',]

metricsAll[1,]
table(metricsAll$training)

load('metricsAll_null.Rd') # see fuse_data.R
nlp_null = metricsAll[metricsAll$training=='NLP_null',]
gesture_null = metricsAll[metricsAll$training=='gesture_null',]
nlp_null$cl[nlp_null$cl == "null/null"] <- "null"
gesture_null$cl[gesture_null$cl == "null/null"] <- "null"


# Results and Plots for All Four Instances (start, max, tmax, end-start)

start <- process_data(nlp_dv = nlp$v0, gesture_dv = gesture$v0, nlp_null_dv = nlp_null$v0, gesture_null_dv = gesture_null$v0, 
                      dv_name = 'Start Performance')
max <- process_data(nlp_dv = nlp$vmax, gesture_dv = gesture$vmax, nlp_null_dv = nlp_null$vmax, gesture_null_dv = gesture_null$vmax, 
                    dv_name = 'Max performance')
tmax <- process_data(nlp_dv = nlp$maxt, gesture_dv = gesture$maxt, nlp_null_dv = nlp_null$maxt, gesture_null_dv = gesture_null$maxt, 
                     dv_name = 'Time at max')
end_start <- process_data(nlp_dv = nlp$vend-nlp$v0, gesture_dv = gesture$vend-gesture$v0, nlp_null_dv = nlp_null$vend-nlp_null$v0, gesture_null_dv = gesture_null$vend-gesture_null$v0, 
                          dv_name = 'End - start', ylim_nlp = c(0, 0.5), ylim_gesture = c(-0.5, 0.5))

print(start)
print(max)
print(tmax)
print(end_start)

# Renaming columns in each tibble to reflect the source
start <- start %>% rename(adj.r.squared_start = adj.r.squared, p.value_start = p.value)
max <- max %>% rename(adj.r.squared_max = adj.r.squared, p.value_max = p.value)
tmax <- tmax %>% rename(adj.r.squared_tmax = adj.r.squared, p.value_tmax = p.value)
end_start <- end_start %>% rename(adj.r.squared_end_start = adj.r.squared, p.value_end_start = p.value)

combined_results <- start %>%
  left_join(max, by = "model") %>%
  left_join(tmax, by = "model") %>%
  left_join(end_start, by = "model")

print(combined_results)
write_xlsx(combined_results, paste0("results/master_result_summary_results.xlsx"))







