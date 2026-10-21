### =========================================================
### GRAPHS SCRIPT — Fire Experiment
### Companion to the analysis script.
### Run the analysis script FIRST so these objects exist:
###   - fixation_features, trial_roi_df
###   - individual_fixations_model, full_trial_model, full_count_model
###   - full_count_species_model, full_trial_model_image
### =========================================================

library(tidyverse)
library(emmeans)
library(lme4)
library(lmerTest)

dir.create("figures", showWarnings = FALSE)

## ---- Helpers ----------------------------------------------

# Standardize emmeans CI column names to lower.CL / upper.CL.
# emmeans sometimes returns asymp.LCL / asymp.UCL (asymptotic z-based CIs),
# especially with large mixed models. This makes plotting consistent.
tidy_emm <- function(emm_obj) {
  df <- as.data.frame(emm_obj)
  if ("asymp.LCL" %in% names(df)) df <- dplyr::rename(df, lower.CL = asymp.LCL)
  if ("asymp.UCL" %in% names(df)) df <- dplyr::rename(df, upper.CL = asymp.UCL)
  df
}

## ---- ROI labeling -----------------------------------------
# roi_type reference level is "1" = Fire (set via relevel in analysis script).
# Map numeric codes to readable group names.
roi_labels <- c("1" = "Fire", "2" = "Vegetation", "3" = "Background")

# Add a readable, ordered ROI label column (Fire first so it anchors the plots).
add_roi_label <- function(df) {
  df %>%
    mutate(ROI_label = factor(roi_labels[as.character(roi_type)],
                              levels = c("Fire", "Vegetation", "Background")))
}

## ---- Theme & color setup ----------------------------------
# Highlight Fire in strong red; mute the others.
roi_fill_colors <- c(
  "Fire"       = "#D7301F",
  "Vegetation" = "#4DAF4A",
  "Background" = "#969696"
)

theme_fire <- function() {
  theme_minimal(base_size = 13) +
    theme(
      panel.grid.minor   = element_blank(),
      panel.grid.major.x = element_blank(),
      legend.position    = "top",
      plot.title         = element_text(face = "bold"),
      axis.title         = element_text(face = "bold")
    )
}

# Reusable emmeans plot: 95% CI errorbars + points, Fire enlarged/highlighted.
plot_emm <- function(emm_df, title, subtitle, ylab, facet = NULL) {
  df <- add_roi_label(emm_df)
  p <- ggplot(df, aes(x = ROI_label, y = emmean, color = ROI_label)) +
    geom_errorbar(aes(ymin = lower.CL, ymax = upper.CL),
                  width = 0.12, linewidth = 1) +
    geom_point(aes(size = ROI_label == "Fire")) +
    scale_color_manual(values = roi_fill_colors, guide = "none") +
    scale_size_manual(values = c(`FALSE` = 3, `TRUE` = 5), guide = "none") +
    labs(title = title, subtitle = subtitle, x = "ROI Type", y = ylab) +
    theme_fire()
  if (!is.null(facet)) p <- p + facet_wrap(facet)
  p
}

# Reusable raw-distribution plot (violin + box).
plot_dist <- function(raw_df, yvar, title, ylab) {
  df <- add_roi_label(raw_df)
  ggplot(df, aes(x = ROI_label, y = .data[[yvar]], fill = ROI_label)) +
    geom_violin(alpha = 0.4, color = NA) +
    geom_boxplot(width = 0.18, outlier.alpha = 0.15, alpha = 0.9) +
    scale_fill_manual(values = roi_fill_colors, guide = "none") +
    labs(title = title, x = "ROI Type", y = ylab) +
    theme_fire()
}

### =========================================================
### 1. INDIVIDUAL FIXATION DURATION (Fire images only)
### =========================================================

ind_fire_df <- fixation_features %>% filter(ImageType == "F")

p_ind_box <- plot_dist(
  ind_fire_df, "FixDur",
  "Individual Fixation Duration by ROI (Fire images)",
  "Fixation duration (ms)"
)
ggsave("figures/01_individual_fixdur_distribution.png",
       p_ind_box, width = 7, height = 5, dpi = 300)

emm_ind_df <- tidy_emm(emmeans(individual_fixations_model, ~ roi_type))
p_ind_emm <- plot_emm(
  emm_ind_df,
  "Estimated Fixation Duration by ROI",
  "Model-adjusted means with 95% CI (controls: area, saliency) — Fire highlighted",
  "Estimated fixation duration (ms)"
)
ggsave("figures/02_individual_fixdur_emmeans.png",
       p_ind_emm, width = 7, height = 5, dpi = 300)

### =========================================================
### 2. TRIAL-LEVEL TOTAL FIXATION DURATION (Fire images only)
### =========================================================

trial_fire_df <- trial_roi_df %>% filter(ImageType == "F")

p_trial_box <- plot_dist(
  trial_fire_df, "total_fixdur",
  "Total Fixation Duration per Trial by ROI (Fire images)",
  "Total fixation duration (ms)"
)
ggsave("figures/03_trial_total_fixdur_distribution.png",
       p_trial_box, width = 7, height = 5, dpi = 300)

emm_trial_df <- tidy_emm(emmeans(full_trial_model, ~ roi_type))
p_trial_emm <- plot_emm(
  emm_trial_df,
  "Estimated Total Fixation Duration by ROI",
  "Model-adjusted means with 95% CI (controls: mean saliency, mean area) — Fire highlighted",
  "Estimated total fixation duration (ms)"
)
ggsave("figures/04_trial_total_fixdur_emmeans.png",
       p_trial_emm, width = 7, height = 5, dpi = 300)

### =========================================================
### 3. FIXATION COUNT (Fire images only)
### =========================================================

p_count_box <- plot_dist(
  trial_fire_df, "fixation_count",
  "Fixation Count per Trial by ROI (Fire images)",
  "Number of fixations"
)
ggsave("figures/05_fixation_count_distribution.png",
       p_count_box, width = 7, height = 5, dpi = 300)

emm_count_df <- tidy_emm(emmeans(full_count_model, ~ roi_type))
p_count_emm <- plot_emm(
  emm_count_df,
  "Estimated Fixation Count by ROI",
  "Model-adjusted means with 95% CI (controls: mean saliency, mean area) — Fire highlighted",
  "Estimated number of fixations"
)
ggsave("figures/06_fixation_count_emmeans.png",
       p_count_emm, width = 7, height = 5, dpi = 300)

### =========================================================
### 4. SPECIES INTERACTION (Fixation count by ROI x Species)
### =========================================================

emm_count_sp_df <- tidy_emm(
  emmeans(full_count_species_model, ~ roi_type | Species, pbkrtest.limit = 3828)
)
p_count_species <- plot_emm(
  emm_count_sp_df,
  "Estimated Fixation Count by ROI and Species",
  "Model-adjusted means with 95% CI — Fire highlighted",
  "Estimated number of fixations",
  facet = vars(Species)
)
ggsave("figures/07_fixation_count_by_species.png",
       p_count_species, width = 9, height = 5, dpi = 300)

### =========================================================
### 5. IMAGE TYPE COMPARISON (Total fixation duration by ROI x ImageType)
### =========================================================

emm_trial_img_df <- tidy_emm(
  emmeans(full_trial_model_image, ~ roi_type | ImageType)
)
p_trial_img <- plot_emm(
  emm_trial_img_df,
  "Estimated Total Fixation Duration by ROI and Image Type",
  "Model-adjusted means with 95% CI — Fire highlighted",
  "Estimated total fixation duration (ms)",
  facet = vars(ImageType)
)
ggsave("figures/08_trial_fixdur_by_imagetype.png",
       p_trial_img, width = 9, height = 5, dpi = 300)

message("All figures written to the 'figures/' directory.")