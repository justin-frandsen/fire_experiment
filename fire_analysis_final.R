library(tidyverse)
library(lme4)
library(lmerTest)   # p-values for lmer
library(performance)  # Model comparison tools
library(emmeans)

getwd()
setwd("Documents/python/repos/fire_experiment")

#raw dataset
fixation_features <- read.csv("fixation_features.csv")

#Converting factors
fixation_features <- fixation_features %>%
  mutate(SubjectName = as.factor(SubjectName),
         Species = as.factor(Species),
         ROI = as.factor(ROI),
         roi_type = as.factor(roi_type)) %>% 
  filter(FixDur >= 50) %>% #remove fixations less than 50ms
  mutate(saliency_scaled = scale(saliency)[,1], #scale predictors
         area_scaled = scale(area)[,1],
         ROI_type_combined = case_when(
           ROI == "Fire" ~ "Fire",
           roi_type == 2 ~ "Veg",
           TRUE ~ "Background"),
         ROI_type_combined = as.factor(ROI_type_combined))

#removing trials where they didn't spend at least 500ms looking
fixation_features <- fixation_features %>%
  group_by(SubjectName, Stimuli, Trial) %>%
  mutate(total_trial_fixation = sum(FixDur, na.rm = TRUE)) %>%
  ungroup() %>%
  filter(total_trial_fixation >= 500)

#making Fire ROIs the baseline
fixation_features$ROI_type_combined <- relevel(fixation_features$ROI_type_combined, ref = "Fire")

# extract trial number
fixation_features <- fixation_features %>%
  group_by(SubjectName) %>%
  mutate(trial_num = cumsum(Stimuli != lag(Stimuli, default = first(Stimuli))) + 1) %>%
  ungroup()

### =========== Individual Fixations Models =======================

## Only fire images
individual_fixations_model <- lmer(FixDur ~ ROI_type_combined + area_scaled + saliency_scaled + (1 | SubjectName),
                       data = fixation_features %>% filter(ImageType == "F"))
summary(individual_fixations_model)

# pairwise comparisons
emmeans(individual_fixations_model, pairwise ~ ROI_type_combined)

## Only Chimps
individual_fixations_model_chimp <- lmer(FixDur ~ ROI_type_combined + area_scaled + saliency_scaled + (1 | SubjectName),
                       data = fixation_features %>% filter(ImageType == "F", Species == "Chimp"))
summary(individual_fixations_model_chimp)

## Only Baboons.  
individual_fixations_model_baboons <- lmer(FixDur ~ ROI_type_combined + area_scaled + saliency_scaled + (1 | SubjectName),
                                         data = fixation_features %>% filter(ImageType == "F", Species == "Baboon"))
summary(individual_fixations_model_baboons)

### =========== Trial Level Fixations Models =======================
# Trial Level Models
trial_roi_df <- fixation_features %>%
  group_by(SubjectName, Species, Stimuli, trial_num, ROI_type_combined, ImageType) %>%
  summarise(total_fixdur = sum(FixDur),
            mean_sal     = mean(saliency),
            mean_area    = mean(area),
            fixation_count = n(),
            .groups = "drop") %>%
  mutate(total_fixdur_scaled = scale(total_fixdur)[,1],
         mean_sal_scaled     = scale(mean_sal)[,1],
         mean_area_scaled    = scale(mean_area)[,1],
         image_type = if_else(str_detect(Stimuli, "SS"), "Scrambled",
                         if_else(str_detect(Stimuli, "FS"), "Fire", NA)))

#create version with 2 types only fire and veg no background
#trial_roi_no_background <- trial_roi_df %>%
#  filter(ROI_type_combined != "Background") %>%
#  group_by(SubjectName, trial_num) %>%
#  filter(
#    n() == 2,
#    all(c("Fire", "Veg") %in% ROI_type_combined)
#  ) %>%
#  ungroup()
  

## Only fire images
full_trial_model <- lmer(total_fixdur ~ ROI_type_combined + mean_sal_scaled + mean_area_scaled + (1 | SubjectName),
                    data = trial_roi_df %>% filter(ImageType == "F"))

summary(full_trial_model)

# pairwise comparisons
emmeans(full_trial_model, pairwise ~ ROI_type_combined)

## Image type comparison model
full_trial_model_image <- lmer(total_fixdur ~ ROI_type_combined * ImageType + mean_sal_scaled + mean_area_scaled + (1 | SubjectName),
                         data = trial_roi_df)

summary(full_trial_model_image)

# pairwise comparisons
emmeans(full_trial_model_image, pairwise ~ ROI_type_combined)

## Only chimps 
full_trial_chimp_model <- lmer(total_fixdur ~ ROI_type_combined + mean_sal_scaled + mean_area_scaled + (1 | SubjectName),
                         data = trial_roi_df %>% filter(ImageType == "F", Species == "Chimp"))

summary(full_trial_chimp_model)

## Only baboons 
full_trial_chimp_model <- lmer(total_fixdur ~ ROI_type_combined + mean_sal_scaled + mean_area_scaled + (1 | SubjectName),
                               data = trial_roi_df %>% filter(ImageType == "F", Species == "Baboon"))

summary(full_trial_chimp_model)

### =========== Number of Fixations Models =======================
## Only fire images
full_trial_model <- lmer(total_fixdur ~ ROI_type_combined + mean_sal_scaled + mean_area_scaled + (1 | SubjectName),
                         data = trial_roi_df %>% filter(ImageType == "F"))

summary(full_trial_model)

# pairwise comparisons
emmeans(model_trial, pairwise ~ ROI_type_combined)

## Image type comparison model
full_trial_model_image <- lmer(fixation_count ~ ROI_type_combined * ImageType + mean_sal_scaled + mean_area_scaled + (1 | SubjectName),
                               data = trial_roi_df)

summary(full_trial_model_image)

# pairwise comparisons
emmeans(model_trial, pairwise ~ ROI_type_combined)

## Only chimps 
full_trial_chimp_model <- lmer(fixation_count ~ ROI_type_combined + mean_sal_scaled + mean_area_scaled + (1 | SubjectName),
                               data = trial_roi_df %>% filter(ImageType == "F", Species == "Chimp"))

summary(full_trial_chimp_model)

## Only baboons 
full_trial_chimp_model <- lmer(fixation_count ~ ROI_type_combined + mean_sal_scaled + mean_area_scaled + (1 | SubjectName),
                               data = trial_roi_df %>% filter(ImageType == "F", Species == "Baboon"))

summary(full_trial_chimp_model)
