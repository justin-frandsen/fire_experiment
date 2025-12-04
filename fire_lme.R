library(tidyverse)
library(lme4)
library(lmerTest)   # p-values for lmer
library(performance)  # Model comparison tools
library(emmeans)

getwd()
setwd("Documents/python/repos/fire_experiment")

fixation_features <- read.csv("fixation_features.csv")
unique(fixation_features$SubjectName)

# roi_type: 0 = background, 1 = fire, 2 = vegetation
# roi_index: unique ID per region, starting from 1

# Convert factors
fixation_features <- fixation_features %>%
  mutate(SubjectName = as.factor(SubjectName),
         Species = as.factor(Species),
         ROI = as.factor(ROI),
         roi_type = as.factor(roi_type)) %>% 
  filter(FixDur >= 50) %>% #remove fixations less than 50ms
  mutate(saliency_scaled = scale(saliency)[,1], #scale predictors
         area_scaled = scale(area)[,1])

# full model with all predictors
## relevel to compare background and vegitation vs fire
fixation_features$roi_type <- relevel(fixation_features$roi_type, ref = "1")
fixation_features %>% filter()

#model
model_fix_full <- lmer(FixDur ~ saliency_scaled + area_scaled + roi_type + (1 | SubjectName),
                       data = fixation_features %>% filter(ImageType == "F"))

summary(model_fix_full)

#pairwise comparisons
emmeans(model_fix_full, pairwise ~ ROI)

# model without first fixations
fix_no_first <- fixation_features %>%
  group_by(SubjectName, Stimuli, Trial) %>%
  mutate(FixIndex = row_number()) %>%
  ungroup() %>%
  filter(FixIndex > 1)

model_no_first <- lmer(FixDur ~ saliency_scaled + area_scaled + roi_type + Species + (1 | SubjectName),
                       data = fix_no_first)

summary(model_no_first)

# Trial Level Models
trial_df <- fixation_features %>%
  group_by(SubjectName, Species, Stimuli, Trial) %>%
  summarise(mean_fixdur   = mean(FixDur),
            median_fixdur = median(FixDur),
            mean_sal      = mean(saliency_scaled),
            mean_area     = mean(area_scaled),
            prop_fire     = mean(as.numeric(as.character(roi_type))), # proportion fixations on fire
            n_fix         = n(),
            .groups = "drop")

model_trial <- lmer(median_fixdur ~ mean_sal + mean_area + prop_fire * Species + (1 | SubjectName),
                    data = trial_df)

summary(model_trial)

#only chimp results
fix_chimp <- fixation_features %>% filter(Species == "Chimp")

model_chimp <- lmer(FixDur ~ saliency_scaled + area_scaled + ROI + (1 | SubjectName),
                    data = fix_chimp)

summary(model_chimp)

#only baboon results
fix_baboon <- fixation_features %>% filter(Species == "Baboon")

model_baboon <- lmer(FixDur ~ saliency_scaled + area_scaled + roi_type + (1 | SubjectName),
                    data = fix_chimp)

summary(model_baboon)









