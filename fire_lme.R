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

fixation_features <- fixation_features %>%
  group_by(SubjectName) %>%
  mutate(
    trial_num = cumsum(Stimuli != lag(Stimuli, default = first(Stimuli))) + 1
  ) %>%
  ungroup()


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
trial_roi_df <- fixation_features %>%
  group_by(SubjectName, Species, Stimuli, trial_num, roi_type) %>%
  summarise(
    total_fixdur = sum(FixDur),
    mean_sal     = mean(saliency),
    mean_area    = mean(area),
    .groups = "drop"
  ) 

trial_roi_df <- fixation_features %>%
  group_by(SubjectName, Species, Stimuli, trial_num, roi_type, ImageType) %>%
  summarise(
    total_fixdur = sum(FixDur),
    mean_sal     = mean(saliency),
    mean_area    = mean(area),
    .groups = "drop"
  ) %>%
  group_by(SubjectName, Species, Stimuli, trial_num) %>%
  complete(
    roi_type,
    fill = list(
      total_fixdur = 0,
      mean_sal = 0,
      mean_area = 0
    )
  ) %>%
  ungroup() %>% # Scale predictors after aggregation
  mutate(
    total_fixdur_scaled = scale(total_fixdur)[,1],
    mean_sal_scaled     = scale(mean_sal)[,1],
    mean_area_scaled    = scale(mean_area)[,1]
  )

fixation_features$roi_type <- relevel(fixation_features$roi_type, ref = "1")

model_trial <- lmer(
  total_fixdur ~ roi_type + mean_sal_scaled + mean_area_scaled +
    (1 | SubjectName),
  data = trial_roi_df)

summary(model_trial)

trial_roi_df %>%
  group_by(roi_type) %>%
  summarise(
    mean_fixdur = mean(total_fixdur),
    sd_fixdur   = sd(total_fixdur),
    median_fixdur = median(total_fixdur),
    n = n()
  )

#compare with and without roi_type

model_with_roi <- lmer(
  total_fixdur ~ roi_type + mean_sal_scaled + mean_area_scaled +
    (1 | SubjectName),
  data = trial_roi_df,
  REML = FALSE
)

model_no_roi <- lmer(
  total_fixdur ~ mean_sal_scaled + mean_area_scaled +
    (1 | SubjectName),
  data = trial_roi_df,
  REML = FALSE
)

anova(model_no_roi, model_with_roi)
AIC(model_no_roi, model_with_roi)
BIC(model_no_roi, model_with_roi)

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









