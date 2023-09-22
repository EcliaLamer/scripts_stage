df = read.csv("loss_K_var_plus_30_35.csv", sep = ";", header = TRUE)


# on vire 30 et 35 pour l'instant
df_2 = df[,-9]
df_sans_K253035 = df_2[,-8] 
df_sans_K253035 = df_sans_K253035[,-7]
names(df_sans_K253035)


"K10" = df$K10
"K12_5" = df$K12_5
"K15" = df$K15
"K17_5" = df$K17_5
"K20" = df$K20
"K22_5" = df$K22_5
"K25" = df$K25

stack_df = stack(df_sans_K253035)
library("ggpubr")
ggboxplot(stack_df,  x = "ind", y = "values", 
          color = "ind",
          ylab = "Performances des reseaux", xlab = "Valeurs de K ") 

# a priori, d'apres le boxplot, les variances sont pas les meme. On va essayer 
# en transformant les donnees avec un log

stack_df_log = stack_df
stack_df_log["values"] = log(stack_df_log["values"])

ggboxplot(stack_df_log,  x = "ind", y = "values", 
          color = "ind",
          #palette = c("#00AFBB", "#E7B800", "#FC4E07"),
          #order = c("ctrl", "trt1", "trt2"),
          ylab = "Performances des reseaux", xlab = "Valeurs de K ") 

# c'est bien mieux !
# On teste la normalite de chacune des distrib

K10_log = log(K10)
K12_5_log = log(K12_5)
K15_log = log(K15)
K17_5_log = log(K17_5)
K20_log = log(K20)
K22_5_log = log(K22_5)
K25_log = log(K25)
summary(K10_log)
sd(K10_log)
summary(K12_5_log)
sd(K12_5_log)
summary(K15_log)
sd(K15_log)
summary(K17_5_log)
sd(K17_5_log)
summary(K20_log)
sd(K20_log)
summary(K22_5_log)
sd(K22_5_log)
summary(K25_log)
sd(K25_log)



shapiro.test(K10_log) # oui, L'hypothèse de normalité est tolérée.
shapiro.test(K12_5_log) # oui
shapiro.test(K15_log) # nop
boxplot(K15_log) # a priori le probleme vient des 2 outliers

shapiro.test(K17_5_log) # oui
shapiro.test(K20_log) # oui
shapiro.test(K22_5_log) #nop
boxplot(K22_5_log) # encore des outliers

# on decide d'enlever les outliers de tout le monde.

# on commence par enlever les outliers de K12_5

outlier = boxplot(K12_5_log)$out

Q1 = quantile(K12_5_log, .25)
Q3 = quantile(K12_5_log, .75)
IQR = IQR(K12_5_log)

K12_5_log_sans_out = subset(K12_5_log, K12_5_log > (Q1 - 1.5*IQR) & K12_5_log < (Q3 + 1.5*IQR))
which(!(K12_5_log > (Q1 - 1.5*IQR) & K12_5_log < (Q3 + 1.5*IQR)))

shapiro.test(K12_5_log_sans_out)# oui

# on enleve les outliers de K15

outlier = boxplot(K15_log)$out

Q1 = quantile(K15_log, .25)
Q3 = quantile(K15_log, .75)
IQR = IQR(K15_log)

K15_log_sans_out = subset(K15_log, K15_log > (Q1 - 1.5*IQR) & K15_log < (Q3 + 1.5*IQR))
which(!(K15_log > (Q1 - 1.5*IQR) & K15_log < (Q3 + 1.5*IQR))) # pour enlever les outliers 'a la main'

shapiro.test(K15_log_sans_out)# oui

#on enleve maintenant  les outliers de K20
outlier = boxplot(K20_log)$out

Q1 = quantile(K20_log, .25)
Q3 = quantile(K20_log, .75)
IQR = IQR(K20_log)

K20_log_sans_out = subset(K20_log, K20_log > (Q1 - 1.5*IQR) & K20_log < (Q3 + 1.5*IQR))
which(!(K20_log > (Q1 - 1.5*IQR) & K20_log < (Q3 + 1.5*IQR)))

shapiro.test(K20_log_sans_out)# oui

#on enleve les outliers de K22_5

outlier = boxplot(K22_5_log)$out

Q1 = quantile(K22_5_log, .25)
Q3 = quantile(K22_5_log, .75)
IQR = IQR(K22_5_log)

K22_5_log_sans_out = subset(K22_5_log, K22_5_log > (Q1 - 1.5*IQR) & K22_5_log < (Q3 + 1.5*IQR))
which(!(K22_5_log > (Q1 - 1.5*IQR) & K22_5_log < (Q3 + 1.5*IQR)))

shapiro.test(K22_5_log_sans_out)# oui

#Finalement, on refait df_stack en enlevant tous ces outliers (avec une methode magnifique)

df_stack_log_sans_out = stack_df_log[c(1:95,
                                       97:117, 
                                       119:136,
                                       138:238,
                                       240:262, 
                                       264:269, 
                                       271:297, 
                                       299:300),]


ggboxplot(df_stack_log_sans_out,  x = "ind", y = "values", 
          color = "ind",
          #palette = c("#00AFBB", "#E7B800", "#FC4E07"),
          #order = c("ctrl", "trt1", "trt2"),
          ylab = "Performances des réseaux", xlab = "Valeurs de K ") 

# on test maintenant l'egalite des variance
bartlett.test(values ~ ind, data = df_stack_log_sans_out) # oui !

# ------------------------------------------------------------------------------
# test variance quand out 15 et 22_5 sont enleve
df_stack_log_sans_15_225_out = stack_df_log[c(1:117, 
                                              119:136,
                                              138:262, 
                                              264:269, 
                                              271:297, 
                                              299:300),]
bartlett.test(values ~ ind, data = df_stack_log_sans_15_225_out) # oui !
# ------------------------------------------------------------------------------

#fit the one-way ANOVA model
model_anova <- aov(values ~ ind, data = df_stack_log_sans_out)
summary(model_anova)
#create Q-Q plot to compare this dataset to a theoretical normal distribution 
qqnorm(model_anova$residuals)
#add straight diagonal line to plot
qqline(model_anova$residuals)

# Extract the residuals
aov_residuals <- residuals(object = model_anova )
# Run Shapiro-Wilk test
shapiro.test(x = aov_residuals )

# l'ANOVA fonctionne. On fait des comparaisons par pair, d'abord avec Tukey puis
# avec des t-tests

TukeyHSD(model_anova,conf.level=.95)


pairwise.t.test(df_stack_log_sans_out$values, df_stack_log_sans_out$ind, pool.sd = TRUE,
                p.adjust.method = "bonferroni")

# On a bien que toutes nos moyennes sont distinctes, meme apres correction de bonferroni.

# -----------------------------------------------------------------------------
# on regarde maintenant si ca continue a s'ameliorer apres 22.5.

df_k25_35 = df[,7:9]
names(df_k25_35)

K25_log = log(df_k25_35$K25)
K30_log = log(df_k25_35$K30)
K35_log = log(df_k25_35$K35)
summary(K30_log)
sd(K30_log)
summary(K35_log)
sd(K35_log)

df_stack_25_35 = stack(df_k25_35)
df_stack_25_35_log = df_stack_25_35
df_stack_25_35_log['values'] = log(df_stack_25_35_log['values'])
ggboxplot(df_stack_25_35_log,  x = "ind", y = "values", 
          color = "ind",
          #palette = c("#00AFBB", "#E7B800", "#FC4E07"),
          #order = c("ctrl", "trt1", "trt2"),
          ylab = "Performances des reseaux", xlab = "Valeurs de K ") 

# on enleve les outliers de 25
outlier = boxplot(K25_log)$out

Q1 = quantile(K25_log, .25)
Q3 = quantile(K25_log, .75)
IQR = IQR(K25_log)

K25_log_sans_out = subset(K25_log, K25_log > (Q1 - 1.5*IQR) & K25_log < (Q3 + 1.5*IQR))
which(!(K25_log > (Q1 - 1.5*IQR) & K25_log < (Q3 + 1.5*IQR)))

shapiro.test(K25_log_sans_out) # ca marche

# on enleve les outliers de 30
outlier = boxplot(K30_log)$out

Q1 = quantile(K30_log, .25)
Q3 = quantile(K30_log, .75)
IQR = IQR(K30_log)

K30_log_sans_out = subset(K30_log, K30_log > (Q1 - 1.5*IQR) & K30_log < (Q3 + 1.5*IQR))
which(!(K30_log > (Q1 - 1.5*IQR) & K30_log < (Q3 + 1.5*IQR)))

shapiro.test(K30_log_sans_out) # ca marche

# on enleve les outliers de 25
outlier = boxplot(K35_log)$out

Q1 = quantile(K35_log, .25)
Q3 = quantile(K35_log, .75)
IQR = IQR(K35_log)

K35_log_sans_out = subset(K35_log, K35_log > (Q1 - 1.5*IQR) & K35_log < (Q3 + 1.5*IQR))
which(!(K35_log > (Q1 - 1.5*IQR) & K35_log < (Q3 + 1.5*IQR)))

shapiro.test(K35_log_sans_out) # ca marche

# on retire l'ensemble des outliers

df_stack_25_35_log_sans_out = df_stack_25_35_log[c(1:2, 4:32, 34:96, 98:113, 115:138, 140:150),]

ggboxplot(df_stack_25_35_log_sans_out,  x = "ind", y = "values", 
          color = "ind",
          #palette = c("#00AFBB", "#E7B800", "#FC4E07"),
          #order = c("ctrl", "trt1", "trt2"),
          ylab = "Performances des reseaux", xlab = "Valeurs de K ") 

bartlett.test(values ~ ind, data =df_stack_25_35_log_sans_out)



t.test(K25_log_sans_out,K30_log_sans_out, var.equal = FALSE)
t.test(K30_log_sans_out,K35_log_sans_out, var.equal = FALSE)
t.test(K25_log_sans_out,K35_log_sans_out, var.equal = FALSE)


################################
df_log_sans_out = data.frame(K10_log, K12_5_log_sans_out, K15_log_sans_out, K17_5_log, K20_log_sans_out, K22_5_log_sans_out)


df_density <- data.frame(K10_log, K12_5_log, K15_log, K17_5_log, K20_log, K22_5_log, K25_log)
summary(df_density)

ggdensity(df_density,
          x = c("K10_log", "K12_5_log", "K15_log" ,"K17_5_log", "K20_log", "K22_5_log"),
          #, "K30", "K35"),
          y = "..density..",
          # Combine the 3 plots
          color = ".x.",fill = ".x.",
          merge = TRUE,
          xlab = "Expression", 
          add = "median",                  # Add median line. 
          palette = "jco",
          rug = TRUE                       # Add marginal rug
)
##########################

#---------------------------





