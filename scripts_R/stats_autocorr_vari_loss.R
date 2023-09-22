df = read.csv("autocorr_var_loss.csv", sep = ";", header = TRUE)
names(df)
summary(df)



A0_2 = df$A0_2
A0_6 = df$A0_6
A0_7 = df$A0_7
A0_8 = df$A0_8
A0_9 = df$A0_9
A1 = df$A1

stack_df = stack(df)

library("ggpubr")
ggboxplot(stack_df, x = "ind", y = "values", 
          color = "ind", 
          #palette = c("#00AFBB", "#E7B800", "#FC4E07"),
          #order = c("ctrl", "trt1", "trt2"),
          ylab = "Performances des reseaux", xlab = "Valeurs de l'autocorrelation ")

# on voit immediatement des problemes avec les outliers de A0_9, mais aussi 
# probablement avec les variances (A0_2 semble etale). On va passer au Log d'abord

A0_2_log = log(df$A0_2)
A0_6_log = log(df$A0_6)
A0_7_log = log(df$A0_7)
A0_8_log = log(df$A0_8)
A0_9_log = log(df$A0_9)
A1_log = log(df$A1)
df_log = data.frame(A0_2_log,A0_6_log,A0_7_log,A0_8_log,A0_9_log,A1_log)
summary(df_log)
sd(A0_2_log)
sd(A0_6_log)
sd(A0_7_log)
sd(A0_8_log)
sd(A0_9_log)
sd(A1_log)



stack_df_log = stack_df
stack_df_log["values"] = log(stack_df_log["values"])

ggboxplot(stack_df_log, x = "ind", y = "values", 
          color = "ind", 
          #palette = c("#00AFBB", "#E7B800", "#FC4E07"),
          #order = c("ctrl", "trt1", "trt2"),
          ylab = "Performances des reseaux", xlab = "Valeurs de l'autocorrelation ")

# il y a clairement un probleme avec les outliers de A0_9. On commence par les 
# enlever

outlier = boxplot(A0_9_log)$out

Q1 = quantile(A0_9_log, .25)
Q3 = quantile(A0_9_log, .75)
IQR = IQR(A0_9_log)

A0_9_log_sans_out = subset(A0_9_log, A0_9_log > (Q1 - 1.5*IQR) & A0_9_log < (Q3 + 1.5*IQR))
which(!(A0_9_log > (Q1 - 1.5*IQR) & A0_9_log < (Q3 + 1.5*IQR))) # pour enlever les outliers 'a la main'

shapiro.test(A0_9_log_sans_out) # juste pour verifier que c'est bien normal

# regardons maintenant 
stack_df_log_test = stack_df_log[c(1:209,
                                   211:219,
                                   221:222,
                                   224:229,
                                   231:234,
                                   236:300),]

ggboxplot(stack_df_log_test, x = "ind", y = "values", 
          color = "ind", 
          #palette = c("#00AFBB", "#E7B800", "#FC4E07"),
          #order = c("ctrl", "trt1", "trt2"),
          ylab = "Performances des reseaux", xlab = "Valeurs de l'autocorrelation ")

# on teste les normalites
shapiro.test(A0_2_log) # nop
shapiro.test(A0_6_log) # oui
shapiro.test(A0_7_log) # oui
shapiro.test(A0_8_log) # oui
shapiro.test(A1_log) # oui 
# On a teste A0_9 plus haut

# on enleve l'outlier de A0_2
outlier = boxplot(A0_2_log)$out

Q1 = quantile(A0_2_log, .25)
Q3 = quantile(A0_2_log, .75)
IQR = IQR(A0_2_log)

A0_2_log_sans_out = subset(A0_2_log, A0_2_log > (Q1 - 1.5*IQR) & A0_2_log < (Q3 + 1.5*IQR))
which(!(A0_2_log > (Q1 - 1.5*IQR) & A0_2_log < (Q3 + 1.5*IQR))) # pour enlever les outliers 'a la main'

shapiro.test(A0_2_log_sans_out) # c'est bien normal

# on enleve tous les outliers

stack_df_log_sans_out = stack_df_log[c(1:49,
                                   51:209,
                                   211:219,
                                   221:222,
                                   224:229,
                                   231:234,
                                   236:300),]


ggboxplot(stack_df_log_sans_out, x = "ind", y = "values", 
          color = "ind", 
          #palette = c("#00AFBB", "#E7B800", "#FC4E07"),
          #order = c("ctrl", "trt1", "trt2"),
          ylab = "Performances des reseaux", xlab = "Valeurs de l'autocorrelation ")


bartlett.test(values ~ ind, data = stack_df_log_sans_out)


#fit the one-way ANOVA model
model_anova <- aov(values ~ ind, data = stack_df_log_sans_out)
summary(model_anova)
#create Q-Q plot to compare this dataset to a theoretical normal distribution 
qqnorm(model_anova$residuals)
#add straight diagonal line to plot
qqline(model_anova$residuals)

# Extract the residuals
aov_residuals <- residuals(object = model_anova )
# Run Shapiro-Wilk test
shapiro.test(x = aov_residuals )

TukeyHSD(model_anova)
pairwise.t.test(stack_df_log_sans_out$values, stack_df_log_sans_out$ind, 
                pool.sd = TRUE, p.adjust.method = "bonferroni")


