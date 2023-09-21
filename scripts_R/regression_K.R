df_K10 = read.csv("C:/Users/fviar/Desktop/stage_alice/donnees_reg_K10.csv", sep = ";", header = TRUE)
df_K12_5 = read.csv("C:/Users/fviar/Desktop/stage_alice/donnees_reg_K12_5.csv", sep = ";", header = TRUE)
df_K15 = read.csv("C:/Users/fviar/Desktop/stage_alice/donnees_reg_K15.csv", sep = ";", header = TRUE)
df_K17_5 = read.csv("C:/Users/fviar/Desktop/stage_alice/donnees_reg_K17_5.csv", sep = ";", header = TRUE)
df_K20 = read.csv("C:/Users/fviar/Desktop/stage_alice/donnees_reg_K20.csv", sep = ";", header = TRUE)
df_K22_5 = read.csv("C:/Users/fviar/Desktop/stage_alice/donnees_reg_K22_5.csv", sep = ";", header = TRUE)
df_K25 = read.csv("C:/Users/fviar/Desktop/stage_alice/donnees_reg_K25.csv", sep = ";", header = TRUE)
df_K25_sans_out = read.csv("C:/Users/fviar/Desktop/stage_alice/donnees_reg_K25_sans_out.csv", sep = ";", header = TRUE)
names(df_K25)
summary(df_K10)

dist_K10 = df_K10$dist_target_pred
weeks_K10 = df_K10$weeks

Q1 = quantile(dist_K10, .25)
Q3 = quantile(dist_K10, .75)
IQR = IQR(dist_K10)

dist_K10_sans_out = subset(dist_K10, dist_K10 > (Q1 - 1.5*IQR) & dist_K10< (Q3 + 1.5*IQR))



#https://delladata.fr/la-regression-lineaire-simple-avec-le-logiciel-r/
library(car)
scatterplot(dist_target_pred~weeks, data=df_K10) 

scatterplot(dist_target_pred~weeks, data=df_K12_5) 

scatterplot(dist_target_pred~weeks, data=df_K15)

scatterplot(dist_target_pred~weeks, data=df_K17_5)

scatterplot(dist_target_pred~weeks, data=df_K20)

scatterplot(dist_target_pred~weeks, data=df_K22_5)

scatterplot(dist_target_pred~weeks, data=df_K25)
scatterplot(dist_target_pred_sans_out~weeks_sans_out, data=df_K25_sans_out)

# nos regressions lineaire simples

lm_0_K10 = lm(dist_target_pred~weeks, data=df_K10)
summary(residuals(lm_0_K10))
plot(lm_0_K10,2) 
# pas du tout normal

lm_1_K10 = lm(dist_target_pred~(weeks*weeks), data=df_K10)
summary(residuals(lm_1_K10))
plot(lm_1_K10,2)
# pas du tout normal


lm_0_K25 = lm(dist_target_pred~weeks, data=df_K25)
summary(residuals(lm_0_K10))
plot(lm_0_K10,2) 
# pas du tout normal

# essai sans outlier
lm_1_K25 = lm(dist_target_pred_sans_out~weeks_sans_out, data=df_K25_sans_out)
summary(residuals(lm_1_K25))
plot(lm_1_K25,2)
# pas du tout normal

# ------------------------------------------------------------------------------
# on tente la correlation de spearman
cor.test(df_K10$weeks,df_K10$dist_target_pred, data=df_K10, method="spearman")
# Ici, la p-value est largement inférieure au seuil de significativité 
# généralement employé de 0.05. On conclut donc à la dépendance monotone 
# significative entre les variables x et y.


# on fait le calcul pour toutes les variables
cor.test(df_K10$weeks,df_K10$dist_target_pred, data=df_K10, method="spearman")
rho_K10 = 0.5802878
cor.test(df_K12_5$weeks,df_K12_5$dist_target_pred, data=df_K12_5, method="spearman")
rho_K12_5 = 0.6448222
cor.test(df_K15$weeks,df_K15$dist_target_pred, data=df_K15, method="spearman")
rho_K15 = 0.6359092
cor.test(df_K17_5$weeks,df_K17_5$dist_target_pred, data=df_K17_5, method="spearman")
rho_K17_5 = 0.6660389
cor.test(df_K20$weeks,df_K20$dist_target_pred, data=df_K20, method="spearman")
rho_K20 = 0.5455219
cor.test(df_K22_5$weeks,df_K22_5$dist_target_pred, data=df_K22_5, method="spearman")
rho_K22_5 = 0.535975
cor.test(df_K25$weeks,df_K25$dist_target_pred, data=df_K25, method="spearman")
rho_K25 = 0.3349086

plot( seq(10, 25, by=2.5), 
      c(rho_K10,rho_K12_5,rho_K15,rho_K17_5,rho_K20, rho_K22_5, rho_K25 ),
      type = 'o', xlab = 'Valeurs de K', ylab = 'Coefficient de Spearman',
      pch = 18, xaxt = 'n')
axis(1, at = seq(10, 25, by=2.5), las=1)

