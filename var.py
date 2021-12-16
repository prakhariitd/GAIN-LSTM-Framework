mod = smt.VAR(X_train_transformed)
res = mod.fit(maxlags=15, ic='aic')
print(res.summary())