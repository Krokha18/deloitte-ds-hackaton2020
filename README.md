# deloitte-ds-hackaton2020
Solutions from Deloitte Data Science Hackathon 2020( Company acceptance prediction )

Решения с Deloitte Data Science Hackathon 2020( Предсказание подходит ли компания по критериям).

(eng):
We took a part in prediction challenge from Deloitte company(4-5 April,2020).

The main task is to classify texts(descriptions of clients of Deloitte) to 3 classes (0 - rejected by product, 1- rejected by function, 2 - accepted). Quality metric: balanced accuracy(sample has disbalance of classes)

Of course, data needs to be preprocessed. We applied lowercase to all the columns, and dropped off some junk words as "and", "or".

In general, our solution uses text vectorizing using tf-idf, then compression of (term-document matrix) using SVD, and finally learning with RandomForestClassifier.

Why Random Forest? Because struct of data is complicated. Linear models such as Logistic Regression, Ridge, Passive Aggressive, are not able to predict with good accuracy. Random Forest is stronger algorithm, has less params, and doesn't overfitting.

Finally, our place is #14 (from 38 teams) in Kaggle with following results: 0.708

In this repository you can find some trials that our team did and try make it better.

Full information about data and competition: https://www.kaggle.com/c/company-acceptance-prediction/overview

Our team (Wiener Sausage) :

[@Krokha5](github.com/Krokha5) - me(Ivan Krokhalyov)

[@sevagul](github.com/sevagul)  - Vsevolod Hulchuk

[@StMichael99](github.com/StMichael99) -  Misha Stolyar
