from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def model_plots(x_train,x_test,y_train,y_test):
        models = {
        "Logistic Regression": LogisticRegression(max_iter=10000), # max_iters for the lbfgs optimization algorithm
        "Bernoulli Naive Bayes": BernoulliNB(),
        "Random Forest": RandomForestClassifier(n_estimators=10, random_state=42),
        "SVM": SVC(kernel='linear', random_state=42)
    }        
        
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        for ax, (name, model) in zip(axs.flat, models.items()):
          model.fit(x_train, y_train)
          y_pred = model.predict(x_test)
          fpr, tpr, _ = roc_curve(y_test, y_pred)
          roc_auc = auc(fpr, tpr)
          ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'CurveArea = {round(roc_auc, 2) * 100}%')
          ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
          ax.set_xlabel('False Positive Rate')
          ax.set_ylabel('True Positive Rate')
          ax.set_title(f'ROC Curve - {name}')
          ax.legend(loc="lower right")

        plt.tight_layout()
        plt.show()