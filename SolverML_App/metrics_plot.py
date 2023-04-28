import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.color_palette("cubehelix")
import scikitplot.metrics
from sklearn import metrics
import statsmodels.api as sm

def plot():
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.flush()
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.flush()
    buffer.close()
    return graph

def plot_classification(model, X_train, X_test, Y_train, Y_test, y_pred, probs):
    fig = plt.figure(figsize=(20, 6))
    gs = fig.add_gridspec(2, 3)
    # fig,axs = plt.subplots(3,2, figsize=(9,8))
    ax1 = fig.add_subplot(gs[0, 0])
    scikitplot.metrics.plot_precision_recall(Y_test, probs, ax=ax1, plot_micro=True, cmap='tab10', text_fontsize=9)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles[0:1], labels[0:1])
    ax1.get_figure()
    ax1.set_title('Precision-Recall Curve', fontweight='bold')
    ax2 = fig.add_subplot(gs[0, 1])
    scikitplot.metrics.plot_roc(Y_test, probs, plot_micro=False, ax=ax2, plot_macro=False, cmap='tab10',
                                text_fontsize=9)
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles[0:1], labels[0:1])
    ax2.get_figure()
    ax2.set_title('ROC Curve', fontweight='bold')
    ax3 = fig.add_subplot(gs[1, 0])
    if len(set(Y_test)) <= 2:
        scikitplot.metrics.plot_ks_statistic(Y_test, probs, ax=ax3, text_fontsize=9)
        ax3.set_title('K-S Statistics', fontweight='bold')
    ax4 = fig.add_subplot(gs[1, 1])
    sns.heatmap(metrics.confusion_matrix(Y_test, y_pred), ax=ax4, annot=True, fmt='d')
    ax4.set_title('Confusion Matrix', fontweight='bold')

    ax5 = fig.add_subplot(gs[:, 2])
    ax5.autoscale(enable=True)
    ax5.set_title('Feature Importance', fontweight='bold')
    feat_importance_init(model, X_train, ax=ax5)
    plt.tight_layout()
    return fig


def plot_regression(model, X_train, X_test, Y_train, Y_test, y_pred):
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('Q-Q Plot', fontweight='bold')
    sm.qqplot(Y_test - y_pred, fit=True, line="45", ax=ax1)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title('Predicted vs Actual Plot', fontweight='bold')
    ax2.set(xlabel='Actual', ylabel='Predicted')
    sns.regplot(Y_test, y_pred, ax=ax2, line_kws={"color": "red"}, order=2)

    ax5 = fig.add_subplot(gs[:, 1])
    ax5.autoscale(enable=True)
    ax5.set_title('Feature Importance', fontweight='bold')
    feat_importance_init(model, X_train, ax=ax5)

    plt.tight_layout()
    return fig


def linear_feat_importance(model, X, ax=None):
    feat_imp = []
    if type(model).__name__ == 'LogisticRegression':
        importance = model.coef_[0]
    elif type(model).__name__ == 'LinearRegression':
        importance = model.coef_
    importance = pd.DataFrame({'Importance':importance,'Columns':X.columns}).sort_values(by='Importance',ascending=False)
    bp = sns.barplot(x=importance['Importance'],y=importance['Columns'], ax=ax)
    #bp = sns.barplot(x=importance, y=X.columns, ax=ax)
    return bp


def tree_feat_importance(model, X, ax=None):
    importance = pd.DataFrame({'Importance':model.feature_importances_,'Columns':X.columns}).sort_values(by='Importance',ascending=False)
    bp = sns.barplot(x=importance['Importance'],y=importance['Columns'], ax=ax)
    return bp


def feat_importance_init(model, X, ax=None):
    if type(model).__name__ in ['LinearRegression', 'LogisticRegression']:
        print('Linear Model')
        return linear_feat_importance(model, X, ax)
    elif type(model).__name__ in ['DecisionTreeRegressor', 'DecisionTreeClassifier', 'RandomForestRegressor',
                                  'RandomForestClassifier', 'XGBRegressor', 'XGBClassifier']:
        print('Tree Model')
        return tree_feat_importance(model, X, ax)