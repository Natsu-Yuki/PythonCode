from sklearn.model_selection import train_test_split


#   K近邻分类
from sklearn.neighbors import KNeighborsClassifier
(n_neighbors=)
#   K近邻分类回归
from sklearn.neighbors import KNeighborsRegressor


#   线性回归
from sklearn.linear_model import LinearRegression
(权重coef_、截距intercept_)
#   岭回归
from sklearn.linear_model import Ridge
(alpha=)
#   Lasso回归
from sklearn.linear_model import Lasso
(alpha=,max_iter)


#   Logistic分类
from sklearn.linear_model import LogisticRegression
(C=)
#   线性向量支持机
from sklearn.svm import LinearSVC
(C=)

#   决策树
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
(max_depth=,random_state=)
(.faeture_importance_)

#   随机森林
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
(n_estimater=,random_state=)
#   梯度提升树
from sklearn.ensemble import  GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
(random_state=,learning_rate=,max_depth=)


#   神经网络
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
(soler=,random_state=,hidden_layer=[],activation=)

#   不确定度
.decision_function(决策函数)
.predict_proba(预测概率)





