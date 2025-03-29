Assess软件包中包含了机器学习任务中常见的评价指标
### 分类指标
- 最近邻分类器
```python
from sklearn.datasets import load_digits
from Assess import K_Nearest_Neighbors
KNN = K_Nearest_Neighbors()
from sklearn.model_selection import train_test_split
name_class = ["Experiment", 1, "PCA", "Digit", "test"]
data, target = load_digits(return_X_y=True)
data_train, data_test, target_train, target_test = train_test_split(data, target, train_size=0.5)
KNN.KNN_predict_odds(data, target, name_class)
KNN.KNN_predict_odds_splited(data_train, data_test, target_train, target_test, name_class)
```
- 支持向量机
```python
from sklearn.datasets import load_digits
from Assess import Support_Vector_Machine
SVM = Support_Vector_Machine()
from sklearn.model_selection import train_test_split
name_class = ["Experiment", 1, "PCA", "Digit", "test"]
data, target = load_digits(return_X_y=True)
data_train, data_test, target_train, target_test = train_test_split(data, target, train_size=0.5)
SVM.SVM_predict_odds(data, target, name_class)
SVM.SVM_predict_odds_splited(data_train, data_test, target_train, target_test, name_class)
```
### 聚类指标
- Adjusted Rand Score
```python
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
data, target = load_digits(return_X_y=True)
KM = KMeans(n_clusters=10).fit(data)
target_pred = KM.labels_
name_cluster = ["Experiment", 2, "KMeans", "Digit", "test"]
from Assess import Adjusted_Rand_Score
ARI = Adjusted_Rand_Score()
ARI.adjusted_rand_score_(target, target_pred, name_cluster)
```
- Ajusted Mutual Info Score
```python
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
data, target = load_digits(return_X_y=True)
KM = KMeans(n_clusters=10).fit(data)
target_pred = KM.labels_
name_cluster = ["Experiment", 2, "KMeans", "Digit", "test"]
from Assess import Ajusted_Mutual_Info_Score
AMI = Ajusted_Mutual_Info_Score()
AMI.adjusted_mutual_info_score_(target, target_pred, name_cluster)
```
- Normalized Mutual Info Score
```python
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
data, target = load_digits(return_X_y=True)
KM = KMeans(n_clusters=10).fit(data)
target_pred = KM.labels_
name_cluster = ["Experiment", 2, "KMeans", "Digit", "test"]
from Assess import Normalized_Mutual_Info_Score
NMI = Normalized_Mutual_Info_Score()
NMI.normalized_mutual_info_score_(target, target_pred, name_cluster)
```
- Homogeneity Score
```python
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
data, target = load_digits(return_X_y=True)
KM = KMeans(n_clusters=10).fit(data)
target_pred = KM.labels_
name_cluster = ["Experiment", 2, "KMeans", "Digit", "test"]
from Assess import Homogeneity_Score
HMI = Homogeneity_Score()
HMI.homogeneity_score_(target, target_pred, name_cluster)
```
- Completeness Score
```python
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
data, target = load_digits(return_X_y=True)
KM = KMeans(n_clusters=10).fit(data)
target_pred = KM.labels_
name_cluster = ["Experiment", 2, "KMeans", "Digit", "test"]
from Assess import Completeness_Score
CMI = Completeness_Score()
CMI.completeness_score_(target, target_pred, name_cluster)
```
- V Measure Score
```python
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
data, target = load_digits(return_X_y=True)
KM = KMeans(n_clusters=10).fit(data)
target_pred = KM.labels_
name_cluster = ["Experiment", 2, "KMeans", "Digit", "test"]
from Assess import V_Measure_Score
VMI = V_Measure_Score()
VMI.v_measure_score_(target, target_pred, name_cluster)
```
- Fowlkes Mallows Score
```python
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
data, target = load_digits(return_X_y=True)
KM = KMeans(n_clusters=10).fit(data)
target_pred = KM.labels_
name_cluster = ["Experiment", 2, "KMeans", "Digit", "test"]
from Assess import Fowlkes_Mallows_Score
FMI = Fowlkes_Mallows_Score()
FMI.fowlkes_mallows_score_(target, target_pred, name_cluster)
```
- 轮廓系数
```python
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
data, target = load_digits(return_X_y=True)
KM = KMeans(n_clusters=10).fit(data)
target_pred = KM.labels_
name_cluster = ["Experiment", 2, "KMeans", "Digit", "test"]
from Assess import Silhouette_Score
SIL = Silhouette_Score()
SIL.silhouette_score_(data, target_pred, name_cluster)
```
- 聚类准确率
```python
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
data, target = load_digits(return_X_y=True)
KM = KMeans(n_clusters=10).fit(data)
target_pred = KM.labels_
name_cluster = ["Experiment", 2, "KMeans", "Digit", "test"]
from Assess import Cluster_Accuracy
CA = Cluster_Accuracy()
CA.calculate_accuracy(target, target_pred, name_cluster)
```
### 降维指标
- 联合排名框架
```python
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
data, target = load_digits(return_X_y=True)
embedding = PCA(n_components=2).fit_transform(data)
from Assess import Q_matrix
QM = Q_matrix(cross_time=1, batch_size=15)
QM.Statistic_Total(data, embedding, name_cluster)
```
- Akaike Information Criterion
```python
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
data, target = load_digits(return_X_y=True)
from Assess import K_Nearest_Neighbors
KNN = K_Nearest_Neighbors()
import numpy as np
from Assess import Akaike_Information_Criterion
AI = Akaike_Information_Criterion(
    filename='AIC-Digit',
    func_name='PCA'
)
AI.max_dimension.append(63)
AI.cut_dimension.append(63)
AI.data_names.append("Digit")
odds = np.zeros((1, 63))
for i in range(1, 64):
    model = PCA(n_components=i)
    model.fit(data_train)
    embedding_train = model.transform(data_train)
    embedding_test = model.transform(data_test)
    KNN.KNN_predict_odds_splited(
        embedding_train, embedding_test,
        target_train, target_test, name=None
    )
    odds[0][i-1] = KNN.accuracy
AI.odds = odds
AI.AIC_Run()
```
