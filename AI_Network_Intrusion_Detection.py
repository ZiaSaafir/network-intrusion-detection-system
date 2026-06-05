#!/usr/bin/env python
# coding: utf-8

# ## Task1 Data Exploration

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("network_traffic.csv")

print("Dataset Shape:")
print(df.shape)


# In[3]:


df.head()


# In[4]:


df.describe()


# In[ ]:





# In[6]:


class_counts = df["label"].value_counts()

plt.figure(figsize=(6,4))

plt.bar(
    ["Normal (0)", "Attack (1)"],
    class_counts.values
)

plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Connections")

plt.show()


# In[7]:


normal = df[df["label"] == 0]
attack = df[df["label"] == 1]

print("Normal Traffic:", len(normal))
print("Attack Traffic:", len(attack))


# In[8]:


plt.figure(figsize=(8,5))

plt.hist(
    normal["src_bytes"],
    bins=30,
    alpha=0.6,
    label="Normal"
)

plt.hist(
    attack["src_bytes"],
    bins=30,
    alpha=0.6,
    label="Attack"
)

plt.title("Source Bytes Distribution")
plt.xlabel("src_bytes")
plt.ylabel("Frequency")
plt.legend()

plt.show()


# In[9]:


plt.figure(figsize=(8,5))

plt.hist(
    normal["count"],
    bins=30,
    alpha=0.6,
    label="Normal"
)

plt.hist(
    attack["count"],
    bins=30,
    alpha=0.6,
    label="Attack"
)

plt.title("Count Distribution")
plt.xlabel("count")
plt.ylabel("Frequency")
plt.legend()

plt.show()


# ## Task 2  Simple Reflex Agent

# In[10]:


def reflex_agent(row):

    if row["serror_rate"] > 0.5:
        return 1

    elif row["count"] > 100:
        return 1

    elif row["same_srv_rate"] < 0.4:
        return 1

    else:
        return 0


# In[11]:


predictions = df.apply(reflex_agent, axis=1)

predictions.head()


# In[12]:


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(
    df["label"],
    predictions
)

print("Accuracy:", accuracy)


# In[13]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(
    df["label"],
    predictions
)

print("Confusion Matrix:")
print(cm)


# ## Task 3  Supervised Learning

# In[14]:


X = df.drop("label", axis=1)
y = df["label"]

print("Features Shape:", X.shape)
print("Labels Shape:", y.shape)


# In[15]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# In[16]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ## KNN Classifier

# In[17]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

k_values = [1, 3, 5, 7, 9, 11]
accuracies = []

for k in k_values:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train_scaled, y_train)

    pred = knn.predict(X_test_scaled)

    acc = accuracy_score(y_test, pred)

    accuracies.append(acc)

    print(f"K = {k}, Accuracy = {acc:.4f}")


# In[18]:


plt.figure(figsize=(7,5))

plt.plot(k_values, accuracies, marker='o')

plt.title("KNN Accuracy vs K")
plt.xlabel("K Value")
plt.ylabel("Accuracy")

plt.grid(True)

plt.show()


# ## Gaussian Naive Bayes

# In[23]:


from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train, y_train)

nb_pred = nb.predict(X_test)


# In[24]:


nb_accuracy = accuracy_score(y_test, nb_pred)

nb_precision = precision_score(y_test, nb_pred)

nb_recall = recall_score(y_test, nb_pred)

nb_f1 = f1_score(y_test, nb_pred)

nb_cm = confusion_matrix(y_test, nb_pred)

print("Accuracy :", nb_accuracy)
print("Precision:", nb_precision)
print("Recall   :", nb_recall)
print("F1 Score :", nb_f1)

print("\nConfusion Matrix")
print(nb_cm)


# ## Logistic Regression

# In[25]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(
    max_iter=1000
)

lr.fit(
    X_train_scaled,
    y_train
)

lr_pred = lr.predict(
    X_test_scaled
)


# In[26]:


lr_accuracy = accuracy_score(y_test, lr_pred)

lr_precision = precision_score(y_test, lr_pred)

lr_recall = recall_score(y_test, lr_pred)

lr_f1 = f1_score(y_test, lr_pred)

lr_cm = confusion_matrix(y_test, lr_pred)

print("Accuracy :", lr_accuracy)
print("Precision:", lr_precision)
print("Recall   :", lr_recall)
print("F1 Score :", lr_f1)

print("\nConfusion Matrix")
print(lr_cm)


# In[27]:


results = pd.DataFrame({
    "Model": [
        "KNN",
        "Naive Bayes",
        "Logistic Regression"
    ],

    "Accuracy": [
        knn_accuracy,
        nb_accuracy,
        lr_accuracy
    ],

    "Precision": [
        knn_precision,
        nb_precision,
        lr_precision
    ],

    "Recall": [
        knn_recall,
        nb_recall,
        lr_recall
    ],

    "F1 Score": [
        knn_f1,
        nb_f1,
        lr_f1
    ]
})

results


# ## TASK 4  K-Means Clustering

# In[28]:


X_cluster = df.drop("label", axis=1)


# In[29]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_cluster)


# In[30]:


from sklearn.cluster import KMeans

kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

print(clusters[:10])


# In[31]:


from sklearn import metrics

cm = metrics.confusion_matrix(
    df["label"],
    clusters
)

print(cm)


# In[32]:


from sklearn.decomposition import PCA

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)


# In[33]:


plt.figure(figsize=(12,5))

plt.subplot(1,2,1)

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=df["label"],
    s=10
)

plt.title("True Labels")

plt.xlabel("PCA 1")
plt.ylabel("PCA 2")


plt.subplot(1,2,2)

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=clusters,
    s=10
)

plt.title("K-Means Clusters")

plt.xlabel("PCA 1")
plt.ylabel("PCA 2")

plt.tight_layout()

plt.show()


# ## Task 5 Genetic Algorithm

# In[34]:


feature_names = X.columns.tolist()

print(feature_names)


# In[35]:


population_size = 10
num_features = 15

population = []

for i in range(population_size):

    chromosome = np.random.randint(
        0,
        2,
        num_features
    )

    population.append(chromosome)


# In[36]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics

def fitness(chromosome):

    selected = []

    for i in range(len(chromosome)):

        if chromosome[i] == 1:
            selected.append(i)

    if len(selected) == 0:
        return 0

    X_train_sub = X_train.iloc[:, selected]
    X_test_sub = X_test.iloc[:, selected]

    scaler = StandardScaler()

    X_train_sub = scaler.fit_transform(X_train_sub)
    X_test_sub = scaler.transform(X_test_sub)

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train_sub, y_train)

    pred = model.predict(X_test_sub)

    acc = metrics.accuracy_score(
        y_test,
        pred
    )

    return acc


# In[37]:


generations = 10

best_chromosome = None
best_score = 0


# In[38]:


for generation in range(generations):

    scores = []

    for chromosome in population:

        score = fitness(chromosome)

        scores.append(score)

        if score > best_score:

            best_score = score
            best_chromosome = chromosome.copy()

    ranked = sorted(
        zip(scores, population),
        reverse=True,
        key=lambda x: x[0]
    )

    parent1 = ranked[0][1]
    parent2 = ranked[1][1]

    new_population = []

    while len(new_population) < population_size:

        point = np.random.randint(
            1,
            num_features
        )

        child = np.concatenate(
            (
                parent1[:point],
                parent2[point:]
            )
        )

        for i in range(num_features):

            if np.random.rand() < 0.05:

                child[i] = 1 - child[i]

        new_population.append(child)

    population = new_population

    print(
        "Generation",
        generation + 1,
        "Best Accuracy:",
        round(best_score, 4)
    )


# In[39]:


selected_features = []

for i in range(num_features):

    if best_chromosome[i] == 1:

        selected_features.append(
            feature_names[i]
        )

print("Selected Features:")

for f in selected_features:
    print(f)


# In[40]:


print("Best Chromosome:")
print(best_chromosome)

print()

print("Number of Features:",
      len(selected_features))

print()

print("Best Accuracy:",
      round(best_score, 4))


# In[41]:


print("All Features Accuracy:",
      round(lr_accuracy, 4))

print("GA Accuracy:",
      round(best_score, 4))


# In[ ]:




