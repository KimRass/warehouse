import numpy as np
from collections import Counter, defaultdict
from sklearn.neighbors import KNeighborsClassifier


def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2, ord=2)


def get_score_from_distance(distance):
    return 1 / distance


def infer_by_knn_numpy(X_test, X_train, y_train, k, weights="uniform"):
    y_test = list()
    for x_test in X_test:
        dist_label = list()
        for x_train, label in zip(X_train, y_train):
            dist = euclidean_distance(x_test, x_train)
            dist_label.append((dist, label))

        dist_label.sort()
        dist_label = dist_label[:k]

        if weights == "uniform":
            counter = Counter([i[1] for i in dist_label])
            pred = counter.most_common(1)[0][0]        
        elif weights == "distance":
            label2score = defaultdict(int)
            for dist, label in dist_label:
                score = get_score_from_distance(dist)
                label2score[label] += score
            pred = sorted(label2score.items(), key=lambda x: x[1])[-1][0]
        else:
            raise ValueError("`weights` shoud be one of the following; ('unifrom', 'distance')")

        y_test.append(pred)
    return np.array(y_test)


def infer_by_knn_sklearn(X_test, X_train, y_train, k, weights="uniform"):
    classifier = KNeighborsClassifier(
        n_neighbors=k, weights="uniform", algorithm="auto"
    )
    classifier.fit(X=X_train, y=y_train)


def main(size_tr, size_te, n_classes):
    X_tr = np.array([[np.random.rand(), np.random.rand()] for _ in range(size_tr)])
    y_tr = np.array([np.random.choice(range(n_classes)) for _ in range(size_tr)])

    X_te = np.array([[np.random.rand(), np.random.rand()] for _ in range(size_te)])

    infer_by_knn_numpy(X_te, X_tr, y_tr, k=5, weights="uniform")
    infer_by_knn_numpy(X_te, X_tr, y_tr, k=5, weights="distance")


if __name__ == "__main__":
    main(size_tr=30, size_te=10, n_classes = 5)
