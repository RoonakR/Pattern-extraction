from pandas import read_csv, DataFrame
from numpy import asarray, transpose, array, linalg, abs, cov, reshape
from sklearn.externals import joblib
from sklearn import mixture
from sklearn.metrics import silhouette_score
from operator import itemgetter
import sympy as sp


def get_dataset(path):
    data = read_csv(path)
    frequency = 5
    step = 60 / frequency
    nw = data.shape[0] / step
    stepFloat = data.shape[0] / nw
    return data, step, nw, stepFloat


def lagrangian_calculator(sample):
    sample = asarray(sample)
    lag = list()
    for j in range(0, sample.shape[1]):
        vector = sample[:, j]
        l = len(vector)
        lam = sp.symbols('lambda', real=True)
        g = -1
        for i in range(0, l):
            g += (vector[i]/(2*lam)) ** 2
        stationary_points = sp.solve(g, lam)
        lamb = max(stationary_points)
        vector = asarray(vector)
        lag.append(vector/(2 * lamb))
    return lag


def pca_calculator(data, step, stepFloat):
    data = asarray(data)
    pca = list()
    section_start = 0
    i = 0
    while section_start <= (data.shape[0]-step):
        section_end = section_start + step
        section = data[int(section_start):int(section_end), :]
        cov_mat = cov(transpose(section))
        eig_vals, eig_vecs = linalg.eig(cov_mat)
        eig_pairs = [(abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
        eig_pairs.sort(key=itemgetter(0))
        eig_pairs.reverse()
        pr = eig_pairs[0]
        pca.append(pr[1])
        section_start = int(i * stepFloat)
        i = i + 1

    return array(pca)


def gmm_calculator(data, n_components=2):
    data = asarray(data)
    model = mixture.GaussianMixture(n_components=n_components, covariance_type='full')
    model.fit(data.astype(float))
    return model


data, step, nw, stepFloat = get_dataset('.../wind.csv')
d = DataFrame(columns=['wind_speed', 'wind_vane'])
for i in range(0, data.shape[0]):
    if data['type'][i] == 'wind_speed':
        d = d.append({'wind_speed': data['value'][i]}, ignore_index=True)

j = 0
for i in range(0, data.shape[0]):
    if data['type'][i] == 'wind_vane':
        d['wind_vane'][j] = data['value'][i]
        j = j+1

lag = lagrangian_calculator(d)
pca = pca_calculator(lag, step, stepFloat)
model = gmm_calculator(pca, n_components=3)
cluster_labels = model.predict(pca.astype(float))
score = silhouette_score(pca.astype(float), array(cluster_labels))
print(score)
filename = 'model_wind(speed,vane).sav'
joblib.dump(model, filename)
