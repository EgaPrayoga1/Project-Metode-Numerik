from sklearn.neural_network import MLPClassifier

# Database: Gerbang Logika AND
# x = Data, y = Target
x = [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0]]  # Prediksi Nomor Anggota Kelompok
y = ['EgaPrayoga', 'IntanAgustin', 'FadliMaulana', 'MilahNurlaela', 'ShilvianaAfisah']  # Nama Anggota Kelompok

# Training and Classify
clf = MLPClassifier(solver='lbfgs', alpha=1e-2,
                    hidden_layer_sizes=(10, 5),
                    random_state=1, max_iter=1000,
                    warm_start=True)
clf.fit(x, y)

# Prediksi
print("Logika AND Metode Artificial Neural Network (ANN)")
print("Logika = Prediksi")
print("1 0 = ", clf.predict([[1, 0]]))
print("2 0 = ", clf.predict([[2, 0]]))
print("3 0 = ", clf.predict([[3, 0]]))
print("4 0 = ", clf.predict([[4, 0]]))
print("5 0 = ", clf.predict([[5, 0]]))
