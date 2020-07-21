#####################################################################################################
# Πρόβλημα Δυαδικής Ταξινόμησης του Iris Dataset με Perceptron, Adaline και Λύση Ελαχίστων Τετραγώνων
#####################################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
def perceptron(xtrain, ttrain, maxepochs, beta):
       # Αρχικοποιώ στο w τυχαίες τιμές
       w = np.random.randn(len(xtrain[0]),1)
       flag = 1
       epoch = 1
       plt.figure()
       plt.ion()    
       while (flag == 1) & (epoch <= maxepochs):
              flag = 0
              # Αρχικοποιώ τους πίνακες v, u
              u = np.zeros(len(xtrain), dtype=float)
              v = np.zeros(len(xtrain), dtype=int)
              for i, x in enumerate(xtrain):
                     u[i] = np.dot(xtrain[i], w)
                     if (u[i] >= 0):
                            v[i] = 1
                     else:
                            v[i] = 0
                     if (v[i] != ttrain[i]):
                            for j in range(len(xtrain[0])):
                                   w[j] = w[j] + beta * (ttrain[i] - v[i]) * xtrain[i,j]
                            flag = 1
              plt.plot(v, 'ro', ttrain, 'b.', markersize=4)
              plt.title("Perceptron")
              plt.show()
              plt.pause(1)
              epoch += 1
              if (epoch <= maxepochs) & (flag == 1):
                     plt.clf()
       return w

def perceptron_w(xtest, ttest, w):
       plt.figure()
       u = np.zeros(len(xtest), dtype=float)
       v = np.zeros(len(xtest), dtype=int)
       for i, x in enumerate(xtest):
              u[i] = np.dot(xtest[i], w)
              if (u[i] >= 0):
                     v[i] = 1
              else:
                     v[i] = 0
       plt.plot(v, 'ro', ttest, 'b.', markersize=4)
       plt.title("Perceptron")
       plt.show()


def adaline(xtrain, ttrain, maxepochs, beta, minmse):
       # Αρχικοποιώ στο w τυχαίες τιμές
       w = np.random.randn(len(xtrain[0]),1)
       flag = 0
       epoch = 1
       mse = []
       # Αρχικοποιώ τους πίνακες v, delta
       v = np.zeros(len(xtrain), dtype=np.float)
       delta = np.zeros(len(xtrain), dtype=np.float)
       plt.figure()
       plt.ion()  
       while (flag == 0) & (epoch <= maxepochs):
              sfalma = 0
              for i, x in enumerate(xtrain):
                     u = np.dot(xtrain[i], w)
                     if (u >= 0):
                            v[i] = 1
                     else:
                            v[i] = -1
                     delta[i] = ttrain[i] - v[i]
                     sfalma = sfalma + delta[i]**2
                     for j in range(len(xtrain[0])):
                            w[j] = w[j] + (beta * delta[i] * xtrain[i,j])
              if (sfalma / len(xtrain) <= minmse):
                     flag = 1
              mse.append(sfalma / len(xtrain))
              plt.plot(v, 'ro', ttrain, 'b.', markersize=4)
              plt.show()
              plt.title("Adaline")
              plt.pause(1)
              epoch += 1
              if (epoch <= maxepochs) & (flag == 0):
                     plt.clf()
       return w
            
def adaline_w(xtest, ttest, w):
       plt.figure()
       u = np.zeros(len(xtest), dtype=float)
       v = np.zeros(len(xtest), dtype=int)
       for i, x in enumerate(xtest):
              u[i] = np.dot(xtest[i], w)
              if (u[i] >= 0):
                     v[i] = 1
              else:
                     v[i] = -1
       plt.plot(v, 'ro', ttest, 'b.', markersize=4)
       plt.title("Perceptron")
       plt.show()

# Διαβάζω το αρχείο προτύπων
data = pd.read_csv('iris.data', header=None).values
# Θέτω τον αριθμό των προτύπων και τον αριθμό των χαρακτηριστικών
NumberOfPatterns, NumberOfAttributes = np.shape(data)
# Κάνω το απαραίτητο χώρισμα
x = data[0:,:4]
pattern = data[0:, 4:]
labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['blue', 'red', 'green']
counter_end = 50
counter_start = 0
# Εμφανίζω τα πρότυπα με βάση το όνομα
for color, label in zip(colors, labels):
       plt.scatter(x[counter_start:counter_end,0], x[counter_start:counter_end,2], color=color, label=label, s=5)
       counter_start+=50
       counter_end+=50
plt.title('Γράφημα όλων των προτύπων')
plt.legend(loc=4)
plt.show()
# Δημιουργώ τον πίνακα των στόχων και προσθέτω την απαραίτητη στήλη στον πίνακα προτύπων
t = np.zeros(NumberOfPatterns, dtype=int)
ones = np.ones(NumberOfPatterns, dtype=int)
x = np.column_stack((x,ones))
ans = 'y'
while (ans == 'y'):
       choice = int(input ('1. Διαχωρισμός Iris-setosa από (Iris-versicolor και Iris-virginica)\n' +
                       '2. Διαχωρισμός Iris-virginica από (Iris-setosa και Iris-versicolor)\n' +
                       '3. Διαχωρισμός Iris-versicolor από (Iris-setosa και Iris-virginica)\n'))
       # Ανάλογα με την επιλογή φτιάχνω τους στόχους
       if choice == 1:
              dictionary = {
                     "Iris-setosa": 1,
                     "Iris-versicolor": 0,
                     "Iris-virginica": 0
              }  
              for i in range(NumberOfPatterns):
                     t[i] = dictionary[pattern.item(i)]
       elif choice == 2:
              dictionary = {
                     "Iris-setosa": 0,
                     "Iris-versicolor": 0,
                     "Iris-virginica": 1
              }       
              for i in range(NumberOfPatterns):
                     t[i] = dictionary[pattern.item(i)]
       else :
              dictionary = {
                     "Iris-setosa": 0,
                     "Iris-versicolor": 1,
                     "Iris-virginica": 0
              }       
              for i in range(NumberOfPatterns):
                     t[i] = dictionary[pattern.item(i)]
       # Χωρίζω σε πρότυπα εκπαίδευσης και test, το ίδιο και για τους στόχους
       xtrain = np.concatenate((x[0:40], x[50:90], x[100:140]), axis=0)
       xtest = np.concatenate((x[40:50], x[90:100], x[140:150]), axis=0)
       ttrain = np.concatenate((t[0:40], t[50:90], t[100:140]), axis=0)
       ttest = np.concatenate((t[40:50], t[90:100], t[140:150]), axis=0)
       colors_2 = ['red', 'blue']
       # Εμφανίζω τα πρότυπα εκπαίδευσης και test
       for color in colors_2:
              if color == 'red':
                     plt.scatter(xtest[:,0], xtest[:,2], color=color, s=5)
              elif color == 'blue':
                     plt.scatter(xtrain[:,0], xtrain[:,2], color=color, s=5)
              else:
                     break
       plt.title("Πρότυπα εκπαίδευσης και του test")
       plt.show()
       while (True):
              choice_2 = int(input ('1. Υλοποίηση με Perceptron\n' +
                                    '2. Υλοποίηση με Adaline\n' +
                                    '3. Υλοποίηση με Λύση Ελαχίστων Τετραγώνων\n' +
                                    '4. Επιστροφή στο αρχικό Menu\n'))
              # Ανάλογα με την 2η επιλογή τρέχω το Perceptron ή το Adaline ή την Λύση Ελαχίστων Τετραγώνων
              if (choice_2 == 1):
                     # Διαβάζω max εποχές και συντελεστή beta
                     maxepochs = int(input ('Δώσε μια ακέραια τιμή για το Μέγιστο Αριθμό επαναλήψεων: '))
                     beta = float(input ('Δώσε μια πραγματική τιμή για το Συντελεστή Εκπαίδευσης: '))

                     # Perceptron για xtrain
                     w = perceptron(xtrain, ttrain, maxepochs, beta)

                     # Perceptron για xtest
                     perceptron_w(xtest, ttest, w)

                     for k in range(9):
                            # Perceptron με train_test_split σε X_train, X_test
                            X_train, X_test, y_train, y_test = ms.train_test_split(x, t, test_size=0.1)

                            # Εμφάνιση των χωρισμένων προτύπων
                            plt.figure()
                            plt.subplot(311)
                            plt.plot(X_train[:,0], X_train[:,2], 'b.')
                            plt.title("X-train με train_test_split - Perceptron - " + str(k+1) + "ο fold")
                            plt.subplot(312)
                            plt.plot(X_test[:,0], X_test[:,2], 'r.')
                            plt.title("X-test με train_test_split - Perceptron - " + str(k+1) + "ο fold")
                            plt.tight_layout()
                            plt.show(block=True)

                            # Perceptron για X_train με train_test_split
                            w = perceptron(X_train, y_train, maxepochs, beta)

                            # Perceptron για X_test με train_test_split
                            perceptron_w(X_test, y_test, w)
              elif (choice_2 == 2):
                     # Αλλαγή πινάκων στόχου απο 0 σε -1
                     ttrain1 = ttrain.copy()
                     ttest1 = ttest.copy()
                     ttrain1[ttrain1 == 0] = -1
                     ttest1[ttest1 == 0] = -1

                     # Διαβάζω max εποχές, συντελεστή beta και το ελάχιστο σφάλμα
                     maxepochs = int(input ('Δώσε μια ακέραια τιμή για το Μέγιστο Αριθμό επαναλήψεων: '))
                     beta = float(input ('Δώσε μια πραγματική τιμή για το Συντελεστή Εκπαίδευσης: '))
                     minmse = float(input ('Δώσε μια πραγματική τιμή για το Ελάχιστο Σφάλμα: '))

                     # Adaline για xtrain
                     w = adaline(xtrain, ttrain1, maxepochs, beta, minmse)

                     # Adaline για xtest
                     adaline_w(xtest, ttest1, w)

                     for k in range(9):
                            # Adaline με train_test_split σε X_train, X_test
                            # Αλλαγή πινάκων στόχου απο 0 σε -1
                            X_train, X_test, y_train, y_test = ms.train_test_split(x, t, test_size=0.1)
                            y_test[y_test == 0] = -1
                            y_train[y_train == 0] = -1

                            # Εμφάνιση των χωρισμένων προτύπων
                            plt.figure()
                            plt.subplot(311)
                            plt.plot(X_train[:,0], X_train[:,2], 'b.')
                            plt.title("X-train με train_test_split - Adaline - " + str(k+1) + "ο fold")
                            plt.subplot(313)
                            plt.plot(X_test[:,0], X_test[:,2], 'r.')
                            plt.title("X-test με train_test_split - Adaline - " + str(k+1) + "ο fold")
                            plt.tight_layout()
                            plt.show(block=True)

                            # Adaline για X_train με train_test_split
                            w = adaline(X_train, y_train, maxepochs, beta, minmse)

                            # Adaline για X_test με train_test_split
                            adaline_w(X_test, y_test, w)
              elif (choice_2 == 3):
                     # Αλλαγή πινάκων στόχου απο 0 σε -1 και θέτω ως τύπο δεδομένων το float64 για να λειτουργήσει το pinv
                     ttrain1 = ttrain.copy()
                     ttest1 = ttest.copy()
                     ttrain1[ttrain1 == 0] = -1
                     ttest1[ttest1 == 0] = -1
                     xtrain = xtrain.astype('float64')
                     xtest = xtest.astype('float64')

                     # Λύση ελαχίστων τετραγώνων για xtrain
                     pinv_xtrain = np.linalg.pinv(xtrain)
                     w = np.dot(pinv_xtrain, ttrain1)
                     v = np.zeros(len(xtrain), dtype=np.float)
                     for i, z in enumerate(xtrain):
                            u = np.dot(xtrain[i], w)
                            if (u >= 0):
                                   v[i] = 1
                            else:
                                   v[i] = -1
                     plt.figure()
                     plt.plot(v, 'ro', ttrain1, 'b.', markersize=4)
                     plt.title("xtrain απλό - Λύση Ελαχίστων Τετραγώνων")
                     plt.show()

                     # Λύση ελαχίστων τετραγώνων για xtest
                     pinv_xtest = np.linalg.pinv(xtest)
                     # w = np.dot(pinv_xtest, ttest1)
                     v = np.zeros(len(xtest), dtype=np.float)
                     for i, z in enumerate(xtest):
                            u = np.dot(xtest[i], w)
                            if (u >= 0):
                                   v[i] = 1
                            else:
                                   v[i] = -1
                     plt.figure()
                     plt.plot(v, 'ro', ttest1, 'b.', markersize=4)
                     plt.title("xtest απλό - Λύση Ελαχίστων Τετραγώνων")
                     plt.show(block=True)

                     for k in range(9):
                            # Λύση ελαχίστων τετραγώνων με train_test_split σε X_train, X_test
                            # Αλλαγή πινάκων στόχου απο 0 σε -1 και θέτω ως τύπο δεδομένων το float64 για να λειτουργήσει το pinv
                            X_train, X_test, y_train, y_test = ms.train_test_split(x, t, test_size=0.1)
                            y_test[y_test == 0] = -1
                            y_train[y_train == 0] = -1
                            X_train = X_train.astype('float64')
                            X_test = X_test.astype('float64')

                          # Εμφάνιση των χωρισμένων προτύπων
                            plt.figure()
                            plt.subplot(311)
                            plt.plot(X_train[:,0], X_train[:,2], 'b.')
                            plt.title("X-train με train_test_split - Λύση Ελαχίστων Τετραγώνων - " + str(k+1) + "ο fold")
                            plt.subplot(313)
                            plt.plot(X_test[:,0], X_test[:,2], 'r.')
                            plt.title("X-test με train_test_split - Λύση Ελαχίστων Τετραγώνων - " + str(k+1) + "ο fold")
                            plt.tight_layout()
                            plt.show()

                            # Λύση ελαχίστων τετραγώνων για X_train με train_test_split
                            pinv_x_train = np.linalg.pinv(X_train)
                            w = np.dot(pinv_x_train, y_train)
                            v = np.zeros(len(X_train), dtype=np.float)
                            for i, z in enumerate(X_train):
                                   u = np.dot(X_train[i], w)
                                   if (u >= 0):
                                          v[i] = 1
                                   else:
                                          v[i] = -1
                            plt.figure()
                            plt.plot(v, 'ro', y_train, 'b.', markersize=4)
                            plt.title("X-train - Λύση Ελαχίστων Τετραγώνων - " + str(k+1) + "ο fold")
                            plt.show()

                            # Λύση ελαχίστων τετραγώνων για X_test με train_test_split
                            pinv_x_test = np.linalg.pinv(X_test)
                            # w = np.dot(pinv_x_test, y_test)
                            v = np.zeros(len(X_test), dtype=np.float)
                            for i, z in enumerate(X_test):
                                   u = np.dot(X_test[i], w)
                                   if (u >= 0):
                                          v[i] = 1
                                   else:
                                          v[i] = -1
                            plt.figure()
                            plt.plot(v, 'ro', y_test, 'b.', markersize=4)
                            plt.title("X-test - Λύση Ελαχίστων Τετραγώνων - " + str(k+1) + "ο fold")
                            plt.show(block=True)
              elif (choice_2 == 4):
                     break
       ans = input ('Αν θέλεις να συνεχίσεις γράψε \'y\'\n')