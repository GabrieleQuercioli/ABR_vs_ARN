import math
import random
import pickle
from timeit import default_timer as timer
import numpy as np
from random import sample
import matplotlib.pyplot as plt


######## CREATION OF ARRAY OF INPUTS ##########

def randomize(dim):
    A = np.random.randint(0, dim * 10, dim)  # può scegliere numeri tra 0 e dim*10
    # print("random: ", A)
    return A


def orderedVec(dim):
    A = np.random.randint(0, dim * 10, dim)
    A = sorted(A)
    # print("ordered: ", A)
    return A


def reversedVec(dim):
    A = np.random.randint(0, dim * 10, dim)
    A = sorted(A, reverse=True)
    # print("ordered dec", A)
    return A


############ ALBERO BINARIO DI RICERCA ############

class AbrNode:
    def __init__(self, key):
        self.p = None
        self.left = None
        self.right = None
        self.key = key


class AbrTree:
    def __init__(self):
        self.root = None

    def setRoot(self, x):
        self.root = x

    def getRoot(self):
        return self.root

    def insert(self, key):
        z = AbrNode(key)
        y = None
        x = self.root
        while x is not None:  # l'input diventa sempre una foglia dell'albero corrente
            y = x  # tiene traccia del padre del nodo corrente x
            if z.key < x.key:
                x = x.left  # se l'input è più piccolo va a sx dell'albero
            else:
                x = x.right
        z.p = y  # imposta y come padre di z
        if y is None:  # se l'albero è vuoto
            self.root = z
        elif z.key < y.key:
            y.left = z  # se l'input è più piccolo imposta z come figlio sx di y
        else:
            y.right = z

    def inorderTreeWalk(self, x):               # attraversamento simmetrico dell'albero
        if x is not None:
            self.inorderTreeWalk(x.left)
            print(x.key)
            self.inorderTreeWalk(x.right)

    def treeSearch(self, x, key):
        while x is not None and x.key != key:
            if key < x.key:
                x = x.left
            else:
                x = x.right
        return x

    def search(self, key):
        return self.treeSearch(self.root, key)

    def maxDepth(self, x):
        if x is None:
            return 0
        else:
            leftHeight = self.maxDepth(x.left)
            rightHeight = self.maxDepth(x.right)
            return max(leftHeight, rightHeight) + 1


########## ALBERO ROSSO-NERO ################

class ArnNode(AbrNode):         # aggiunge il colore al nodo di un albero Binario
    def __init__(self, key):
        super().__init__(key)
        self.color = None

    def setColor(self, color):
        if color == "R" or color == 1:              # nodo rosso
            self.color = 1
        if color == "N" or color == 0:              # nodo nero
            self.color = 0
        else:
            print("Color error")


class ArnTree:
    def __init__(self):
        self.nil = ArnNode(None)
        self.nil.color = 0
        self.root = self.nil

    def inorderTreeWalk(self, x):
        if x != self.nil:
            self.inorderTreeWalk(x.left)
            print(x.key)
            self.inorderTreeWalk(x.right)

    def insert(self, k):
        z = ArnNode(k)
        y = self.nil                      # funziona anche sostituiendo None a self.nil?
        x = self.root
        while x != self.nil:
            y = x                          # tiene traccia del padre
            if z.key < y.key:
                x = x.left
                # print(k, "left ARN")
            else:
                x = x.right
                # print(k, "right ARN")
        z.p = y                          # assegna il padre al nuovo nodo z
        if y == self.nil:                # se l'albero era vuoto
            self.root = z
        elif z.key < y.key:              # assegna la corretta pos del nuovo nodo come figlio di y
            y.left = z
        else:
            y.right = z
        z.left = self.nil               # z deve essere una foglia
        z.right = self.nil
        z.color = 1                     # setta a rosso il colore del nuovo nodo
        self.insert_fixup(z)            # funzione che serve a far soddisfare le 5 proprietà di ARN a insert

    def insert_fixup(self, z):          # dopo insert il nuovo nodo è al giusto posto ma non soddisfa ancora le proprietà di ARN
        while z.p.color == 1:           # finchè il padre di z è rosso (z non potrebbe essere rosso a sua volta in tal caso)
            if z.p == z.p.p.left:       # se il padre di z è il figlio sx di suo padre
                y = z.p.p.right
                if y.color == 1:        # se lo zio di z è rosso rendo neri lui e il padre di z, e rosso il nonno
                    z.p.color = 0
                    y.color = 0
                    z.p.p.color = 1
                    z = z.p.p
                else:
                    if z == z.p.right:      # se z è il figlio dx
                        z = z.p
                        self.leftRotate(z)  # facendo un rotazione a sx ora z è il padre e z.p è suo figlio sx
                    z.p.color = 0           # il padre di z diventa nero e il nonno rosso
                    z.p.p.color = 1
                    self.rightRotate(z.p.p)   # ora il nonno è figlio dx del padre di z, e z il figlio sx
            else:                               # se il padre di z è il figlio dx di suo padre
                y = z.p.p.left                 # da qui in poi è speculare (cambio dx con sx e viceversa)
                if y.color == 1:
                    z.p.color = 0
                    y.color = 0
                    z.p.p.color = 1
                    z = z.p.p
                else:
                    if z == z.p.left:
                        z = z.p
                        self.rightRotate(z)
                    z.p.color = 0
                    z.p.p.color = 1
                    self.leftRotate(z.p.p)
        self.root.color = 0                     # la radice è sempre nera

    def leftRotate(self, x):
        # print("left Rotate")
        y = x.right
        x.right = y.left                        # sposta il sotto albero sx del figlio dx di x nel figlio dx di x
        if y.left != self.nil:                  # se y ha un sotto albero sx allora x ne diventa il padre
            y.left.p = x
        y.p = x.p                               # collega il padre di x a y
        if x.p == self.nil:                     # se x non ha padre y diventa root
            self.root = y
        elif x == x.p.left:                     # se x era figlio sx y diventa il figlio sx
            x.p.left = y
        else:
            x.p.right = y
        y.left = x                              # x diventa figlio sx e y il padre
        x.p = y

    def rightRotate(self, y):
        # print("right rotate")
        x = y.left
        y.left = x.right
        if x.right != self.nil:
            x.right.p = y
        x.p = y.p
        if y.p.key == self.nil:                 # giusto key?
            self.root = x
        elif y == y.p.left:
            y.p.left = x
        else:
            y.p.right = x
        x.right = y
        y.p = x

    def treeSearch(self, x, key):
        while x.key is not None and x.key != key:
            if key < x.key:
                x = x.left
            else:
                x = x.right
            return x

    def search(self, key):
        return self.treeSearch(self.root, key)

    def maxDepth(self, x):
        if x is None:
            return 0
        else:
            leftHeight = self.maxDepth(x.left)
            rightHeight = self.maxDepth(x.right)
            return max(leftHeight, rightHeight) + 1


########## TESTING ##########################

# insert e search possono essere usati anche da un ARN?
def timeInsert(T, A):  # fa insert di un array in un ABR e calcola il tempo tot che ci mette
    startTimer = timer()
    for i in range(0, len(A)):
        T.insert(A[i])
    time = timer() - startTimer
    return time


def timeSearch(T, nums):
    Time = []
    startTimer = timer()
    for i in range(0, len(nums)):
        T.search(nums[i])
    endTimer = timer()
    Time.append(endTimer - startTimer)
    return np.mean(Time)


def AbrRandom():        # costo stimato O(h) con h altezza dell'albero, nel caso migliore dove l'ordine di input
    Tins = []           # mantiene bilanciato l'albero h = log(n)
    Ts = []
    AvgIns = []
    AvgS = []
    dim = 10
    for i in range(0, 5):
        numsToSearch = randomize(dim)       # crea un vettore di numeri casuali da ricercare in seq in ABR
        for j in range(0, 5):           # esegue 5 prove di creazione di un ABR con stessa dim
            A = randomize(dim)
            ABR = AbrTree()
            Tins.append(timeInsert(ABR, A))
            # print("time to insert: ", timeInsert(ABR, A))
            Ts.append(timeSearch(ABR, numsToSearch))
            # print("time to search: ", timeSearch(ABR, numsToSearch))
        sumI = 0
        sumS = 0
        for k in range(0, len(Tins)):
            sumI += Tins[k]
            sumS += Ts[k]
        mediaI = sumI / len(Tins)          # salva il tempo medio per creare un ABR con stessa dim
        mediaS = sumS / len(Ts)          # salva il tempo medio per cercare in ABR con stessa dim
        AvgIns.append(mediaI)               # salva i tempi medi per dim crescente
        print("Avg random insert", AvgIns)
        AvgS.append(mediaS)
        print("Avg random search", AvgS)
        dim += 1000
        Tins[0:len(Tins)] = []              # resetta il vettore di tempi parziali
        Ts[0:len(Ts)] = []
    pickle.dump(AvgIns, open("m1.p", "wb"))     # salva il valore così che poi lo possa usare per i plots (wb modalità write)
    pickle.dump(AvgS, open("m2.p", "wb"))       # dato che sono 2 i parametri da ritornare non potevo usare return


def ArnRandom():        # costo stimato è di O(log(n)) per insert + O(log(n)) per il fixup (uguale per il caso ordinato)
    Tins = []           # costo ricerca anche O(log(n))
    Ts = []
    AvgIns = []
    AvgS = []
    dim = 10
    for i in range(0, 5):
        numsToSearch = randomize(dim)       # crea un vettore di numeri casuali da ricercare in seq in ARN
        for j in range(0, 5):           # esegue 5 prove di creazione di un ARN con stessa dim
            A = randomize(dim)
            ARN = ArnTree()
            Tins.append(timeInsert(ARN, A))
            Ts.append(timeSearch(ARN, numsToSearch))
        sumI = 0
        sumS = 0
        for k in range(0, len(Tins)):
            sumI += Tins[k]
            sumS += Ts[k]
        mediaI = sumI / len(Tins)          # salva il tempo medio per creare un ARN con stessa dim
        mediaS = sumS / len(Ts)          # salva il tempo medio per cercare in ARN con stessa dim
        AvgIns.append(mediaI)               # salva i tempi medi per dim crescente
        print("Avg random insert", AvgIns)
        AvgS.append(mediaS)
        print("Avg random search", AvgS)
        dim += 1000
        Tins[0:len(Tins)] = []              # resetta il vettore di tempi parziali
        Ts[0:len(Ts)] = []
    pickle.dump(AvgIns, open("m6.p", "wb"))
    pickle.dump(AvgS, open("m8.p", "wb"))


def AbrOrdered():       # caso peggiore: l'albero è sempre sbilanciato, il costo stimato è O(h) con h = n numero di elementi input
    Tins = []
    Ts = []
    AvgIns = []
    AvgS = []
    dim = 10
    for i in range(0, 5):
        numsToSearch = randomize(dim)       # crea un vettore di numeri casuali da ricercare in seq in ABR
        for j in range(0, 5):           # esegue 5 prove di creazione di un ABR con stessa dim
            A = orderedVec(dim)             # crea un array ordinato
            ABR = AbrTree()
            Tins.append(timeInsert(ABR, A))
            Ts.append(timeSearch(ABR, numsToSearch))
        sumI = 0
        sumS = 0
        for k in range(0, len(Tins)):
            sumI += Tins[k]
            sumS += Ts[k]
        mediaI = sumI / len(Tins)          # salva il tempo medio per creare un ABR con stessa dim
        mediaS = sumS / len(Ts)          # salva il tempo medio per cercare in ABR con stessa dim
        AvgIns.append(mediaI)               # salva i tempi medi per dim crescente
        print("Avg inc insert", AvgIns)
        AvgS.append(mediaS)
        print("Avg inc search", AvgS)
        dim += 1000
        Tins[0:len(Tins)] = []              # resetta il vettore di tempi parziali
        Ts[0:len(Ts)] = []
    pickle.dump(AvgIns, open("m5.p", "wb"))
    pickle.dump(AvgS, open("m3.p", "wb"))


def ArnOrdered():
    Tins = []
    Ts = []
    AvgIns = []
    AvgS = []
    dim = 10
    for i in range(0, 5):
        numsToSearch = randomize(dim)       # crea un vettore di numeri casuali da ricercare in seq in ABR
        for j in range(0, 5):           # esegue 5 prove di creazione di un ABR con stessa dim
            A = orderedVec(dim)             # crea un array ordinato
            ARN = ArnTree()
            Tins.append(timeInsert(ARN, A))
            Ts.append(timeSearch(ARN, numsToSearch))
        sumI = 0
        sumS = 0
        for k in range(0, len(Tins)):
            sumI += Tins[k]
            sumS += Ts[k]
        mediaI = sumI / len(Tins)          # salva il tempo medio per creare un ABR con stessa dim
        mediaS = sumS / len(Ts)          # salva il tempo medio per cercare in ABR con stessa dim
        AvgIns.append(mediaI)               # salva i tempi medi per dim crescente
        print("Avg inc insert", AvgIns)
        AvgS.append(mediaS)
        print("Avg inc search", AvgS)
        dim += 1000
        Tins[0:len(Tins)] = []              # resetta il vettore di tempi parziali
        Ts[0:len(Ts)] = []
    pickle.dump(AvgIns, open("m7.p", "wb"))
    pickle.dump(AvgS, open("m9.p", "wb"))


######### RUNNING TESTS & CREATING PLOTS ###########

def runTests():
    print("ABR incremental")
    AbrOrdered()
    print("ABR random")
    AbrRandom()
    print("ARN incremental")
    ArnOrdered()
    print("ARN random")
    ArnRandom()


def showPlots():
    runTests()
    A = pickle.load(open("m1.p", "rb"))         # riprendo i valori che avevo messo nei pickle
    B = pickle.load(open("m6.p", "rb"))         # rb è per modalità read
    C = pickle.load(open("m5.p", "rb"))
    D = pickle.load(open("m7.p", "rb"))
    E = pickle.load(open("m2.p", "rb"))
    F = pickle.load(open("m3.p", "rb"))
    G = pickle.load(open("m8.p", "rb"))
    H = pickle.load(open("m9.p", "rb"))
    y = [10, 1010, 2010, 3010, 4010]
    plt.plot(y, A,  label="ABR: inserimento elementi random")
    plt.plot(y, B,   label="ARN: inserimento elementi random")
    plt.legend()
    plt.xlabel("Numero elementi")
    plt.ylabel("Tempo esecuzione")
    plt.show()

    plt.plot(y, C, label="ABR: inserimento elementi ordinati")
    plt.plot(y, D, label="ARN: inserimento elementi ordinati")
    plt.legend()
    plt.xlabel("Numero elementi")
    plt.ylabel("Tempo esecuzione")
    plt.show()

    plt.plot(y, E, label="ABR: ricerca random")
    plt.plot(y, G, label="ARN: ricerca random")
    plt.legend()
    plt.xlabel("Numero Elementi")
    plt.ylabel("Tempo di Esecuzione")
    plt.show()

    plt.plot(y, F, label="ABR: ricerca ordinata")
    plt.plot(y, H, label="ARN: ricerca ordinata")
    plt.legend()
    plt.xlabel("Numero Elementi")
    plt.ylabel("Tempo di Esecuzione")
    plt.show()


if __name__ == '__main__':
    showPlots()
