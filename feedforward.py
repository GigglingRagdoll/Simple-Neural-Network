import numpy as np

class FeedForward:
    def __init__(self, layer_sizes, f, fprime, learn_rate=1):
        self.layer_sizes = layer_sizes
        self.layers = len(self.layer_sizes)
        self.w_matrices = self.init_weights()

        self.f = f
        self.fprime = fprime

        self.learn_rate = learn_rate

    def init_weights(self):
        matrices = []

        for i in range(self.layers-1):
            rows = self.layer_sizes[i+1]
            cols = self.layer_sizes[i]

            matrix = np.random.uniform(-0.1,0.1,size=(rows, cols))
            matrices.append(matrix)

        return matrices

    def predict(self, X):
        return self.feed_helper(X, train=False)[0][-1][0, 0]

    def feed_helper(self, X, train=True):
        Ys = [X]
        Zs = []

        for W in self.w_matrices:
            Y = Ys[-1]
            Z = np.dot(W, Y)
            if W is self.w_matrices[-1]:
                Y = Z
            else:
                Y = self.f(Z)# if np.random.uniform() > 0.5 else np.zeros(Z.shape)
            
            Ys.append(Y)
            Zs.append(Z)

        return Ys, Zs

    def feed(self, Xs):
        # easier to backpropagate if we keep track
        # of the intermediate steps
        Yn = [ [] for X in range(self.layers) ]
        Zn = [ [] for Z in range(self.layers-1) ]
        
        for X in Xs:
            Ys, Zs = self.feed_helper(X)

            for i in range(len(Ys)):
                if Yn[i] == []:
                   Yn[i] = Ys[i]

                else:
                   Yn[i] = np.append(Yn[i], Ys[i], axis=1)

            for i in range(len(Zs)):
                if Zn[i] == []:
                   Zn[i] = Zs[i]

                else:
                   Zn[i] = np.append(Zn[i], Zs[i], axis=1)

        return Yn, Zn

    def backprop(self, T, Yn, Zn):
        # calculate delta for output layer
        Y = Yn.pop(-1).T
        Z = Zn.pop(-1).T
        delta = np.multiply((Y-T), self.fprime(Z))

        # calculate change for output weights
        Y = Yn.pop(-1)
        change = np.dot(Y, delta)
        changes = [change]

        # copy weights here
        Ws = list(np.copy(self.w_matrices))

        while Yn:
            # calculate delta for hidden layer
            Z = Zn.pop(-1)
            W = Ws.pop(-1)
            Y = Yn.pop(-1)

            delta = np.dot(delta, W)
            delta = np.multiply(delta, self.fprime(Z.T))
            
            # calculate change for hidden layer weights
            change = np.dot(Y, delta)
            changes.append(change)

        for i in range(len(changes)):
            self.w_matrices[-(i+1)] = self.w_matrices[-(i+1)] - self.learn_rate*changes[i].T
            
    def train(self, Xs, T, acc=95, max_iter=2000):
        for i in range(max_iter):
            Yn, Zn = self.feed(Xs)
            #print(Yn[-1])
            self.backprop(T, Yn, Zn)
            #if self.score(T, Xs) >= acc:
            #    break

    def score(self, T, Xs):
        score = 0

        Yn, Zn = self.feed(Xs)
        Ys = Yn[-1].T
        for i in range(len(T)):
            score += 100 - np.abs((Ys[i]-T[i])/T[i])*100

        #print(score, len(T))
        #score = (len(T)*100 - score)/len(T)
        score /= len(T)
        #print(score)
        return score[0, 0]
        
def tanhprime(x):
    return np.multiply(1/np.cosh(x), 1/np.cosh(x))
    
