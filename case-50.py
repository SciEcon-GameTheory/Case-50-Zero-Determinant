import numpy as np
import scipy.sparse.linalg as sla


class SelfishMiner:
    def __init__(self, strategy):
        self.s = strategy
        self.w = np.array([[5/8],
                           [1/2],
                           [-1/8],
                           [0]])

class LoyalMiner:
    def __init__(self):
        self.s = np.array([1,0,0,1])
        self.w = np.array([[5/8],
                           [-1/8],
                           [1/2],
                           [0]])

    def update(self, new_strategy):
        self.s = new_strategy

class ZD:
    def __init__(self, SM, LM):
        self.c = 5/8
        self.r = 5/3
        self.theta = 1/20
        self.a = 8
        self.b = -18
        self.SM = SM
        self.LM = LM

        self.x_axes = []
        self.y_axes = []
        self.M = None

    def generate_M(self):
        lm = self.LM.s; sm = self.SM.s
        a1 = lm[0]; a2 = lm[1]; a3 = lm[2]; a4 = lm[3]
        b1 = sm[0]; b2 = sm[1]; b3 = sm[2]; b4 = sm[3]

        M = np.array([[a1*b1, a1*(1-b1), (1-a1)*b1, (1-a1)*(1-b1)],
                      [a2*b3, a2*(1-b3), (1-a2)*b3, (1-a2)*(1-b3)],
                      [a3*b2, a3*(1-b2), (1-a3)*b2, (1-a3)*(1-b2)],
                      [a4*b4, a4*(1-b4), (1-a4)*b4, (1-a4)*(1-b4)]])
        M = M.astype('float64')
        self.M = M

    def find_eigenvector(self):
        M = self.M
        eval, evec = sla.eigs(M.T, k=1, which='LM')
        u = (evec/evec.sum()).real
        vector = u.T[0]

        return vector

    def generate_gama(self):
        a = self.a; b = self.b
        # calculate UL
        sL = self.find_eigenvector(); wL = self.LM.w
        UL = np.dot(sL, wL)[0]
        # calculate US
        sS = self.find_eigenvector(); wS = self.SM.w
        US = np.dot(sS, wS)[0]
        gama = a * UL + b * US

        return gama

    def generate_strategy(self, gama):
        gama = gama
        theta = self.theta; a = self.a; b = self.b
        WL = self.LM.w; WS = self.SM.w
        lm = theta * (a * WL + b * WS - gama * np.array([[1],
                                                         [1],
                                                         [1],
                                                         [1]]))

        strategy = np.transpose(lm)[0]
        # print(strategy)
        return strategy

    def main(self):
        loyal_miner = self.LM
        selfish_miner = self.SM

        for i in range(0, 50):
            self.generate_M()

            print("vector: ", self.find_eigenvector())
            # print("result: ", np.dot(self.find_eigenvector(), self.M))
            print("M:", self.M)

            eigen_vector = self.find_eigenvector()
            gama = self.generate_gama()
            new_strategy = self.generate_strategy(gama)

            UL =  np.dot(eigen_vector, loyal_miner.w)[0]
            US = np.dot(eigen_vector, selfish_miner.w)[0]

            total_reward = UL + US
            loyal_miner.update(new_strategy)

            # for drawing purpose
            self.y_axes.append(total_reward)
            self.x_axes.append(i+1)

            # print(loyal_miner.s)
            # print(self.M)
            print("Ul: ", UL)
            print("US: ", US)
            print("new_strategy", new_strategy)

            # print(loyal_miner.w)
            # print(total_reward)

if __name__ == '__main__':
    strategy = np.array([1,0,0,1])
    SM = SelfishMiner(strategy)
    LM = LoyalMiner()
    ZD = ZD(SM, LM)
    ZD.main()

    print("Done")