import numpy as np
import matplotlib.pyplot as plt

# Funções auxiliares
def sigmoid(t):
    return 1 / (1 + np.exp(-t))

def sigmoid_derivative(p):
    return sigmoid(p) * (1 - sigmoid(p))

def mse(y1, y2):
    return np.mean(np.square(y1 - y2))

def Vc_RC(t, r=5, c=0.1, vin=1):
    tau = -t / (r * c)
    vc = vin * (1 - np.exp(tau))
    return vc

# Classe da rede neural
class NeuralNetwork:
    def __init__(self, x, y, n=15):
        self.entrada = x
        self.pesos_0 = np.random.rand(self.entrada.shape[1], n)
        self.pesos_1 = np.random.rand(n, 1)
        self.y = y
        self.saida = np.zeros(y.shape)

    def feedforward(self):
        self.pot_ativ_0 = np.dot(self.entrada, self.pesos_0)
        self.camada_0 = sigmoid(self.pot_ativ_0)
        self.pot_ativ_1 = np.dot(self.camada_0, self.pesos_1)
        self.camada_1 = sigmoid(self.pot_ativ_1)
        return self.camada_1

    def backprop(self):
        erro_saida = 2 * (self.y - self.saida) * sigmoid_derivative(self.pot_ativ_1)
        d_pesos_1 = np.dot(self.camada_0.T, erro_saida)
        erro_oculta = np.dot(erro_saida, self.pesos_1.T) * sigmoid_derivative(self.pot_ativ_0)
        d_pesos_0 = np.dot(self.entrada.T, erro_oculta)

        self.pesos_0 += d_pesos_0 * 0.1
        self.pesos_1 += d_pesos_1 * 0.1

    def train(self):
        self.saida = self.feedforward()
        self.backprop()

    def predict(self, x):
        self.entrada = x
        self.saida = self.feedforward()
        return self.saida

# Dados
t = np.arange(0, 3, 0.1)
vc = Vc_RC(t)

# Normalização
t = t / np.amax(t)

# Divisão treino/teste
porcent_treino = 60
tam_treino = int(len(vc) * porcent_treino / 100)

x_train = t[:tam_treino].reshape(-1, 1)
y_train = vc[:tam_treino].reshape(-1, 1)
x_test = t[tam_treino:].reshape(-1, 1)
y_test = vc[tam_treino:]

# Criação da rede
nn = NeuralNetwork(x_train, y_train, n=15)

# Treinamento
erro = []
for i in range(500):
    saida = nn.feedforward()
    erro.append(mse(y_train, saida))
    nn.train()

# Gráfico do erro
plt.plot(erro, 'r')
plt.xlabel("Época")
plt.ylabel("Erro quadrático médio")
plt.title("Evolução do erro")
plt.grid()
plt.show()

# Previsão e gráfico final
saida_rede = nn.predict(x_test)

plt.plot(x_test.flatten(), y_test, 'b', label="Tensão real")
plt.plot(x_test.flatten(), saida_rede.flatten(), 'r', label="Tensão prevista")
plt.plot(x_train.flatten(), y_train.flatten(), 'g--', label="Treinamento")
plt.xlabel("Tempo (normalizado)")
plt.ylabel("Tensão (Vc)")
plt.title("Modelo de circuito RC")
plt.legend()
plt.grid()
plt.show()
