import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  

def carregar_dados(caminho_arquivo):
    dados = np.loadtxt(caminho_arquivo)
    entradas_x = dados[:, :2]
    desejado_d = dados[:, 2]
    return entradas_x, desejado_d

def treinar_adaline(entradas_x, desejado_d, taxa_aprendizado, max_epocas, tolerancia):
    num_amostras = entradas_x.shape[0]
    
    bias = -np.ones((num_amostras, 1))
    # Adiciona o -1 para cada amostra, criando uma nova matriz com o Bias na primeira coluna e as entradas_x nas colunas seguintes
    x_com_bias = np.hstack((bias, entradas_x))
    
    pesos = np.random.rand(3)
    pesos_iniciais = np.copy(pesos) 
    
    historico_eqm = []
    
    for epoca in range(max_epocas):
        erro_quadratico_soma = 0
        
        for i in range(num_amostras):
            amostra_atual = x_com_bias[i]
            d_atual = desejado_d[i]

            u = np.dot(pesos, amostra_atual)
            erro = d_atual - u
            pesos = pesos + taxa_aprendizado * erro * amostra_atual   #Delta Rule
            erro_quadratico_soma += (erro ** 2)
            
        eqm = erro_quadratico_soma / num_amostras
        historico_eqm.append(eqm)
        
        if eqm <= tolerancia:
            break
            
    numero_de_epocas_que_rodou = epoca + 1
    return pesos_iniciais, pesos, numero_de_epocas_que_rodou, historico_eqm

def classificar (amostra_x, pesos):
    amostra_bias = np.insert(amostra_x, 0, -1)  
    u = np.dot(pesos, amostra_bias)
    return 1 if u >= 0.5 else 0


x_treino, d_treino = carregar_dados('dados2-tra.txt')
x_teste, d_teste = carregar_dados('dados2-tst.txt')

taxa_aprendizado = 0.01
max_epocas = 50
tolerancia = 0.001

pesos_salvos = {}
historico_eqm_salvo = {}

for i in range(1, 6):
    nome_treino = f"Treino {i}"
    pesos_iniciais, pesos_finais, epocas, historico_eqm = treinar_adaline(x_treino, d_treino, taxa_aprendizado, max_epocas, tolerancia)

    pesos_salvos[nome_treino] = pesos_finais
    historico_eqm_salvo[nome_treino] = historico_eqm
    erro_final = historico_eqm[-1]
    
    print(f"\nTreinamento {nome_treino}:")
    print(f"Vetor Inicial: [{pesos_iniciais[0]:.4f}, {pesos_iniciais[1]:.4f}, {pesos_iniciais[2]:.4f}]")
    print(f"Vetor Final:   [{pesos_finais[0]:.4f}, {pesos_finais[1]:.4f}, {pesos_finais[2]:.4f}]")
    print(f"Épocas:        {epocas}")
    print(f"Taxa (eta):    {taxa_aprendizado}")
    print(f"Erro Final:    {erro_final:.6f}")

print("Amostra | y(T1) | y(T2) | y(T3) | y(T4) | y(T5) | Real")

for i, amostra in enumerate(x_teste):
    resposta_Real = int(d_teste[i])

    y_t1 = classificar(amostra, pesos_salvos["Treino 1"])
    y_t2 = classificar(amostra, pesos_salvos["Treino 2"])
    y_t3 = classificar(amostra, pesos_salvos["Treino 3"])
    y_t4 = classificar(amostra, pesos_salvos["Treino 4"])
    y_t5 = classificar(amostra, pesos_salvos["Treino 5"])

    print(f"   {i+1:02d}   |   {y_t1}   |   {y_t2}   |   {y_t3}   |   {y_t4}   |   {y_t5}   |  {resposta_Real}")

# Plotar o gráfico do EQM para cada treinamento
plt.figure(figsize=(10, 6))
for nome_treino, historico_eqm in historico_eqm_salvo.items():
    plt.plot(historico_eqm, label=f"Treinamento {nome_treino}")

plt.title("Evolução do Erro Quadrático Médio (EQM) durante o Treinamento")
plt.xlabel("Época")
plt.ylabel("EQM")
plt.legend()
plt.grid(True)
plt.savefig("grafico_eqm.png")