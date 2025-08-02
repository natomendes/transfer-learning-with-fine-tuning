# Classificação de Imagens: Gatos vs. Cachorros com Transfer Learning e Fine-Tuning

Este projeto demonstra a construção de um classificador de imagens de aprendizado profundo para distinguir entre gatos e cachorros. A abordagem utiliza **Transfer Learning** e **Fine-Tuning** sobre a arquitetura VGG16, adaptada para processar imagens em escala de cinza.

O pipeline completo, desde o download e limpeza dos dados até o treinamento e avaliação do modelo, está implementado no notebook `transfer_learning_fine_tuning.ipynb`.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/natomendes/transfer-learning-with-fine-tuning/blob/main/transfer_learning_fine_tuning.ipynb)

---

## 📜 Visão Geral

O objetivo é classificar imagens com alta acurácia, aproveitando o conhecimento de um modelo pré-treinado (VGG16) na vasta base de dados ImageNet. O processo é dividido em duas fases principais de treinamento:

1.  **Transfer Learning:** O modelo base (VGG16) é "congelado", e apenas a cabeça de classificação (camadas adicionadas no topo) é treinada. Isso adapta o classificador para a nossa tarefa específica sem alterar os extratores de características já aprendidos.
2.  **Fine-Tuning:** Algumas das camadas superiores do modelo base são "descongeladas" e treinadas com uma taxa de aprendizado muito baixa. Isso ajusta sutilmente os extratores de características para se especializarem ainda mais no nosso dataset.

## 🛠️ Tecnologias Utilizadas

*   **TensorFlow & Keras:** Para a construção, treinamento e avaliação do modelo de deep learning.
*   **VGG16:** Arquitetura de rede neural convolucional usada como base para o transfer learning.
*   **NumPy:** Para manipulação de arrays e operações numéricas.
*   **Matplotlib & Seaborn:** Para visualização de dados e resultados do treinamento.
*   **Scikit-learn:** Para gerar métricas de avaliação como o `classification_report`.
*   **Pillow (PIL):** Para manipulação e verificação de imagens.

---

## 🚀 O Pipeline do Projeto

O fluxo de trabalho do projeto segue as seguintes etapas:

### 1. Preparação do Ambiente e Dados

*   **Download do Dataset:** O script baixa automaticamente o dataset "Cats and Dogs" da Microsoft.
*   **Limpeza de Dados:** Uma etapa crucial é a verificação e remoção de imagens corrompidas ou em formatos inesperados. O notebook implementa duas rotinas de limpeza para garantir a integridade dos dados que alimentam o modelo.
*   **Organização:** As imagens são divididas em diretórios de `treino` e `validação`, mantendo a estrutura esperada pelo Keras (`train/cats`, `train/dogs`, etc.).

### 2. Criação do Modelo com Transfer Learning

*   **Adaptação para Escala de Cinza:** Embora o VGG16 tenha sido treinado com imagens coloridas (3 canais), este projeto o adapta para aceitar imagens em escala de cinza (1 canal). Isso é feito modificando a primeira camada convolucional e ajustando seus pesos (calculando a média dos pesos dos 3 canais originais).
*   **Congelamento do Modelo Base:** A base VGG16 é congelada (`trainable = False`) para a primeira fase de treinamento.
*   **Adição da Cabeça de Classificação:** Camadas `GlobalAveragePooling2D`, `Dropout` e `Dense` são adicionadas no topo do modelo base para realizar a classificação final.

### 3. Treinamento em Duas Fases

O treinamento é estrategicamente dividido para maximizar a performance:

*   **Fase 1 (Transfer Learning):** O modelo é treinado por algumas épocas com o modelo base congelado. O objetivo é treinar apenas o classificador adicionado.
*   **Fase 2 (Fine-Tuning):** As últimas 4 camadas do VGG16 são descongeladas, e o modelo é treinado novamente com uma **taxa de aprendizado muito baixa** (`learning_rate / 10`). Isso permite que os extratores de características mais complexos se ajustem ao nosso dataset específico sem o risco de "esquecer" o que aprenderam com o ImageNet.

### 4. Callbacks e Otimizações

*   **EarlyStopping:** Interrompe o treinamento se a perda no conjunto de validação não melhorar, evitando overfitting.
*   **ReduceLROnPlateau:** Reduz a taxa de aprendizado automaticamente se o treinamento estagnar.
*   **tf.data.Dataset:** O pipeline de dados é otimizada com `.cache()` e `.prefetch()` para garantir que a GPU não fique ociosa esperando por dados.

---

## 📊 Resultados

O modelo atinge uma alta acurácia no conjunto de validação, demonstrando a eficácia da abordagem de transfer learning e fine-tuning. Os gráficos de treinamento mostram a evolução da acurácia e da perda ao longo das duas fases.

**Exemplo de Gráfico de Treinamento:**

  <!-- Placeholder for an actual training graph image -->

A linha vermelha pontilhada indica o início da fase de fine-tuning, onde geralmente observamos uma melhoria adicional na performance do modelo.

---

## ⚙️ Como Executar

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/natomendes/transfer-learning-with-fine-tuning.git
    cd transfer-learning-with-fine-tuning
    ```

2.  **Instale as dependências:**
    É recomendado criar um ambiente virtual.
    ```bash
    pip install tensorflow numpy matplotlib requests scikit-learn seaborn pillow
    ```

3.  **Execute o Notebook:**
    Abra e execute o `transfer_learning_fine_tuning.ipynb` em um ambiente como Jupyter Notebook, JupyterLab ou Google Colab.

    *   **Primeira Execução:** Descomente a linha `base_dir = download_and_extract_dataset()` para baixar e extrair os dados.
    *   **Execuções Subsequentes:** Comente a linha de download e use `base_dir = 'PetImages'` para pular o download.

---

## 📁 Estrutura de Arquivos

```
├── README.md                                     # Este arquivo
└── transfer_learning_fine_tuning.ipynb           # Notebook com todo o código do projeto
