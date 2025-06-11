# DatathonFiap
Projeto: Previsão de Contratação de Candidatos
Descrição
Este projeto tem como foco desenvolver uma solução para prever a contratação de candidatos com base em informações como perfil pessoal, formação acadêmica, idiomas e experiências profissionais. A proposta utiliza algoritmos de aprendizado de máquina para resolver um problema de classificação binária, comparando o desempenho dos modelos Random Forest e XGBoost.

Objetivo
Criar uma ferramenta preditiva que auxilie recrutadores e gestores na identificação de candidatos com maior chance de contratação. A ideia é tornar o processo seletivo mais eficiente, reduzindo o tempo de análise e os custos envolvidos.

Tecnologias Utilizadas
Python 3.10+

Pandas e NumPy

Scikit-learn

XGBoost

Matplotlib e Seaborn (visualização de dados)

Streamlit (opcional, para interface web)

Joblib (armazenamento do modelo)

Jupyter Notebook

Modelos Utilizados
Foram testados dois modelos de machine learning com foco em classificação:

Modelo	AUC ROC	Recall (Classe 1)	F1-score (Classe 1)
Random Forest	0.767	0.45	0.28
XGBoost (final)	0.788	0.60	0.28

O modelo final escolhido foi o XGBoost, por apresentar maior sensibilidade na identificação de candidatos contratados.

Etapas do Projeto
Coleta e Preparação dos Dados

Escolha de colunas relevantes

Tratamento de dados ausentes

Pré-processamento

Codificação de variáveis categóricas (One-Hot Encoding)

Separação entre variáveis preditoras (X) e alvo (y)

Modelagem Preditiva

Random Forest (com e sem técnicas de balanceamento)

XGBoost (com ajuste de scale_pos_weight)

Otimização de hiperparâmetros com GridSearchCV (para Random Forest)

Avaliação dos Modelos

Métricas: Acurácia, Recall, F1-score

Curva ROC e AUC

Matriz de confusão

Exportação e Deploy Simulado

Salvamento do modelo final com Joblib

Criação de função para prever contratação com base nos dados de um novo candidato

Deploy Simulado
Foi implementada uma função para simular o uso do modelo em produção. Ela recebe os dados de um candidato em formato de dicionário e retorna a previsão de contratação (sim ou não), além da probabilidade associada.
