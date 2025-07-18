import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

caminho_vagas = r'vagas.json'
caminho_prospects = r'prospects.json'
caminho_applicants = r'applicants.json'

with open(caminho_vagas, encoding='utf-8') as f:
    raw_vagas = json.load(f)
    
lista_vagas = []
for id_vaga, dados in raw_vagas.items():
    dados['id_vaga'] = id_vaga
    lista_vagas.append(dados)
    
df_vagas = pd.json_normalize(lista_vagas)
print(f'Vagas carregadas: {df_vagas.shape}')

with open(caminho_prospects, encoding='utf-8') as f:
    raw_prospects = json.load(f)

lista_prospects = []
for id_vaga, conteudo in raw_prospects.items():
    if isinstance(conteudo, list):
        for candidato in conteudo:
            if isinstance(candidato, dict):
                candidato['id_vaga'] = id_vaga
                lista_prospects.append(candidato)
    elif isinstance(conteudo, dict) and 'prospects' in conteudo:
        for candidato in conteudo['prospects']:
            if isinstance(candidato, dict):
                candidato['id_vaga'] = id_vaga
                lista_prospects.append(candidato)

df_prospeccoes = pd.json_normalize(lista_prospects)
print(f'Prospecções carregadas: {df_prospeccoes.shape}')

lista_candidatos = []
with open(caminho_applicants, encoding='utf-8') as f:
    try:
        raw_candidatos = json.load(f)
    except json.JSONDecodeError as e:
        print("Erro ao carregar JSON completo. Tentando extrair manualmente...")

        f.seek(0)
        for i, linha in enumerate(f):
            try:
                candidato = json.loads(linha)
                lista_candidatos.append(candidato)
            except:
                print(f"Linha inválida na {i}")
            
with open(caminho_applicants, encoding='utf-8') as f:
    raw_candidatos = json.load(f)

lista_candidatos = []
for id_candidato, dados in raw_candidatos.items():
    dados['id_candidato'] = id_candidato
    lista_candidatos.append(dados)

df_candidatos = pd.json_normalize(lista_candidatos)
print(f'Candidatos carregados: {df_candidatos.shape}')

print("\nVagas:")
print(df_vagas.head())

print("\nProspecções:")
print(df_prospeccoes.head())

print("\nCandidatos:")
print(df_candidatos.head())

print("Vagas:")
df_vagas.info()

print("\nProspecções:")
df_prospeccoes.info()

print("\nCandidatos:")
df_candidatos.info()
# %% [markdown]
# ### 6.2 Verificação dos nomes das colunas
# %%
print("Colunas disponíveis em df_vagas:")
print(df_vagas.columns.tolist())
# %%
print("Colunas disponíveis em df_prospeccoes:")
print(df_prospeccoes.columns.tolist())
# %%
print("Colunas disponíveis em df_candidatos:")
print(df_candidatos.columns.tolist())

print(f"Total de vagas: {df_vagas['id_vaga'].nunique():,.0f}".replace(',', '.'))

print(f"Total de candidatos: {df_candidatos['id_candidato'].nunique():,.0f}".replace(',', '.'))

print(f"Total de prospecções: {df_prospeccoes.shape[0]:,.0f}".replace(',', '.'))

if 'situacao_candidado' in df_prospeccoes.columns:
    print("Distribuição da situação dos candidatos:")
    print(df_prospeccoes['situacao_candidado'].value_counts(dropna=False))
else:
    print("Coluna 'situacao' não encontrada. Verifique o nome correto.")
    
df_candidatos['formacao_e_idiomas.ano_conclusao'].unique()

df_vagas['informacoes_basicas.tipo_contratacao'].unique()

df_vagas['informacoes_basicas.prioridade_vaga'].unique()

df_vagas['perfil_vaga.estado'].unique()

df_vagas['perfil_vaga.cidade'].unique()

df_vagas['perfil_vaga.regiao'].unique()

df_vagas['perfil_vaga.nivel_academico'].unique()

df_vagas['perfil_vaga.nivel_ingles'].unique()

df_vagas['perfil_vaga.nivel_espanhol'].unique()

df_vagas['perfil_vaga.outro_idioma'].unique()

df_vagas['perfil_vaga.areas_atuacao'].unique()

df_candidatos['informacoes_profissionais.area_atuacao'].unique()

df_candidatos['informacoes_profissionais.nivel_profissional'].unique()

df_candidatos['informacoes_profissionais.nivel_profissional'].unique()

df_candidatos['formacao_e_idiomas.nivel_academico'].unique()

df_candidatos['formacao_e_idiomas.nivel_ingles'].unique()

df_candidatos['formacao_e_idiomas.nivel_espanhol'].unique()

df_candidatos['formacao_e_idiomas.outro_idioma'].unique()
     
df_candidatos['formacao_e_idiomas.instituicao_ensino_superior'].unique()

print("Valores nulos em df_vagas:")
print(df_vagas.isnull().sum().sort_values(ascending=False))

print("\nValores nulos em df_prospeccoes:")
print(df_prospeccoes.isnull().sum().sort_values(ascending=False))

print("\nValores nulos em df_candidatos:")
print(df_candidatos.isnull().sum().sort_values(ascending=False))

if 'situacao_candidado' in df_prospeccoes.columns:
    print("Distribuição da situação dos candidatos:")
    print(df_prospeccoes['situacao_candidado'].value_counts(dropna=False))
else:
    print("Coluna 'situacao' não encontrada. Verifique o nome correto.")
    
if 'situacao_candidado' in df_prospeccoes.columns:
    situacao_counts = df_prospeccoes['situacao_candidado'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=situacao_counts.values, y=situacao_counts.index, palette="viridis")
    plt.title("Distribuição da Situação dos Candidatos")
    plt.xlabel("Quantidade")
    plt.ylabel("Situação")
    plt.tight_layout()
    plt.show()
else:
    print("Coluna 'situacao' não encontrada. Verifique o nome correto.")
    print("Coluna 'situacao' não encontrada. Verifique o nome correto.")

percentual = (situacao_counts / situacao_counts.sum() * 100).round(2)
df_percentual = pd.DataFrame({
    'Situação': situacao_counts.index,
    'Quantidade': situacao_counts.values,
    'Percentual (%)': percentual.values
})
print("Tabela com percentuais da situação dos candidatos em relação ao total de candidaturas:")
print(df_percentual)

df_prospeccoes = df_prospeccoes.rename(columns={'codigo': 'id_candidato'})
     
df_completo = df_prospeccoes.merge(df_vagas, on='id_vaga', how='left')

df_completo = df_completo.merge(df_candidatos, on='id_candidato', how='left')

print(f"Shape final do DataFrame consolidado: {df_completo.shape}")
print(df_completo[['id_vaga', 'id_candidato', 'situacao_candidado']].head())

print("Colunas disponíveis em df_completo:")
print(df_completo.columns.tolist())

print("Registros em df_prospeccoes:", df_prospeccoes.shape[0])
print("Registros após merge (df_completo):", df_completo.shape[0])

print("Vagas ausentes no merge:", df_completo['informacoes_basicas.titulo_vaga'].isnull().sum())
print("Candidatos ausentes no merge:", df_completo['infos_basicas.nome'].isnull().sum())

print(df_prospeccoes['id_candidato'].dtype)
print(df_candidatos['id_candidato'].dtype)

df_completo[['id_candidato', 'id_vaga', 'situacao_candidado',
             'perfil_vaga.nivel_ingles', 'formacao_e_idiomas.nivel_ingles',
             'informacoes_profissionais.area_atuacao']].sample(5)

situacoes_contratado = ['Contratado pela Decision', 'Contratado como Hunting']

df_completo['foi_contratado'] = df_completo['situacao_candidado'].isin(situacoes_contratado).astype(int)

print("Distribuição da variável `foi_contratado`:")
print(df_completo['foi_contratado'].value_counts())

percentual = (df_completo['foi_contratado'].value_counts(normalize=True) * 100).round(2)
print("\nPercentual:")
print(percentual)

import matplotlib.pyplot as plt
import seaborn as sns

contagem = df_completo['foi_contratado'].value_counts().sort_index()
labels = ['Não contratado', 'Contratado']

plt.figure(figsize=(8, 4))
sns.barplot(x=contagem.values, y=labels, palette="crest")
plt.title("Distribuição da Variável Alvo: Foi Contratado")
plt.xlabel("Quantidade")
plt.ylabel("Situação")
plt.tight_layout()
plt.show()

contagem = df_completo['foi_contratado'].value_counts().sort_index()
# Percentual
percentual = df_completo['foi_contratado'].value_counts(normalize=True).sort_index() * 100

# Monta DataFrame
df_alvo = pd.DataFrame({
    'Classe': ['Não Contratado', 'Contratado'],
    'Quantidade': contagem.values,
    'Percentual (%)': percentual.round(2).values
})

print("\nTabela da variável alvo (`foi_contratado`):")
print(df_alvo)

colunas_modelo = [
    'perfil_vaga.estado',
    'perfil_vaga.cidade',
    'perfil_vaga.regiao',
    'perfil_vaga.nivel_academico',
    'perfil_vaga.nivel_ingles',
    'perfil_vaga.nivel_espanhol',
    'perfil_vaga.areas_atuacao',
    'perfil_vaga.vaga_especifica_para_pcd',
    'informacoes_basicas.prioridade_vaga',
    'informacoes_profissionais.area_atuacao',
    'informacoes_profissionais.nivel_profissional',
    'formacao_e_idiomas.nivel_academico',
    'formacao_e_idiomas.nivel_ingles',
    'formacao_e_idiomas.nivel_espanhol',
    'formacao_e_idiomas.outro_idioma',
    'formacao_e_idiomas.instituicao_ensino_superior',
    'formacao_e_idiomas.ano_conclusao',
    'foi_contratado']  # variável alvo
    
    
df_modelo = df_completo[colunas_modelo].copy()
 
colunas_categoricas = df_modelo.select_dtypes(include=['object', 'category']).columns.tolist()
colunas_numericas = df_modelo.select_dtypes(include=['int64', 'float64']).columns.tolist()

from sklearn.impute import SimpleImputer

# Detectar colunas primeiro
colunas_categoricas = df_modelo.select_dtypes(include=['object', 'category']).columns.tolist()
colunas_numericas = df_modelo.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Imputação categórica
if colunas_categoricas:
    imp_cat = SimpleImputer(strategy='constant', fill_value='Desconhecido')
    df_modelo[colunas_categoricas] = imp_cat.fit_transform(df_modelo[colunas_categoricas])

# Imputação numérica
if colunas_numericas:
    imp_num = SimpleImputer(strategy='median')
    df_modelo[colunas_numericas] = imp_num.fit_transform(df_modelo[colunas_numericas])

print("Valores nulos restantes por coluna:")
print(df_modelo.isnull().sum().sort_values(ascending=False))
     
# ========================================
# 0 - IMPORTS
# ========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, roc_auc_score, recall_score, f1_score
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier
import joblib

# ========================================
# 1 - PREPARAÇÃO DOS DADOS
# ========================================

# Ajuste o seu df_modelo aqui


y = df_modelo['foi_contratado']
X = df_modelo.drop(columns='foi_contratado')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Identificar colunas categóricas e numéricas
colunas_categoricas = X_train.select_dtypes(include='object').columns.tolist()
colunas_numericas = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

# ========================================
# 2 - PRÉ-PROCESSAMENTO (Pipelines)
# ========================================

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Desconhecido')),
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

preprocessador = ColumnTransformer([
    ('num', num_pipeline, colunas_numericas),
    ('cat', cat_pipeline, colunas_categoricas)
])

# ========================================
# 3 - RANDOM FOREST COM PIPELINE + GRIDSEARCH
# ========================================

pipeline_rf = Pipeline([
    ('preprocessamento', preprocessador),
    ('classificador', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

param_grid = {
    'classificador__n_estimators': [100, 200],
    'classificador__max_depth': [None, 10],
    'classificador__max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(
    estimator=pipeline_rf,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=3,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
melhor_pipeline_rf = grid_search.best_estimator_

print("\nMelhores parâmetros RF:", grid_search.best_params_)
print("Melhor AUC validação cruzada RF:", grid_search.best_score_)

# Avaliação RF
y_pred_rf = melhor_pipeline_rf.predict(X_test)
y_proba_rf = melhor_pipeline_rf.predict_proba(X_test)[:, 1]

print("\nRandom Forest Final")
print("Acurácia:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
auc_rf = roc_auc_score(y_test, y_proba_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, label=f'AUC RF = {auc_rf:.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Falso Positivo')
plt.ylabel('Verdadeiro Positivo')
plt.title('Curva ROC - Random Forest')
plt.legend()
plt.grid()
plt.show()

# ========================================
# 4 - XGBOOST (com pré-processamento aplicado)
# ========================================

# Pré-processando dados para XGBoost (XGBoost não aceita strings)
X_train_proc = preprocessador.fit_transform(X_train)
X_test_proc = preprocessador.transform(X_test)

modelo_xgb = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

modelo_xgb.fit(X_train_proc, y_train)

X_test_proc = preprocessador.transform(X_test)
y_pred_xgb = modelo_xgb.predict(X_test_proc)
y_proba_xgb = modelo_xgb.predict_proba(X_test_proc)[:, 1]


print("\nXGBoost")
print("Acurácia:", accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))
print(confusion_matrix(y_test, y_pred_xgb))

fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_proba_xgb)
auc_xgb = roc_auc_score(y_test, y_proba_xgb)

plt.figure(figsize=(8, 6))
plt.plot(fpr_xgb, tpr_xgb, label=f'AUC XGB = {auc_xgb:.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Falso Positivo')
plt.ylabel('Verdadeiro Positivo')
plt.title('Curva ROC - XGBoost')
plt.legend()
plt.grid()
plt.show()

# ========================================
# 5 - COMPARAÇÃO FINAL
# ========================================
df_resultados = pd.DataFrame([
    {
        'Modelo': 'Random Forest',
        'Acurácia': accuracy_score(y_test, y_pred_rf),
        'Recall Classe 1': recall_score(y_test, y_pred_rf),
        'F1 Classe 1': f1_score(y_test, y_pred_rf),
        'AUC': auc_rf
    },
    {
        'Modelo': 'XGBoost',
        'Acurácia': accuracy_score(y_test, y_pred_xgb),
        'Recall Classe 1': recall_score(y_test, y_pred_xgb),
        'F1 Classe 1': f1_score(y_test, y_pred_xgb),
        'AUC': auc_xgb
    }
])

print("\n📊 Comparativo Final:")
print(df_resultados)

# ========================================
# 6 - SALVANDO OS MODELOS (produção)
# ========================================

# Salvando o pré-processador e modelos
joblib.dump(preprocessador, 'preprocessador.pkl')
joblib.dump(melhor_pipeline_rf, 'random_forest_pipeline.pkl')
joblib.dump(modelo_xgb, 'modelo_xgb.pkl')
joblib.dump(X_train.columns.tolist(), 'colunas_modelo.pkl')
X_train.to_csv('dados_treinamento.csv', index=False)

# ========================================
# 7 - TESTE COM NOVO CANDIDATO
# ========================================

def prever_candidato_rf(dados_dict, pipeline_rf, colunas_modelo):
    df_input = pd.DataFrame([dados_dict])
    
    for col in colunas_modelo:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[colunas_modelo]
    
    classe_predita = pipeline_rf.predict(df_input)[0]
    probabilidade = pipeline_rf.predict_proba(df_input)[0][1]
    
    return classe_predita, probabilidade

def prever_candidato_xgb(dados_dict, preprocessador, modelo_xgb, colunas_modelo):
    df_input = pd.DataFrame([dados_dict])
    
    for col in colunas_modelo:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[colunas_modelo]
    
    X_proc = preprocessador.transform(df_input)
    
    classe_predita = modelo_xgb.predict(X_proc)[0]
    probabilidade = modelo_xgb.predict_proba(X_proc)[0][1]
    
    return classe_predita, probabilidade

# Exemplo de uso com novo candidato
exemplo_candidato = {
    'formacao_e_idiomas.nivel_ingles': 3,
    'formacao_e_idiomas.nivel_espanhol': 1,
    'formacao_e_idiomas.nivel_academico': 4,
    'informacoes_profissionais.nivel_profissional': 2,
    'perfil_vaga.estado': 'SP'  
}

# Random Forest:
classe_rf, prob_rf = prever_candidato_rf(
    dados_dict=exemplo_candidato,
    pipeline_rf=melhor_pipeline_rf,
    colunas_modelo=X_train.columns.tolist()
)

print(f"\nRandom Forest: Classe (0 não contratado, 1 contratado): {classe_rf}, Probabilidade de contratação: {prob_rf:.2%}")

# XGBoost:
classe_xgb, prob_xgb = prever_candidato_xgb(
    dados_dict=exemplo_candidato,
    preprocessador=preprocessador,
    modelo_xgb=modelo_xgb,
    colunas_modelo=X_train.columns.tolist()
)

print(f"XGBoost: Classe (0 não contratado, 1 contratado): {classe_xgb}, Probabilidade de contratação: {prob_xgb:.2%}")


