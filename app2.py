import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

caminho_vagas = r'/content/vagas.json'
caminho_prospects = r'/content/prospects.json'
caminho_applicants = r'/content/applicants.json'

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
    'foi_contratado'  # variável alvo
    
    
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
     
y = df_modelo['foi_contratado']
X = df_modelo.drop(columns='foi_contratado')

X_encoded = pd.get_dummies(X, drop_first=True)

print(f"Formato final das features codificadas: {X_encoded.shape}")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# Identifica colunas categóricas e numéricas
colunas_categoricas = X_train.select_dtypes(include='object').columns.tolist()
colunas_numericas = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Pipeline para colunas numéricas
num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# Pipeline para colunas categóricas
cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Desconhecido')),
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

# Combinando ambos
preprocessador = ColumnTransformer(transformers=[
    ('num', num_pipeline, colunas_numericas),
    ('cat', cat_pipeline, colunas_categoricas)
])

modelo_rf = RandomForestClassifier(random_state=42)
modelo_rf.fit(X_train, y_train)

y_pred = modelo_rf.predict(X_test)

print("Acurácia:", accuracy_score(y_test, y_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

modelo_rf_bal = RandomForestClassifier(random_state=42, class_weight='balanced')
modelo_rf_bal.fit(X_train, y_train)

y_pred_bal = modelo_rf_bal.predict(X_test)

print("Acurácia:", accuracy_score(y_test, y_pred_bal))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred_bal))
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred_bal))

importances = modelo_rf_bal.feature_importances_
features = X_train.columns

df_importancia = pd.DataFrame({
    'Feature': features,
    'Importância': importances
}).sort_values(by='Importância', ascending=False)

top_n = 20
plt.figure(figsize=(10, 8))
sns.barplot(data=df_importancia.head(top_n), x='Importância', y='Feature', palette='viridis')
plt.title(f'Top {top_n} Variáveis mais Importantes')
plt.xlabel('Importância')
plt.ylabel('Variável')
plt.tight_layout()
plt.show()

from sklearn.metrics import roc_curve, roc_auc_score

y_probs = modelo_rf_bal.predict_proba(X_test)[:, 1]  # Prob da classe 1

fpr, tpr, thresholds = roc_curve(y_test, y_probs)
auc = roc_auc_score(y_test, y_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("Taxa de Falsos Positivos (FPR)")
plt.ylabel("Taxa de Verdadeiros Positivos (TPR)")
plt.title("Curva ROC")
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100],
    'max_depth': [None, 10],
    'max_features': ['sqrt']
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, class_weight='balanced'),
    param_grid=param_grid,
    scoring='roc_auc',
    cv=3,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("Melhores hiperparâmetros (versão reduzida):", grid_search.best_params_)
print("Melhor AUC (validação cruzada):", grid_search.best_score_)

modelo_final = RandomForestClassifier(
    random_state=42,
    class_weight='balanced',
    n_estimators=100,
    max_depth=10,
    max_features='sqrt'
)

modelo_final.fit(X_train, y_train)

y_pred_final = modelo_final.predict(X_test)
y_proba_final = modelo_final.predict_proba(X_test)[:, 1]

print("Acurácia:", accuracy_score(y_test, y_pred_final))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred_final))
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred_final))

fpr, tpr, _ = roc_curve(y_test, y_proba_final)
auc = roc_auc_score(y_test, y_proba_final)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Falso Positivo')
plt.ylabel('Verdadeiro Positivo')
plt.title('Curva ROC - Modelo Final')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

from xgboost import XGBClassifier

modelo_xgb = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1])  # compensando desbalanceamento
)

from xgboost import XGBClassifier

modelo_xgb = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

modelo_xgb.fit(X_train, y_train)


y_pred_xgb = modelo_xgb.predict(X_test)
y_proba_xgb = modelo_xgb.predict_proba(X_test)[:, 1]

print("Acurácia:", accuracy_score(y_test, y_pred_xgb))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred_xgb))
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred_xgb))

fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_proba_xgb)
auc_xgb = roc_auc_score(y_test, y_proba_xgb)

plt.figure(figsize=(8, 6))
plt.plot(fpr_xgb, tpr_xgb, label=f'AUC = {auc_xgb:.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Falso Positivo')
plt.ylabel('Verdadeiro Positivo')
plt.title('Curva ROC - Modelo XGBoost')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score

df_resultados = pd.DataFrame([
    {
        'Modelo': 'Random Forest',
        'Acurácia': accuracy_score(y_test, y_pred_final),
        'Recall Classe 1': recall_score(y_test, y_pred_final),
        'F1 Classe 1': f1_score(y_test, y_pred_final),
        'AUC': roc_auc_score(y_test, y_proba_final)
    },
    {
        'Modelo': 'XGBoost',
        'Acurácia': accuracy_score(y_test, y_pred_xgb),
        'Recall Classe 1': recall_score(y_test, y_pred_xgb),
        'F1 Classe 1': f1_score(y_test, y_pred_xgb),
        'AUC': roc_auc_score(y_test, y_proba_xgb)
    }
])

print("\n📈 Comparação de modelos:")
print(df_resultados)

def prever_contratacao(dados_dict, modelo, colunas_modelo):
    """
    Recebe um dicionário com os dados do candidato e retorna a predição do modelo treinado.
    """
    import pandas as pd
    import numpy as np

    # Cria o DataFrame e garante todas as colunas, na ordem correta
    df_input = pd.DataFrame([dados_dict])
    df_input = df_input.reindex(columns=colunas_modelo, fill_value=0)

    # Predição
    proba = modelo.predict_proba(df_input)[0][1]
    classe = int(proba >= 0.5)
     
def prever_contratacao(dados_dict, modelo, colunas_modelo):
    import pandas as pd

    df = pd.DataFrame([dados_dict])
    # Adiciona colunas faltantes com valor 0 ou NaN
    for col in colunas_modelo:
        if col not in df.columns:
            df[col] = 0  # ou np.nan

    # Reordena as colunas para garantir a ordem correta
    df = df[colunas_modelo]

    classe = modelo.predict(df)[0]
    prob = modelo.predict_proba(df)[0][1]

    return classe, prob

exemplo_candidato = {
    'formacao_e_idiomas.nivel_ingles': 3,
    'formacao_e_idiomas.nivel_espanhol': 1,
    'formacao_e_idiomas.nivel_academico': 4,
    'informacoes_profissionais.nivel_profissional': 2,
    'perfil_vaga.estado': 'SP',
    # Adicione mais campos se necessário...
}

classe_predita, probabilidade = prever_contratacao(
    dados_dict=exemplo_candidato,
    modelo=modelo_xgb,
    colunas_modelo=X_train.columns
)

print(f"Classe prevista: {classe_predita} (0 = Não contratado, 1 = Contratado)")
print(f"Probabilidade de contratação: {probabilidade}")

import joblib

joblib.dump(modelo_xgb, 'modelo_xgb.pkl')

['modelo_xgb.pkl']

joblib.dump(X_train.columns.tolist(), 'colunas_modelo.pkl')

['colunas_modelo.pkl']

X_train.to_csv('dados_treinamento.csv', index=False)