import streamlit as st
import joblib
import pandas as pd

# Define os campos e nomes amigáveis
campos = {
    'perfil_vaga.estado': 'Estado da vaga',
    'perfil_vaga.cidade': 'Cidade da vaga',
    'perfil_vaga.regiao': 'Região da vaga',
    'perfil_vaga.nivel_academico': 'Nível acadêmico da vaga',
    'perfil_vaga.nivel_ingles': 'Nível de inglês da vaga',
    'perfil_vaga.nivel_espanhol': 'Nível de espanhol da vaga',
    'perfil_vaga.areas_atuacao': 'Área de atuação da vaga',
    'perfil_vaga.vaga_especifica_para_pcd': 'Vaga específica para PCD?',
    'informacoes_basicas.prioridade_vaga': 'Prioridade da vaga',
    'informacoes_profissionais.area_atuacao': 'Área de atuação do candidato',
    'informacoes_profissionais.nivel_profissional': 'Nível profissional do candidato',
    'formacao_e_idiomas.nivel_academico': 'Nível acadêmico do candidato',
    'formacao_e_idiomas.nivel_ingles': 'Nível de inglês do candidato',
    'formacao_e_idiomas.nivel_espanhol': 'Nível de espanhol do candidato',
    'formacao_e_idiomas.outro_idioma': 'Outro idioma',
    'formacao_e_idiomas.instituicao_ensino_superior': 'Instituição de ensino superior',
    'formacao_e_idiomas.ano_conclusao': 'Ano de conclusão da formação'
}

# Listas de opções
estados = ['Acre', 'Alagoas', 'Amapá', 'Amazonas', 'Bahia', 'Ceará', 'Distrito Federal', 'Espirito Santo', 
           'Goiás', 'Maranhão', 'Mato Grosso', 'Mato Grosso do Sul', 'Minas Gerais', 'Pará', 'Paraíba', 
           'Paraná', 'Pernambuco', 'Piauí', 'Rio de Janeiro', 'Rio Grande do Norte', 'Rio Grande do Sul', 
           'Rondônia', 'Roraima', 'Santa Catarina', 'São Paulo', 'Sergipe', 'Tocantins']

regioes = ['Norte', 'Nordeste', 'Centro-Oeste', 'Sudeste', 'Sul']

niveis_idioma_vaga = ['Avançado', 'Fluente', 'Nenhum', 'Básico', 'Intermediário', 'Técnico']
niveis_idioma_candidato = ['Nenhum', 'Intermediário', 'Básico', 'Avançado', 'Fluente']

outros_idiomas_candidato = [
    'Português - Fluente', 'Francês - Intermediário', 'Francês - Básico', 'Italiano - Avançado', 'Português -',
    'Italiano - Básico', 'Português - Avançado', 'Alemão - Básico', 'Alemão -', 'Japonês - Básico', 'Japonês -',
    'Alemão - Avançado', 'Japonês - Avançado', 'Alemão - Intermediário', 'Russo - Básico', 'Russo - Intermediário',
    'Italiano - Intermediário', 'Português - Intermediário', 'Japonês - Intermediário', 'Mandarim - Básico',
    'Francês - Avançado', 'Português - Nenhum', 'Italiano -', 'Português - Básico', 'Francês - Fluente',
    'Italiano - Fluente', 'Japonês - Fluente', 'Japonês - Nenhum', 'Russo - Fluente', 'Alemão - Nenhum',
    'Francês -', 'Alemão - Fluente'
]

outros_idiomas_vaga = [
    'Mandarim ou Russo', 'Nenhum', 'Avançado', 'Português Fluente', 'Português Avançado', 'Português Básico',
    'Francês Básico', 'Fluente', 'Básico', 'Francês Intermediário', 'Mandarim - Desejável', 'Alemão Fluente',
    'Francês Fluente', 'Alemão Avançado', 'Português Intermediário', 'Português'
]

areas_atuacao_vaga = [
    'TI - Sistemas e Ferramentas', 'TI - Desenvolvimento/Programação', 'TI - Projetos', 'TI - SAP',
    'TI - Infraestrutura', 'Gestão e Alocação de Recursos de TI', 'Administrativa',
    'TI - Processos e Negócios', 'Recursos Humanos', 'TI - Desenvolvimento/Design',
    'TI - Suporte', 'Financeira/Controladoria', 'TI - Desenvolvimento/Mobile', 'TI - Qualidade/Testes',
    'Comercial', 'TI - Banco de Dados', 'TI - Arquitetura', 'TI - Governança', 'TI - Telecom',
]

# Carrega o modelo treinado
modelo = joblib.load('modelo_xgb.pkl')

# Interface Streamlit
st.title('🧠 Previsão de Contratação - Decision AI')
st.markdown("### Preencha os dados da vaga e do candidato")

# Captura as entradas do usuário com widgets
estado = st.selectbox(campos['perfil_vaga.estado'], estados)
cidade = st.text_input(campos['perfil_vaga.cidade'])
regiao = st.selectbox(campos['perfil_vaga.regiao'], regioes)
nivel_academico_vaga = st.selectbox(campos['perfil_vaga.nivel_academico'], niveis_idioma_vaga)
nivel_ingles_vaga = st.selectbox(campos['perfil_vaga.nivel_ingles'], niveis_idioma_vaga)
nivel_espanhol_vaga = st.selectbox(campos['perfil_vaga.nivel_espanhol'], niveis_idioma_vaga)
area_atuacao_vaga = st.selectbox(campos['perfil_vaga.areas_atuacao'], areas_atuacao_vaga)
vaga_pcd = st.selectbox(campos['perfil_vaga.vaga_especifica_para_pcd'], ['Sim', 'Não'])
prioridade_vaga = st.selectbox(campos['informacoes_basicas.prioridade_vaga'], ['Alta', 'Média', 'Baixa'])
area_atuacao_candidato = st.selectbox(campos['informacoes_profissionais.area_atuacao'], areas_atuacao_vaga)
nivel_profissional_candidato = st.selectbox(campos['informacoes_profissionais.nivel_profissional'], ['Júnior', 'Pleno', 'Sênior'])
nivel_academico_candidato = st.selectbox(campos['formacao_e_idiomas.nivel_academico'], niveis_idioma_candidato)
nivel_ingles_candidato = st.selectbox(campos['formacao_e_idiomas.nivel_ingles'], niveis_idioma_candidato)
nivel_espanhol_candidato = st.selectbox(campos['formacao_e_idiomas.nivel_espanhol'], niveis_idioma_candidato)
outro_idioma = st.selectbox(campos['formacao_e_idiomas.outro_idioma'], outros_idiomas_candidato)
instituicao_ensino = st.text_input(campos['formacao_e_idiomas.instituicao_ensino_superior'])
ano_conclusao = st.number_input(campos['formacao_e_idiomas.ano_conclusao'], min_value=1900, max_value=2025, step=1)

# Botão para previsão
if st.button('Prever Contratação'):
    # Cria um DataFrame com as entradas no formato que seu modelo espera
    dados = {
        'perfil_vaga.estado': estado,
        'perfil_vaga.cidade': cidade,
        'perfil_vaga.regiao': regiao,
        'perfil_vaga.nivel_academico': nivel_academico_vaga,
        'perfil_vaga.nivel_ingles': nivel_ingles_vaga,
        'perfil_vaga.nivel_espanhol': nivel_espanhol_vaga,
        'perfil_vaga.areas_atuacao': area_atuacao_vaga,
        'perfil_vaga.vaga_especifica_para_pcd': vaga_pcd,
        'informacoes_basicas.prioridade_vaga': prioridade_vaga,
        'informacoes_profissionais.area_atuacao': area_atuacao_candidato,
        'informacoes_profissionais.nivel_profissional': nivel_profissional_candidato,
        'formacao_e_idiomas.nivel_academico': nivel_academico_candidato,
        'formacao_e_idiomas.nivel_ingles': nivel_ingles_candidato,
        'formacao_e_idiomas.nivel_espanhol': nivel_espanhol_candidato,
        'formacao_e_idiomas.outro_idioma': outro_idioma,
        'formacao_e_idiomas.instituicao_ensino_superior': instituicao_ensino,
        'formacao_e_idiomas.ano_conclusao': ano_conclusao
    }
    df = pd.DataFrame([dados])

    # Pré-processar df conforme seu modelo espera (exemplo: encoding, normalização)
    # Aqui você deve colocar as mesmas transformações que aplicou no treinamento do modelo
    # Por simplicidade, vamos assumir que o modelo aceita esses dados diretamente.

    # Faz a previsão (exemplo para um modelo xgboost)
    pred = modelo.predict(df)

    st.success(f'A previsão de contratação é: {"Sim" if pred[0] == 1 else "Não"}')


