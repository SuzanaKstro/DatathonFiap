import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

modelo = joblib.load('modelo_xgb.pkl')

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


st.title('Previsão de Contratação - Decision AI')
st.markdown("### Preencha os dados da vaga e do candidato")
