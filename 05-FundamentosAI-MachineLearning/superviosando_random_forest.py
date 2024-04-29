import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Importar o classificador de Florestas Aleatórias
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Função para carregar os dados
def load_data(file_path):
    return pd.read_csv(file_path)

# Função para preparar os dados
def prepare_data(data):
    data_clean = data.dropna()
    data_clean = data_clean.copy()
    le = LabelEncoder()
    features = data_clean[['De_CodFolha', 'De_CodEvento', 'De_TipoEvento', 'DataUltimaUtilizacao']].copy()
    features['De_TipoEvento'] = le.fit_transform(features['De_TipoEvento'])
    X = features
    y = data_clean['Para_CodEvento']
    return X, y, le

# Carregar os dados
file_path = 'C:\\Dev\\Projects\\PERSONAL-TrilhaDataScience\\05-FundamentosAI-MachineLearning\\depara-v2.csv'
data = load_data(file_path)
X, y, le = prepare_data(data)

# Dividir os dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo usando Florestas Aleatórias
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Avaliar o modelo
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy:.2f}')

# Preparar DataFrame de previsões para exportação, mantendo os dados originais
X_test_original = data.loc[X_test.index]  # Recuperar as linhas originais sem transformação
X_test_original['Pred_Para_CodEvento'] = y_pred  # Adicionar coluna de previsões

# Salvar as previsões em um arquivo CSV
predictions_output_path = 'C:\\Dev\\Projects\\PERSONAL-TrilhaDataScience\\05-FundamentosAI-MachineLearning\\predictions_random_forest.csv'
X_test_original.to_csv(predictions_output_path, index=False)

print(f'Previsões salvas em: {predictions_output_path}')
