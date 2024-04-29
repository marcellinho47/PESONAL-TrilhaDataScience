import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder

# Caminho base para os arquivos
base_path = "C:\\Dev\\Projects\\PERSONAL-TrilhaDataScience\\05-FundamentosAI-MachineLearning\\"

# Carregar os dados de treinamento e teste
train_data_path = base_path + 'depara-v2-treinamento.csv'
test_data_path = base_path + 'depara-v2-teste.csv'
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)
test_data_original = test_data.copy()

# Colunas categóricas para serem codificadas
categorical_cols = ['De_DescFolha', 'De_DescEvento', 'De_TipoEvento']

# Inicializar o OrdinalEncoder para as colunas categóricas
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
encoder.fit(train_data[categorical_cols])

# Aplicar a transformação tanto nos dados de treinamento quanto de teste
train_data[categorical_cols] = encoder.transform(train_data[categorical_cols])
test_data[categorical_cols] = encoder.transform(test_data[categorical_cols])

# Colunas de destino a serem previstas
target_cols = ['Para_CodFolha', 'Para_DescFolha', 'Para_CodEvento', 'Para_DescEvento']

# Definir as colunas de origem que serão usadas como features para os modelos
features_cols = categorical_cols + ['De_CodFolha', 'De_CodEvento', 'DataUltimaUtilizacao']

# Treinar um modelo de DecisionTreeClassifier para cada coluna de destino
for target_col in target_cols:
    print(f"Treinando modelo para prever {target_col}...")

    # Preparar o conjunto de dados de treinamento, excluindo linhas com valores NaN na coluna alvo
    filtered_train_data = train_data.dropna(subset=[target_col])
    X_train = filtered_train_data[features_cols]
    y_train = filtered_train_data[target_col]

    # Treinar o modelo
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Aplicar o modelo aos dados de teste para fazer previsões
    X_test = test_data[features_cols]
    test_data_original[f'Pred_{target_col}'] = model.predict(X_test)

# Salvar o arquivo de resultado com as previsões
output_file = base_path + 'depara-v2-teste-com-predicoes-formato-original.csv'
test_data_original.to_csv(output_file, index=False)
print(f"Previsões salvas em: {output_file}")

