# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
import pandas as pd
from pypmml import Model

app = Flask(__name__)

# Carregar o modelo PMML
model = Model.load('pmmlMLP.pmml')

# Coeficientes de normalização
normalization_coefficients = {
    'Idade': {'orig': (0.0, 1.0), 'norm': (-4.479687889667643, -4.3617256523674675)},
    'Genero': {'orig': (0.0, 1.0), 'norm': (-1.046824317027073, 0.9494453107919978)},
    'Renda': {'orig': (0.0, 1.0), 'norm': (-2.5807823367269376, -2.5807515270233363)},
    'Escolaridade': {'orig': (0.0, 1.0), 'norm': (-1.5226122511936537, -0.7903295262794034)},
    'Estado_Civil': {'orig': (0.0, 1.0), 'norm': (-1.0597078266478577, 0.9379023293320119)},
    'Numero_filhos': {'orig': (0.0, 1.0), 'norm': (-0.7385993375745517, 0.3934594602032657)},
    'Casa_propria': {'orig': (0.0, 1.0), 'norm': (-1.4427644503136394, 0.688887530329936)},
}

# Função para normalizar os dados
def normalize_data(data):
    normalized_data = {}
    for key, value in data.items():
        if key in normalization_coefficients:
            norm_params = normalization_coefficients[key]
            orig_min, orig_max = norm_params['orig']
            norm_min, norm_max = norm_params['norm']
            normalized_data[key] = ((float(value) - orig_min) / (orig_max - orig_min)) * (norm_max - norm_min) + norm_min
        else:
            normalized_data[key] = float(value)
    return normalized_data

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        try:
            # Receber e processar os dados do formulário
            idade = int(request.form['idade'])
            genero = int(request.form['genero'])
            renda = float(request.form['renda'])
            escolaridade = int(request.form['escolaridade'])
            estado_civil = int(request.form['estado_civil'])
            numero_filhos = int(request.form['numero_filhos'])
            casa_propria = int(request.form['casa_propria'])
            
            # Preparar dados para a normalização
            input_data = {
                'Idade': idade,
                'Genero': genero,
                'Renda': renda,
                'Escolaridade': escolaridade,
                'Estado_Civil': estado_civil,
                'Numero_filhos': numero_filhos,
                'Casa_propria': casa_propria
            }
            
            # Normalizar os dados
            normalized_data = normalize_data(input_data)
            
            # Converter para DataFrame
            df_input = pd.DataFrame([normalized_data])
            
            # Fazer a predição
            result = model.predict(df_input)
            
            # Imprimir o resultado para depuração
            print("Resultado da predição:", result)
            
            # Ajustar o acesso ao resultado com base na estrutura observada
            # Dependendo da estrutura, ajuste a forma como você acessa o valor da predição
            if isinstance(result, pd.DataFrame):
                score = result.iloc[0, 0]  # Supondo que o valor está na primeira linha e primeira coluna
            else:
                # Ajuste para outros formatos se necessário
                score = result
            
            # Interpretar a predição
            if score < 0.33:
                prediction = 'Baixo'
            elif score < 0.66:
                prediction = 'Médio'
            else:
                prediction = 'Alto'
        
        except Exception as e:
            # Usar a sintaxe antiga para formatação de strings
            prediction = 'Erro: {}'.format(str(e))

    return render_template('form.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
