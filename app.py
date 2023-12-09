import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import io
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


df = pd.read_csv('Salary Data.csv')

buffer = io.StringIO()
df.info(buf=buffer)
info = buffer.getvalue()


st.title(' Apresentação de Resultados da Análise de Dados Salariais ')

st.write('## Link do GitHub')
st.link_button('GitHub', url='https://github.com/pedromvba/projeto_regressao_infnet')

st.write('## Objetivo')

st.write('Busca-se a partir do dataset analisado, verificar: (i) quais são os itens que mais influenciam o salário de um empregado e (ii) identificar qual característica influencia mais no salário, escolaridade ou experiência, de forma a direcionar os esforços das personas.')

st.write('Assim, o objetivo do projeto foi verificar, a partir de uma base de dados salarial, os componentes que mais influenciavam no salário pago a uma pessoa a partir do perfil/características apresentadas, bem como gerar um modelo de regressão linear sobre o caso')


st.write('## Amostra')
st.write('Uma amostra dos dados iniciais pode ser observada abaixo:')
st.write(df, width=700)


st.write('## Análise Exploratória')

st.write('### Principais Características Descritivas')
st.text(info)
st.write('Conforme registrado acima, observou-se dois registros nulos que foram excluídos do dataset.')


st.write('### Análise da Distribuição das Principais Variáveis')
# Age Distribution
fig_1 = plt.figure(figsize=(10,6), dpi=200)
# creating mean line
plt.vlines(x=df['Age'].mean(), ymin=0, ymax=60, colors='red', label='média')
# creating median line
plt.vlines(x=df['Age'].median(), ymin=0, ymax=60, colors='green', label ='mediana')
# histogram plot
sns.histplot(data=df, x='Age', bins=15)
# adjusting plot
plt.ylabel('Frequência')
plt.legend()
plt.title('Distribuição de Age')

st.write('A análise foi realizada de forma completa para as outras variáveis, sendo verificado inclusive o equilíbrio da base em termos de número de registos para cada categoria. Todavia, Age e Years of Experience se mostraram como as que necessitavam de maior atenção no processo, por isso o destaque.')

# Years of Experience Distribution
fig_2 = plt.figure(figsize=(10,6), dpi=200)

# creating mean line
plt.vlines(x=df['Years of Experience'].mean(), ymin=0, ymax=60, colors='red', label='média')
# creating median line
plt.vlines(x=df['Years of Experience'].median(), ymin=0, ymax=60, colors='green', label ='mediana')
# histogram plot
sns.histplot(data=df, x='Years of Experience', bins=12)
# adjusting plot
plt.ylabel('Frequência')
plt.legend()
plt.title('Distribuição de Years of Experience')


col1, col2  = st.columns(2, gap = 'medium')

col1.write('Age Distribution:')
col1.pyplot(fig_1)

col2.write('Years of Experience Distribution:')
col2.pyplot(fig_2)

st.write('Durante a análise, pôde-se observar que, tanto na feature Age quanto na feature Years of Experience, os valores da média e mediana são próximos, com uma diferença percentual de 4,10% e 13,08% respectivamente. Outro ponto a ser observado é que nos 2 casos a média é maior que a mediana. Essas observações indicam, no caso de Age, uma distribuição próxima a uma normal, com dados concentrados próximos à média e mediana e uma assimetria à direta no caso de Year of Experience, bem como a possibilidade de outliers à direita.')
st.write('Por fim, o desvio padrão de Age é maior do que o de Years of Experience, indicando um maior espalhamento dos valores com relação à média.')


st.write('### Identificação de Outliers')

# Age Outliers
fig_3 = plt.figure(figsize=(10,6), dpi=200)
# box plot
sns.boxplot(data=df, x='Age')
# adjusting plot
plt.title('Age')



# Years os Experience Outliers
fig_4 = plt.figure(figsize=(8,8), dpi=200)
# box plot
sns.boxplot(data=df, x='Years of Experience')
# adjusting plot
plt.title('Box Plot de Years of Experience')

col3, col4  = st.columns(2, gap = 'medium')

col3.write('Age Outliers:')
col3.pyplot(fig_3)

col4.write('Years of Experience Outliers:')
col4.pyplot(fig_4)

st.write('A partir dos box plots pode-se observar, pelo critério de 1,5 IQR, a existência de 2 outliers nos dados de Years of Experience. Os 2 outliers foram retirados para o treinamento do modelo')


st.write('## Dados após limpeza e remoção de outliers')
st.write('Conforme registrado, foram removidos duas entradas nulas e dois outliers de Years of Experience. A seguir os dados bem como a comparação entre a distribuição de Years of Experience nos 2 momentos.')

cleandf = pd.read_csv('clean_dataframe.csv')

buffer = io.StringIO()
cleandf.info(buf=buffer)
info2 = buffer.getvalue()

st.text(info2)


st.dataframe(cleandf)


fig_7 = plt.figure(figsize=(10,6), dpi=200)

# creating mean line
plt.vlines(x=cleandf['Years of Experience'].mean(), ymin=0, ymax=60, colors='red', label='média')
# creating median line
plt.vlines(x=cleandf['Years of Experience'].median(), ymin=0, ymax=60, colors='green', label ='mediana')
# histogram plot
sns.histplot(data=cleandf, x='Years of Experience', bins=12)
# adjusting plot
plt.ylabel('Frequência')
plt.legend()
plt.title('Distribuição de Years of Experience')




col5, col6  = st.columns(2, gap = 'medium')

col5.write('Years of Experience Distribution Inicial:')
col5.pyplot(fig_2)

col6.write('Years of Experience Distribution Após Ajustes:')
col6.pyplot(fig_7)


# Checking Data Correlations With Salary

st.write('### Correlação dos Dados com a Variável Alvo (Salary)')
transformed_df = pd.read_csv('modeling_dataframe.csv')

fig_5 = plt.figure(figsize=(10,8), dpi=300)
sns.heatmap(transformed_df.corr(), annot=True, fmt=".2f")
plt.title('Matriz de Correlação dos Dados')
st.pyplot(fig_5)
st.write('Assim observa-se uma forte correlação entre nossa variável alvo (Salary) e Years of Experience, Job Category Encoded, Education Level Encoded e Age. Por outro lado, a variável representativa do gênero (Gender Male) não possui alta correlação com Salary.')



st.write('## Preparação dos Dados')
st.write('De forma a preparar os dados para a modelagem, foi necessário transformar os dados categóricos em numéricos. Nesse ponto, para o caso de gender, utilizou-se a técnica do One Hot Encoding e excluiu-se uma das resultantes visto que a classificação em homens e mulheres é complementar.')
st.write('Para o caso de Education Level, utilizou-se o Label Encoding, considerando que existe uma relação de maior valor entre os diferentes níveis de especialização, sendo o mais valioso o PhD.')
st.write('Por fim, para o caso de Job Title, dividiu-se os dados em 4 categorias utilizando a própria , nomenclatura e também foi executado um Ordinal Encoding considerando a hierarquia entre as posições: (i) , profissionais que possuem junior no nome serão tratados como junior; (ii) profissionais com senior no nome, serão tratados como senior; (iii) profissionais com director, VP e CEO no nome serão tratados como diretores e (iv) outros serão tratados como analitas/plenos, uma faixa intermediária entre o junior e o senior.')

# Modeling

st.write('## Modelagem')
st.write('Nas modelagens realizadas, não se observou diferença significativa nos resultados devido à aplicação do standard scaler, assim optou-se, considerando o critério de simplicidade, por apresentar o modelo sem a aplicação da transformação. Adicionalmente, em testes realizados observou-se uma melhoria do modelo sem a inclusão da variável age.')
st.write('Inicialmente optou-se por fazer o teste sem age, pois age e years of experience são duas variáveis altamente correlacionadas. Em modelos de regressão linear variáveis altamente correlacionadas podem trazer ruído para a modelagem, prejudicando o modelo. No caso em análise, observou-se uma melhoria do modelo sem age.')
st.write('Outro ponto interessante observado foi que em que pese a baixa correlação entre o gênero da pessoa e o salário, o modelo performou melhor com a variável Gender_Male do que sem ela.')
st.write('Por fim, foram testados diferentes percentuais de split, 30:70, 25:75 e 20:80, sendo o último o que apresentou melhor desempenho. Acreditamos que isso ocorreu devido à baixa quantidade de dados disponíveis.')
st.write('Nesse sentido, o modelo apresentado a seguir não possui age como feature e foi gerado com um split 20:80.')

X = transformed_df.drop(['Salary', 'Age'],axis=1)
y = transformed_df['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)
mse = mean_squared_error(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
    

st.write('### Resultado da Regressão Linear')

coeficients = linreg.coef_
intercept = linreg.intercept_

st.write('Os coeficientes da Regressão Linear são:')
st.dataframe(coeficients)

st.write('O ponto de interseção com o eixo Y é:')
st.write(round(intercept,2))


st.write('### Métricas')   

metrics= pd.DataFrame({
    'Mean Squared Error': round(mse,2),
    'Mean Absolut Error': round(mae,2),
    'Root Mean Squared Error': round(rmse,2),
    'R2 Score': round(r2,2),
    'Salário Médio': round(df['Salary'].mean(),2)
}, index=['Valores'])

st.dataframe(metrics)

st.write('### Exemplo')
st.write(' Predição do salário de um homem com 15 anos de experiência, em uma posição de senior e com mestrado:')

prediction = linreg.predict(pd.DataFrame(
    { 
        'Education Level Encoded': 1, 
        'Gender_Male': True,
        'Job Category Encoded': 2,
        'Years of Experience': 15,
    
}, index = ['persona']))

st.write(f'Salário = {round(prediction[0],2)}')


st.write('## Conclusão') 

st.write('Durante a análise realizada, observou-se que os itens que mais influenciam o salário são os anos de experiência, bem como a categoria do trabalho (junior, analista, senior diretor). Por outro lado, o gênero não influenciou o salário. Sendo assim, no caso em análise, seria mais interessante a pessoa se inserir no mercado e começar a trabalhar do que buscar uma maior especialização.')
st.write('Por fim, foi gerado um modelo que permite predizer o salário de um individuo a partir da apresentação de: (i) nível de educação, (ii) gênero da pessoa, (iii) categoria do trabalho e (iv) anos de experiência.')
