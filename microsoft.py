import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
import numpy as np

st.set_page_config(
    page_title='Microsoft Stock',
    page_icon='💸'
    )
# Carica il DataFrame con i dati delle azioni di Microsoft
df = pd.read_csv('Microsoft_Stock.csv', parse_dates=['Date'])

# Converte la colonna 'Date' in tipo datetime

df['Date'] = pd.to_datetime(df['Date'])

# Imposta il titolo principale della pagina
st.title('Microsoft Stock Price')

# Aggiungi un sottotitolo usando markdown
st.markdown("### Dataset Analysis")

# Sidebar
st.sidebar.header('Opzioni di Visualizzazione')

# Seleziona l'intervallo di analisi o un intervallo personalizzato
selected_interval = st.sidebar.selectbox('Intervalli di Analisi', ['3 mesi', '6 mesi', '1 anno', '3 anni', '5 anni', 'Massimo', 'Personalizzato'], index=5)

if selected_interval == 'Personalizzato':
    # Imposta l'intervallo personalizzato
    start_date = pd.to_datetime(st.sidebar.date_input('Data di inizio', df['Date'].min()))
    end_date = pd.to_datetime(st.sidebar.date_input('Data di fine', df['Date'].max()))
else:
    # Calcola la data di inizio dell'intervallo in base alla selezione predefinita
    end_date = df['Date'].max()
    if selected_interval == '3 mesi':
        start_date = end_date - pd.DateOffset(months=3)
    elif selected_interval == '6 mesi':
        start_date = end_date - pd.DateOffset(months=6)
    elif selected_interval == '1 anno':
        start_date = end_date - pd.DateOffset(years=1)
    elif selected_interval == '3 anni':
        start_date = end_date - pd.DateOffset(years=3)
    elif selected_interval == '5 anni':
        start_date = end_date - pd.DateOffset(years=5)
    else:
        start_date = df['Date'].min()

# Filtra il DataFrame in base all'intervallo di tempo selezionato
filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

filtered_2=filtered_df.copy()
filtered_2['Date']=filtered_2['Date'].dt.date

# Visualizza la tabella con i dati nel DataFrame
with st.expander("### Financial Data"):
    st.table(filtered_2)

# Seleziona il campo "Volume" per l'istogramma
volume_field = 'Volume'  # Cambia qui se il nome della colonna è diverso

# Istogramma per il campo "Volume"
volume_histogram_fig = px.histogram(
    filtered_2,
    x=filtered_2['Date'],
    y=volume_field,
    nbins=30,# Numero di barre nell'istogramma
    title=f'Volume trend - from {start_date.date()} to {end_date.date()}'
)

volume_histogram_fig.update_traces(marker=dict(color='aqua'))
st.plotly_chart(volume_histogram_fig)

#-------------------- SCATTER PLOT ------------------------------------
# Seleziona il campo per lo scatter graph tramite radio nella sidebar
selected_scatter_field = st.sidebar.radio('Seleziona Campo per lo Scatter Graph', ['Low', 'Open', 'High'], index=0)

# Scatter Plot in funzione del Volume con gradazioni di colore diverse per ogni campo selezionato
scatter_fig_volume = px.scatter(
    filtered_2,
    x='Volume',
    y=selected_scatter_field,
    size='Volume',
    color='Close',  # Usa il campo selezionato per la gradazione di colore
    color_continuous_scale='Viridis',  # Scegli una gradazione di colore diversa (puoi cambiarla)
    title=f"Scatter Plot of {selected_scatter_field} against Volume - from {start_date.date()} to {end_date.date()}",
    labels={'Volume': 'Volume', selected_scatter_field: selected_scatter_field}
)
scatter_fig_volume.update_traces(marker=dict(symbol='circle'), mode='markers')  # Imposta il simbolo dei pallini
scatter_fig_volume.update_yaxes(type='linear')  # Per garantire che i campi selezionati vengano visualizzati correttamente
st.plotly_chart(scatter_fig_volume)




# Escludi la colonna 'Date' dai campi selezionabili
fields_to_select = [col for col in df.columns if col != 'Date']
# Seleziona i campi per il line chart tramite checkbox nella sidebar
# Imposta 'Close' come campo di default
default_fields = ['Close']
selected_fields = st.sidebar.multiselect('Seleziona Campo per il Line Chart', fields_to_select, default=default_fields)

# Line Chart per i campi selezionati
fig = px.line(
    filtered_2,
    x="Date",
    y=selected_fields,
    title=f"Line Chart of {selected_fields} Price - from {start_date.date()} to {end_date.date()}",
)
fig.update_xaxes(type='category')  # Per garantire che le date vengano visualizzate correttamente
st.plotly_chart(fig)


# Grafico a Candele (Candlestick) per il campo selezionato
selected_field = st.sidebar.selectbox('Seleziona Campo per il grafico a Candele', fields_to_select ,index=fields_to_select.index('Close'))
candlestick_fig = go.Figure(data=[go.Candlestick(x=filtered_df['Date'],
                                                  open=filtered_df[selected_field],
                                                  high=filtered_df[selected_field],
                                                  low=filtered_df[selected_field],
                                                  close=filtered_df[selected_field],
                                                  name=f'Candele {selected_field}')])
candlestick_fig.update_layout(title=f'Candle Graph of {selected_field} Price - from {start_date.date()} to {end_date.date()}',
                               xaxis_title='Data', yaxis_title='Valore')
st.plotly_chart(candlestick_fig)

#----------------------------------------------- FORECESTING -------------------------------------------------------------
df_for=df.copy()
df_for["Date"]=df_for["Date"].dt.date

# Prophet sidebar
st.sidebar.subheader('Prophet parameters configuration')
horizon_selection = st.sidebar.slider('Forecasting horizon (days)', min_value=1, max_value=365, value=90)   # valore di default 90 giorni
growth_selection = st.sidebar.radio(label='Growth', options=['linear', 'logistic'])



if growth_selection == 'logistic':
    st.sidebar.info('Configure logistic growth saturation as a percentage of latest Close')
    cap = st.sidebar.slider('Constant carrying capacity', min_value=1.0, max_value=1.5, value=1.2)    #cap valore da moltiplicare all'ultimo elemento
    cap_close = cap*df_for['Close'].iloc[-1]     # lo moltiplica per l'ultimo elemento 
    df_for['cap']=cap_close   # aggiunge una colonna al dataframe ed è una richiesta di Prophet che prevede una variabile detta CAP
                                 # che deve essere fornita time-stamp per time-stamp e in questo caso è costante

#---------------------------------------------------------------------- SPLITTING ------------------------------------------------------------------

# Definisci il numero di righe da utilizzare per il test set (horizontal_selection)
test_set_size = horizon_selection

# Calcola l'indice da cui iniziare il train set
start_index_train = len(df_for) - test_set_size - 1

# Estrai il train set e il test set
train_set = df_for.iloc[:start_index_train]
test_set = df_for.iloc[start_index_train:start_index_train + test_set_size]

train_RF = train_set.copy()  #.drop(columns=['cap'])
test_RF = test_set.copy()   #.drop(columns=['cap'])


#----------------------------------------------------------------------------------------------------------------------------------------------------

seasonality_selection = st.sidebar.radio(label='Seasonality', options=['additive', 'multiplicative'])

with st.sidebar.expander('Seasonality components'):
    # variabili booleane --> vengono trasformati in chiamate a funzione dove necessario
    weekly_selection = st.checkbox('Weekly')
    monthly_selection = st.checkbox('Monthly', value=True)
    yearly_selection = st.checkbox('Yearly', value=True)
with open('holiday_countries.txt', 'r') as fp:
    holiday_country_list = fp.read().split('\n')
    holiday_country_list.insert(0, 'None')  # dato che holiday puo anche non essere specificato metto campo none
holiday_country_selection = st.sidebar.selectbox(label="Holiday country", options=holiday_country_list)  


#Forecasting
st.header('Forecasting')

# Stampare i set di allenamento e test
with st.expander("### Test Seleted Data"):
    st.table(test_set)

# Prophet model fitting
with st.spinner('Model fitting..'):   # fin quando non ha eseguito il blocco di codice fa girare lo spinner
    prophet_df = train_set.rename(columns={'Date': 'ds', 'Close': 'y'})   # il modello prophet richiede due colonne ds--> data e y --> target
    model = Prophet(
        seasonality_mode=seasonality_selection,  # obligatoria
        weekly_seasonality=weekly_selection,   # boolenao e noi gli diamo la checkbox booleana 
        yearly_seasonality=yearly_selection,
        growth=growth_selection,     # oblgatoria 
        )
    if holiday_country_selection != 'None':
        model.add_country_holidays(country_name=holiday_country_selection)  
    # la stagionalità mensile non è prevista va aggiunta a mano     
    if monthly_selection:
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.fit(prophet_df)  # prende il dataframe e fa il fitting dei modelli



# Creare un DataFrame future basato sulle date del test set
future = pd.DataFrame(test_set['Date'])
future.columns = ['ds']  # Rinomina la colonna 'Date' a 'ds'

# Aggiungere la colonna 'cap' se growth_selection è 'logistic'
if growth_selection == 'logistic':
    future['cap'] = cap_close

# Fare previsioni sul DataFrame future
with st.spinner('Making predictions..'):
    forecast = model.predict(future)




# Visualizzare il DataFrame future e le previsioni
with st.expander("### Prophet Predictions"):
    st.table(forecast)

st.write("##### Prophet Forecasting")
# Prophet forecast plot
fig = px.scatter(prophet_df, x='ds', y='y', labels={'ds': 'Day', 'y': 'Close'})  # sono i puntini del grafico a dispersione
fig.add_scatter(x=forecast['ds'], y=forecast['yhat'], name='yhat')
fig.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'], name='yhat_lower')  # intervalli di confidenza
fig.add_scatter(x=forecast['ds'], y=forecast['yhat_upper'], name='yhat_upper')
st.plotly_chart(fig)




# Calcolare le metriche sul test set
test_true_values = test_set['Close'].values
test_predicted_values = forecast['yhat'].values


# Calcolare l'MSE, RMSE, MAE, MSLE e R-squared
mse = mean_squared_error(test_true_values, test_predicted_values)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test_true_values, test_predicted_values)
msle = mean_squared_log_error(test_true_values, test_predicted_values)
r_squared = r2_score(test_true_values, test_predicted_values)


# Creare un grafico Plotly
metrics_fig = go.Figure()

# Aggiungere le metriche al grafico
metrics_fig.add_trace(go.Scatter(x=['MSE', 'RMSE', 'MAE', 'MSLE', 'R-squared'],
                                 y=[mse, rmse, mae, msle, r_squared],
                                 mode='lines+markers',
                                 name='Metrics'))

# Aggiornare il layout del grafico
metrics_fig.update_layout(title='Regression Metrics on Test Set',
                          xaxis_title='Metrics',
                          yaxis_title='Values')

# Visualizzare il grafico
st.plotly_chart(metrics_fig)



# Creare liste per ciascuna metrica
mse_list = (test_true_values - test_predicted_values) ** 2
rmse_list = np.sqrt(mse_list)
mae_list = np.abs(test_true_values - test_predicted_values)
msle_list = np.log(1 + mse_list)
r_squared_list = (test_true_values - test_predicted_values.mean())**2

# Sidebar: Selettore metrica
st.sidebar.header('Metrics Configuration')
selected_metric = st.sidebar.selectbox('Select Metrics for both models',['MSE', 'RMSE', 'MAE', 'MSLE', 'R-squared'], index=2)

# Seleziona la lista corrispondente alla metrica scelta
selected_list = mse_list if selected_metric == 'MSE' else rmse_list if selected_metric == 'RMSE' else mae_list if selected_metric == 'MAE' else msle_list if selected_metric == 'MSLE' else r_squared_list

# Creare un grafico Plotly
metric_chart = go.Figure()
metric_chart.add_trace(go.Scatter(x=test_set['Date'], y=selected_list.tolist(), mode='lines', name=selected_metric))

# Aggiornare il layout del grafico
metric_chart.update_layout(title=f'{selected_metric} Curve on Test Set',
                           xaxis_title='Date',
                           yaxis_title=f'{selected_metric} Value')

# Visualizzare il grafico
st.plotly_chart(metric_chart)



#-------------------------------------------------------------- RANDOM FOREST ------------------------------------------------------------------------------

#FieldDate=test_RF["Date"]


train_RF.set_index('Date', inplace=True)
test_RF.set_index('Date', inplace= True)



# Creare una copia del DataFrame train_RF
train_RF_pred = train_RF.copy()

# Rimuovere la colonna "Close" dalla copia
train_RF_pred = train_RF_pred.drop(columns=['Close'])  # train set con tutte le colonne tranne close
test_RF_pred = test_RF.copy()

# Rimuovere la colonna "Close" dalla copia
test_RF_pred = test_RF_pred.drop(columns=['Close'])   # test set con tutte le colonne tranne close

# Creare una copia del DataFrame train_RF
train_RF_label = train_RF.copy()

# Eliminare le colonne specificate ("Open", "Low", "High", "Volume")
colonne_da_elim = ["Open", "Low", "High", "Volume"]
train_RF_label = train_RF_label.drop(columns=colonne_da_elim)  # train solo con label Close

test_RF_label= test_RF.copy()
test_RF_label= test_RF_label.drop(columns=colonne_da_elim)     # test solo con label Close


# --------------------------------- Definizione Iperparametri --------------------------------------

# Sidebar

with st.sidebar.expander("Random Forest Hyperparameters", expanded=True):
    n_estimators = st.slider('Number of Estimators (n_estimators)', min_value=1, max_value=500, value=100, step=1)
    random_state = st.slider('Random State (random_state)', min_value=1, max_value=100, value=42, step=1)
    criterion = st.selectbox('Criterion (criterion)', ['friedman_mse', 'poisson', 'absolute_error', 'squared_error'], index=3)
    max_depth = st.slider('Max Depth (max_depth)', min_value=1, max_value=100, value=30, step=1)



# Creazione del modello RandomForestRegressor con i parametri dalla sidebar
model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, criterion=criterion, max_depth=max_depth)

with st.spinner('Model fitting..'): 
    # Addestramento del modello
    model.fit(train_RF_pred, train_RF_label.values.ravel())

    # Effettuare previsioni sul set di test
    predictions = model.predict(test_RF_pred)


#print(predictions)
    
predictions_cleaned = predictions.ravel()

df_temp=test_RF_label.copy()
df_temp["Predictions"]= predictions_cleaned


with st.expander("### Random Forest Predictions"):
    st.table(df_temp)

# Prophet forecast plot
fig = px.scatter()

# Traccia per i dati di allenamento (giallo)
fig.add_scatter(x=train_RF.index, y=train_RF['Close'], mode='markers', name='Train Data', line=dict(color='mediumseagreen'))

# Traccia per la previsione sui dati di test (verde chiaro)
fig.add_scatter(x=test_RF.index, y=predictions, mode='lines', name='RF forecast (Test Data)', line=dict(color='red'))


# Aggiorna il layout del grafico
fig.update_layout(title='Random Forest Forecasting',
                  xaxis_title='Date',
                  yaxis_title='Close Price')

# Visualizza il grafico
st.plotly_chart(fig)

# --------------------------------------------------- METRICHE ------------------------------------------------------------------------


# Calcolare le metriche sul test set
test_true_values = test_RF_label['Close'].values
test_predicted_values = predictions_cleaned


# Calcolare l'MSE, RMSE, MAE, MSLE e R-squared
mse = mean_squared_error(test_true_values, test_predicted_values)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test_true_values, test_predicted_values)
msle = mean_squared_log_error(test_true_values, test_predicted_values)
r_squared = r2_score(test_true_values, test_predicted_values)


# Creare un grafico Plotly
metrics_fig = go.Figure()

# Aggiungere le metriche al grafico
light_green_color = 'lightgreen'
metrics_fig.add_trace(go.Scatter(x=['MSE', 'RMSE', 'MAE', 'MSLE', 'R-squared'],
                                 y=[mse, rmse, mae, msle, r_squared],
                                 mode='lines+markers',
                                 name='Metrics',
                                 line=dict(color=light_green_color)))

# Aggiornare il layout del grafico
metrics_fig.update_layout(title='Regression Metrics on Test Set',
                          xaxis_title='Metrics',
                          yaxis_title='Values')

# Visualizzare il grafico
st.plotly_chart(metrics_fig)



# Creare liste per ciascuna metrica
mse_list = (test_true_values - test_predicted_values) ** 2
rmse_list = np.sqrt(mse_list)
mae_list = np.abs(test_true_values - test_predicted_values)
msle_list = np.log(1 + mse_list)
r_squared_list = (test_true_values - test_predicted_values.mean())**2


# Seleziona la lista corrispondente alla metrica scelta
selected_list = mse_list if selected_metric == 'MSE' else rmse_list if selected_metric == 'RMSE' else mae_list if selected_metric == 'MAE' else msle_list if selected_metric == 'MSLE' else r_squared_list

# Creare un grafico Plotly
metric_chart = go.Figure()
metric_chart.add_trace(go.Scatter(x=test_RF.index, y=selected_list.tolist(), mode='lines', name=selected_metric, line=dict(color=light_green_color)))


# Aggiornare il layout del grafico
metric_chart.update_layout(title=f'{selected_metric} Curve on Test Set',
                           xaxis_title='Date',
                           yaxis_title=f'{selected_metric} Value')

# Visualizzare il grafico
st.plotly_chart(metric_chart)
