
import pandas as pd
from pycaret.regression import load_model, predict_model
from fastapi import FastAPI
import uvicorn

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model('api_eff')

# Define predict function
@app.post('/predict')
def predict(Pattern_x, Puissance_x, Cact_x, Norme_x, E1, Qprcal, Thm, ThOut1, ThOutn, V40ps, QPR, TYPE_TH, Lg_HE, Th_Pos_Regul, SETING__EN_CHIFFRE, lg_doigt_de_gant):
    data = pd.DataFrame([[Pattern_x, Puissance_x, Cact_x, Norme_x, E1, Qprcal, Thm, ThOut1, ThOutn, V40ps, QPR, TYPE_TH, Lg_HE, Th_Pos_Regul, SETING__EN_CHIFFRE, lg_doigt_de_gant]])
    data.columns = ['Pattern_x', 'Puissance_x', 'Cact_x', 'Norme_x', 'E1', 'Qprcal', 'Thm', 'ThOut1', 'ThOutn', 'V40ps', 'QPR', 'TYPE TH', 'Lg HE', 'Th Pos Regul', 'SETING  EN CHIFFRE', 'lg doigt de gant']
    predictions = predict_model(model, data=data) 
    return {'prediction': list(predictions['Label'])}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)