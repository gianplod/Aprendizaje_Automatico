import requests
import json
import urllib

# Request data goes here
# The example below assumes JSON formatting which may be updated
# depending on the format your endpoint expects.
# More information can be found here:
# https://docs.microsoft.com/azure/machine-learning/how-to-deploy-advanced-entry-script
data = [{"Id": 1, "MSSubClass": 60, "MSZoning": "RL", "LotFrontage": 65.0, "LotArea": 8450, "Street": "Pave", "Alley": None, "LotShape": "Reg", "LandContour": "Lvl", "Utilities": "AllPub", "LotConfig": "Inside", "LandSlope": "Gtl", "Neighborhood": "CollgCr", "Condition1": "Norm", "Condition2": "Norm", "BldgType": "1Fam", "HouseStyle": "2Story", "OverallQual": 7, "OverallCond": 5, "YearBuilt": 2003, "YearRemodAdd": 2003, "RoofStyle": "Gable", "RoofMatl": "CompShg", "Exterior1st": "VinylSd", "Exterior2nd": "VinylSd", "MasVnrType": "BrkFace", "MasVnrArea": 196.0, "ExterQual": "Gd", "ExterCond": "TA", "Foundation": "PConc", "BsmtQual": "Gd", "BsmtCond": "TA", "BsmtExposure": "No", "BsmtFinType1": "GLQ", "BsmtFinSF1": 706, "BsmtFinType2": "Unf", "BsmtFinSF2": 0, "BsmtUnfSF": 150, "TotalBsmtSF": 856,
         "Heating": "GasA", "HeatingQC": "Ex", "CentralAir": "Y", "Electrical": "SBrkr", "1stFlrSF": 856, "2ndFlrSF": 854, "LowQualFinSF": 0, "GrLivArea": 1710, "BsmtFullBath": 1, "BsmtHalfBath": 0, "FullBath": 2, "HalfBath": 1, "BedroomAbvGr": 3, "KitchenAbvGr": 1, "KitchenQual": "Gd", "TotRmsAbvGrd": 8, "Functional": "Typ", "Fireplaces": 0, "FireplaceQu": None, "GarageType": "Attchd", "GarageYrBlt": 2003.0, "GarageFinish": "RFn", "GarageCars": 2, "GarageArea": 548, "GarageQual": "TA", "GarageCond": "TA", "PavedDrive": "Y", "WoodDeckSF": 0, "OpenPorchSF": 61, "EnclosedPorch": 0, "3SsnPorch": 0, "ScreenPorch": 0, "PoolArea": 0, "PoolQC": None, "Fence": None, "MiscFeature": None, "MiscVal": 0, "MoSold": 2, "YrSold": 2000, "SaleType": "WD", "SaleCondition": "Normal"}]

body = str.encode(json.dumps(data))

url = 'http://24e47ee2-9c96-468b-9f06-66a72050a27a.westus2.azurecontainer.io/score'


headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}

try:
    response = requests.post(url, body, headers)
    prediction = response.json()
    print(f"Precio de venta estimado: {json.loads(prediction)[0]['prediction_label']}")
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))
    print(error.info())
    print(error.read().decode("utf8", 'ignore'))
