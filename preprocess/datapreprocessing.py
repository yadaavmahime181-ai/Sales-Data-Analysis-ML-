import pandas as pd
from Connection import DBconnect
class Preprocess:
    def cleaning(path):

        customer = pd.read_excel(path, sheet_name = "CustomerDetail")
        orders = pd.read_excel(path, sheet_name = "Jan")

        # feature engineering(Customer's Data) 
        customer = customer.drop(columns= ["CustomerID", "Month", "MonthName", "DateInText", "Gender", "Gender.1"])
        customer = customer.rename(columns= {"CustomerID.1": "CustomerID", "Gender.1":"Gender"})
        customer = customer.dropna()

        # feature engineering(Order's Data) 
        orders = orders.rename(columns = {"Quantity ordered new": "Qty"})
        orders = orders.dropna()
        
        return customer, orders
    
    
    def storeInDB(customer, orders):
        conn = DBconnect.getConnection()
        cur = conn.cursor()
        
        cur.close()
        conn.close()