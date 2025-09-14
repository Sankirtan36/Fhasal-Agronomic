import flask
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
data=pd.read_csv("./crop_detect/Assets/Crop_recommendation.csv")

# print(data.head())
# print(data.shape)
# print(data.isnull()) #only checks whether null or not
# print(data.isnull().sum())#sum's all the null values present in any column
# spitting labels and featues
# pahle colon ka matlab hota hai ki sare columns access karna aur 2nd  colon ka matlab hota hai ki konsi include nahi karni hai usse n samjho aur soch ki sirf n-1 tak he jayegi 
# 2nd case mae pahle colon ka matlab hai sari columnsko access karo aur -1 ka matlab ki sirf last wali column he lo
fet=data.iloc[:,:-1]

lab=data.iloc[:,-1]
fet_train,fet_test,lab_train,lab_test=train_test_split(fet,lab,test_size=0.25,random_state=42)
"""
random_state batata hai ki jo data liya hai sabne (variables) wo same rahe har bar change na ho 
agar isse nahi likhte ho tou har baar alag alag data lenege saab means train hone ke liye alag aur test karne ke liye aalg ,iske andar no. kuch bhi de sakte ho bass iska kam hai ki har bar data same dena har ko jo pichle bar diya tha (matlab agar refresh hojaye tou har bar data same mile)
ye model ko train karta hai 
"""
model=RandomForestClassifier()
print(model.fit(fet_train,lab_train))#trains model
predictions=model.predict(fet_test)
accuracy=model.score(fet_test,lab_test)
print("Accuracy: ",accuracy)
new_features=[[23,34,14,34.432123123,81.322354322,8.2342242,300.234234]]#value di hai 
predict_crop=model.predict(new_features)
print("Predicted Crop: ",predict_crop)
