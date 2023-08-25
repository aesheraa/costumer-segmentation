
#Yapay zeka ile m��teri segmentasyonu
# merhaba bu projede bir avm kart�na sahip m��terilerin analiz edip s�n�fland�ean yapay zeka kodluyorum


#K- Means Clustering :  �ok fazla say�daki ham veriyi gruplara ay�rmak i�in kullan�l�r. Ham veriler hakk�nda �nceden sizin s�n�fland�rma yapmaman�z gerekmektedir. 
#Yapay zeka sizin i�in s�n�fland�rmay� yapar ona yol g�stermenize gerek yoktur 
#t�m i�lemi kendisi yapar( unsupervised learning modeller grubuna girer) Bu i�lemi yaparken centroid sistemini kullan�r (yuvarlak �ekiller)


#Ham datay� grupland�rmak i�in �nce ka� adet grup kullan�laca��n� algoritmaya s�ylememiz gerekiyor. 
#K-Means kelimesindeki K adet centroidi belirtir. K-Means algoritmas� �u �ekilde �al���r;
#1-) K adet centroid ( merkez noktas� ) se�ilir.
#2-) Her veriyi( noktay�) kendisine en yak�n centroid grubun dahil eder.
#3-) Her centroidin merkezini kendi grubundaki noktalara bakarak tekrar hesaplar ve centroidi yeni merkez lokasyonuna koyar.
#4-) 2. ve 3. ad�m� centroidler art�k yer de�i�tirmeyene kadar devam ettirir.
#5-) Art�k centroidlerin merkezi de�i�memektedir, grupland�rmay� bitirir. Grupland�rma bitti�inde algoritmaya yeni bir veri(nokta) geldi�inde en yak�n centroidi belirleyerek o gruba dahil eder.



#K de�erini belirlerken elbow metodu kullan�l�r. K de�erleri distortion de�erleriyle birlikte grafi�e d�kt���n�zde dirsek noktas�nda bulunan K de�eri en optimal K de�eri olarak kabul edilir.

#Bu projede konsept : AVM y�netimi m��terilerde bulunan AVM kartlar� sayesinde m��terilere ait baz� bilgileri kaydetmi�tir ancak hen�z herhangi bir gruplama yapmam��t�r, 
#projede bu m��teriler i�in grupland�rma yapaca��z. K-Means algoritmas� kullanarak �nce bu m��terileri ka� gruba ay�raca��m�z� belirleyece�iz sonra da gruplara dahil edece�iz. 
#Toplam 201 adet m��teri bilgisinden olu�an veri seti ile �al��aca��m.


import numpy
import os

os.environ["OMP_NUM_THREADS"] = "1"


from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

df = pd.read_csv("Avm_Musterileri.csv")
df.head()



plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

# Veri setimize bir g�z atal�m:

#ilk genel bir bak�� att���mda m��teri numaras�, y�ll�k geliri ve harcama skoru �n�me geliyor. 

# Baz� s�tun isimleri �ok uzun onlar� k�saltal�m:
df.rename(columns = {'Annual Income (k$)':'income'}, inplace = True)
df.rename(columns = {'Spending Score (1-100)':'score'}, inplace = True)



# modelimize verileri basmadan �nce verileri normalize etmeliyim


# Ge�en sefer normalizasyonu ben kendim  yapm��t�m bu sefer sklearn k�t�phanesi i�inde bulunan MinMaxScaler() fonksiyonunu kullan�yorum

scaler = MinMaxScaler()

scaler.fit(df[['income']])
df['income'] = scaler.transform(df[['income']])

scaler.fit(df[['score']])
df['score'] = scaler.transform(df[['score']])

df.head()


df.tail()
# �nce K de�erini belirliyorum Elbow y�ntemi olarak 

k_range = range(1,11)

list_dist = []

for k in k_range:
    kmeans_modelim = KMeans(n_clusters=k)
    kmeans_modelim.fit(df[['income','score']])
    list_dist.append(kmeans_modelim.inertia_)



plt.xlabel('K')
plt.ylabel('Distortion de�eri (inertia)')
plt.plot(k_range,list_dist)
plt.show()



# en iyi K de�erinin 5 oldu�unu g�zlemledim 
# K = 5 i�in bir K-Means modeli olu�tural�m:
kmeans_modelim = KMeans(n_clusters = 5)
y_predicted = kmeans_modelim.fit_predict(df[['income','score']])
y_predicted



df['cluster'] = y_predicted
df.head()



# Centroidleri g�relim:
kmeans_modelim.cluster_centers_   




df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
df4 = df[df.cluster==3]
df5 = df[df.cluster==4]


plt.xlabel('income')
plt.ylabel('score')
plt.scatter(df1['income'],df1['score'],color='green')
plt.scatter(df2['income'],df2['score'],color='red')
plt.scatter(df3['income'],df3['score'],color='black')
plt.scatter(df4['income'],df4['score'],color='orange')
plt.scatter(df5['income'],df5['score'],color='purple')



# kmeans_modelim.cluster_centers_ numpy 2 boyutlu array oldu�u i�in x ve y s�tunlar�n� kmeans_modelim.cluster_centers_[:,0] 
# ve kmeans_modelim.cluster_centers_[:,1] �eklinde scatter plot i�in al�yoruz:
plt.scatter(kmeans_modelim.cluster_centers_[:,0], kmeans_modelim.cluster_centers_[:,1], color='blue', marker='X', label='centroid')
plt.legend()
plt.show()

