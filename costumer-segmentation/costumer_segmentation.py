
#Yapay zeka ile müþteri segmentasyonu
# merhaba bu projede bir avm kartýna sahip müþterilerin analiz edip sýnýflandýean yapay zeka kodluyorum


#K- Means Clustering :  çok fazla sayýdaki ham veriyi gruplara ayýrmak için kullanýlýr. Ham veriler hakkýnda önceden sizin sýnýflandýrma yapmamanýz gerekmektedir. 
#Yapay zeka sizin için sýnýflandýrmayý yapar ona yol göstermenize gerek yoktur 
#tüm iþlemi kendisi yapar( unsupervised learning modeller grubuna girer) Bu iþlemi yaparken centroid sistemini kullanýr (yuvarlak þekiller)


#Ham datayý gruplandýrmak için önce kaç adet grup kullanýlacaðýný algoritmaya söylememiz gerekiyor. 
#K-Means kelimesindeki K adet centroidi belirtir. K-Means algoritmasý þu þekilde çalýþýr;
#1-) K adet centroid ( merkez noktasý ) seçilir.
#2-) Her veriyi( noktayý) kendisine en yakýn centroid grubun dahil eder.
#3-) Her centroidin merkezini kendi grubundaki noktalara bakarak tekrar hesaplar ve centroidi yeni merkez lokasyonuna koyar.
#4-) 2. ve 3. adýmý centroidler artýk yer deðiþtirmeyene kadar devam ettirir.
#5-) Artýk centroidlerin merkezi deðiþmemektedir, gruplandýrmayý bitirir. Gruplandýrma bittiðinde algoritmaya yeni bir veri(nokta) geldiðinde en yakýn centroidi belirleyerek o gruba dahil eder.



#K deðerini belirlerken elbow metodu kullanýlýr. K deðerleri distortion deðerleriyle birlikte grafiðe döktüðünüzde dirsek noktasýnda bulunan K deðeri en optimal K deðeri olarak kabul edilir.

#Bu projede konsept : AVM yönetimi müþterilerde bulunan AVM kartlarý sayesinde müþterilere ait bazý bilgileri kaydetmiþtir ancak henüz herhangi bir gruplama yapmamýþtýr, 
#projede bu müþteriler için gruplandýrma yapacaðýz. K-Means algoritmasý kullanarak önce bu müþterileri kaç gruba ayýracaðýmýzý belirleyeceðiz sonra da gruplara dahil edeceðiz. 
#Toplam 201 adet müþteri bilgisinden oluþan veri seti ile çalýþacaðým.


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

# Veri setimize bir göz atalým:

#ilk genel bir bakýþ attýðýmda müþteri numarasý, yýllýk geliri ve harcama skoru önüme geliyor. 

# Bazý sütun isimleri çok uzun onlarý kýsaltalým:
df.rename(columns = {'Annual Income (k$)':'income'}, inplace = True)
df.rename(columns = {'Spending Score (1-100)':'score'}, inplace = True)



# modelimize verileri basmadan önce verileri normalize etmeliyim


# Geçen sefer normalizasyonu ben kendim  yapmýþtým bu sefer sklearn kütüphanesi içinde bulunan MinMaxScaler() fonksiyonunu kullanýyorum

scaler = MinMaxScaler()

scaler.fit(df[['income']])
df['income'] = scaler.transform(df[['income']])

scaler.fit(df[['score']])
df['score'] = scaler.transform(df[['score']])

df.head()


df.tail()
# önce K deðerini belirliyorum Elbow yöntemi olarak 

k_range = range(1,11)

list_dist = []

for k in k_range:
    kmeans_modelim = KMeans(n_clusters=k)
    kmeans_modelim.fit(df[['income','score']])
    list_dist.append(kmeans_modelim.inertia_)



plt.xlabel('K')
plt.ylabel('Distortion deðeri (inertia)')
plt.plot(k_range,list_dist)
plt.show()



# en iyi K deðerinin 5 olduðunu gözlemledim 
# K = 5 için bir K-Means modeli oluþturalým:
kmeans_modelim = KMeans(n_clusters = 5)
y_predicted = kmeans_modelim.fit_predict(df[['income','score']])
y_predicted



df['cluster'] = y_predicted
df.head()



# Centroidleri görelim:
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



# kmeans_modelim.cluster_centers_ numpy 2 boyutlu array olduðu için x ve y sütunlarýný kmeans_modelim.cluster_centers_[:,0] 
# ve kmeans_modelim.cluster_centers_[:,1] þeklinde scatter plot için alýyoruz:
plt.scatter(kmeans_modelim.cluster_centers_[:,0], kmeans_modelim.cluster_centers_[:,1], color='blue', marker='X', label='centroid')
plt.legend()
plt.show()

