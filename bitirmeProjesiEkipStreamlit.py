
import numpy as np
import pandas as pd
import streamlit as st

# Sayfa Ayarları
st.set_page_config(
    page_title="fetal health",
    page_icon="https://miro.medium.com/v2/resize:fit:2400/1*rGi8_JUoGX0L3W6nivmIAg@2x.png",
    menu_items={
        'Get Help': 'mailto:ata.ozarslan@istdsa.com',
        "About": "For More Information\n" + "https://github.com/ataozarslan/DSNov22"
    }
)

st.title("İşemrine Göre Ekiplerin Çalışma Performansları")



# Header Ekleme
st.header("Data Dictionary")
st.markdown("EKIP_ADI")
st.markdown("EMAIL")
st.markdown("EKIP_NO")
st.markdown("IRTIBAT_TEL_NO1")
st.markdown("GOREV_TURU_ACIKLAMA")
st.markdown("BILDIRIM_ZAMANI")
st.markdown("TAMAMLANMA_ZAMANI")
st.markdown("GOREV_DURUM")
st.markdown("TAMAMLANMA_SURESI")
st.markdown("ISEMRI_ID")
st.markdown("EKIP_ID")
st.markdown("BAGLI_OLDUGU_BIRIM_ID")
st.markdown("GOREV_TURU_ID")
st.markdown("TAMAMLANMA_HIZI")



# Pandasla veri setini okuyalım
df = pd.read_pickle("/Users/niluferceylan/Desktop/yapay zeka proje3/ekip_train_df8.pkl")




# Tablo Ekleme
st.dataframe(df.sample(50))

#---------------------------------------------------------------------------------------------------------------------

# Sidebarda Markdown Oluşturma
st.sidebar.markdown("**Choose** the features below to see the result!")

# Sidebarda Kullanıcıdan Girdileri Alma


isemri_id = st.sidebar.number_input("ISEMRI_ID")
ekip_id = st.sidebar.number_input("EKIP_ID")
gorev_turu_id =st.sidebar.number_input("GOREV_TURU_ID")


#---------------------------------------------------------------------------------------------------------------------
# Pickle kütüphanesi kullanarak eğitilen modelin tekrardan kullanılması
from joblib import load

logreg_model = load('ekip_logreg_model8.pkl')

input_df = pd.DataFrame({
    'ISEMRI_ID': [isemri_id],
    'EKIP_ID': [ekip_id],
    'GOREV_TURU_ID' : [gorev_turu_id]

})

# Verilerimizi ölçeklendirmeyi unutmuyoruz!
std_scale = load('ekip_scaler4.pkl')
scaled_input_df = std_scale.transform(input_df)
pred = logreg_model.predict(scaled_input_df)

pred_probability = np.round(logreg_model.predict_proba(scaled_input_df), 2)



#---------------------------------------------------------------------------------------------------------------------

st.header("Results")


# Sonuç Ekranı
if st.sidebar.button("Submit"):
   

       
    # Info mesajı oluşturma
            st.info("You can find the result below.")

        # Sorgulama zamanına ilişkin bilgileri elde etme
            from datetime import date, datetime

    
    


        # Sonuçları Görüntülemek için DataFrame
            results_df = pd.DataFrame({
            'Isemri ID': [isemri_id],

            'Ekip ID' : [ekip_id],
            'Gorev Turu ID' : [gorev_turu_id],
        
            'Tamamlama Durumu': [pred],
  

            })
            if pred == 1:
                results_df["Tamamlama Durumu"] = "Zamanında Tamamlandı"
           
            else:
                results_df["Tamamlama Durumu"] = "Geç Tamamlandı"
        
            st.table(results_df)
  
       

    
       
else:
    st.markdown("Please click the *Submit Button*!")
