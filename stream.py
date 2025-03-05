import streamlit as st
import geopandas as gpd
import pandas as pd
import plotly.express as px
from PIL import Image
import base64
from streamlit_option_menu import option_menu
import plotly.io as pio
tabs=["Ana Sayfa","Türkiye Haritası"]
tabs = option_menu(
    menu_title=None,
    options=["Ana Sayfa","Türkiye Haritası","Partilerin Oy Trendi","Partilerin MV Sayısı Trendi","Cumhurbaşkanlığı Seçimi"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#d6094d"},
        "icon": {"color": "orange", "font-size": "18px"},
        "nav-link": {
            "font-size": "20px", 
            "text-align": "center", 
            "margin": "0px", 
            "--hover-color": "#444", 
            "padding-left": "20px",  # Add padding for consistent spacing
            "padding-right": "20px",  # Add padding for consistent spacing
            "height": "85px",  # Set a fixed height for all buttons
            "min-width": "150px",  # Ensure buttons do not shrink too small
            "white-space": "normal",  # Allow text to wrap if necessary
            "display": "inline-flex",  # Use inline-flex to adjust width to text content
            "justify-content": "center",
            "align-items": "center",
        },
        "nav-link-selected": {"background-color": "orange"},
    }
)


st.markdown(
    """
    <style>
    .title {
        font-size: 36px;  
        font-family: 'Freestyle Script', Courier !important;  
        color: red !important;  
        text-align: center;  
    }
    </style>
    <h1 class="title">Hazırlayan: Bora Kaya</h1>
    """, 
    unsafe_allow_html=True)
page=st.sidebar.radio("Sekmeler",tabs)
import logging
import streamlit as st

if page=="Cumhurbaşkanlığı Seçimi":
    import streamlit as st
    import pandas as pd

    # Başlık
    st.title("Cumhurbaşkanı Adayları ve Parti Destek Oranı Simülasyonu")

    # Parti ve Aday Listesi
    parties = ['A Partisi', 'B Partisi', 'C Partisi']
    candidates = ['X Adayı', 'Y Adayı', 'Z Adayı']

    # Başlangıçta DataFrame oluşturma
    data = {
        'A Partisi': [0, 0, 0],
        'B Partisi': [0, 0, 0],
        'C Partisi': [0, 0, 0]
    }

    # DataFrame oluştur
    df = pd.DataFrame(data, index=candidates)

    # Tabloyu gösterme ve veri girişi
    st.header("Destek Oranlarını Girin")

    # Kullanıcıdan destek oranlarını almak
    for i, candidate in enumerate(candidates):
        for party in parties:
            df.at[candidate, party] = st.number_input(f"{party} - {candidate} Destek Oranı (%)", min_value=0, max_value=100, step=1, key=f"{party}_{candidate}")

    # Verileri işleme ve gösterme
    if st.button("Verileri Gönder"):
        # Destek oranlarının toplamı %100 olmalı
        for party in parties:
            total_support = df[party].sum()
            if total_support != 100:
                st.warning(f"{party} için girilen destek oranları toplamı {total_support}%. Lütfen oranları düzelterek toplamın 100 olmasını sağlayın.")
            else:
                st.success(f"{party} için oranlar doğru şekilde girildi!")

        # Kullanıcıdan alınan verileri dataframe olarak gösterme
        st.write("Girilen Destek Oranları Tablosu:")
        st.dataframe(df)





# Streamlit log seviyelerini ayarlamak için logging modülünü kullanıyoruz
logging.basicConfig(level=logging.ERROR)
if page=="Ana Sayfa":
    
    import logging

    # Set logging level to suppress warnings
    logging.basicConfig(level=logging.ERROR)
            
    import sys
    import streamlit as st

    # Standart hata akışını geçici olarak yok sayalım
    class NullIO:
        def write(self, msg):
            pass

    # Hataları gizlemek için stderr'yi geçici olarak NullIO'ya yönlendiriyoruz
    sys.stderr = NullIO()
    # Sayfanın üst kısmında parti logolarını ve oy oranlarını ekleyelim
    st.header("Parti Oy Oranları")

    # Parti logolarını yükleyelim
    party_logos = {
        'AKP': 'akp_logo.png',  # AKP logosu
        'CHP': 'chp_logo.png',  # CHP logosu
        'MHP': 'mhp_logo.png',  # MHP logosu
        'DEM': 'dem_logo.png',  # DEM logosu
        'TİP': 'tip_logo.png' ,  # TİP logosu
        'ZP':'zafer_logo.png',
        'YRP':'refah_logo.png',
        'İYİ':'iyi_logo.png'
    }
    türkiyeoranlar=pd.read_csv("türkiyeoranlar.csv",index_col=0)
    fark=türkiyeoranlar.diff()

    cols = st.columns(8)  # 5 sütunlu bir yapı oluşturuyoruz

    for idx, (party, logo) in enumerate(party_logos.items()):
        logo_img = Image.open(logo)
        with cols[idx]:
            # Logoları aynı boyutta göstermek
            st.image(logo_img, width=50)
            
            # Oy oranını yazıyoruz
            st.markdown(f"""
                <h5 style="text-align: center; font-size: 16px; margin-top: 10px;">
                    {party}: {türkiyeoranlar[party].iloc[-1]:.2f}%
                </h5>
            """, unsafe_allow_html=True)

            # Fark değerini yazıyoruz
            fark_value = fark[party].iloc[-1]
            fark_color = 'green' if fark_value > 0 else 'red'  # Fark pozitifse yeşil, negatifse kırmızı
            if fark_value>0:
                st.markdown(f"""
                    <h6 style="text-align: center; font-size: 14px; color: {fark_color};">
                        (+{fark_value:.2f})
                    </h6>
                """, unsafe_allow_html=True)
            elif fark_value<=0:
                
                st.markdown(f"""
                    <h6 style="text-align: center; font-size: 14px; color: {fark_color};">
                        (-{fark_value:.2f})
                    </h6>
                """, unsafe_allow_html=True)


    
    import numpy as np
    import matplotlib.pyplot as plt


    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import math

    
    # Streamlit Başlık
    st.title("Parlamento Sandalye Dağılımı Simülasyonu")

    scenario = st.radio(
        "Senaryo Seçin",
        ("İYİ-ZP ittifak yapmıyor", "İYİ-ZP ittifak yapıyor")
    )

    if scenario == "İYİ-ZP ittifak yapmıyor":
        from collections import namedtuple, defaultdict
        mv=pd.read_csv("MV_Şub 2025.csv",index_col=0)
        mv=mv.sum()
        mv=pd.DataFrame(mv)

        # Parlamento Verisi (Örnek)
        data = {
            "PARTIDO": list(mv.index.values),
            "COLOR": [ '#FFCC00','#FF5733', '#0000FF', '#800080', '#6600CC'],
            "NAME": list(mv.index.values),
            "SEATS": list(mv.values.reshape(1,-1)[0]),
        }

        df = pd.DataFrame(data)

        Party = namedtuple('Party', ['name', 'color', 'size'])
        parties = [Party(row['PARTIDO'], row['COLOR'], row['SEATS']) for _, row in df.iterrows()]
        NUM_DEPUTIES = sum(p.size for p in parties)

        party_seats = [f"**{party.name}: {party.size}**" for party in parties]

        # Yazıları yan yana yazdırmak için markdown kullanma
        st.markdown(" | ".join(party_seats))
        st.image("parlemento.png")
    elif scenario== "İYİ-ZP ittifak yapıyor":
        mv=pd.read_csv("MV_Şub 2025_iyizp.csv",index_col=0)
        mv=mv.sum()
        mv=pd.DataFrame(mv)

        # Parlamento Verisi (Örnek)
        data = {
            "PARTIDO": list(mv.index.values),
            "COLOR": [ '#FFCC00','#FF5733', '#0000FF', '#800080', '#6600CC','#17e8e8','#9d0404'],
            "NAME": list(mv.index.values),
            "SEATS": list(mv.values.reshape(1,-1)[0]),
        }

        df = pd.DataFrame(data)

        import streamlit as st
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from collections import namedtuple, defaultdict

        
        # Parti ve sandalye sayıları için namedtuple tanımla
        Party = namedtuple('Party', ['name', 'color', 'size'])
        parties = [Party(row['PARTIDO'], row['COLOR'], row['SEATS']) for _, row in df.iterrows()]
        NUM_DEPUTIES = sum(p.size for p in parties)

        # Fonksiyonlar
        def calculate_radius(num_rows, initial_radius, radius_increment):
            """Calculate the radius for each row."""
            return [initial_radius + i * radius_increment for i in range(num_rows)]

        def calculate_arc_lengths(radius):
            """Calculate the arc lengths for each radius."""
            return [r * np.pi for r in radius]

        def calculate_deputies_per_row(num_deputies, arc_lengths, total_arc_length):
            """Calculate the number of deputies per row."""
            deputies_per_row = [int(num_deputies * (arc_length / total_arc_length)) for arc_length in arc_lengths]

            # Distribute the remaining deputies
            diff = num_deputies - sum(deputies_per_row)
            deputies_per_row[-1] += diff
            return deputies_per_row

        def generate_points(num_rows, radii, deputies_per_row):
            """Generate the points for each deputy."""
            points = []
            for row in range(num_rows):
                radius = radii[row]
                num_deputies_row = deputies_per_row[row]
                angles = np.linspace(0, np.pi, num_deputies_row)  # Spread deputies across the semicircle
                x = radius * np.cos(angles)
                y = radius * np.sin(angles)
                for i in range(num_deputies_row):
                    points.append((radius, angles[i], x[i], y[i]))
            return sorted(points, key=lambda x: x[1], reverse=True)

        def main(num_rows, initial_radius, radius_increment):
            """Main function to generate deputies' positions."""
            radius = calculate_radius(num_rows, initial_radius, radius_increment)
            arc_lengths = calculate_arc_lengths(radius)
            total_arc_length = sum(arc_lengths)
            deputies_per_row = calculate_deputies_per_row(NUM_DEPUTIES, arc_lengths, total_arc_length)
            points = generate_points(num_rows, radius, deputies_per_row)

            Deputy = namedtuple('Deputy', ['x', 'y', 'radius', 'angle'])
            return [Deputy(x, y, radius, angle) for (radius, angle, x, y) in points]

        def plot_deputies(deputies, parties, POINT_SIZE):
            """Plot the deputies on a chart."""
            deputies_by_party = defaultdict(list)

            current_index = 0

            for party in parties:
                party_deputies = deputies[current_index:current_index + party.size]
                deputies_by_party[party.name].extend(party_deputies)
                current_index += party.size

            deputies_by_party = dict(deputies_by_party)

            fig, ax = plt.subplots(figsize=(12, 6))

            for party in parties:
                color = party.color
                party_deputies = deputies_by_party[party.name]

                label = party.name

                for deputy in party_deputies:
                    if color == 'unassigned':
                        ax.scatter(deputy.x, deputy.y, s=POINT_SIZE, facecolors='none', edgecolors='grey', linewidth=0.75, label=label)
                    else:
                        ax.scatter(deputy.x, deputy.y, s=POINT_SIZE, alpha=1, color=color, label=label)
                    label = ""

            ax.set_aspect('equal')
            ax.axis('off')

            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0), ncol=6, frameon=False)

        # Constants
        NUM_ROWS = 12
        INITIAL_RADIUS = 30
        RADIUS_INCREMENT = 5
        POINT_SIZE = 100

        # Streamlit Başlık
        import warnings
        warnings.filterwarnings("ignore")
        import logging
        import streamlit as st

        # Streamlit log seviyelerini ayarlamak için logging modülünü kullanıyoruz
        logging.basicConfig(level=logging.ERROR)

        deputies = main(NUM_ROWS, INITIAL_RADIUS, RADIUS_INCREMENT)


        party_seats = [f"**{party.name}: {party.size}**" for party in parties]

        # Yazıları yan yana yazdırmak için markdown kullanma
        st.markdown(" | ".join(party_seats))
        st.image("parlemento_zpiyip.png")
 

    # Parti isimleri ve sandalye sayıları
    


    


    
    # Plotly ile interaktif harita oluşturmak

if page=="Partilerin MV Sayısı Trendi":
    import os
    import plotly.graph_objects as go

    data=pd.read_csv("türkiyeoranlar.csv",index_col=0)
    data_index=data.index.values
    data.index=pd.date_range(start="2024-01-31",freq="M",periods=len(data))
    dağılım=pd.read_csv("dağılım.csv")
    dağılım.index=pd.date_range(start="2024-01-31",freq="M",periods=len(dağılım))
    folder_path = './'  # Aynı dizinde bulunan dosyalar için

    # weighted_averages.index değerlerini alalım (bunları dışarıdan alıyoruz)
    weighted_averages_index = data_index
    # Klasördeki tüm CSV dosyalarını listeleyelim
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # Dosya isimlerinden ".csv" uzantısını çıkararak karşılaştırma yapalım
    matching_files = []

    for csv_file in csv_files:
        # ".csv" uzantısını çıkarıyoruz
        cleaned_csv_file = csv_file.replace('.csv', '')
        
        # weighted_averages.index değerleriyle karşılaştırma yapıyoruz
        if cleaned_csv_file in weighted_averages_index:
            matching_files.append(csv_file)

    matching_files=["MV_"+file for file in matching_files]
    matching_filesiyizp = [file.replace(".csv", "_iyizp.csv") for file in matching_files]
    şehirler=["TÜRKİYE GENELİ"]
    şehirler.extend(pd.read_csv("Şub 2025.csv")["İl Adı"].values)
    

    şehir = st.sidebar.selectbox("İl Seçin:", şehirler)

    scenario = st.radio(
        "Senaryo Seçin",
        ("İYİ-ZP ittifak yapmıyor", "İYİ-ZP ittifak yapıyor")
    )
    if scenario=="İYİ-ZP ittifak yapıyor":
        dağılımiyizp=pd.read_csv("dağılımiyizp.csv")
        dağılımiyizp.index=pd.date_range(start="2024-01-31",freq="M",periods=len(dağılımiyizp))

        if şehir=="TÜRKİYE GENELİ":
            fig= go.Figure()
            fig.add_trace(go.Scatter(
        x=[min(dağılımiyizp.index), max(dağılımiyizp.index)],  # x aralığı
        y=[300, 300],  # Yatay çizgi
        mode='lines',  # Sadece çizgi
        name="Meclis Çoğunluğu (300)",  # Etiket
        line=dict(color="black", dash="dash", width=5)  # Kırmızı, kesikli çizgi, kalınlık 6
    ))

            fig.add_trace(go.Scatter(
                        x=dağılımiyizp.index,
                        y=(dağılımiyizp["AKP"]+dağılım["MHP"]).values,
                        mode='lines+markers',
                        name="Cumhur İttifakı",
                        line=dict(color='#c26c11', width=4),
                        marker=dict(size=8, color="black"),
                        hovertemplate='Cumhur İttifakı<br>%{x|%Y-%m}<br>%{y:.2f}<extra></extra>'
                    ))
            fig.add_trace(go.Scatter(
                        x=dağılımiyizp.index,
                        y=dağılımiyizp["AKP"].values,
                        mode='lines+markers',
                        name="AKP",
                        line=dict(color='orange', width=4),
                        marker=dict(size=8, color="black"),
                        hovertemplate='AKP<br>%{x|%Y-%m}<br>%{y:.2f}<extra></extra>'
                    ))
            fig.add_trace(go.Scatter(
                        x=dağılımiyizp.index,
                        y=dağılımiyizp["CHP"].values,
                        mode='lines+markers',
                        name="CHP",
                        line=dict(color='red', width=4),
                        marker=dict(size=8, color="black"),
                        hovertemplate='CHP<br>%{x|%Y-%m}<br>%{y:.2f}<extra></extra>'
                    ))
            
            fig.add_trace(go.Scatter(
                        x=dağılımiyizp.index,
                        y=dağılımiyizp["MHP"].values,
                        mode='lines+markers',
                        name="MHP",
                        line=dict(color='blue', width=4),
                        marker=dict(size=8, color="black"),
                        hovertemplate='MHP<br>%{x|%Y-%m}<br>%{y:.2f}<extra></extra>'
                    ))
            fig.add_trace(go.Scatter(
                        x=dağılımiyizp.index,
                        y=dağılımiyizp["DEM"].values,
                        mode='lines+markers',
                        name="DEM",
                        line=dict(color='purple', width=4),
                        marker=dict(size=8, color="black"),
                        hovertemplate='DEM<br>%{x|%Y-%m}<br>%{y:.2f}<extra></extra>'
                    ))
            
            fig.add_trace(go.Scatter(
                        x=dağılımiyizp.index,
                        y=dağılımiyizp["TİP"].values,
                        mode='lines+markers',
                        name="TİP",
                        line=dict(color='black', width=4),
                        marker=dict(size=8, color="black"),
                        hovertemplate='TİP<br>%{x|%Y-%m}<br>%{y:.2f}<extra></extra>'
                    ))
            
            fig.add_trace(go.Scatter(
                        x=dağılımiyizp.index,
                        y=dağılımiyizp["İYİ"].values,
                        mode='lines+markers',
                        name="İYİ",
                        line=dict(color='#17e8e8', width=4),
                        marker=dict(size=8, color="black"),
                        hovertemplate='İYİ<br>%{x|%Y-%m}<br>%{y:.2f}<extra></extra>'
                    ))
            
            fig.add_trace(go.Scatter(
                        x=dağılımiyizp.index,
                        y=dağılımiyizp["ZP"].values,
                        mode='lines+markers',
                        name="ZP",
                        line=dict(color='#9d0404', width=4),
                        marker=dict(size=8, color="black"),
                        hovertemplate='ZP<br>%{x|%Y-%m}<br>%{y:.2f}<extra></extra>'
                    ))
            
            
            
            
            
            fig.update_layout(
                    xaxis=dict(
                        tickvals=dağılım.index.strftime("%Y-%m"),  # Original datetime index
                        ticktext=dağılım.index.strftime("%Y-%m"),
                        tickfont=dict(size=14, family="Arial Black", color="black")
                    ),
                    yaxis=dict(
                        tickfont=dict(size=14, family="Arial Black", color="black")
                    ),
                    font=dict(family="Arial", size=14, color="black"),
                    height=700,  # Grafik boyutunu artırma
                )
            st.plotly_chart(fig)
        else:
            
            şehirdata=pd.DataFrame()
            for file in matching_filesiyizp:

                df=pd.read_csv(file)
                df=df[df["Unnamed: 0"]==şehir]
                şehirdata=pd.concat([şehirdata,df],axis=0)
                şehirdata=şehirdata.set_index(pd.date_range(start="2024-01-31",freq="M",periods=len(şehirdata)))
            fig= go.Figure()
            def apply_offset(values, offset_value=0):
                """
                Aynı değere sahip olan yerlere ofset ekler.
                Bu ofset değeri küçük tutarak çizgiler net bir şekilde görünür olacak.
                """
                adjusted_values = values.copy()
                for i in range(1, len(values)):
                    if values[i] == values[i-1]:  # Aynı değeri bulduğunda
                        adjusted_values[i] += offset_value  # Küçük bir ofset ekler
                return adjusted_values

            # AKP Partisi
            fig.add_trace(go.Scatter(
                x=şehirdata.index,
                y=apply_offset(şehirdata["AKP"].values),  # Y değerlerine ofset uygula
                mode='lines+markers',
                name="AKP",
                line=dict(color='orange', width=4),
                marker=dict(size=8, color="black", symbol='circle'),
                hovertemplate='AK Parti<br>%{x|%Y-%m}<br>%{y:.2f}<extra></extra>'
            ))

            # CHP Partisi
            fig.add_trace(go.Scatter(
                x=şehirdata.index,
                y=apply_offset(şehirdata["CHP"].values),  # Y değerlerine ofset uygula
                mode='lines+markers',
                name="CHP",
                line=dict(color='red', width=4),
                marker=dict(size=8, color="black", symbol='square'),
                hovertemplate='CHP<br>%{x|%Y-%m}<br>%{y:.2f}<extra></extra>'
            ))

            # MHP Partisi
            fig.add_trace(go.Scatter(
                x=şehirdata.index,
                y=apply_offset(şehirdata["MHP"].values),  # Y değerlerine ofset uygula
                mode='lines+markers',
                name="MHP",
                line=dict(color='blue', width=4),
                marker=dict(size=8, color="black", symbol='star'),
                hovertemplate='MHP<br>%{x|%Y-%m}<br>%{y:.2f}<extra></extra>'
            ))

            # DEM Partisi
            fig.add_trace(go.Scatter(
                x=şehirdata.index,
                y=apply_offset(şehirdata["DEM"].values),  # Y değerlerine ofset uygula
                mode='lines+markers',
                name="DEM",
                line=dict(color='purple', width=4),
                marker=dict(size=8, color="black", symbol='diamond'),
                hovertemplate='DEM Parti<br>%{x|%Y-%m}<br>%{y:.2f}<extra></extra>'
            ))

            # İYİ Partisi
            fig.add_trace(go.Scatter(
                x=şehirdata.index,
                y=apply_offset(şehirdata["İYİ"].values),  # Y değerlerine ofset uygula
                mode='lines+markers',
                name="İYİ",
                line=dict(color='#17e8e8', width=4),
                marker=dict(size=8, color="black", symbol='cross'),
                hovertemplate='İYİ Parti<br>%{x|%Y-%m}<br>%{y:.2f}<extra></extra>'
            ))

            # ZP Partisi
            fig.add_trace(go.Scatter(
                x=şehirdata.index,
                y=apply_offset(şehirdata["ZP"].values),  # Y değerlerine ofset uygula
                mode='lines+markers',
                name="ZP",
                line=dict(color='#9d0404', width=4),
                marker=dict(size=8, color="black", symbol='x'),
                hovertemplate='Zafer Partisi<br>%{x|%Y-%m}<br>%{y:.2f}<extra></extra>'
            ))
      
            
            
            
            fig.update_layout(
                    xaxis=dict(
                        tickvals=şehirdata.index.strftime("%Y-%m"),  # Original datetime index
                        ticktext=şehirdata.index.strftime("%Y-%m"),
                        tickfont=dict(size=14, family="Arial Black", color="black")
                    ),
                    yaxis=dict(
                        tickfont=dict(size=14, family="Arial Black", color="black")
                    ),
                    font=dict(family="Arial", size=14, color="black"),
                    height=700,  # Grafik boyutunu artırma
                )
            st.markdown(f"<h2 style='text-align:left; color:black;'>{şehir} Milletvekili Dağılımı</h2>", unsafe_allow_html=True)
            st.plotly_chart(fig)



    else:

        import os
        import plotly.graph_objects as go

        data=pd.read_csv("türkiyeoranlar.csv",index_col=0)
        data_index=data.index.values
        data.index=pd.date_range(start="2024-01-31",freq="M",periods=len(data))
        dağılım=pd.read_csv("dağılım.csv")
        dağılım.index=pd.date_range(start="2024-01-31",freq="M",periods=len(dağılım))
        folder_path = './'  # Aynı dizinde bulunan dosyalar için

        # weighted_averages.index değerlerini alalım (bunları dışarıdan alıyoruz)
        weighted_averages_index = data_index
        # Klasördeki tüm CSV dosyalarını listeleyelim
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

        # Dosya isimlerinden ".csv" uzantısını çıkararak karşılaştırma yapalım
        matching_files = []

        for csv_file in csv_files:
            # ".csv" uzantısını çıkarıyoruz
            cleaned_csv_file = csv_file.replace('.csv', '')
            
            # weighted_averages.index değerleriyle karşılaştırma yapıyoruz
            if cleaned_csv_file in weighted_averages_index:
                matching_files.append(csv_file)

        matching_files=["MV_"+file for file in matching_files]



        if şehir=="TÜRKİYE GENELİ":
            fig= go.Figure()
            fig.add_trace(go.Scatter(
        x=[min(dağılım.index), max(dağılım.index)],  # x aralığı
        y=[300, 300],  # Yatay çizgi
        mode='lines',  # Sadece çizgi
        name="Meclis Çoğunluğu (300)",  # Etiket
        line=dict(color="black", dash="dash", width=5)  # Kırmızı, kesikli çizgi, kalınlık 6
    ))

            fig.add_trace(go.Scatter(
                        x=dağılım.index,
                        y=(dağılım["AKP"]+dağılım["MHP"]).values,
                        mode='lines+markers',
                        name="Cumhur İttifakı",
                        line=dict(color='#c26c11', width=4),
                        marker=dict(size=8, color="black"),
                        hovertemplate='Cumhur İttifakı<br>%{x|%Y-%m}<br>%{y:.2f}<extra></extra>'
                    ))
            fig.add_trace(go.Scatter(
                        x=dağılım.index,
                        y=dağılım["AKP"].values,
                        mode='lines+markers',
                        name="AKP",
                        line=dict(color='orange', width=4),
                        marker=dict(size=8, color="black"),
                        hovertemplate='AKP<br>%{x|%Y-%m}<br>%{y:.2f}<extra></extra>'
                    ))
            fig.add_trace(go.Scatter(
                        x=dağılım.index,
                        y=dağılım["CHP"].values,
                        mode='lines+markers',
                        name="CHP",
                        line=dict(color='red', width=4),
                        marker=dict(size=8, color="black"),
                        hovertemplate='CHP<br>%{x|%Y-%m}<br>%{y:.2f}<extra></extra>'
                    ))
            
            fig.add_trace(go.Scatter(
                        x=dağılım.index,
                        y=dağılım["MHP"].values,
                        mode='lines+markers',
                        name="MHP",
                        line=dict(color='blue', width=4),
                        marker=dict(size=8, color="black"),
                        hovertemplate='MHP<br>%{x|%Y-%m}<br>%{y:.2f}<extra></extra>'
                    ))
            fig.add_trace(go.Scatter(
                        x=dağılım.index,
                        y=dağılım["DEM"].values,
                        mode='lines+markers',
                        name="DEM",
                        line=dict(color='purple', width=4),
                        marker=dict(size=8, color="black"),
                        hovertemplate='DEM<br>%{x|%Y-%m}<br>%{y:.2f}<extra></extra>'
                    ))
            
            fig.add_trace(go.Scatter(
                        x=dağılım.index,
                        y=dağılım["TİP"].values,
                        mode='lines+markers',
                        name="TİP",
                        line=dict(color='black', width=4),
                        marker=dict(size=8, color="black"),
                        hovertemplate='TİP<br>%{x|%Y-%m}<br>%{y:.2f}<extra></extra>'
                    ))
            
            
            
            fig.update_layout(
                    xaxis=dict(
                        tickvals=dağılım.index.strftime("%Y-%m"),  # Original datetime index
                        ticktext=dağılım.index.strftime("%Y-%m"),
                        tickfont=dict(size=14, family="Arial Black", color="black")
                    ),
                    yaxis=dict(
                        tickfont=dict(size=14, family="Arial Black", color="black")
                    ),
                    font=dict(family="Arial", size=14, color="black"),
                    height=700,  # Grafik boyutunu artırma
                )
            st.plotly_chart(fig)
        else:
            
            şehirdata=pd.DataFrame()
            for file in matching_filesiyizp:

                df=pd.read_csv(file)
                df=df[df["Unnamed: 0"]==şehir]
                şehirdata=pd.concat([şehirdata,df],axis=0)
                şehirdata=şehirdata.set_index(pd.date_range(start="2024-01-31",freq="M",periods=len(şehirdata)))
            fig= go.Figure()
            fig.add_trace(go.Scatter(
                        x=şehirdata.index,
                        y=şehirdata["AKP"].values,
                        mode='lines+markers',
                        name="AKP",
                        line=dict(color='orange', width=4),
                        marker=dict(size=8, color="black"),
                        hovertemplate='AK Parti<br>%{x|%Y-%m}<br>%{y:.2f}<extra></extra>'
                    ))
            fig.add_trace(go.Scatter(
                        x=şehirdata.index,
                        y=şehirdata["CHP"].values,
                        mode='lines+markers',
                        name="CHP",
                        line=dict(color='red', width=4),
                        marker=dict(size=8, color="black"),
                        hovertemplate='CHP<br>%{x|%Y-%m}<br>%{y:.2f}<extra></extra>'
                    ))
            
            fig.add_trace(go.Scatter(
                        x=şehirdata.index,
                        y=şehirdata["MHP"].values,
                        mode='lines+markers',
                        name="MHP",
                        line=dict(color='blue', width=4),
                        marker=dict(size=8, color="black"),
                        hovertemplate='MHP<br>%{x|%Y-%m}<br>%{y:.2f}<extra></extra>'
                    ))
            fig.add_trace(go.Scatter(
                        x=şehirdata.index,
                        y=şehirdata["DEM"].values,
                        mode='lines+markers',
                        name="DEM",
                        line=dict(color='purple', width=4),
                        marker=dict(size=8, color="black"),
                        hovertemplate='DEM Parti<br>%{x|%Y-%m}<br>%{y:.2f}<extra></extra>'
                    ))
            
            
            
            
            
            
            fig.update_layout(
                    xaxis=dict(
                        tickvals=şehirdata.index.strftime("%Y-%m"),  # Original datetime index
                        ticktext=şehirdata.index.strftime("%Y-%m"),
                        tickfont=dict(size=14, family="Arial Black", color="black")
                    ),
                    yaxis=dict(
                        tickfont=dict(size=14, family="Arial Black", color="black")
                    ),
                    font=dict(family="Arial", size=14, color="black"),
                    height=700,  # Grafik boyutunu artırma
                )
            st.markdown(f"<h2 style='text-align:left; color:black;'>{şehir} Milletvekili Dağılımı </h2>", unsafe_allow_html=True)
            st.plotly_chart(fig)


if page=="Partilerin Oy Trendi":
    import os
    import plotly.graph_objects as go
    data=pd.read_csv("türkiyeoranlar.csv",index_col=0)
    data_index=data.index.values
    data.index=pd.date_range(start="2024-01-31",freq="M",periods=len(data))

    folder_path = './'  # Aynı dizinde bulunan dosyalar için

    # weighted_averages.index değerlerini alalım (bunları dışarıdan alıyoruz)
    weighted_averages_index = data_index
    # Klasördeki tüm CSV dosyalarını listeleyelim
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # Dosya isimlerinden ".csv" uzantısını çıkararak karşılaştırma yapalım
    matching_files = []

    for csv_file in csv_files:
        # ".csv" uzantısını çıkarıyoruz
        cleaned_csv_file = csv_file.replace('.csv', '')
        
        # weighted_averages.index değerleriyle karşılaştırma yapıyoruz
        if cleaned_csv_file in weighted_averages_index:
            matching_files.append(csv_file)
    şehirler=["TÜRKİYE GENELİ"]
    şehirler.extend(pd.read_csv("Şub 2025.csv")["İl Adı"].values)
    

    şehir = st.sidebar.selectbox("İl Seçin:", şehirler)



    if şehir=="TÜRKİYE GENELİ":
        fig= go.Figure()
        fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data["AKP"].values,
                    mode='lines+markers',
                    name="AKP",
                    line=dict(color='orange', width=4),
                    marker=dict(size=8, color="black"),
                    hovertemplate='AKP<br>%{x|%Y-%m}<br>%{y:.2f}<extra></extra>'
                ))
        fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data["CHP"].values,
                    mode='lines+markers',
                    name="CHP",
                    line=dict(color='red', width=4),
                    marker=dict(size=8, color="black"),
                    hovertemplate='CHP<br>%{x|%Y-%m}<br>%{y:.2f}<extra></extra>'
                ))
        
        fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data["MHP"].values,
                    mode='lines+markers',
                    name="MHP",
                    line=dict(color='blue', width=4),
                    marker=dict(size=8, color="black"),
                    hovertemplate='MHP<br>%{x|%Y-%m}<br>%{y:.2f}<extra></extra>'
                ))
        fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data["DEM"].values,
                    mode='lines+markers',
                    name="DEM",
                    line=dict(color='purple', width=4),
                    marker=dict(size=8, color="black"),
                    hovertemplate='DEM<br>%{x|%Y-%m}<br>%{y:.2f}<extra></extra>'
                ))
        
        fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data["İYİ"].values,
                    mode='lines+markers',
                    name="İYİ",
                    line=dict(color='#17e8e8', width=4),
                    marker=dict(size=8, color="black"),
                    hovertemplate='İYİ<br>%{x|%Y-%m}<br>%{y:.2f}<extra></extra>'
                ))
        
        fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data["ZP"].values,
                    mode='lines+markers',
                    name="ZP",
                    line=dict(color='#9d0404', width=4),
                    marker=dict(size=8, color="black"),
                    hovertemplate='ZP<br>%{x|%Y-%m}<br>%{y:.2f}<extra></extra>'
                ))
        
        fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data["YRP"].values,
                    mode='lines+markers',
                    name="YRP",
                    line=dict(color='#0f9b4a', width=4),
                    marker=dict(size=8, color="black"),
                    hovertemplate='YRP<br>%{x|%Y-%m}<br>%{y:.2f}<extra></extra>'
                ))
        
        
        fig.update_layout(
                xaxis=dict(
                    tickvals=data.index.strftime("%Y-%m"),  # Original datetime index
                    ticktext=data.index.strftime("%Y-%m"),
                    tickfont=dict(size=14, family="Arial Black", color="black")
                ),
                yaxis=dict(
                    tickfont=dict(size=14, family="Arial Black", color="black")
                ),
                font=dict(family="Arial", size=14, color="black"),
                height=700,  # Grafik boyutunu artırma
            )
        st.markdown(f"<h2 style='text-align:left; color:black;'>{şehir} Partilerin Oy Oranı </h2>", unsafe_allow_html=True)
        st.plotly_chart(fig)
    else:
        
        şehirdata=pd.DataFrame()
        for file in matching_files:

            df=pd.read_csv(file)
            df=df[df["İl Adı"]==şehir]
            şehirdata=pd.concat([şehirdata,df],axis=0)
            şehirdata=şehirdata.set_index(pd.date_range(start="2024-01-31",freq="M",periods=len(şehirdata)))
        fig= go.Figure()
        fig.add_trace(go.Scatter(
                    x=şehirdata.index,
                    y=şehirdata["AKP"].values,
                    mode='lines+markers',
                    name="AKP",
                    line=dict(color='orange', width=4),
                    marker=dict(size=8, color="black"),
                    hovertemplate='AK Parti<br>%{x|%Y-%m}<br>%{y:.2f}<extra></extra>'
                ))
        fig.add_trace(go.Scatter(
                    x=şehirdata.index,
                    y=şehirdata["CHP"].values,
                    mode='lines+markers',
                    name="CHP",
                    line=dict(color='red', width=4),
                    marker=dict(size=8, color="black"),
                    hovertemplate='CHP<br>%{x|%Y-%m}<br>%{y:.2f}<extra></extra>'
                ))
        
        fig.add_trace(go.Scatter(
                    x=şehirdata.index,
                    y=şehirdata["MHP"].values,
                    mode='lines+markers',
                    name="MHP",
                    line=dict(color='blue', width=4),
                    marker=dict(size=8, color="black"),
                    hovertemplate='MHP<br>%{x|%Y-%m}<br>%{y:.2f}<extra></extra>'
                ))
        fig.add_trace(go.Scatter(
                    x=şehirdata.index,
                    y=şehirdata["DEM"].values,
                    mode='lines+markers',
                    name="DEM",
                    line=dict(color='purple', width=4),
                    marker=dict(size=8, color="black"),
                    hovertemplate='DEM Parti<br>%{x|%Y-%m}<br>%{y:.2f}<extra></extra>'
                ))
        
        fig.add_trace(go.Scatter(
                    x=şehirdata.index,
                    y=şehirdata["İYİ"].values,
                    mode='lines+markers',
                    name="İYİ",
                    line=dict(color='#17e8e8', width=4),
                    marker=dict(size=8, color="black"),
                    hovertemplate='İYİ Parti<br>%{x|%Y-%m}<br>%{y:.2f}<extra></extra>'
                ))
        
        fig.add_trace(go.Scatter(
                    x=şehirdata.index,
                    y=şehirdata["ZP"].values,
                    mode='lines+markers',
                    name="ZP",
                    line=dict(color='#9d0404', width=4),
                    marker=dict(size=8, color="black"),
                    hovertemplate='Zafer Partisi<br>%{x|%Y-%m}<br>%{y:.2f}<extra></extra>'
                ))
        
        fig.add_trace(go.Scatter(
                    x=şehirdata.index,
                    y=şehirdata["YRP"].values,
                    mode='lines+markers',
                    name="YRP",
                    line=dict(color='#0f9b4a', width=4),
                    marker=dict(size=8, color="black"),
                    hovertemplate='Yeniden Refah Partisi<br>%{x|%Y-%m}<br>%{y:.2f}<extra></extra>'
                ))
        
        
        fig.update_layout(
                xaxis=dict(
                    tickvals=şehirdata.index.strftime("%Y-%m"),  # Original datetime index
                    ticktext=şehirdata.index.strftime("%Y-%m"),
                    tickfont=dict(size=14, family="Arial Black", color="black")
                ),
                yaxis=dict(
                    tickfont=dict(size=14, family="Arial Black", color="black")
                ),
                font=dict(family="Arial", size=14, color="black"),
                height=700,  # Grafik boyutunu artırma
            )
        st.markdown(f"<h2 style='text-align:left; color:black;'>{şehir} Oy Oranları</h2>", unsafe_allow_html=True)
        st.plotly_chart(fig)
    
    
if page=="Türkiye Haritası":
    gdf = gpd.read_file('turkey_administrativelevels0_1_2/tur_polbnda_adm1.shp')
    oranlar=pd.read_csv("Şub 2025.csv")
    oranlar.loc[oranlar["İl Adı"]=="ESKİŞEHİR","İl Adı"]="ESKİŞEHIR"
    for col in oranlar.columns[1:]:
        oranlar[col]=oranlar[col].round(2)
    # Shapefile ile veri birleştirme (il adı üzerinden)
    gdf = gdf.merge(oranlar, left_on='adm1_tr', right_on='İl Adı')

    # Harita verilerinin hazırlanması
    gdf['Partiler'] = gdf[['AKP', 'CHP', 'MHP', 'DEM', 'TİP']].idxmax(axis=1)

    # Renkler için bir sözlük oluştur
    party_colors = {
        'AKP': 'orange',
        'CHP': 'red',
        'MHP': 'blue',
        'DEM': 'purple'
    }

    # Renkleri yeni bir sütunda belirleyin
    gdf['color'] = gdf['Partiler'].map(party_colors)
    gdf=gdf.set_index("İl Adı")

    # Streamlit Uygulaması Başlığı
    st.title("Türkiye Seçim Haritası")
    st.write("Harita üzerinde her ildeki partilerin oy oranlarını görebilirsiniz.")
    fig = px.choropleth(gdf, 
                        geojson=gdf.geometry, 
                        locations=gdf.index,
                        color='Partiler',
                        hover_name=gdf.index,  # Hoverda il adı görünsün
                        hover_data={  # Hoverda sadece partilerin oy oranları görünsün
                            'AKP': True,
                            'CHP': True,
                            'MHP': True,
                            'DEM': True,
                            'TİP': True,
                            'ZP': True,  # Burada ZP'yi istemiyorsanız False yapabilirsiniz
                            'YRP': True ,  # Burada YRP'yi istemiyorsanız False yapabilirsiniz
                            'İYİ':True
                        },
                        color_discrete_map=party_colors,
                        title="Türkiye Seçim Haritası")

    # Harita'nın doğru görünebilmesi için geojson özelliğini ayarlıyoruz
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(
        autosize=True,
        height=800,  # Yüksekliği artırdık
        margin={"r":0,"t":40,"l":0,"b":0}  # Kenar boşluklarını sıfırladık
    )
    # Plotly figürünü HTML formatına dönüştür
    html_str = pio.to_html(fig, full_html=False)

    # HTML formatındaki haritayı tam ekran olarak Streamlit'e ekleyelim
    st.components.v1.html(html_str, height=800, width=1000)

