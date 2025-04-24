import streamlit as st
import pandas as pd
import joblib
import openai
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from itertools import product
import os
import plotly.express as px
from dotenv import load_dotenv

#gen. setup
load_dotenv()



                                       #ladet api key aus .env file |bei bedarf fallback bauen mit model unter openai.api_key
openai.api_key = os.getenv("OPENAI_API_KEY")


default_values = {          # fallback für fehlende Eingaben , werden ERST aktiviert wenn user checkbox aktiviert
    "alter_jahre": 40,
    "wohnlage": "neutral",
    "entfernung_km": 20,
    "homeoffice_tage": 2,
    "beziehungsstatus": "ledig",
    "kinder": 1
}



model = joblib.load(Path("./models/rf_regressor.pkl"))              #modell laden (regression)
#modell ist ein RFRegr., der auf den Daten trainiert wurde (aus der dbeaver path)
#2000 simulierte Profile mit 7 Features (6 input, 1 output) 
#aus/für der DB(data) generiert = siehe generate_employees.py!!!


#features sind: alter, wohnlag, km entfernung, homeofficeanspruch die woche/monat, beziehungsstatus, etc ...... 

wohnlage_vals = ["gut", "neutral", "schlecht"]
beziehungsstatus_vals = ["ledig", "verheiratet", "geschieden", "verwitwet"]
encoder = OneHotEncoder(drop="first", sparse_output=False)
encoder.fit(pd.DataFrame(product(wohnlage_vals, beziehungsstatus_vals), columns=["wohnlage", "beziehungsstatus"]))

# config für streamlit
st.set_page_config(
    page_title="DAGENT - Krankheitsprognose",
    page_icon="imgs/icon2.png",  
    layout="wide"
)

if "show_csv" not in st.session_state:
    st.session_state["show_csv"] = False
if "show_llm" not in st.session_state:
    st.session_state["show_llm"] = False

# ui
st.title("decision-agent@DAGENT ")
st.markdown("Krankheitstage-Prognose für Bewerber mit anschließendem Vergleich und Analyse durch ein LLM.")
st.markdown("Anwendung zur (sim.) datengestützten Entscheidungshilfe bei Kandidatenvergleichen (z.b. für HR).")

num_candidates = st.slider("Wieviele Kandidaten sollen insgesamt verglichen werden?", 1, 5, 2)
columns = st.columns(num_candidates)
candidate_data, names = [], []


#schlefie für die kandidaten
#für eingabemaske der kandidaten
for i in range(num_candidates):
    with columns[i]:
        with st.container(border=True):
            st.subheader(f"Kandidat {i+1}")
            #slle inputfelder für die kandidaten, also JE kandidaten die selbe eingabemaske
            name = st.text_input(f"Name", f"Kandidat {i+1}", key=f"name_{i}")

            
            alter_missing = st.checkbox("Angabe fehlt (Alter)", key=f"alter_missing_{i}")
            alter = (
                default_values["alter_jahre"]
                if alter_missing else st.slider("Alter", 18, 65, 30, key=f"alter_{i}")
            )

            wohnlage_missing = st.checkbox("Angabe fehlt (Wohnlage)", key=f"wohnlage_missing_{i}")
            wohnlage = (
                default_values["wohnlage"]
                if wohnlage_missing else st.selectbox("Wohnlage", wohnlage_vals, key=f"wohnlage_{i}")
            )

            
            entf_missing = st.checkbox("Angabe fehlt (Entfernung)", key=f"entf_missing_{i}")
            entfernung = (
                default_values["entfernung_km"]
                if entf_missing else st.slider("Entfernung zum Betrieb (km)", 0, 100, 10, key=f"entf_{i}")
            )

            
            home_missing = st.checkbox("Angabe fehlt (Homeoffice)", key=f"home_missing_{i}")
            homeoffice = (
                default_values["homeoffice_tage"]
                if home_missing else st.slider("Homeoffice-Tage pro Woche", 0, 5, 2, key=f"homeoffice_{i}")
            )

            #
            bz_missing = st.checkbox("Angabe fehlt (Beziehungsstatus)", key=f"bz_missing_{i}")
            beziehungsstatus = (
                default_values["beziehungsstatus"]
                if bz_missing else st.selectbox("Beziehungsstatus", beziehungsstatus_vals, key=f"bz_{i}")
            )

            
            kinder_missing = st.checkbox("Angabe fehlt (Kinder)", key=f"kinder_missing_{i}")
            kinder = (
                default_values["kinder"]
                if kinder_missing else st.slider("Kinder", 0, 4, 1, key=f"kinder_{i}")
            )




            names.append(name)#speichert die namen der kandidaten in eine liste
            candidate_data.append({
                "alter_jahre": alter,
                "wohnlage": wohnlage,
                "entfernung_km": entfernung,
                "homeoffice_tage": homeoffice,
                "beziehungsstatus": beziehungsstatus,
                "kinder": kinder
            })


if st.button("Prognose berechnen & vergleichen"):                   #knopf für prognose
    df_input = pd.DataFrame(candidate_data)
    encoded = encoder.transform(df_input[["wohnlage", "beziehungsstatus"]])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(["wohnlage", "beziehungsstatus"]))
    X = pd.concat([df_input.drop(columns=["wohnlage", "beziehungsstatus"]).reset_index(drop=True), encoded_df], axis=1)
    predictions = model.predict(X)

    # results
    st.subheader("Ergebnisse")
    for i, pred in enumerate(predictions):
        st.info(f"{names[i]}: Erwartete Krankheitstage pro Jahr: {pred:.1f} Tage")

    best_idx = predictions.argmin()
    st.success(f"{names[best_idx]} hat die geringste prognostizierte Krankheitsanzahl und ist damit aktuell am geeignetsten.")
    st.caption("Hinweis: Modell trainiert auf 2000 simulierten Profilen.")

    #visualsierung der ergebnisse
    #via plotly  
    fig = px.bar(x=names, y=predictions, labels={"x": "Kandidat", "y": "Krankheitstage (pro Jahr)"}, title="Vergleich der Prognosen")
    st.plotly_chart(fig, use_container_width=True)

    
    st.session_state["last_predictions"] = predictions.tolist()
    st.session_state["last_inputs"] = df_input.to_dict(orient="records")            #wichtig für sessuion_state
    st.session_state["last_names"] = names

# csv export der ergebnisse
st.session_state["show_csv"] = st.checkbox("CSV-Download der Ergebnisse", value=st.session_state["show_csv"])
if st.session_state["show_csv"] and "last_predictions" in st.session_state:

                                                              #daataframe für csv export mit namen und kh tagen
    csv_df = pd.DataFrame({
        "Name": st.session_state["last_names"],
        "Erwartete Krankheitstage": st.session_state["last_predictions"]
    })
    st.download_button("CSV herunterladen", data=csv_df.to_csv(index=False), file_name="prognose.csv", mime="text/csv")

# llm analyse   | 
st.session_state["show_llm"] = st.checkbox("KI-Analyse anzeigen", value=st.session_state["show_llm"])
if st.session_state["show_llm"] and "last_predictions" in st.session_state:

    #beschreibungtext aus eigenen daten generieren
    
    beschreibung = ""
    for i, (data, pred) in enumerate(zip(st.session_state["last_inputs"], st.session_state["last_predictions"])):
        beschreibung += (
            f"{st.session_state['last_names'][i]}:\n"
            f"- Alter: {data['alter_jahre']} Jahre\n"
            f"- Wohnlage: {data['wohnlage']}\n"
            f"- Entfernung: {data['entfernung_km']} km\n"
            f"- Homeoffice: {data['homeoffice_tage']} Tage/Woche\n"
            f"- Beziehungsstatus: {data['beziehungsstatus']}\n"
            f"- Kinder: {data['kinder']}\n"
            f"- Prognose: {pred:.1f} Krankheitstage/Jahr\n\n"
        )


    prompt = (                      #TODO prompt ggfs später in mcp umwandeln´, um use case transformierbart zu machen 
        "Analysiere folgende Profile und bewerte, wer am wenigsten krank sein wird und warum. "
        "Beziehe dich auf Homeoffice, Entfernung, Familienstand etc. Erkläre nachvollziehbar.\n\n" + beschreibung
    )
    try:

#anfrage wird hier an die LLM geschickt

        with st.spinner("KI analysiert..."):
            response = openai.chat.completions.create(
                model="gpt-4-0125-preview",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            antwort = response.choices[0].message.content
            st.subheader("KI-Analyse")
            st.write(antwort)
    except Exception as e:


        #fehlermeldungen 
        #bei modell fehlern hat streamlit einen eigenen error handler, der die meldung anzeigt
        #fehler in abfrage von data ODER path (whatever) werden hier behandelt
        st.error(f"Fehler bei der LLM-Abfrage: {e}")


# Footer-Link
st.markdown(
    "<p style='text-align: center; font-size: 12px; color: gray;'>Mehr Infos auf <a href='https://github.com/eneav/opi-bot' target='_blank' style='color: gray;'>GitHub</a></p>",
    unsafe_allow_html=True
)