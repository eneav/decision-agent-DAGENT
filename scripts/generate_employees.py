import sqlite3
import random
from pathlib import Path

#  pfad zur datenbank 
db_path = Path(__file__).resolve().parent.parent / "data" / "personal.db"

#  Verbindung herstellen
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("DELETE FROM mitarbeiter")



#frei skalierbar wieviele mitarbeiter generiert werden sollen
#ABER 
#vorher in db die vorher generierten daten löschen, damit die db nicht überquillt
# DROP TABLE IF EXISTS bewerber;

num_employees = 2000



# mögliche werte für die Attribute
wohnlagen = ['gut', 'neutral', 'schlecht']
beziehungsstatus_options = ['ledig', 'verheiratet', 'geschieden', 'verwitwet']

#  random  generieren 
for _ in range(num_employees):
    alter = random.randint(20, 65)
    wohnlage = random.choices(wohnlagen, weights=[0.4, 0.4, 0.2])[0]
    entfernung = round(random.uniform(0.5, 50.0), 1)
    homeoffice = random.randint(0, 20)
    beziehungsstatus = random.choices(beziehungsstatus_options, weights=[0.5, 0.3, 0.15, 0.05])[0]
    kinder = random.choices([0, 1, 2, 3, 4], weights=[0.5, 0.2, 0.2, 0.08, 0.02])[0]

    krankheitstage = (
        random.randint(0, 5)
        + (5 if wohnlage == 'schlecht' else 0)
        + (3 if homeoffice < 5 else 0)
        + (2 if kinder >= 2 else 0)
        + random.randint(0, 4)  # streuung
    )

    # einfügen in ddb 
    cursor.execute("""
    INSERT INTO mitarbeiter (alter_jahre, wohnlage, entfernung_km, homeoffice_tage, beziehungsstatus, kinder, krankheitstage)
    VALUES (?, ?, ?, ?, ?, ?, ?)
""", (alter, wohnlage, entfernung, homeoffice, beziehungsstatus, kinder, krankheitstage))


#speichern und schließen 
conn.commit()
conn.close()

print(f" {num_employees} Mitarbeiter erfolgreich generiert und eingefügt!")
