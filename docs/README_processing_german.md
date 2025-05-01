# Datenverarbeitungsprozess

Dieses Dokument beschreibt den Datenverarbeitungsprozess für das Dengue-Fieber Vorhersagemodell.

## Übersicht

Der Datenverarbeitungsprozess besteht aus mehreren Schritten, die nacheinander ausgeführt werden:

1. Datenuntersuchung (`data_investigation.py`)
2. Datenbereinigung (`data_cleaner.py`)
3. Feature-Kombination (`data_cleaner_combinations.py`)
4. Feature-Engineering (`data_features.py`)
5. Analyse der bereinigten Daten (`data_cleaner_investigation.py`)

## Schritt 1: Datenuntersuchung

**Datei:** `src/data_investigation.py`

Dieser Schritt analysiert die Rohdaten und identifiziert:
- Datenstruktur und -typen
- Fehlende Werte
- Ausreißer
- Korrelationen zwischen Features
- Verteilungen der Features

**Ausgaben:**
- Statistiken zu fehlenden Werten
- Korrelationsmatrizen
- Verteilungsplots
- Empfehlungen für die Datenbereinigung

## Schritt 2: Datenbereinigung

**Datei:** `src/data_cleaner.py`

Basierend auf den Ergebnissen der Datenuntersuchung werden folgende Bereinigungen durchgeführt:
- Behandlung fehlender Werte
- Entfernung von Ausreißern
- Konvertierung von Datentypen
- Normalisierung von Features
- Erstellung von `weekofyear` aus dem Datum

**Ausgaben:**
- `cleaned_train_data.csv`
- `cleaned_test_features.csv`

## Schritt 3: Feature-Kombination

**Datei:** `src/data_cleaner_combinations.py`

Kombiniert verwandte Features, um Redundanz zu reduzieren:
- Temperatur-Features:
  - Reanalysis-Temperaturen (K)
  - Station-Temperaturen (C)
- Konvertierung von Datumsformaten
- Berechnung von Durchschnittswerten und Spannen

**Ausgaben:**
- `combined_train_data.csv`
- `combined_test_features.csv`

## Schritt 4: Feature-Engineering

**Datei:** `src/data_features.py`

Erstellt neue Features für das Modell:
- Zeitliche Features:
  - `dayofyear`
  - `quarter`
  - `is_month_start`
  - `is_month_end`
- Wetter-Features:
  - Durchschnittliche Temperatur
  - Temperaturbereich
  - Niederschlagstage
  - Luftfeuchtigkeit
- NDVI-Features:
  - Mittelwert
  - Standardabweichung
  - Bereich
- Lag-Features für `total_cases`

**Ausgaben:**
- `featured_train_data.csv`
- `featured_test_features.csv`

## Schritt 5: Analyse der bereinigten Daten

**Datei:** `src/data_cleaner_investigation.py`

Überprüft die Qualität der bereinigten Daten:
- Verteilung der Features
- Korrelationen
- Fehlende Werte
- Ausreißer
- Datenqualität

**Ausgaben:**
- Verteilungsplots in `data/processed/png/`
- Statistiken und Metriken

## Ausführungsreihenfolge

1. Führen Sie zuerst `data_investigation.py` aus
2. Basierend auf den Ergebnissen, führen Sie `data_cleaner.py` aus
3. Führen Sie `data_cleaner_combinations.py` aus
4. Führen Sie `data_features.py` aus
5. Abschließend führen Sie `data_cleaner_investigation.py` aus

## Verzeichnisstruktur

```
data/
├── raw/                    # Rohdaten
├── processed/              # Verarbeitete Daten
│   ├── cleaned_*.csv      # Bereinigte Daten
│   ├── combined_*.csv     # Kombinierte Features
│   ├── featured_*.csv     # Feature-Engineering
│   └── png/               # Analyseplots
└── submission/            # Vorhersagen
```

## Nächste Schritte

Nach Abschluss der Datenverarbeitung:
1. Trainieren Sie das Modell mit den verarbeiteten Daten
2. Evaluieren Sie die Modellleistung
3. Optimieren Sie die Features basierend auf der Modellleistung 