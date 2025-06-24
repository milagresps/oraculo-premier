import streamlit as st
import requests
import pandas as pd
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingRegressor

st.set_page_config(page_title="Oráculo com Estatísticas Reais", layout="centered")

API_KEY = st.text_input("🔐 Digite sua API-Football Key:", type="password")

def carregar_dados_modelo():
    df = pd.read_csv("https://www.football-data.co.uk/mmz4281/2324/E0.csv")
    df = df[['FTHG','FTAG','HS','AS','HST','AST','HF','AF','HC','AC','FTR']].dropna()
    df['Result'] = df['FTR'].map({'H': 0, 'D': 1, 'A': 2})
    X = df.drop(['FTR','Result'], axis=1)
    y_cls = df['Result']
    y_home = df['FTHG']
    y_away = df['FTAG']
    return X, y_cls, y_home, y_away

def treinar_modelos():
    X, y_cls, y_home, y_away = carregar_dados_modelo()
    modelo_cls = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    modelo_cls.fit(X, y_cls)
    modelo_home = GradientBoostingRegressor()
    modelo_home.fit(X, y_home)
    modelo_away = GradientBoostingRegressor()
    modelo_away.fit(X, y_away)
    return modelo_cls, modelo_home, modelo_away

def buscar_partidas(api_key):
    url = "https://v3.football.api-sports.io/fixtures"
    headers = {"x-apisports-key": api_key}
    params = {"league": 39, "season": 2023, "next": 10}
    res = requests.get(url, headers=headers, params=params)
    if res.status_code != 200:
        return None
    data = res.json()["response"]
    jogos = []
    for m in data:
        hora = datetime.fromisoformat(m['fixture']['date'].replace("Z", "+00:00")).strftime("%d/%m %H:%M")
        jogos.append({
            "Data": hora,
            "Time Casa": m['teams']['home']['name'],
            "Time Fora": m['teams']['away']['name'],
            "ID_Casa": m['teams']['home']['id'],
            "ID_Fora": m['teams']['away']['id'],
            "ID_Jogo": m['fixture']['id']
        })
    return pd.DataFrame(jogos)

def stats_reais_time(team_id, api_key):
    url = "https://v3.football.api-sports.io/teams/statistics"
    headers = {"x-apisports-key": api_key}
    params = {"team": team_id, "season": 2023, "league": 39}
    res = requests.get(url, headers=headers, params=params)
    if res.status_code != 200:
        return None
    data = res.json()['response']
    estat = {
        "HS": data['shots']['total'],
        "HST": data['shots']['on'],
        "HF": data['fouls']['total'],
        "HC": data['corners']['total'],
        "FTAG": 0,  # placeholder
        "FTHG": 0   # placeholder
    }
    return estat

if API_KEY:
    st.title("🔮 Oráculo com IA + Estatísticas Reais")
    modelo_cls, modelo_home, modelo_away = treinar_modelos()
    df_jogos = buscar_partidas(API_KEY)

    if df_jogos is not None:
        jogo_sel = st.selectbox("Escolha um jogo:", df_jogos["Data"] + " | " + df_jogos["Time Casa"] + " vs " + df_jogos["Time Fora"])
        dados_jogo = df_jogos[df_jogos["Data"] + " | " + df_jogos["Time Casa"] + " vs " + df_jogos["Time Fora"] == jogo_sel].iloc[0]
        
        casa_stats = stats_reais_time(dados_jogo["ID_Casa"], API_KEY)
        fora_stats = stats_reais_time(dados_jogo["ID_Fora"], API_KEY)

        if casa_stats and fora_stats:
            entrada = pd.DataFrame([{
                "HS": casa_stats["HS"],
                "HST": casa_stats["HST"],
                "HF": casa_stats["HF"],
                "HC": casa_stats["HC"],
                "AS": fora_stats["HS"],
                "AST": fora_stats["HST"],
                "AF": fora_stats["HF"],
                "AC": fora_stats["HC"],
                "FTHG": 0,
                "FTAG": 0
            }])

            if st.button("🔎 Prever com IA"):
                probs = modelo_cls.predict_proba(entrada)[0]
                gols_casa = int(round(modelo_home.predict(entrada)[0]))
                gols_fora = int(round(modelo_away.predict(entrada)[0]))

                st.markdown("### 🎯 Previsão com dados reais:")
                st.write(f"🏠 Vitória Casa: {round(probs[0]*100, 2)}%")
                st.write(f"🤝 Empate: {round(probs[1]*100, 2)}%")
                st.write(f"🛫 Vitória Fora: {round(probs[2]*100, 2)}%")
                st.success(f"Placar provável: {gols_casa} x {gols_fora}")

                st.markdown("### 💸 Simule suas odds:")
                col1, col2, col3 = st.columns(3)
                odd1 = col1.number_input("Odd Casa", min_value=1.0, value=1.9)
                odd2 = col2.number_input("Odd Empate", min_value=1.0, value=3.2)
                odd3 = col3.number_input("Odd Fora", min_value=1.0, value=4.0)

                def ev(p, o): return round((p * o) - 1, 3)

                st.write(f"EV Casa: {ev(probs[0], odd1)} | EV Empate: {ev(probs[1], odd2)} | EV Fora: {ev(probs[2], odd3)}")
        else:
            st.warning("Não foi possível carregar estatísticas para este jogo.")
    else:
        st.error("Erro ao buscar jogos. Verifique sua chave da API.")
else:
    st.info("Insira sua chave da API-Football acima para começar.")
