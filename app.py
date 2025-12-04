# -*- coding: utf-8 -*-
"""
競馬：完全統合テンプレート（netkeiba取得 + スコア計算 + 手動調整 + 馬券配分 + Excel出力）
- 簡易版の「昨日のシステム」を再現するテンプレート
- 実運用では netkeiba ページのセレクタを実際のHTMLに合わせて調整してください
"""
import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import io
from itertools import combinations, permutations

st.set_page_config(page_title="競馬：完全統合システム", layout="wide")

# ----------------- 設定 -----------------
SAMPLE_CSV_PATH = "/mnt/data/sample_keiba.csv"  # テスト用CSV（内部テストパス）
TOTAL_INVEST_DEFAULT = 5000
MAX_TOP_FOR_TRIO = 6  # 3連系は上位何頭まで考慮するか
# -----------------------------------------

st.title("競馬：完全統合システム（netkeiba取得 + スコア計算）")

col_left, col_right = st.columns([2,1])
with col_left:
    st.markdown("### レース入力 / データ取得")
    race_url = st.text_input("netkeibaのレースURL（空欄でデモCSV使用）", value="")
    use_sample = st.checkbox("サンプルCSVを使う（デバッグ）", value=False)
    # surface選択（芝/ダート）で年齢点の尺度を変えたいという要望に対応
    surface = st.selectbox("馬場（surface）", ["auto","芝","ダート"])
with col_right:
    total_invest = st.number_input("総投資額（円）", min_value=0, value=TOTAL_INVEST_DEFAULT, step=500)
    desired_multiplier = st.slider("希望払戻倍率（目安）", 1.0, 10.0, 1.5, 0.5)
    show_debug = st.checkbox("デバッグログを表示", value=False)

# ----------------- ユーティリティ -----------------
def send_line_notify(token, message: str):
    if not token:
        return False
    try:
        url = "https://notify-api.line.me/api/notify"
        headers = {"Authorization": f"Bearer {token}"}
        requests.post(url, headers=headers, data={"message": message})
        return True
    except Exception:
        return False

# 年齢スコア（surfaceによって4段階や3段階に切替可能）
def age_score(age:int, surface:str="auto"):
    # デフォルト（芝向け：3段評価近似）
    if surface=="dirt":
        # ユーザー要望でダートは4点系に変えられるようにする（例）
        if age<=4: return 4.0
        elif age==5: return 3.0
        elif age==6: return 2.0
        else: return 1.0
    else:
        # 芝またはautoのデフォルト（以前の指定に合わせた近似）
        if age<=4: return 3.0
        elif age==5: return 2.0
        elif age==6: return 1.5
        else: return 1.0

# 通算成績スコア（直近5年等はデータが必要、ここは仮でレーティング入力欄も用意）
def record_score(rank_value):
    # rank_value は1～100のような総合評価を期待
    if rank_value>=80: return 3.0
    elif rank_value>=50: return 2.0
    else: return 1.0

# 簡易スクレイピング（実際のページに合わせてセレクタ調整が必要）
def fetch_race_data_from_netkeiba(url:str):
    try:
        res = requests.get(url, timeout=10)
        res.encoding = res.apparent_encoding
        soup = BeautifulSoup(res.text, "lxml")

        # === ここは要カスタマイズ ===
        # 仮：出走馬名は .HorseName のようなセレクタ（実際は確認必須）
        horse_tags = soup.select("a.HorseName")  # ← 要修正
        odds_tags = soup.select(".Odds")          # ← 要修正
        # fallback if selectors fail:
        if not horse_tags:
            return None

        horses = []
        n = min(len(horse_tags), 18)
        for i in range(n):
            name = horse_tags[i].get_text(strip=True)
            try:
                odds = float(odds_tags[i].get_text(strip=True).replace(",",""))
            except:
                odds = 5.0
            horses.append({
                "horse_name": name,
                "age": np.random.choice([3,4,5,6,7]),  # placeholder
                "score_base": 2.0 + np.random.random()*4.0,
                "odds": odds,
                "last_weight": int(480 + np.random.randint(-10,40)),
                "running_style": np.random.choice(["先行","差し","追込","逃げ"]),
            })
        return horses
    except Exception as e:
        if show_debug:
            st.error(f"スクレイピング失敗: {e}")
        return None

# Demo fallback (if scraping fails)
def demo_horses():
    return [
        {"horse_name":"馬A","age":3,"score_base":23.0,"odds":3.0,"last_weight":500,"running_style":"差し"},
        {"horse_name":"馬B","age":4,"score_base":20.0,"odds":5.0,"last_weight":480,"running_style":"先行"},
        {"horse_name":"馬C","age":5,"score_base":17.0,"odds":10.0,"last_weight":520,"running_style":"追込"},
        {"horse_name":"馬D","age":6,"score_base":15.0,"odds":12.0,"last_weight":495,"running_style":"先行"},
    ]

# convert to dataframe for display
def horses_to_df(horses):
    df = pd.DataFrame(horses)
    df = df.rename(columns={"horse_name":"馬名","score_base":"基礎スコア","odds":"オッズ","last_weight":"前走体重","running_style":"脚質","age":"年齢"})
    return df[["馬名","年齢","基礎スコア","オッズ","前走体重","脚質"]]

# 自動単勝配分（スコアに比例）
def auto_allocate(horses, total_invest):
    total_score = sum(max(0.001, h.get("final_score", h.get("score_base",1))) for h in horses)
    alloc = {}
    for h in horses:
        s = h.get("final_score", h.get("score_base",1))
        amt = total_invest * (s/total_score)
        alloc[h["horse_name"]] = int(round(amt/50.0)*50)
    return alloc

# 3連系生成（上位N頭から）
def generate_trio_bets(horses, total_invest, bet_type="3連複"):
    sorted_h = sorted(horses, key=lambda x: x.get("final_score", x.get("score_base",0)), reverse=True)
    top = sorted_h[:MAX_TOP_FOR_TRIO]
    names = [h["horse_name"] for h in top]
    combos = []
    if bet_type=="3連複":
        combos = list(combinations(names,3))
    else:
        combos = list(permutations(names,3))
    alloc = {}
    total_score = sum(h.get("final_score",h.get("score_base",0)) for h in top) or 1.0
    for combo in combos:
        score_sum = sum(next(h.get("final_score",h.get("score_base",0)) for h in top if h["horse_name"]==c) for c in combo)
        amt = total_invest * 0.15 * (score_sum/total_score)  # 3連系は総投資の一部（例：15%）
        alloc[combo] = int(round(amt/50.0)*50)
    return alloc

# time-score quartile calculation for same distance races (expects a list of times)
def time_quartile_score(times, target_time):
    # higher is better -> convert to quartile points 4-1
    if not times:
        return 1
    arr = np.array(times)
    q1 = np.percentile(arr,25)
    q2 = np.percentile(arr,50)
    q3 = np.percentile(arr,75)
    if target_time <= q1: return 4
    elif target_time <= q2: return 3
    elif target_time <= q3: return 2
    else: return 1

# ---------------- UI 動作 ----------------
if st.button("最新オッズ取得・スコア計算"):
    # 1) データ取得
    horses = None
    if use_sample:
        try:
            df_sample = pd.read_csv(SAMPLE_CSV_PATH)
            horses = []
            for _,r in df_sample.iterrows():
                horses.append({
                    "horse_name": r.get("horse_name") or r.get("horse") or f"馬{_}",
                    "age": int(r.get("age", np.random.choice([3,4,5,6]))),
                    "score_base": float(r.get("score", 2.0)),
                    "odds": float(r.get("odds",5.0)),
                    "last_weight": int(r.get("last_weight", r.get("weight",480))),
                    "running_style": r.get("style", "差し"),
                })
        except Exception as e:
            st.warning("サンプルCSV読み込み失敗。デモデータを使用します。")
            horses = demo_horses()
    else:
        if race_url:
            horses = fetch_race_data_from_netkeiba(race_url)
            if horses is None:
                st.warning("スクレイピングでデータ取得できませんでした。デモデータを使用します。")
                horses = demo_horses()
        else:
            st.info("race_url が空欄のためデモデータを使用します。")
            horses = demo_horses()

    # 2) ベースのDataFrame表示
    df = horses_to_df(horses)
    st.subheader("全頭基礎データ")
    st.dataframe(df)

    # 3) スコア計算（年齢・騎手(入力)・通算成績(入力)・馬場/距離適性(簡易)）
    st.subheader("自動スコア計算：パラメータ入力")
    # ここでは簡易的に騎手/通算成績欄をUIで入力してもらう
    jockey_rating_input = {}
    record_rating_input = {}
    for h in horses:
        name = h["horse_name"]
        jockey_rating_input[name] = st.slider(f"{name} - 騎手評価 (1-100)", 1, 100, 60, key=f"j_{name}")
        record_rating_input[name] = st.slider(f"{name} - 通算成績評価 (1-100)", 1, 100, 60, key=f"r_{name}")

    # compute final score
    for h in horses:
        ag = h.get("age",4)
        base = h.get("score_base",2.0)
        a_sc = age_score(ag, surface if surface!="auto" else "芝")
        j_sc = 3.0 * (jockey_rating_input[h["horse_name"]] / 100.0)  # 0-3 scale
        r_sc = record_score(record_rating_input[h["horse_name"]])
        # distance/track適性は未知の場合は baseに小加点
        final = base + a_sc + j_sc + r_sc
        h["final_score"] = round(final, 2)

    # 4) 全頭スコア表示 + 手動調整
    st.subheader("スコア一覧（手動で微調整可）")
    df_scores = pd.DataFrame([{
        "馬名": h["horse_name"], "年齢": h["age"], "基礎スコア": h["score_base"],
        "最終スコア": h["final_score"], "オッズ": h["odds"], "前走体重": h["last_weight"], "脚質": h["running_style"]
    } for h in horses])
    st.dataframe(df_scores)

    # manual adjustments
    st.subheader("手動調整")
    manual_adjust = {}
    for h in horses:
        name = h["horse_name"]
        adj = st.number_input(f"{name} の補正値（マイナス可）", value=0, step=1, key=f"adj_{name}")
        h["final_score"] = float(round(h["final_score"] + adj,2))
        manual_adjust[name] = adj

    # 5) 時間スコア（距離同条件のタイム上位四分位で4-1点付与）
    st.subheader("タイムスコア（距離同条件）")
    # ユーザーが同条件の過去タイムを入力できる簡易UI（CSVから抽出する実装は要追加）
    # ここでは入力フォームでカンマ区切りのタイム群を受け取り、各馬のtarget_time列が必要
    times_input = st.text_area("同条件の過去タイムをカンマ区切りで入力（例：91.3,92.0,92.5）", value="")
    times_list = []
    if times_input.strip():
        try:
            times_list = [float(x.strip()) for x in times_input.split(",") if x.strip()]
        except:
            st.warning("タイム入力の形式が正しくありません。")
    # example: if each horse has a 'time' key (not present in demo), compute time_score
    for h in horses:
        t = h.get("time", None)
        if t is not None and times_list:
            h["time_score"] = time_quartile_score(times_list, t)
        else:
            h["time_score"] = 2  # default neutral

    # 6) 自動配分（単勝）
    st.subheader("単勝：自動配分（スコア比例）")
    allocation = auto_allocate(horses, total_invest)
    # show sliders for adjustment
    adjusted = {}
    for h in horses:
        name = h["horse_name"]
        default = allocation.get(name, 0)
        adjusted[name] = st.slider(f"{name}（スコア: {h['final_score']}、オッズ: {h['odds']}）",
                                   0, total_invest, int(default), step=50, key=f"alloc_{name}")
    total_alloc = sum(adjusted.values())
    st.write(f"合計投資額: {total_alloc} / 設定総額: {total_invest}")
    if total_alloc>total_invest:
        st.warning("合計が総投資額を超えています。調整してください。")

    # 7) 3連系（自動案）
    st.subheader("3連複 / 3連単：自動案（上位組合せ）")
    trio_type = st.radio("券種選択", ("3連複","3連単"))
    trio_alloc = generate_trio_bets(horses, total_invest, bet_type=trio_type)
    # display top combos
    for combo, amt in list(trio_alloc.items())[:20]:
        st.write(f"{combo}: {amt}円")

    # 8) Excel出力（投票表）
    st.subheader("Excel出力")
    out_df = pd.DataFrame([{
        "馬名": name,
        "投資額": adjusted[name],
        "オッズ": next(h["odds"] for h in horses if h["horse_name"]==name),
        "期待払戻": adjusted[name] * next(h["odds"] for h in horses if h["horse_name"]==name),
        "スコア": next(h["final_score"] for h in horses if h["horse_name"]==name)
    } for name in adjusted])
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        out_df.to_excel(writer, index=False, sheet_name="allocations")
    buffer.seek(0)
    st.download_button("Excelをダウンロード", data=buffer, file_name="allocations.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # 9) LINE通知（任意）
    st.subheader("通知（LINE）")
    line_token = st.text_input("LINE Notify トークン（任意）", value="", type="password")
    if st.button("上位3頭をLINEに送信"):
        top3 = sorted(horses, key=lambda x:x.get("final_score",0), reverse=True)[:3]
        msg = f"レース: {race_url or '手動入力'}\n上位3頭:\n"
        for t in top3:
            msg += f"{t['horse_name']} スコア:{t['final_score']} オッズ:{t['odds']}\n"
        ok = send_line_notify(line_token, msg)
        if ok:
            st.success("LINE送信しました")
        else:
            st.info("LINE送信に失敗しました（トークン未設定など）")

st.info("注意: netkeibaスクレイピングはページ構造に依存します。`fetch_race_data_from_netkeiba` のセレクタを実際のページに合わせてください。")
