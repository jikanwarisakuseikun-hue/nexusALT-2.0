import streamlit as st
import pandas as pd
import time
import os
import datetime
import pytz
import hashlib
import threading
import queue
import time as _time
from gtts import gTTS
import tempfile
import json
import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from streamlit_mic_recorder import mic_recorder  # 🎙️ 波形が出るマイク
from google import genai
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from gspread.exceptions import APIError

# ページ設定
st.set_page_config(
    page_title="Nexus English 2.0",
    page_icon="🎤",
    layout="centered"
)

# -------------------------------------------------------------
# 認証・設定ヘルパー (Secrets)
# -------------------------------------------------------------
def get_secrets():
    try:
        return {
            "default_gemini_api_key": st.secrets["GEMINI_API_KEY"],
            "drive_folder_id": st.secrets["GOOGLE_DRIVE_FOLDER_ID"],
            "master_spreadsheet_name": st.secrets["MASTER_SPREADSHEET_NAME"],
            "service_account_info": dict(st.secrets["connections"]["gsheets"]),
            "teacher_api_keys": dict(st.secrets.get("TEACHER_API_KEYS", {})),
        }
    except Exception as e:
        st.error(f"Streamlit Secretsの設定が不足しています: {e}")
        st.stop()

SECRETS = get_secrets()

# gspreadのAPIError（429など）が出たときだけ、ジッター付き指数バックオフでリトライする
SHEETS_RETRY = dict(
    stop=stop_after_attempt(6),
    wait=wait_random_exponential(multiplier=1, max=30),
    retry=retry_if_exception_type(APIError),
    reraise=True,
)

def _retry_gemini_with_backoff(func, max_retries=4, base_delay=1.5):
    """Gemini API側の429/RESOURCE_EXHAUSTED時に指数バックオフでリトライする
    （例外クラスがSDKバージョンで変わりうるため、メッセージ内容で判定する）"""
    last_err = None
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_err = e
            msg = str(e)
            is_rate_limit = ("429" in msg) or ("RESOURCE_EXHAUSTED" in msg) or ("rate limit" in msg.lower())
            if is_rate_limit and attempt < max_retries - 1:
                _time.sleep(base_delay * (2 ** attempt))
                continue
            raise
    raise last_err

# -------------------------------------------------------------
# Google認証クライアント（プロセス全体で1回だけ生成）
# -------------------------------------------------------------
@st.cache_resource
def get_gspread_client():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = Credentials.from_service_account_info(SECRETS["service_account_info"], scopes=scopes)
    return gspread.authorize(creds)

@st.cache_resource
def get_drive_service():
    scopes = ["https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(SECRETS["service_account_info"], scopes=scopes)
    return build('drive', 'v3', credentials=creds)

# -------------------------------------------------------------
# スプレッドシート／ワークシートのハンドルをキャッシュする
#   client.open()やsheet.worksheet()はそれ自体がAPI呼び出しのため、
#   操作のたびに開き直さず使い回すことでAPI呼び出し回数そのものを減らす。
# -------------------------------------------------------------
@st.cache_resource(ttl=600)
def get_spreadsheet(spreadsheet_name_or_id):
    client = get_gspread_client()
    target = str(spreadsheet_name_or_id)
    if target.startswith("1") and len(target) > 20:
        return client.open_by_key(target)
    else:
        return client.open(target)

@st.cache_resource(ttl=600)
def get_worksheet(spreadsheet_name_or_id, title):
    sheet = get_spreadsheet(spreadsheet_name_or_id)
    return sheet.worksheet(title)

# -------------------------------------------------------------
# 全体マスタの読み込み (Schoolsシート)
#   ログイン画面の学校プルダウンで使う。頻繁には変わらないので長めにキャッシュ。
# -------------------------------------------------------------
@st.cache_data(ttl=300)
@retry(**SHEETS_RETRY)
def load_master_schools():
    target = SECRETS["master_spreadsheet_name"]
    try:
        ws = get_worksheet(target, "Schools")
        return pd.DataFrame(ws.get_all_records())
    except Exception as e:
        st.error(f"【全体マスタ読み込みエラー】スプレッドシート '{target}' の取得に失敗しました。\n原因: {e}")
        raise e

# -------------------------------------------------------------
# クラス設定（ClassConfigシート）とクラス名簿（Rosterシート、任意）
#   student_count・teacher_id・氏名は、先生がダッシュボードから
#   直接編集できるよう学校ごとのスプレッドシートで管理する。
#   ログイン頻度に比べて変更頻度は圧倒的に低いので、長め(1時間)にキャッシュする。
# -------------------------------------------------------------
@st.cache_data(ttl=3600)
@retry(**SHEETS_RETRY)
def load_class_config_from_sheet(spreadsheet_name):
    try:
        ws = get_worksheet(spreadsheet_name, "ClassConfig")
        df = pd.DataFrame(ws.get_all_records())
    except Exception:
        df = pd.DataFrame(columns=["target_class", "student_count", "teacher_id"])
    if df.empty:
        df = pd.DataFrame(columns=["target_class", "student_count", "teacher_id"])
    return df

@retry(**SHEETS_RETRY)
def save_class_config_to_sheet(spreadsheet_name, df_class_config):
    sheet = get_spreadsheet(spreadsheet_name)
    try:
        ws = sheet.worksheet("ClassConfig")
    except gspread.exceptions.WorksheetNotFound:
        ws = sheet.add_worksheet(title="ClassConfig", rows="50", cols="10")

    ws.clear()
    ws.update([df_class_config.columns.values.tolist()] + df_class_config.values.tolist())
    # 保存内容がすぐログイン画面・PIN一覧に反映されるよう、関連キャッシュを破棄する
    load_class_config_from_sheet.clear()
    get_worksheet.clear()

@st.cache_data(ttl=3600)
@retry(**SHEETS_RETRY)
def load_roster_from_sheet(spreadsheet_name):
    """氏名の一覧（任意）。設定しなくても「n番」表示でテストは実施できる。"""
    try:
        ws = get_worksheet(spreadsheet_name, "Roster")
        df = pd.DataFrame(ws.get_all_records())
    except Exception:
        df = pd.DataFrame(columns=["target_class", "student_number", "student_name"])
    if df.empty:
        df = pd.DataFrame(columns=["target_class", "student_number", "student_name"])
    return df

@retry(**SHEETS_RETRY)
def save_roster_to_sheet(spreadsheet_name, df_roster):
    sheet = get_spreadsheet(spreadsheet_name)
    try:
        ws = sheet.worksheet("Roster")
    except gspread.exceptions.WorksheetNotFound:
        ws = sheet.add_worksheet(title="Roster", rows="500", cols="5")

    ws.clear()
    ws.update([df_roster.columns.values.tolist()] + df_roster.values.tolist())
    load_roster_from_sheet.clear()
    get_worksheet.clear()

# -------------------------------------------------------------
# 生徒ログイン用PIN（Sheetsを一切読まずに照合する）
#   secretsのschool_pepper（学校ごとの秘密の種）から、
#   学校×クラス×出席番号 に対して一意な4桁PINを計算する。
#   同じ入力からは常に同じPINが再現されるので、事前にリストを
#   どこかに保存しておく必要がない＝生徒ログイン時のSheets読み取りがゼロになる。
# -------------------------------------------------------------
def generate_pin(school_id: str, class_name: str, student_number, pepper: str) -> str:
    raw = f"{school_id}:{class_name}:{student_number}:{pepper}"
    digest = hashlib.sha256(raw.encode()).hexdigest()
    return str(int(digest, 16) % 10000).zfill(4)

def get_classes_for_school(spreadsheet_name: str):
    df = load_class_config_from_sheet(spreadsheet_name)
    if df.empty or "target_class" not in df.columns:
        return []
    return sorted(df["target_class"].astype(str).unique().tolist())

def get_class_config(spreadsheet_name: str, class_name: str):
    df = load_class_config_from_sheet(spreadsheet_name)
    if df.empty or "target_class" not in df.columns:
        return {}
    row = df[df["target_class"].astype(str) == str(class_name)]
    if row.empty:
        return {}
    r = row.iloc[0]
    try:
        student_count = int(r.get("student_count", 0))
    except Exception:
        student_count = 0
    return {"student_count": student_count, "teacher_id": str(r.get("teacher_id", "")).strip()}

def get_student_name(spreadsheet_name: str, class_name: str, student_number) -> str:
    """Rosterシートに氏名が設定されていればそれを表示名に使う。
    未設定の場合は「n番」にフォールバックする（動作に支障はない）。"""
    df = load_roster_from_sheet(spreadsheet_name)
    if not df.empty and {"target_class", "student_number", "student_name"}.issubset(df.columns):
        row = df[
            (df["target_class"].astype(str) == str(class_name)) &
            (df["student_number"].astype(str) == str(student_number))
        ]
        if not row.empty:
            name = str(row.iloc[0].get("student_name", "")).strip()
            if name:
                return name
    return f"{student_number}番"

# -------------------------------------------------------------
# 教職員ログイン（Usersシート、キャッシュ付き）
#   生徒はもうUsersシートを読まないため、ここに残る負荷は教職員数人分だけ。
#   ttl=30でも十分に安全（同時ログインしても1回のAPI呼び出しに集約される）。
# -------------------------------------------------------------
@st.cache_data(ttl=30)
@retry(**SHEETS_RETRY)
def load_teacher_users(spreadsheet_name_or_id):
    ws = get_worksheet(spreadsheet_name_or_id, "Users")
    return pd.DataFrame(ws.get_all_records())

def authenticate_teacher(school_row, input_id, input_pw):
    s_id = school_row["school_id"]
    s_name = school_row["school_name"]
    target_ss = str(school_row["spreadsheet_name_or_id"]).strip()

    try:
        users_df = load_teacher_users(target_ss)
        if not users_df.empty:
            role_series = users_df.get("role", pd.Series([""] * len(users_df))).astype(str).str.strip().str.lower()
            matched = users_df[
                (users_df["user_id"].astype(str) == input_id) &
                (users_df["password"].astype(str) == input_pw) &
                (role_series != "student")
            ]
            if not matched.empty:
                u_info = matched.iloc[0]
                return {
                    "authenticated": True,
                    "user_id": u_info["user_id"],
                    "user_name": u_info["user_name"],
                    "school_id": s_id,
                    "school_name": s_name,
                    "spreadsheet_name": target_ss,
                    "role": u_info.get("role", "teacher"),
                }
    except Exception:
        pass

    return {"authenticated": False}

# -------------------------------------------------------------
# 学校別 Config（横長お題）の読み込み・保存
# -------------------------------------------------------------
@st.cache_data(ttl=60)
@retry(**SHEETS_RETRY)
def load_config_from_sheet(spreadsheet_name):
    try:
        ws = get_worksheet(spreadsheet_name, "Config")
        return pd.DataFrame(ws.get_all_records())
    except Exception as e:
        st.warning(f"Configシートが見つからないか読み込めません。デフォルト設定を表示します。 (詳細: {e})")
        default_data = [
            {
                "target_class": "2-1", "teacher_id": "teacher_1", "num_questions": 2,
                "q1_text": "Please introduce yourself in English.", "q1_criteria": "挨拶や名前・趣味を話せているか (A/B/C)",
                "q2_text": "What do you want to do during your summer vacation?", "q2_criteria": "未来の表現を用いて計画を説明できているか (A/B/C)",
                "q3_text": "", "q3_criteria": "", "q4_text": "", "q4_criteria": "", "q5_text": "", "q5_criteria": ""
            }
        ]
        return pd.DataFrame(default_data)

@retry(**SHEETS_RETRY)
def save_config_to_sheet(spreadsheet_name, df_config):
    sheet = get_spreadsheet(spreadsheet_name)
    try:
        config_ws = sheet.worksheet("Config")
    except gspread.exceptions.WorksheetNotFound:
        config_ws = sheet.add_worksheet(title="Config", rows="50", cols="15")

    config_ws.clear()
    config_ws.update([df_config.columns.values.tolist()] + df_config.values.tolist())
    # 保存内容がすぐ生徒画面に反映されるよう、関連キャッシュを破棄する
    load_config_from_sheet.clear()
    get_worksheet.clear()

# -------------------------------------------------------------
# Google Drive アップロード
# -------------------------------------------------------------
@retry(stop=stop_after_attempt(4), wait=wait_random_exponential(multiplier=1, max=20), reraise=True)
def upload_audio_to_drive(file_path, file_name):
    service = get_drive_service()
    folder_id = SECRETS["drive_folder_id"]

    try:
        file_metadata = {'name': file_name, 'parents': [folder_id]}
        media = MediaFileUpload(file_path, mimetype='audio/wav', resumable=True)

        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, webViewLink',
            supportsAllDrives=True
        ).execute()

        try:
            service.permissions().create(
                fileId=file.get('id'),
                body={'role': 'reader', 'type': 'anyone'},
                supportsAllDrives=True
            ).execute()
        except Exception:
            pass
        return file.get('webViewLink')
    except Exception as e:
        st.error(f"【Googleドライブ アップロードエラー】詳細: {e}")
        raise e

# -------------------------------------------------------------
# 結果保存 ― 非同期バッチ書き込みキュー
#   採点結果が出たらキューに積むだけ（インメモリなので一瞬）。
#   実際のSheets書き込みはバックグラウンドスレッドが3秒ごと/5件たまるごとに
#   append_rowsでまとめて行うので、クラス全員がほぼ同時に送信しても
#   Sheets APIへの書き込み回数を大幅に減らせる。
#   ※ 書き込みに失敗した行は諦める（生徒のテスト進行自体は止めない）
# -------------------------------------------------------------
RESULT_HEADER = [
    "タイムスタンプ(JST)", "学校名", "クラス", "出席番号", "氏名", "ログインID",
    "問題番号", "質問文", "質問文を見たか", "文字起こし", "評価(A/B/C)", "アドバイス", "音声URL", "解答時間(秒)"
]

def _write_with_backoff(func, max_retries=5, base_delay=1.0):
    for attempt in range(max_retries):
        try:
            return func()
        except APIError as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status == 429 and attempt < max_retries - 1:
                _time.sleep(base_delay * (2 ** attempt))
                continue
            raise

@st.cache_resource
def get_result_queue():
    """アプリプロセス全体で1つだけ生成される結果書き込みキュー＋ワーカースレッド"""
    q = queue.Queue()

    def worker():
        buffer = {}  # (spreadsheet_name, target_class) -> [row, row, ...]
        last_flush = _time.time()

        while True:
            try:
                item = q.get(timeout=2)
                key = (item["spreadsheet_name"], item["target_class"])
                buffer.setdefault(key, []).append(item["row"])
            except queue.Empty:
                pass

            now = _time.time()
            should_flush = buffer and (
                now - last_flush >= 3
                or any(len(rows) >= 5 for rows in buffer.values())
            )

            if should_flush:
                for (spreadsheet_name, target_class), rows in list(buffer.items()):
                    try:
                        def _write():
                            sheet = get_spreadsheet(spreadsheet_name)
                            sheet_title = str(target_class).strip()
                            try:
                                ws = sheet.worksheet(sheet_title)
                            except gspread.exceptions.WorksheetNotFound:
                                ws = sheet.add_worksheet(title=sheet_title, rows="1000", cols="15")
                                ws.append_row(RESULT_HEADER)
                            ws.append_rows(rows, value_input_option="USER_ENTERED")

                        _write_with_backoff(_write)
                    except Exception:
                        pass
                buffer = {}
                last_flush = now

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return q

def queue_result_row(spreadsheet_name, target_class, result_row):
    get_result_queue().put({
        "spreadsheet_name": spreadsheet_name,
        "target_class": target_class,
        "row": result_row,
    })

# -------------------------------------------------------------
# Gemini API 評価（音声）
# -------------------------------------------------------------
def evaluate_audio_with_gemini(audio_path, question_text, criteria, api_key):
    def _call():
        client = genai.Client(api_key=api_key)
        audio_file = client.files.upload(file=audio_path)

        prompt = f"""
        あなたは中学校英語科の厳格かつ親切なAI英語スピーキングテスト採点官です。
        以下の質問と評価基準に基づいて、生徒の音声を文字起こしし、評価を行ってください。

        【質問】
        {question_text}

        【評価基準】
        {criteria}

        以下のJSON形式のみで正確に出力してください（マークダウンのコードブロックは含めない）。
        {{
          "transcript": "文字起こしされた英語テキスト",
          "evaluation": "A または B または C",
          "advice": "日本語での丁寧なアドバイスと良かった点・改善点"
        }}
        """

        response = client.models.generate_content(
            model='gemini-3.5-flash-lite',
            contents=[audio_file, prompt]
        )

        try:
            client.files.delete(name=audio_file.name)
        except Exception:
            pass

        text_res = response.text.strip()
        if text_res.startswith("```json"):
            text_res = text_res[7:]
        if text_res.endswith("```"):
            text_res = text_res[:-3]
        text_res = text_res.strip()

        res_json = json.loads(text_res)
        return res_json.get("transcript", ""), res_json.get("evaluation", "C"), res_json.get("advice", "評価生成エラー")

    try:
        return _retry_gemini_with_backoff(_call)
    except Exception as e:
        st.error(f"Gemini API通信エラーの詳細: {e}")
        raise e

# -------------------------------------------------------------
# Gemini API 評価（テキスト） ※音声を使わない解答用
# -------------------------------------------------------------
def evaluate_text_with_gemini(text_answer, question_text, criteria, api_key):
    def _call():
        client = genai.Client(api_key=api_key)

        prompt = f"""
        あなたは中学校英語科の厳格かつ親切なAI英語スピーキングテスト採点官です。
        以下は生徒が英語で書いたテキストの解答です。質問と評価基準に基づいて評価してください。

        【質問】
        {question_text}

        【生徒の解答（テキスト）】
        {text_answer}

        【評価基準】
        {criteria}

        以下のJSON形式のみで正確に出力してください（マークダウンのコードブロックは含めない）。
        {{
          "evaluation": "A または B または C",
          "advice": "日本語での丁寧なアドバイスと良かった点・改善点"
        }}
        """

        response = client.models.generate_content(
            model='gemini-3.5-flash-lite',
            contents=prompt
        )

        text_res = response.text.strip()
        if text_res.startswith("```json"):
            text_res = text_res[7:]
        if text_res.endswith("```"):
            text_res = text_res[:-3]
        text_res = text_res.strip()

        res_json = json.loads(text_res)
        return res_json.get("evaluation", "C"), res_json.get("advice", "評価生成エラー")

    try:
        return _retry_gemini_with_backoff(_call)
    except Exception as e:
        st.error(f"Gemini API通信エラーの詳細: {e}")
        raise e

def get_api_key_for_class(all_config, target_class):
    """Configシートのteacher_idに応じたAPIキーを選択する（音声/文字どちらの評価でも共通で使う）"""
    api_key_to_use = SECRETS["default_gemini_api_key"]
    teacher_keys = SECRETS.get("teacher_api_keys", {})
    try:
        target_class_row = all_config[all_config["target_class"].astype(str) == str(target_class)]
        if not target_class_row.empty:
            t_id = str(target_class_row.iloc[0].get("teacher_id", "")).strip()
            if t_id in teacher_keys and teacher_keys[t_id]:
                api_key_to_use = teacher_keys[t_id]
    except Exception:
        pass
    return api_key_to_use

# -------------------------------------------------------------
# 問題読み上げ音声（TTS）のキャッシュ
# -------------------------------------------------------------
@st.cache_data(ttl=3600)
def generate_tts_audio(text: str) -> bytes:
    tts = gTTS(text=str(text), lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_audio:
        tts.save(tmp_audio.name)
        tmp_path = tmp_audio.name
    try:
        with open(tmp_path, "rb") as f:
            data = f.read()
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    return data

# -------------------------------------------------------------
# メインアプリケーション
# -------------------------------------------------------------
def main():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("🔒 Nexus English 2.0 ログイン")

        try:
            master_df = load_master_schools()
        except Exception:
            master_df = pd.DataFrame()

        school_names = master_df["school_name"].tolist() if not master_df.empty else []

        if not school_names:
            st.warning("学校一覧を取得できませんでした。管理者に連絡してください。")
            st.stop()

        selected_school_name = st.selectbox("学校を選択してください：", school_names)
        school_row = master_df[master_df["school_name"] == selected_school_name].iloc[0]
        s_id = str(school_row["school_id"])
        target_ss = str(school_row["spreadsheet_name_or_id"]).strip()
        talky_sheet_id = str(school_row["talky_sheet_id"]).strip()   # ← 追加

        login_type = st.radio("ログイン種別を選択してください：", ["生徒", "教職員"], horizontal=True)

        if login_type == "生徒":
            class_list = get_classes_for_school(target_ss)

            if not class_list:
                st.warning("この学校のクラス設定が見つかりません。先生に「クラス設定」タブでの登録を依頼してください。")

            selected_class = st.selectbox("クラスを選択してください：", class_list) if class_list else None
            class_cfg = get_class_config(target_ss, selected_class) if selected_class else {}
            student_count = int(class_cfg.get("student_count", 0))

            selected_number = st.selectbox(
                "出席番号を選択してください：",
                list(range(1, student_count + 1))
            ) if student_count > 0 else None
            if student_count == 0 and selected_class:
                st.warning("このクラスの人数設定が見つかりません。管理者に連絡してください。")

            input_pin = st.text_input("PIN（4桁）：", type="password", max_chars=4)

            if st.button("ログイン"):
                if not (selected_class and selected_number and input_pin):
                    st.error("クラス・出席番号・PINをすべて入力してください。")
                else:
                    # Talky AI 2.0とPINを共通化するため、スプレッドシートIDを鍵にする
                    pepper = st.secrets.get("school_pepper", {}).get(target_ss, "")
                    expected_pin = generate_pin(target_ss, selected_class, selected_number, pepper)

                    if pepper and str(input_pin).strip() == expected_pin:
                        st.session_state.authenticated = True
                        st.session_state.role = "student"
                        st.session_state.user_id = f"{selected_class}-{selected_number}"
                        st.session_state.user_name = get_student_name(target_ss, selected_class, selected_number)
                        st.session_state.school_id = s_id
                        st.session_state.school_name = selected_school_name
                        st.session_state.spreadsheet_name = target_ss
                        st.session_state.talky_sheet_id = talky_sheet_id   # ← 追加（PIN一覧タブで使う）
                        st.session_state.assigned_class = selected_class
                        st.session_state.attendance_number = selected_number

                        st.toast("ログインしました！", icon="✅")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("PINが正しくありません。")

        else:  # 教職員
            input_id = st.text_input("ログインID")
            input_pw = st.text_input("パスワード", type="password")

            if st.button("ログイン"):
                auth_result = authenticate_teacher(school_row, input_id, input_pw)
                if auth_result["authenticated"]:
                    st.session_state.authenticated = True
                    st.session_state.user_id = auth_result["user_id"]
                    st.session_state.user_name = auth_result["user_name"]
                    st.session_state.school_id = auth_result["school_id"]
                    st.session_state.school_name = auth_result["school_name"]
                    st.session_state.spreadsheet_name = auth_result["spreadsheet_name"]
                    st.session_state.assigned_class = ""
                    st.session_state.role = auth_result["role"]
                    st.session_state.attendance_number = ""
                    st.session_state.talky_sheet_id = str(school_row["talky_sheet_id"]).strip()

                    st.toast("ログインしました！", icon="✅")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("学校・IDまたはパスワードが正しくありません。")
        return

    role = st.session_state.get("role", "student")
    ss_name = st.session_state.get("spreadsheet_name")
    s_id = st.session_state.get("school_id")

    # 生徒用のテスト進行状態は、サイドバーからも参照できるよう先に初期化しておく
    if role != "teacher":
        if "test_step" not in st.session_state:
            st.session_state.test_step = 0
        if "test_results" not in st.session_state:
            st.session_state.test_results = []
        if "question_viewed" not in st.session_state:
            st.session_state.question_viewed = {}  # {question_index: True/False}

    with st.sidebar:
        st.write(f"学校: **{st.session_state.get('school_name')}**")
        st.write(f"ユーザー: **{st.session_state.get('user_name')}** (`{st.session_state.get('user_id')}`)")
        if role != "teacher":
            st.write(f"出席番号: **{st.session_state.get('attendance_number')}番**")
        st.write(f"権限: **{'先生' if role == 'teacher' else '生徒'}**")

        # 生徒向け：質問文を表示するかどうかをサイドバーで選択できるようにする
        if role != "teacher":
            st.markdown("---")
            current_step = st.session_state.get("test_step", 0)
            show_question = st.checkbox(
                "📄 質問文を表示する",
                key=f"show_question_{current_step}",
                help="オンにすると、この設問の質問文が画面に表示されます。表示したことは記録され、先生に共有されるデータにも残ります。"
            )
            if show_question:
                st.session_state.question_viewed[current_step] = True

        st.markdown("---")
        if st.button("ログアウト"):
            st.session_state.clear()
            st.rerun()

    # 👨‍🏫 先生用画面
    if role == "teacher":
        st.title(f"👨‍🏫 教師ダッシュボード ({st.session_state.get('school_name')})")

        tab_config, tab_class, tab_pin = st.tabs(["📝 お題設定", "🏫 クラス設定", "🔑 生徒PIN一覧"])

        with tab_config:
            st.markdown("クラスごとの担当教師、問題数 (`num_questions`)、各問題の質問文・評価基準を横長テーブルで編集できます。")
            config_df = load_config_from_sheet(ss_name)
            edited_config_df = st.data_editor(config_df, num_rows="dynamic", use_container_width=True)

            if st.button("💾 変更をスプレッドシートに保存する", type="primary"):
                with st.spinner("保存中..."):
                    save_config_to_sheet(ss_name, edited_config_df)
                    st.success("Config設定が正常に更新されました！")

        with tab_class:
            st.markdown(
                "クラスの人数（`student_count`）と担当教師ID（`teacher_id`）を管理します。"
                "ここで登録したクラスが、生徒のログイン画面のプルダウンにそのまま反映されます。"
            )
            class_config_df = load_class_config_from_sheet(ss_name)
            if class_config_df.empty:
                class_config_df = pd.DataFrame([{"target_class": "2-1", "student_count": 30, "teacher_id": "teacher_1"}])
            edited_class_config_df = st.data_editor(
                class_config_df, num_rows="dynamic", use_container_width=True, key="class_config_editor"
            )

            if st.button("💾 クラス設定を保存する", type="primary", key="save_class_config"):
                with st.spinner("保存中..."):
                    save_class_config_to_sheet(ss_name, edited_class_config_df)
                    st.success("クラス設定を更新しました！")

            st.markdown("---")
            st.markdown("#### 👤 氏名の設定（任意）")
            st.caption("設定しなくても「n番」という表示のままテストは実施できます。氏名を表示したいクラスだけ入力してください。")
            roster_df = load_roster_from_sheet(ss_name)
            if roster_df.empty:
                roster_df = pd.DataFrame([{"target_class": "2-1", "student_number": 1, "student_name": ""}])
            edited_roster_df = st.data_editor(
                roster_df, num_rows="dynamic", use_container_width=True, key="roster_editor"
            )

            if st.button("💾 氏名一覧を保存する", key="save_roster"):
                with st.spinner("保存中..."):
                    save_roster_to_sheet(ss_name, edited_roster_df)
                    st.success("氏名一覧を更新しました！")

        with tab_pin:
            st.caption("PINはどこにも保存されておらず、secretsのschool_pepperから毎回その場で計算しています。")
            class_list_for_pin = get_classes_for_school(ss_name)

            if not class_list_for_pin:
                st.warning("クラス設定が見つかりません。「🏫 クラス設定」タブでクラスを登録してください。")
            else:
                pin_selected_class = st.selectbox("クラスを選択してください：", class_list_for_pin, key="pin_class_select")
                pin_class_cfg = get_class_config(ss_name, pin_selected_class)
                pin_student_count = int(pin_class_cfg.get("student_count", 0))
                talky_sheet_id = st.session_state.get("talky_sheet_id", "")               # ← 変更
        　　　　　pepper = st.secrets.get("school_pepper", {}).get(talky_sheet_id, "")      # ← 変更
                # Talky AI 2.0とPINを共通化するため、スプレッドシートIDを鍵にする
                pepper = st.secrets.get("school_pepper", {}).get(ss_name, "")

                if not pepper:
                    st.error("この学校のschool_pepperがsecretsに設定されていません。管理者に設定を依頼してください。")
                elif pin_student_count == 0:
                    st.warning("このクラスのstudent_countが0です。「🏫 クラス設定」タブで人数を設定してください。")
                else:
                    pin_rows = [
                        {
                            "出席番号": n,
                            "氏名": get_student_name(ss_name, pin_selected_class, n),
                            "PIN": generate_pin(talky_sheet_id, pin_selected_class, n, pepper),   # ← 変更
                        }
                        for n in range(1, pin_student_count + 1)
                    ]
                    st.dataframe(pd.DataFrame(pin_rows), hide_index=True, use_container_width=True)
                    st.caption("この一覧を印刷・配布してください。PINは学校・クラス・出席番号ごとに固定です（school_pepperを変更しない限り変わりません）。")

    # 🎙️⌨️ 生徒用画面
    else:
        st.title("🎤 Nexus English 2.0 受験画面")

        my_class = st.session_state.get("assigned_class", "2-1")
        all_config = load_config_from_sheet(ss_name)

        class_row = all_config[all_config["target_class"].astype(str) == str(my_class)]

        if class_row.empty:
            st.warning(f"現在、あなたのクラス（{my_class}）のお題データが設定されていません。")
            return

        c_data = class_row.iloc[0]
        try:
            num_q = int(c_data.get("num_questions", 1))
        except Exception:
            num_q = 1

        questions_list = []
        for i in range(1, num_q + 1):
            q_text = c_data.get(f"q{i}_text", "")
            q_crit = c_data.get(f"q{i}_criteria", "英語で適切に返答できているか。")
            if str(q_text).strip():
                questions_list.append({"id": i, "text": q_text, "criteria": q_crit})

        if not questions_list:
            st.warning("有効な質問が設定されていません。")
            return

        # 学生情報の表示（入力不要の自動設定）
        s_school = st.session_state.get("school_name", "")
        s_name = st.session_state.get("user_name", "")
        s_class = my_class
        s_number = st.session_state.get("attendance_number", 1)

        st.info(f"👤 受験者: **{s_class} {s_number}番 {s_name} さん**")

        total_questions = len(questions_list)
        current_step = st.session_state.test_step

        if current_step < total_questions:
            q_info = questions_list[current_step]
            q_id = q_info["id"]
            q_text = q_info["text"]
            q_criteria = q_info["criteria"]

            st.progress((current_step) / total_questions, text=f"進捗: 質問 {current_step + 1} / {total_questions}")

            st.markdown(f"### 質問 {current_step + 1}")
            st.write("🔊 音声をよく聞いて、英語で答えてください。（サイドバーで質問文を表示することもできます）")

            # サイドバーのチェックボックスがオンの場合、質問文を画面にも表示する
            if st.session_state.get(f"show_question_{current_step}", False):
                st.info(f"📖 質問文: {q_text}")

            # 音声の自動再生（テキストは画面に出さない。同じ質問文なら再生成しない）
            tts_bytes = generate_tts_audio(str(q_text))
            st.audio(tts_bytes, format="audio/mp3", autoplay=True)

            # 回答方法の選択：音声 or 文字
            answer_mode = st.radio(
                "回答方法を選んでください：",
                ["🎙️ 音声で答える", "⌨️ 文字で入力する"],
                horizontal=True,
                key=f"answer_mode_{current_step}"
            )

            if answer_mode == "🎙️ 音声で答える":
                st.markdown("#### 🎙️ 録音スタート")
                st.markdown("ここを押して英語を読んでね")

                # 波形が出るマイクコンポーネント (streamlit-mic-recorder)
                audio_info = mic_recorder(
                    start_prompt="🔴 録音開始",
                    stop_prompt="⏹️ 録音終了",
                    just_once=False,
                    use_container_width=True,
                    key=f"mic_{current_step}"
                )

                # 音声データが取得できている場合
                if audio_info and "bytes" in audio_info and audio_info["bytes"]:
                    audio_bytes = audio_info["bytes"]
                    st.audio(audio_bytes, format="audio/wav")

                    if st.button("📤 この音声を送信して次へ進む", key=f"submit_btn_{current_step}", type="primary"):
                        with st.spinner("音声をアップロードし、AI採点中..."):
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f_wav:
                                f_wav.write(audio_bytes)
                                wav_path = f_wav.name

                            try:
                                file_name = f"{s_school}_{s_class}_{s_number}_{s_name}_Q{q_id}.wav"
                                audio_url = upload_audio_to_drive(wav_path, file_name)

                                api_key_to_use = get_api_key_for_class(all_config, s_class)

                                transcript, evaluation, advice = evaluate_audio_with_gemini(wav_path, str(q_text), str(q_criteria), api_key_to_use)
                            finally:
                                try:
                                    os.remove(wav_path)
                                except Exception:
                                    pass

                            jst = pytz.timezone('Asia/Tokyo')
                            timestamp = datetime.datetime.now(jst).strftime('%Y-%m-%d %H:%M:%S')

                            # この設問で質問文を表示したかどうか（サイドバーのチェックボックス由来）
                            viewed_flag = st.session_state.question_viewed.get(current_step, False)
                            viewed_text = "はい" if viewed_flag else "いいえ"

                            result_row = [
                                timestamp, s_school, s_class, s_number, s_name,
                                st.session_state.get("user_id"), q_id, q_text, viewed_text,
                                transcript, evaluation, advice, audio_url, 10
                            ]
                            # 実際のSheets書き込みはバックグラウンドキューに任せる（ここでは一瞬で完了）
                            queue_result_row(ss_name, s_class, result_row)

                            st.session_state.test_results.append({
                                "question": q_text,
                                "transcript": transcript,
                                "evaluation": evaluation,
                                "advice": advice,
                                "viewed": viewed_text
                            })

                            st.session_state.test_step += 1
                            st.rerun()

            else:  # ⌨️ 文字で入力する
                text_answer = st.text_area("英語で解答を入力してください：", key=f"text_answer_{current_step}")

                if st.button("📤 この解答を送信して次へ進む", key=f"submit_text_btn_{current_step}", type="primary"):
                    if not text_answer.strip():
                        st.warning("解答を入力してください。")
                    else:
                        with st.spinner("AI採点中..."):
                            api_key_to_use = get_api_key_for_class(all_config, s_class)

                            evaluation, advice = evaluate_text_with_gemini(text_answer, str(q_text), str(q_criteria), api_key_to_use)

                            jst = pytz.timezone('Asia/Tokyo')
                            timestamp = datetime.datetime.now(jst).strftime('%Y-%m-%d %H:%M:%S')

                            viewed_flag = st.session_state.question_viewed.get(current_step, False)
                            viewed_text = "はい" if viewed_flag else "いいえ"

                            result_row = [
                                timestamp, s_school, s_class, s_number, s_name,
                                st.session_state.get("user_id"), q_id, q_text, viewed_text,
                                text_answer, evaluation, advice, "(文字入力のため音声なし)", 0
                            ]
                            queue_result_row(ss_name, s_class, result_row)

                            st.session_state.test_results.append({
                                "question": q_text,
                                "transcript": text_answer,
                                "evaluation": evaluation,
                                "advice": advice,
                                "viewed": viewed_text
                            })

                            st.session_state.test_step += 1
                            st.rerun()
        else:
            st.balloons()
            st.success("🎉 すべての質問が終了しました！お疲れ様でした。")
            st.markdown("### 📊 今回のテスト結果サマリー")
            for idx, res in enumerate(st.session_state.test_results):
                with st.expander(f"質問 {idx + 1} の結果"):
                    st.write(f"**文字起こし:** {res['transcript']}")
                    st.write(f"**評価:** {res['evaluation']}")
                    st.write(f"**アドバイス:** {res['advice']}")
                    st.write(f"**質問文を見たか:** {res.get('viewed', 'いいえ')}")

            if st.button("テストをやり直す"):
                st.session_state.pop("test_step", None)
                st.session_state.pop("test_results", None)
                st.session_state.pop("question_viewed", None)
                st.rerun()

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray; font-size: 0.9em;'>"
        "© 2026 Shogo Takeuchi. All Rights Reserved."
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
