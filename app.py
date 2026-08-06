import streamlit as st
import pandas as pd
import time
import os
import datetime
import pytz
from gtts import gTTS
import tempfile
import json
import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from audio_recorder_streamlit import audio_recorder
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

# ページ設定
st.set_page_config(
    page_title="中学校英語スピーキングテスト",
    page_icon="🎤",
    layout="centered"
)

# -------------------------------------------------------------
# 認証・設定ヘルパー (Retries & Secrets)
# -------------------------------------------------------------
def get_secrets():
    try:
        return {
            "default_gemini_api_key": st.secrets["GEMINI_API_KEY"],
            "drive_folder_id": st.secrets["GOOGLE_DRIVE_FOLDER_ID"],
            "master_spreadsheet_name": st.secrets["MASTER_SPREADSHEET_NAME"],
            "service_account_info": dict(st.secrets["connections"]["gsheets"]),
            "teacher_api_keys": st.secrets.get("TEACHER_API_KEYS", {})
        }
    except Exception as e:
        st.error(f"Streamlit Secretsの設定が不足しています: {e}")
        st.stop()

SECRETS = get_secrets()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_gspread_client():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    creds_info = SECRETS["service_account_info"]
    creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
    return gspread.authorize(creds)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_drive_service():
    scopes = ["https://www.googleapis.com/auth/drive"]
    creds_info = SECRETS["service_account_info"]
    creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
    return build('drive', 'v3', credentials=creds)

# -------------------------------------------------------------
# 全体マスタの読み込み (Schoolsシート)
# -------------------------------------------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def load_master_schools():
    client = get_gspread_client()
    target = SECRETS["master_spreadsheet_name"]
    try:
        if target.startswith("1") and len(target) > 20:
            sheet = client.open_by_key(target)
        else:
            sheet = client.open(target)
        ws = sheet.worksheet("Schools")
        return pd.DataFrame(ws.get_all_records())
    except Exception as e:
        st.error(f"【全体マスタ読み込みエラー】スプレッドシート '{target}' の取得に失敗しました。\n原因: {e}")
        raise e

# -------------------------------------------------------------
# ログイン認証
# -------------------------------------------------------------
def authenticate_user(input_id, input_pw):
    try:
        master_df = load_master_schools()
    except Exception:
        return {"authenticated": False}
        
    client = get_gspread_client()
    
    for _, school_row in master_df.iterrows():
        s_id = school_row["school_id"]
        s_name = school_row["school_name"]
        target_ss = str(school_row["spreadsheet_name_or_id"]).strip()
        
        try:
            if target_ss.startswith("1") and len(target_ss) > 20:
                ss = client.open_by_key(target_ss)
            else:
                ss = client.open(target_ss)
                
            users_ws = ss.worksheet("Users")
            users_df = pd.DataFrame(users_ws.get_all_records())
            
            if not users_df.empty:
                matched = users_df[(users_df["user_id"].astype(str) == input_id) & (users_df["password"].astype(str) == input_pw)]
                if not matched.empty:
                    u_info = matched.iloc[0]
                    return {
                        "authenticated": True,
                        "user_id": u_info["user_id"],
                        "user_name": u_info["user_name"],
                        "school_id": s_id,
                        "school_name": s_name,
                        "spreadsheet_name": target_ss,
                        "assigned_class": u_info["assigned_class"],
                        "role": u_info.get("role", "student")
                    }
        except Exception:
            continue
            
    return {"authenticated": False}

# -------------------------------------------------------------
# 学校別 Config（横長お題）の読み込み・保存
# -------------------------------------------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def load_config_from_sheet(spreadsheet_name):
    client = get_gspread_client()
    try:
        if spreadsheet_name.startswith("1") and len(spreadsheet_name) > 20:
            sheet = client.open_by_key(spreadsheet_name)
        else:
            sheet = client.open(spreadsheet_name)
        config_ws = sheet.worksheet("Config")
        return pd.DataFrame(config_ws.get_all_records())
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

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def save_config_to_sheet(spreadsheet_name, df_config):
    client = get_gspread_client()
    if spreadsheet_name.startswith("1") and len(spreadsheet_name) > 20:
        sheet = client.open_by_key(spreadsheet_name)
    else:
        sheet = client.open(spreadsheet_name)
    try:
        config_ws = sheet.worksheet("Config")
    except gspread.exceptions.WorksheetNotFound:
        config_ws = sheet.add_worksheet(title="Config", rows="50", cols="15")
    
    config_ws.clear()
    config_ws.update([df_config.columns.values.tolist()] + df_config.values.tolist())

# -------------------------------------------------------------
# Google Drive アップロード & 結果保存
# -------------------------------------------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def upload_audio_to_drive(file_path, file_name):
    service = get_drive_service()
    folder_id = SECRETS["drive_folder_id"]
    
    try:
        file_metadata = {'name': file_name, 'parents': [folder_id]}
        media = MediaFileUpload(file_path, mimetype='audio/wav', resumable=True)
        file = service.files().create(body=file_metadata, media_body=media, fields='id, webViewLink').execute()
        
        try:
            service.permissions().create(fileId=file.get('id'), body={'role': 'reader', 'type': 'anyone'}).execute()
        except Exception:
            pass
        return file.get('webViewLink')
    except Exception as e:
        st.error(f"【Googleドライブ アップロードエラー】詳細: {e}")
        raise e

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def save_result_to_sheet(spreadsheet_name, target_class, result_row):
    client = get_gspread_client()
    if spreadsheet_name.startswith("1") and len(spreadsheet_name) > 20:
        spreadsheet = client.open_by_key(spreadsheet_name)
    else:
        spreadsheet = client.open(spreadsheet_name)
    sheet_title = str(target_class).strip()
    
    try:
        worksheet = spreadsheet.worksheet(sheet_title)
    except gspread.exceptions.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(title=sheet_title, rows="100", cols="15")
        header = [
            "タイムスタンプ(JST)", "学校名", "クラス", "出席番号", "氏名", "ログインID",
            "問題番号", "質問文", "文字起こし", "評価(A/B/C)", "アドバイス", "音声URL", "解答時間(秒)"
        ]
        worksheet.append_row(header)
        
    worksheet.append_row(result_row)

# -------------------------------------------------------------
# Gemini API 評価
# -------------------------------------------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def evaluate_audio_with_gemini(audio_path, question_text, criteria, api_key):
    genai.configure(api_key=api_key)
    audio_file = genai.upload_file(path=audio_path)
    
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
    
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content([audio_file, prompt])
    
    try:
        genai.delete_file(audio_file.name)
    except Exception:
        pass
        
    text_res = response.text.strip()
    if text_res.startswith("```json"):
        text_res = text_res[7:]
    if text_res.endswith("```"):
        text_res = text_res[:-3]
    text_res = text_res.strip()
    
    try:
        res_json = json.loads(text_res)
        return res_json.get("transcript", ""), res_json.get("evaluation", "C"), res_json.get("advice", "評価生成エラー")
    except Exception as e:
        return f"Parse Error: {response.text}", "C", f"解析エラー: {e}"

# -------------------------------------------------------------
# メインアプリケーション
# -------------------------------------------------------------
def main():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("🔒 中学校英語スピーキングテスト ログイン")
        with st.form("login_form"):
            input_id = st.text_input("ログインID")
            input_pw = st.text_input("パスワード", type="password")
            submitted = st.form_submit_button("ログイン")
            
            if submitted:
                auth_result = authenticate_user(input_id, input_pw)
                if auth_result["authenticated"]:
                    st.session_state.authenticated = True
                    st.session_state.user_id = auth_result["user_id"]
                    st.session_state.user_name = auth_result["user_name"]
                    st.session_state.school_id = auth_result["school_id"]
                    st.session_state.school_name = auth_result["school_name"]
                    st.session_state.spreadsheet_name = auth_result["spreadsheet_name"]
                    st.session_state.assigned_class = auth_result["assigned_class"]
                    st.session_state.role = auth_result["role"]
                    
                    teacher_keys = SECRETS.get("teacher_api_keys", {})
                    st.session_state.gemini_api_key = teacher_keys.get(auth_result["user_id"], SECRETS["default_gemini_api_key"])
                    
                    st.success(f"ログイン成功: {st.session_state.user_name} ({st.session_state.school_name})")
                    st.rerun()
                else:
                    st.error("IDまたはパスワードが正しくありません。（スピーキングテストのマスタ設定もご確認ください）")
        return

    with st.sidebar:
        st.write(f"学校: **{st.session_state.get('school_name')}**")
        st.write(f"ユーザー: **{st.session_state.get('user_name')}**")
        st.write(f"権限: **{'先生' if st.session_state.get('role') == 'teacher' else '生徒'}**")
        st.markdown("---")
        if st.button("ログアウト"):
            st.session_state.clear()
            st.rerun()

    role = st.session_state.get("role", "student")
    ss_name = st.session_state.get("spreadsheet_name")

    # 👨‍🏫 先生用画面（横長Configの編集管理）
    if role == "teacher":
        st.title(f"👨‍🏫 教師ダッシュボード ({st.session_state.get('school_name')})")
        st.markdown("クラスごとの担当教師、問題数 (`num_questions`)、各問題の質問文・評価基準を横長テーブルで編集できます。")
        
        config_df = load_config_from_sheet(ss_name)
        edited_config_df = st.data_editor(config_df, num_rows="dynamic", use_container_width=True)
        
        if st.button("💾 変更をスプレッドシートに保存する", type="primary"):
            with st.spinner("保存中..."):
                save_config_to_sheet(ss_name, edited_config_df)
                st.success("Config設定が正常に更新されました！")

    # 🎙️ 生徒用画面（横長Configから問題数を読み込んで出題）
    else:
        st.title("🎤 中学校英語スピーキングテスト 受験画面")
        
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

        col1, col2 = st.columns(2)
        with col1:
            s_school = st.text_input("学校名", st.session_state.get("school_name", ""))
            s_name = st.text_input("生徒氏名", st.session_state.get("user_name", ""))
        with col2:
            s_class = st.text_input("クラス", value=my_class)
            s_number = st.number_input("出席番号", min_value=1, max_value=50, step=1)

        if "test_step" not in st.session_state:
            st.session_state.test_step = 0
        if "test_results" not in st.session_state:
            st.session_state.test_results = []

        total_questions = len(questions_list)
        current_step = st.session_state.test_step

        if current_step < total_questions:
            q_info = questions_list[current_step]
            q_id = q_info["id"]
            q_text = q_info["text"]
            q_criteria = q_info["criteria"]

            st.progress((current_step) / total_questions, text=f"進捗: 質問 {current_step + 1} / {total_questions}")
            
            st.markdown(f"### 質問 {current_step + 1} (問題番号: {q_id})")
            st.info("🔊 以下の英語の質問をよく聞いて、英語で答えてください。")
            
            tts = gTTS(text=str(q_text), lang='en')
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_audio:
                tts.save(tmp_audio.name)
                tmp_audio_path = tmp_audio.name
                
            st.audio(tmp_audio_path, format="audio/mp3", autoplay=False)
            st.markdown(f"**質問文:** `{q_text}`")

            if f"thinking_done_{current_step}" not in st.session_state:
                if st.button("▶️ 音声を再生してシンキングタイム(5秒)を開始する", key=f"start_btn_{current_step}"):
                    with st.spinner("シンキングタイム中... 準備をしてください"):
                        bar = st.progress(0)
                        for i in range(5):
                            time.sleep(1)
                            bar.progress((i + 1) / 5)
                    st.session_state[f"thinking_done_{current_step}"] = True
                    st.rerun()
            else:
                st.success("✨ シンキングタイム終了！録音ボタンを押して話してください。")

                st.markdown("#### 🎙️ 録音エリア")
                audio_bytes = audio_recorder(text="クリックして録音開始/停止", recording_color="#e8b62c", neutral_color="#6aa36f", icon_size="2x")

                if audio_bytes:
                    st.audio(audio_bytes, format="audio/wav")
                    
                    if st.button("📤 この音声を送信して次へ進む", key=f"submit_btn_{current_step}"):
                        if not s_name.strip():
                            st.warning("氏名を入力してください。")
                        else:
                            with st.spinner("音声をアップロードし、AI採点中..."):
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f_wav:
                                    f_wav.write(audio_bytes)
                                    wav_path = f_wav.name

                                file_name = f"{s_school}_{s_class}_{s_number}_{s_name}_Q{q_id}.wav"
                                audio_url = upload_audio_to_drive(wav_path, file_name)

                                api_key_to_use = st.session_state.get("gemini_api_key", SECRETS["default_gemini_api_key"])
                                transcript, evaluation, advice = evaluate_audio_with_gemini(wav_path, str(q_text), str(q_criteria), api_key_to_use)

                                jst = pytz.timezone('Asia/Tokyo')
                                timestamp = datetime.datetime.now(jst).strftime('%Y-%m-%d %H:%M:%S')

                                result_row = [
                                    timestamp, s_school, s_class, s_number, s_name,
                                    st.session_state.get("user_id"), q_id, q_text,
                                    transcript, evaluation, advice, audio_url, 10
                                ]
                                save_result_to_sheet(ss_name, s_class, result_row)

                                st.session_state.test_results.append({
                                    "question": q_text,
                                    "transcript": transcript,
                                    "evaluation": evaluation,
                                    "advice": advice
                                })

                                st.session_state.test_step += 1
                                st.rerun()
        else:
            st.balloons()
            st.success("🎉 すべての質問が終了しました！お疲れ様でした。")
            st.markdown("### 📊 今回のテスト結果サマリー")
            for idx, res in enumerate(st.session_state.test_results):
                with st.expander(f"質問 {idx + 1} の結果"):
                    st.write(f"**質問:** {res['question']}")
                    st.write(f"**文字起こし:** {res['transcript']}")
                    st.write(f"**評価:** {res['evaluation']}")
                    st.write(f"**アドバイス:** {res['advice']}")
                    
            if st.button("テストをやり直す"):
                st.session_state.pop("test_step", None)
                st.session_state.pop("test_results", None)
                for key in list(st.session_state.keys()):
                    if key.startswith("thinking_done_"):
                        del st.session_state[key]
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
