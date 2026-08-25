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
from streamlit_mic_recorder import mic_recorder  # 🎙️ 波形が出るマイク
from google import genai
from tenacity import retry, stop_after_attempt, wait_exponential

# ページ設定
st.set_page_config(
    page_title="Nexus English 2.0",
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
            "teacher_api_keys": dict(st.secrets.get("TEACHER_API_KEYS", {}))
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
#   ログイン画面の学校プルダウンで使うため、キャッシュして頻繁に読まないようにする
# -------------------------------------------------------------
@st.cache_data(ttl=300)
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
#   以前は「全学校をループしてUsersシートを検索」していたため、
#   クラス一斉ログイン時にSheets APIのレート制限(429)にかかりやすかった。
#   学校をプルダウンで先に選んでもらい、選んだ1校のUsersシートだけを見る。
# -------------------------------------------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def authenticate_user(school_row, input_id, input_pw):
    client = get_gspread_client()

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

                # 出席番号の取得（列がない場合はデフォルトで1）
                att_num = 1
                if "attendance_number" in u_info and pd.notna(u_info["attendance_number"]):
                    try:
                        att_num = int(u_info["attendance_number"])
                    except Exception:
                        att_num = 1

                return {
                    "authenticated": True,
                    "user_id": u_info["user_id"],
                    "user_name": u_info["user_name"],
                    "school_id": s_id,
                    "school_name": s_name,
                    "spreadsheet_name": target_ss,
                    "assigned_class": u_info["assigned_class"],
                    "role": u_info.get("role", "student"),
                    "attendance_number": att_num
                }
    except Exception:
        pass

    return {"authenticated": False}

# -------------------------------------------------------------
# 学校別 Config（横長お題）の読み込み・保存
#   読み込みはキャッシュして、生徒が問題を1問進めるたびに
#   Sheetsへ再アクセスしないようにする。
#   保存後はキャッシュを明示的にクリアして、最新のお題がすぐ反映されるようにする。
# -------------------------------------------------------------
@st.cache_data(ttl=60)
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
    # 保存した内容がすぐ生徒画面に反映されるよう、Configのキャッシュを破棄する
    load_config_from_sheet.clear()

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
            "問題番号", "質問文", "質問文を見たか", "文字起こし", "評価(A/B/C)", "アドバイス", "音声URL", "解答時間(秒)"
        ]
        worksheet.append_row(header)

    worksheet.append_row(result_row)

# -------------------------------------------------------------
# Gemini API 評価（音声）
# -------------------------------------------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def evaluate_audio_with_gemini(audio_path, question_text, criteria, api_key):
    try:
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
            model='gemini-3.5-flash',
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

    except Exception as e:
        st.error(f"Gemini API通信エラーの詳細: {e}")
        raise e

# -------------------------------------------------------------
# Gemini API 評価（テキスト） ※音声を使わない解答用
# -------------------------------------------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def evaluate_text_with_gemini(text_answer, question_text, criteria, api_key):
    try:
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
            model='gemini-3.5-flash',
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
#   同じ質問文なら再生成しない。rerunのたびにgTTSへ通信しないようにする。
#   一時ファイルは読み込み後すぐ削除する。
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

        with st.form("login_form"):
            selected_school_name = st.selectbox("学校を選択してください：", school_names)
            input_id = st.text_input("ログインID")
            input_pw = st.text_input("パスワード", type="password")
            submitted = st.form_submit_button("ログイン")

            if submitted:
                school_row = master_df[master_df["school_name"] == selected_school_name].iloc[0]
                auth_result = authenticate_user(school_row, input_id, input_pw)
                if auth_result["authenticated"]:
                    st.session_state.authenticated = True
                    st.session_state.user_id = auth_result["user_id"]
                    st.session_state.user_name = auth_result["user_name"]
                    st.session_state.school_id = auth_result["school_id"]
                    st.session_state.school_name = auth_result["school_name"]
                    st.session_state.spreadsheet_name = auth_result["spreadsheet_name"]
                    st.session_state.assigned_class = auth_result["assigned_class"]
                    st.session_state.role = auth_result["role"]
                    st.session_state.attendance_number = auth_result["attendance_number"]

                    st.toast("ログインしました！", icon="✅")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("学校・IDまたはパスワードが正しくありません。")
        return

    role = st.session_state.get("role", "student")
    ss_name = st.session_state.get("spreadsheet_name")

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
        st.write(f"出席番号: **{st.session_state.get('attendance_number')}番**")
        st.write(f"権限: **{'先生' if st.session_state.get('role') == 'teacher' else '生徒'}**")

        # 🆕 生徒向け：質問文を表示するかどうかをサイドバーで選択できるようにする
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
        st.markdown("クラスごとの担当教師、問題数 (`num_questions`)、各問題の質問文・評価基準を横長テーブルで編集できます。")

        config_df = load_config_from_sheet(ss_name)
        edited_config_df = st.data_editor(config_df, num_rows="dynamic", use_container_width=True)

        if st.button("💾 変更をスプレッドシートに保存する", type="primary"):
            with st.spinner("保存中..."):
                save_config_to_sheet(ss_name, edited_config_df)
                st.success("Config設定が正常に更新されました！")

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

            # 🆕 サイドバーのチェックボックスがオンの場合、質問文を画面にも表示する
            if st.session_state.get(f"show_question_{current_step}", False):
                st.info(f"📖 質問文: {q_text}")

            # 音声の自動再生（テキストは画面に出さない。同じ質問文なら再生成しない）
            tts_bytes = generate_tts_audio(str(q_text))
            st.audio(tts_bytes, format="audio/mp3", autoplay=True)

            # 🆕 回答方法の選択：音声 or 文字
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

                            # 🆕 この設問で質問文を表示したかどうか（サイドバーのチェックボックス由来）
                            viewed_flag = st.session_state.question_viewed.get(current_step, False)
                            viewed_text = "はい" if viewed_flag else "いいえ"

                            result_row = [
                                timestamp, s_school, s_class, s_number, s_name,
                                st.session_state.get("user_id"), q_id, q_text, viewed_text,
                                transcript, evaluation, advice, audio_url, 10
                            ]
                            save_result_to_sheet(ss_name, s_class, result_row)

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
                            save_result_to_sheet(ss_name, s_class, result_row)

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
