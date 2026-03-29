import streamlit as st
import base64
import json
from streamlit_lottie import st_lottie
from modules.processor import HRDataProcessor
from modules.llm_engine import HRAnalyzer

st.set_page_config(page_title="AI HR Coach", page_icon="", layout="wide")

# код
def set_bg_hack(main_bg):
    """Устанавливает картинку на фон"""
    try:
        with open(main_bg, "rb") as f:
            b64_img = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background: url(data:image/jpeg;base64,{b64_img});
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            /* Полупрозрачный фон для основного контента, чтобы текст читался */
            .main > div {{
                background-color: rgba(255, 255, 255, 0.9);
                padding: 2rem;
                border-radius: 15px;
            }}
            .stButton>button {{ width: 100%; border-radius: 20px; font-weight: bold; }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        pass  # Если картинки нет, просто игнорируем

def load_lottieurl(filepath: str):
    """Загрузка локальной Lottie анимации"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

# Применяем фон (положи картинку background.jpg в корень проекта)
set_bg_hack("background.jpg")

# Основной интерфейс
st.title("ИИ-Тренажер: Собеседование с HR")
st.write("Сравните свои ответы с реальным опытом десятков HR-специалистов.")

# Боковая панель
with st.sidebar:
    # Добавляем анимацию (скачай любой .json с lottiefiles.com и назови hr_anim.json)
    anim = load_lottieurl("hr_anim.json")
    if anim:
        st_lottie(anim, speed=1, height=150, key="hr_animation")
        
    st.header("Настройки доступа")
    api_key = st.text_input("Введите ваш GigaChat API Key", type="password")
    st.info("Ключ нужен для работы нейросети. Мы его не сохраняем.")
    
    uploaded_file = st.file_uploader("Загрузите файл с ответами (CSV/XLSX)", type=["csv", "xlsx"])

# Основной экран
if uploaded_file and api_key:
    if 'processor' not in st.session_state:
        st.session_state.processor = HRDataProcessor(uploaded_file)
    
    proc = st.session_state.processor
    
    col1, col2 = st.columns([1, 1.2], gap="large")
    
    with col1:
        st.subheader("Практика")
        selected_q = st.selectbox("Выберите вопрос из базы:", proc.questions)
        
        # 👉 ДОБАВЛЕНО: полный текст вопроса
        st.markdown(f"**Вопрос:** {selected_q}")
        
        hr_list = proc.get_valid_answers(selected_q)
        
        st.caption(f"🔍 Найдено экспертных мнений по вопросу: {len(hr_list)}")
        
        student_ans = st.text_area(
            "Ваш ответ:",
            height=250,
            placeholder="Напишите, как бы вы ответили на этот вопрос..."
        )
        
        run_btn = st.button("Проверить мой ответ", type="primary")

    with col2:
        st.subheader("Анализ и интерпретация")
        if run_btn:
            if not student_ans:
                st.error("Пожалуйста, напишите ответ!")
            else:
                analyzer = HRAnalyzer(api_key)
                with st.spinner("Нейросеть анализирует кластеры мнений..."):
                    result = analyzer.analyze(selected_q, hr_list, student_ans)
                    
                    if result["status"] == "error":
                        st.error(result["message"])
                    else:
                        data = result["data"]
                        
                        # Красивый вывод кластеров
                        st.write("### Как ответили HR-директора:")
                        for cluster in data.get("clusters", []):
                            st.info(
                                f"**{cluster.get('name')} ({cluster.get('percentage')})**\n\n"
                                f"{cluster.get('description')}"
                            )
                        
                        st.write("---")
                        
                        # Позиция студента
                        st.write("### Оценка вашего ответа:")
                        st.markdown(f"**Совпадение с рынком:** {data.get('student_match')}")
                        
                        # Критика в желтом блоке
                        st.warning(
                            f" **Зоны роста (критика):**\n\n{data.get('critique')}"
                        )
                        
                        # Золотой стандарт в зеленом блоке
                        st.success(
                            f" **Золотой стандарт:**\n\n{data.get('gold_standard')}"
                        )

        else:
            st.info("👈 Выберите вопрос, напишите ответ и нажмите кнопку проверки. Здесь появится подробный разбор.")

else:
    # Заглушка, если ничего не загружено
    st.warning("Для начала работы введите API-ключ и загрузите файл базы знаний в левом меню.")
