import pandas as pd

class HRDataProcessor:
    def __init__(self, file_object):
        try:
            # Пытаемся прочитать как CSV
            self.df = pd.read_csv(file_object)
        except Exception:
            # ВАЖНО: сбрасываем указатель файла в начало перед второй попыткой
            file_object.seek(0)
            # Если не вышло, пробуем как Excel
            self.df = pd.read_excel(file_object)
        
        # Список вопросов — это заголовки колонок
        self.questions = self.df.columns.tolist()

    def get_valid_answers(self, question_name: str):
        """Возвращает только заполненные ответы HR, пропуская пустые."""
        if question_name not in self.df.columns:
            return []
        # dropna() убирает пустые клетки
        return self.df[question_name].dropna().astype(str).tolist()
