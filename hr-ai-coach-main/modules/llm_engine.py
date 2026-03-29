import json
from gigachat import GigaChat


class HRAnalyzer:
    def __init__(self, api_key: str):
        self.client = GigaChat(
            credentials=api_key,
            scope="GIGACHAT_API_PERS",
            verify_ssl_certs=False
        )

    def analyze(self, question, hr_answers, student_answer):
        hr_context = "\n".join([f"ЭКСПЕРТ {i+1}: {a}" for i, a in enumerate(hr_answers)])

        # 👉 Новый безопасный промпт
        prompt = f"""Ты — экспертный HR-аналитик.

Твоя задача — разобрать ответы HR и сравнить с ответом студента.

ВАЖНО: НЕ используй JSON. НЕ используй markdown.

Верни результат СТРОГО в формате:

CLUSTERS:
1) Название | Процент | Описание
2) Название | Процент | Описание

MATCH:
текст

CRITIQUE:
текст

GOLD:
Сначала добавь дисклеймер:
Это не единственно верный и субъективный вариант ответа, однако он отражает наиболее сильную позицию на рынке.

Затем идеальный ответ.

------------------------------------

ВОПРОС:
{question}

ОТВЕТЫ HR:
{hr_context}

ОТВЕТ СТУДЕНТА:
{student_answer}
"""

        try:
            response = self.client.chat(prompt)
            content = response.choices[0].message.content.strip()

            parsed = self._parse_response(content)

            return {"status": "success", "data": parsed}

        except Exception as e:
            return {
                "status": "error",
                "message": f"Ошибка при обработке ответа модели: {str(e)}"
            }

    def _parse_response(self, text: str):
        lines = text.split("\n")

        clusters = []
        student_match = ""
        critique = ""
        gold = ""

        mode = None

        for line in lines:
            line = line.strip()

            if not line:
                continue

            if line.startswith("CLUSTERS"):
                mode = "clusters"
                continue
            elif line.startswith("MATCH"):
                mode = "match"
                continue
            elif line.startswith("CRITIQUE"):
                mode = "critique"
                continue
            elif line.startswith("GOLD"):
                mode = "gold"
                continue

            # --- Парсинг ---
            if mode == "clusters":
                if "|" in line:
                    parts = [p.strip() for p in line.split("|")]

                    if len(parts) >= 3:
                        clusters.append({
                            "name": parts[0].lstrip("1234567890). "),
                            "percentage": parts[1],
                            "description": parts[2]
                        })

            elif mode == "match":
                student_match += line + " "

            elif mode == "critique":
                critique += line + " "

            elif mode == "gold":
                gold += line + " "

        return {
            "clusters": clusters,
            "student_match": student_match.strip(),
            "critique": critique.strip(),
            "gold_standard": gold.strip()
        }
