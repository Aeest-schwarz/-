import re
from collections import Counter
from gigachat import GigaChat


class HRAnalyzer:
    MAX_HR_ANSWERS = 10

    def __init__(self, api_key: str, scope: str = "GIGACHAT_API_PERS"):
        self.api_key = api_key
        self.scope = scope

    def _expert_consensus_rate(self, hr_answers: list[str]) -> float:
        """
        Грубая оценка согласованности экспертов:
        считаем долю экспертов, упомянувших самое частотное слово (3+ символов).
        Возвращает значение от 0.0 до 1.0.
        """
        if not hr_answers:
            return 0.0

        all_words = []
        for answer in hr_answers:
            words = [w.lower().strip(".,!?;:\"'()") for w in answer.split() if len(w) > 3]
            all_words.extend(words)

        if not all_words:
            return 0.0

        counter = Counter(all_words)
        most_common_word, _ = counter.most_common(1)[0]

        mentions = sum(
            1 for answer in hr_answers
            if most_common_word in answer.lower()
        )
        return mentions / len(hr_answers)

    def analyze(self, question: str, hr_answers: list[str], student_answer: str) -> dict:
        # hr_answers — ответы HR-экспертов (не студентов)
        hr_answers = hr_answers[:self.MAX_HR_ANSWERS]
        hr_context = "\n".join([f"ЭКСПЕРТ {i+1}: {a}" for i, a in enumerate(hr_answers)])

        consensus_rate = self._expert_consensus_rate(hr_answers)
        # Если более 95% экспертов дали однозначный ответ — предупреждение не нужно
        use_disclaimer = consensus_rate < 0.95

        disclaimer_instruction = (
            """Сначала добавь предупреждение:
Это не единственно верный и субъективный вариант ответа, однако он отражает наиболее сильную позицию на рынке.

Затем идеальный ответ."""
            if use_disclaimer
            else "Напиши идеальный ответ без предупреждений."
        )

        prompt = f"""Ты — экспертный HR-аналитик.

Твоя задача — разобрать ответы HR-экспертов и сравнить с ответом студента.

ВАЖНО: НЕ используй JSON. НЕ используй markdown. Строго следуй структуре ниже.
Каждый раздел должен содержать ТОЛЬКО свой контент — не повторяй вопрос и не копируй ответ студента в раздел MATCH или CRITIQUE.

Верни результат СТРОГО в формате:

CLUSTERS:
1) Название | Процент | Описание
2) Название | Процент | Описание

MATCH:
[Только анализ совпадения ответа студента с позицией экспертов. Не добавляй заголовок "Совпадение с рынком". Не повторяй вопрос и не цитируй ответ студента дословно.]

CRITIQUE:
[Только критика и замечания по ответу студента. Не переходи в раздел GOLD.]

GOLD:
{disclaimer_instruction}

------------------------------------

ВОПРОС:
{question}

ОТВЕТЫ ЭКСПЕРТОВ:
{hr_context}

ОТВЕТ СТУДЕНТА:
{student_answer}
"""

        try:
            with GigaChat(
                credentials=self.api_key,
                scope=self.scope,
                verify_ssl_certs=False
            ) as client:
                response = client.chat(prompt)
                content = response.choices[0].message.content.strip()

            parsed = self._parse_response(content, question, student_answer)
            return {"status": "success", "data": parsed}

        except Exception as e:
            return {
                "status": "error",
                "message": f"Ошибка при обработке ответа модели: {str(e)}"
            }

    def _parse_response(self, text: str, question: str = "", student_answer: str = "") -> dict:
        lines = text.split("\n")

        clusters = []
        match_lines, critique_lines, gold_lines = [], [], []
        mode = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            normalized = line.upper().rstrip(":")
            if normalized.startswith("CLUSTERS"):
                mode = "clusters"
                continue
            elif normalized.startswith("MATCH"):
                mode = "match"
                continue
            elif normalized.startswith("CRITIQUE"):
                mode = "critique"
                continue
            elif normalized.startswith("GOLD"):
                mode = "gold"
                continue

            if mode == "clusters" and "|" in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 3:
                    clusters.append({
                        "name": parts[0].lstrip("1234567890). "),
                        "percentage": parts[1],
                        "description": parts[2]
                    })
            elif mode == "match":
                match_lines.append(line)
            elif mode == "critique":
                critique_lines.append(line)
            elif mode == "gold":
                gold_lines.append(line)

        student_match = self._sanitize_match(
            " ".join(match_lines), question, student_answer
        )

        # Фикс: если GOLD пустой, а CRITIQUE подозрительно длинный —
        # последние 40% строк critique отдаём в gold
        if not gold_lines and len(critique_lines) > 4:
            split_at = int(len(critique_lines) * 0.6)
            gold_lines = critique_lines[split_at:]
            critique_lines = critique_lines[:split_at]

        return {
            "clusters": clusters,
            "student_match": student_match,
            "critique": " ".join(critique_lines),
            "gold_standard": " ".join(gold_lines),
        }

    def _sanitize_match(self, match_text: str, question: str, student_answer: str) -> str:
        """
        Очищает поле MATCH от:
        - фразы "Совпадение с рынком" в любом регистре и с любым разделителем
        - случайно попавшего текста вопроса или ответа студента
        """
        if not match_text:
            return match_text

        # Убираем фразу "Совпадение с рынком" в любом регистре и с любым разделителем после
        match_text = re.sub(
            r'(?i)совпадение\s+с\s+рынком\s*[:\-–—]?\s*',
            '',
            match_text
        ).strip()

        match_lower = match_text.lower().strip()
        question_lower = question.lower().strip()
        student_lower = student_answer.lower().strip()

        def overlap_ratio(a: str, b: str) -> float:
            if not a or not b:
                return 0.0
            shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
            return len(shorter) / len(longer) if shorter in longer else 0.0

        if overlap_ratio(match_lower, question_lower) > 0.8:
            return ""
        if overlap_ratio(match_lower, student_lower) > 0.8:
            return ""

        return match_text
