 def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Define offensive words and sensitive topics for validation
        self.offensive_words = ["badword1", "badword2", "hack", "bypass"]
        self.sensitive_topics = ["politics", "violence", "hate speech"]

    # Guardrail: Inappropriate Content Filter
    def inappropriate_content_filter(self, text):
        """Filters input for inappropriate or offensive content."""
        if any(word in text.lower() for word in self.offensive_words):
            return "[Content removed due to inappropriate language]"
        return text

    # Guardrail: Prompt Injection Shield
    def prompt_injection_shield(self, text):
        """Protects against malicious or harmful instructions."""
        forbidden_phrases = ["delete all data", "bypass security", "unauthorized access"]
        if any(phrase in text.lower() for phrase in forbidden_phrases):
            return "[Prompt rejected due to security risks]"
        return text

    # Guardrail: Offensive Language Filter
    def offensive_language_filter(self, text):
        """Filters offensive or disrespectful content in responses."""
        pattern = re.compile(r'\b(?:' + '|'.join(self.offensive_words) + r')\b', re.IGNORECASE)
        filtered_text = pattern.sub("[censored]", text)
        return filtered_text

    # Guardrail: Sensitive Content Scanner
    def sensitive_content_scanner(self, text):
        """Detects sensitive topics and flags them."""
        if any(topic in text.lower() for topic in self.sensitive_topics):
            return "[Content flagged due to sensitive topics]"
        return text

    # Unified Input Validation Guardrail
    def validate_input(self, text):
        """Validates input through all input guardrails."""
        text = self.inappropriate_content_filter(text)
        text = self.prompt_injection_shield(text)
        return text

    # Unified Output Validation Guardrail
    def validate_output(self, text):
        """Validates output through all output guardrails."""
        text = self.offensive_language_filter(text)
        text = self.sensitive_content_scanner(text)
        return text
