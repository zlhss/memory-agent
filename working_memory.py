class WorkingMemory:
    def __init__(self, max_messages=6):
        self.max_messages = max_messages
        self.messages = []

    def add_message(self, role, content):
        self.messages.append({
            "role": role,
            "content": content
        })

        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
            
    def get_messages(self):
        return self.messages