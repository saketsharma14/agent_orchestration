from models.model_manager import ModelManager

manager = ModelManager()
response = manager.generate("Explain Adam optimizer in one paragraph.")
print(response)