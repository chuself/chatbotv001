import tkinter as tk

from tkinter import filedialog, messagebox
import json
import os

class DataProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data to JSON Processor")

        # Text area for manual text input
        self.text_area = tk.Text(root, height=15, width=60)
        self.text_area.pack(padx=10, pady=10)

        # File upload button
        self.upload_button = tk.Button(root, text="Upload Text File", command=self.upload_file)
        self.upload_button.pack(pady=5)

        # Process button to structure data
        self.process_button = tk.Button(root, text="Process Data", command=self.process_data)
        self.process_button.pack(pady=5)

        # Save button to save as JSON file
        self.save_button = tk.Button(root, text="Save as JSON", command=self.save_as_json)
        self.save_button.pack(pady=5)

        # Placeholder for storing structured data
        self.intents_data = None

    def upload_file(self):
        """
        Opens a file dialog for the user to upload a text file.
        The content of the file is loaded into the text area.
        """
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
            self.text_area.insert(tk.END, content)
        else:
            messagebox.showerror("Error", "Failed to load file")

    def process_data(self):
        """
        Processes the text from the text area and converts it into structured JSON format.
        Assumes text is in the format of: tag:pattern:response.
        """
        text_data = self.text_area.get("1.0", tk.END).strip()
        if not text_data:
            messagebox.showerror("Error", "No data to process.")
            return
        
        lines = text_data.split("\n")  # Split the input into lines
        intents = []
        for line in lines:
            try:
                # Split the line by ":" assuming format tag:pattern:response
                tag, pattern, response = line.split(":")
                tag = tag.strip()
                pattern = pattern.strip()
                response = response.strip()

                # Check if tag already exists, append patterns/responses if it does
                found = False
                for intent in intents:
                    if intent['tag'] == tag:
                        intent['patterns'].append(pattern)
                        intent['responses'].append(response)
                        found = True
                        break

                if not found:
                    intents.append({
                        "tag": tag,
                        "patterns": [pattern],
                        "responses": [response]
                    })
            except ValueError:
                messagebox.showerror("Error", f"Invalid format in line: {line}")
                return

        self.intents_data = {"intents": intents}
        messagebox.showinfo("Success", "Data successfully processed!")

    def save_as_json(self):
        """
        Saves the structured data (intents) into a JSON file.
        """
        if self.intents_data is None:
            messagebox.showerror("Error", "No data to save.")
            return
        
        save_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if save_path:
            with open(save_path, "w", encoding="utf-8") as json_file:
                json.dump(self.intents_data, json_file, indent=4)
            messagebox.showinfo("Success", "Data saved as JSON.")
        else:
            messagebox.showerror("Error", "Failed to save file")


if __name__ == "__main__":
    root = tk.Tk()
    app = DataProcessingApp(root)
    root.mainloop()
