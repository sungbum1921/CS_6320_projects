from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load from directory
# model = T5ForConditionalGeneration.from_pretrained(
#     "sql_generator",
#     local_files_only=True
# )

# tokenizer = T5Tokenizer.from_pretrained(
#     "sql_generator",
#     local_files_only=True
# )
model = T5ForConditionalGeneration.from_pretrained(
    "sql_generator-nl2sql",
    local_files_only=True
)

tokenizer = T5Tokenizer.from_pretrained(
    "sql_generator-nl2sql",
    local_files_only=True
)

# Verify loading
def generate_sql(question):
    inputs = tokenizer(f"Translate to SQL: {question}", return_tensors="pt")
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# print(generate_sql("get me teams from table sporst_team who play in premier league and have won more than 4 championships"))
# print(generate_sql("show employees and departments with salaries greater than 50000"))
print(generate_sql("give me list of students whose GPA is above 3.5; order by gpa"))
