from PyPDF2 import PdfReader
import re
import ollama

reader = PdfReader("WHO_classification_textbook.pdf")

# pages 293 to 299
text = ""
for page in reader.pages[292:299]:
    text += page.extract_text()

# Remove citations 
text = re.sub(r'\{\d+(?:,\d+)*\}', '', text)
text = re.sub(r'\[\d+(?:,\d+)*\]', '', text)
# Removes ¬ \n 
text = re.sub(r'¬\s*\n', '', text)
# Removes lists of numbers
text = re.sub(r'\(\s*\d+(?:\s*,\s*\d+)*\s*\)', '', text)

# Remove Table and Figure captions and the line after them (if short)
def remove_caption_and_next_line(text, pattern):
    lines = text.split('\n')
    cleaned_lines = []
    skip_next = False
    for i, line in enumerate(lines):
        if skip_next:
            skip_next = False
            continue
        if re.match(pattern, line):
            # If next line is short, skip it too
            if i+1 < len(lines) and len(lines[i+1].strip()) < 60:
                skip_next = True
            continue
        cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)

text = remove_caption_and_next_line(text, r'^(Table|Fig)[^\n]*')

# Remove page numbers, headers, and footers
text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
text = re.sub(r'^.*Mature B-cell neoplasms.*$', '', text, flags=re.MULTILINE)
text = re.sub(r'^Diffuse large B-cell lymphoma.*$', '', text, flags=re.MULTILINE)

# Fix hyphenation at line breaks (e.g. sub-\ntype -> subtype)
text = re.sub(r'-\s*\n\s*', '', text)

# Join all single newlines (not double newlines) to spaces
text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

# Remove references with odd brackets or pipes
text = re.sub(r'\{[\d,\s]+\}', '', text)
text = re.sub(r'\[[\d,\s]+\]', '', text)
text = re.sub(r'\|[\d,\s]+\}', '', text)

# Remove special characters and artifacts
text = re.sub(r'[|!;<>^*’‘“”©®]', '', text)

# Remove codes like 9680/3, t(1418)(q32q21.3), etc.
text = re.sub(r'\b\d{3,4}/\d\b', '', text)
text = re.sub(r't\(\d{2,5}\)\(q\d{2,5}q\d{2,5}\)', '', text)

# Remove extra spaces and leading/trailing whitespace
text = re.sub(r' +', ' ', text)
text = re.sub(r'\n+', '\n', text)
text = text.strip()

# Remove or merge random short lines (less than 5 chars and not part of a paragraph)
def remove_short_lines(text):
    lines = text.split('\n')
    merged = []
    for line in lines:
        if len(line.strip()) < 5:
            continue
        merged.append(line)
    return '\n'.join(merged)

text = remove_short_lines(text)

# Split into sentences/paragraphs for dataset
# (You can adjust this to split by paragraphs or sentences as needed)
dataset = [line.strip() for line in re.split(r'\.|\n', text) if line.strip()]

print(text)
# print(dataset)
with open('dataset.txt', 'w') as f:
    for item in dataset:
        f.write(item + '\n')

from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,      # number of characters or tokens per chunk
    chunk_overlap=200     # overlap between chunks
)
dataset = splitter.split_text(text)

print("Loaded", len(dataset), "entries")

# Implement the retrieval system

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

# Each element in the VECTOR_DB will be a tuple (chunk, embedding)
# The embedding is a list of floats, for example: [0.1, 0.04, -0.34, 0.21, ...]
VECTOR_DB = []

def add_chunk_to_database(chunk):
  embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
  VECTOR_DB.append((chunk, embedding))

for i, chunk in enumerate(dataset):
  add_chunk_to_database(chunk)
  print(f'Added chunk {i+1}/{len(dataset)} to the database')

def cosine_similarity(a, b):
  dot_product = sum([x * y for x, y in zip(a, b)])
  norm_a = sum([x ** 2 for x in a]) ** 0.5
  norm_b = sum([x ** 2 for x in b]) ** 0.5
  return dot_product / (norm_a * norm_b)

def retrieve(query, top_n=3):
  query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
  # temporary list to store (chunk, similarity) pairs
  similarities = []
  for chunk, embedding in VECTOR_DB:
    similarity = cosine_similarity(query_embedding, embedding)
    similarities.append((chunk, similarity))
  # sort by similarity in descending order, because higher similarity means more relevant chunks
  similarities.sort(key=lambda x: x[1], reverse=True)
  # finally, return the top N most relevant chunks
  return similarities[:top_n]



# Chatbot

input_query = input('Ask me a question: ')
retrieved_knowledge = retrieve(input_query)

print('Retrieved knowledge:')
for chunk, similarity in retrieved_knowledge:
  print(f' - (similarity: {similarity:.2f}) {chunk}')

context = '\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])
instruction_prompt = f'''You are a helpful chatbot.
Use only the following pieces of context to answer the question. Don't make up any new information:
{context}
'''
# print(instruction_prompt)

stream = ollama.chat(
  model=LANGUAGE_MODEL,
  messages=[
    {'role': 'system', 'content': instruction_prompt},
    {'role': 'user', 'content': input_query},
  ],
  stream=True,
)

# print the response from the chatbot in real-time
print('Chatbot response:')
for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)


