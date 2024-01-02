from bs4 import BeautifulSoup
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
from langchain.chat_models import ChatOpenAI

# load the API token from .env file for replicate llama LLM 
load_dotenv()

# declare output json schema
class RelatedCompany(BaseModel):
    company_name: str
    company_domain: str

class OutputSchema(BaseModel):
    related_companies: List[RelatedCompany]
    topic: str

# define llm
chat_model = ChatOpenAI(
    model="gpt-3.5-turbo",
    max_tokens=1000
)

# Read the HTML content from the text file
with open("web.txt", "r", encoding="utf-8") as file:
    html_content = file.read()

# Use bs4 to parse html and extract needed content only
soup = BeautifulSoup(html_content, 'html.parser')
# Extract all address div
content = soup.find("body")
# getting title and article content
title = soup.select_one("#tc-main-content > div > div > div > article.article-container.article--post > div.article__content-outer > div:nth-child(2) > div:nth-child(1) > header > div.article__title-wrapper > h1").text
article_content = soup.select_one("#tc-main-content > div > div > div > article.article-container.article--post > div.article__content-outer > div:nth-child(2) > div:nth-child(2)").text

parser = PydanticOutputParser(pydantic_object=OutputSchema)

prompt = PromptTemplate(
    template="""
        You are required to identify the relevant company mentioned and its website domain in the following article text./
        unstructured text from article: {article}. Topic: {topic}. /
        \n {format_instructions}
        """,
            input_variables = ["article", "topic"],
            partial_variables = {"format_instructions": parser.get_format_instructions()}
)

prompt_formatted_str = prompt.format_prompt(article=article_content, topic=title)

output = chat_model(prompt_formatted_str.to_messages())

parsed = parser.parse(output.content)
print(parsed.model_dump())

with open('output.txt', 'w') as f:
    f.write(str(parsed.model_dump()))


