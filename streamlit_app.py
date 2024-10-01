import anthropic
import os
import requests
import streamlit as st

from bs4 import BeautifulSoup
from openai import OpenAI

MODEL_NAME = "claude-3-5-sonnet-20240620"


def scrape_website(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        return text
    else:
        return f"Failed to retrieve the webpage. Status code: {response.status_code}"


def get_completion(client, prompt, max_tokens=2048):
    return client.messages.create(
        model=MODEL_NAME,
        max_tokens=max_tokens,
        messages=[{
            "role": 'user', "content":  prompt
        }]
    ).content 


def get_query(client, text, max_tokens=64):
    query = get_completion(
        client=client,
        prompt=f"""Here are the contents of a website: <website>{text}</website>
    
        Summarize the contents of this website to form a search query that will be used to find related content.
        Output only the search query without any other extra text in the following format: 
        
        Answer: query
        """,
        max_tokens=max_tokens,
    )[0].text.split(':', 1)[1]

    return query


def search(query):
    messages = [
        {
            "role": "system",
            "content": (
                """You are an artificial intelligence assistant and you need to 
                give the following:
                1. A comprehensive summary of all the results from the user's query.
                2. A list of references for all external sources. Be sure to include the URL.
                For each reference, use the following format:
                "<Title>". <Publication>, <Date Published>. <Authors>. <URL>"""
            ),
        },
        {
            "role": "user",
            "content": (
                query
            ),
        },
    ]

    client = OpenAI(api_key=os.environ['PERPLEXITY_API_KEY'], base_url="https://api.perplexity.ai")
    response = client.chat.completions.create(
        model="llama-3.1-sonar-small-128k-online",
        messages=messages,
    )

    return response.choices[0].message.content


def analyze(client, site_text, search_text, max_tokens=8192):
    return get_completion(
        client=client,        
        prompt=f"""Here are the contents of a website: <website>{site_text}</website>

        Here are the results of a search on the topic: <search>{search_text}</search>
    
        Create a separate section for each of the following:
        Website Summary: Summarize the contents of the website.
        Comparison with External Sources: Compare the information in the website to the search results.
        """,
        max_tokens=max_tokens,
    )[0].text.split(':', 1)[1]


def main():
    os.environ['ANTHROPIC_API_KEY'] = st.secrets['ANTHROPIC_API_KEY']
    os.environ['PERPLEXITY_API_KEY'] = st.secrets['PERPLEXITY_API_KEY']
    st.title("Seeing the Sites")

    with st.form(key='site_analyzer'):
        url = st.text_input(label='Site URL')
        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
        scraped_text = scrape_website(url)
        query = get_query(client=client, text=scraped_text)
        search_results = search(query)
        analysis = analyze(client=client, site_text=scraped_text, search_text=search_results)

        print({
            "query": query,
            "analysis": analysis
            })
        
        st.header(query.title())
        st.write(analysis)
        st.write("Summary of External Sources:")
        st.write(search_results)



if __name__ == "__main__":
    main()
