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
                """You are a search engine AI. Run a search on the user's query and output the following:
                1. Summary: A comprehensive summary of all the results from the user's query.
                2. References: A list of references for all external sources.
                It is VERY IMPORTANT that you include the URL for each reference.
                Use the following format:
                "<Title>". <Publication>, <Date Published>. <Authors>. <URL>
                The URL should be clickable andstart with 'https://www.'
                
                For example: "Biden Will Visit Appalachia Region Ravaged by Hurricane Helene". The New York Times, 2024-09-30. https://www.nytimes.com/2024/09/30/us/biden-helene-north-carolina.html
                """
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


def summarize(client, site_text, max_tokens=2048):
    return get_completion(
        client=client,        
        prompt=f"""Summarize the contents of this website: <website>{site_text}</website>
        """,
        max_tokens=max_tokens,
    )[0].text.split(':', 1)[1]


def compare(client, site_text, search_text, max_tokens=4096):
    return get_completion(
        client=client,        
        prompt=f"""Here are the contents of a website: <website>{site_text}</website>
        Here are the results of a search on the topic: <search>{search_text}</search>
        Compare the information in the website to the search results.
        Be as comprehensive and accurate as possible with all relevant details.
        """,
        max_tokens=max_tokens,
    )[0].text.split(':', 1)[1]


def main():
    os.environ['ANTHROPIC_API_KEY'] = st.secrets['ANTHROPIC_API_KEY']
    os.environ['PERPLEXITY_API_KEY'] = st.secrets['PERPLEXITY_API_KEY']

    st.title("Seeing the Sites")
    st.write("Enter the URL of a website to see a summary and comparison with external sources.")

    with st.form(key='site_analyzer'):
        url = st.text_input(label='Site URL')
        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
        scraped_text = scrape_website(url)
        summary = summarize(client=client, site_text=scraped_text)
        query = get_query(client=client, text=scraped_text)
        search_results = search(query)
        comparison = compare(client=client, site_text=scraped_text, search_text=search_results)

        print(f"Query:\n{query}\n")
        print(f"Summary:\n{summary}\n")
        print(f"Comparison:\n{comparison}\n")
        print(f"Search Results:\n{search_results}")
        references = search_results.split('References')[-1]
        
        st.header(query.title())
        st.header("Website Summary")
        st.write(summary)
        st.header("Comparison with External Sources")
        st.write(comparison)
        st.header("References")
        st.write(references)

if __name__ == "__main__":
    main()
