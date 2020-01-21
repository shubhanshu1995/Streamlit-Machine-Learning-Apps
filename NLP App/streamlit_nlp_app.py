import streamlit as st

# NLP Packages
import spacy
from textblob import TextBlob
from gensim.summarization import summarize

# Sumy Pkg
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

@st.cache
def sumy_summarizer(docx):
	parser = PlaintextParser.from_string(docx,Tokenizer("english"))
	lex_summarizer = LexRankSummarizer()
	summary = lex_summarizer(parser.document,3)
	summary_list = [str(sentence) for sentence in summary]
	result = ' '.join(summary_list)
	return result

@st.cache
def text_analyzer(my_text):
	nlp = spacy.load('en_core_web_sm')
	docx = nlp(my_text)
	# tokens = [token.text for token in docx]
	allData = ['Token:{},\nLemma:{}'.format(token.text,token.lemma_) for token in docx]
	return allData

@st.cache
def entity_analyzer(my_text):
	nlp = spacy.load('en_core_web_sm')
	docx = nlp(my_text)
	tokens = [token.text for token in docx]
	entities = [(entity.text,entity.label_) for entity in docx.ents ]
	allData = ['"Tokens":{},\n"Entities":{}'.format(tokens,entities)]
	return allData
	
# Pkgs

def main():
	""" NLP App with Streamlit """
	st.title("NLP App made with ❤ by Shubhanshu using Streamlit")
	st.subheader("Natural Language Processing on the Go")

	# SIDEBARS
	st.sidebar.subheader("About the App")
	st.sidebar.text("NLP App made using Streamlit Framework")
	st.sidebar.info("Streamlit is Awesome !!...Kudos to the Streamlit Team :)")

	# Tokenization
	if st.checkbox("Show Tokens and Lemma"):
		st.subheader("Tokenize Your Text")
		message = st.text_area("Enter Your Text","Type Here",key=0)
		if st.button("Analyze",key=1):
			nlp_result = text_analyzer(message)
			st.json(nlp_result)


	# Named Entity
	if st.checkbox("Show Named Entities"):
		st.subheader("Extract Entities From Your Text")
		message = st.text_area("Enter Your Text","Type Here",key=2)
		if st.button("Extract",key=3):
			nlp_result = entity_analyzer(message)
			st.json(nlp_result)

	# Sentiment Analysis
	if st.checkbox("Show Sentiment Analysis"):
		st.subheader("Sentiment of Your Text")
		message = st.text_area("Enter Your Text","Type Here",key=4)
		if st.button("Analyze",key=5):
			blob = TextBlob(message)
			result_sentiment = blob.sentiment
			st.success(result_sentiment)

	# Text Summarization
	if st.checkbox("Show Text Summarization"):
		st.subheader("Summarize your text")
		message = st.text_area("Enter Your Text","Type Here",key=6)
		summary_options = st.selectbox("Choose Your Summarizer",("gensim","sumy"))
		if st.button("Summarize",key=7):
			if summary_options == 'gensim':
				st.text("Using Gensim..")
				summary_result = summarize(message)
			elif summary_options == 'sumy':
				st.text("Using Sumy..")
				summary_result = sumy_summarizer(message)
			else:
				st.warning("Using Default Summarizer")
				st.text("Using Gensim")
				summary_result = summarize(message)
			st.success(summary_result)

if __name__ == '__main__':
	main()