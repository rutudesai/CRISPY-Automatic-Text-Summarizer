#Importing Libraries:
import pandas as pd
import numpy as np
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)  
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
from io import StringIO
import matplotlib.pyplot as plt
# %matplotlib inline
from wordcloud import WordCloud

# Importing package and summarizer
import gensim
from gensim.summarization import summarize

import sumy
# Importing the parser and tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
# Import the LexRank summarizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# Import the summarizer
from sumy.summarizers.lsa import LsaSummarizer

# Import the summarizer
from sumy.summarizers.luhn import LuhnSummarizer

# Import the summarizer
from sumy.summarizers.kl import KLSummarizer

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Import the scoring method-ROUGE
from rouge import Rouge
ROUGE = Rouge()

#This part of code creates a sidebar and even gives it a heading 
st.sidebar.header('User Input Parameters')

Page=st.sidebar.selectbox('Select Page:',['Home Page','About the Data','Visualization','Predictor'])

if(Page=='Home Page'):
    #The below code is used to write stuff on Web App
    st.write("""
    # “CRISPy”-Automatic Text Summarizer
    
    Text summarization is a method to compress the given text into a diminished version
    conserving its information content and overall meaning. Text summarization methods can be
    classified into “extractive” and “abstractive” summarization.

    Extractive Summarization: Takes top-k sentences to form the article based on the
    frequency and combines the sentences to summarize of the article. The summary may or
    may not make any sense.
    Abstractive Summarization: In this approach the model predicts summary based on
    context and tries to provide shortened version of the article.

    This app summarizes the text provided on the basis of Abstarctive and Extractive Summarization.
    """)
    #The below code will show an image and below it there will be caption
    #image=Image.open('Colleges.JPG')
    #st.image(image)


if(Page=='About the Data'):

    st.write("""
    # About The Data
    """)
    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")

    
    st.write("""
    2. Dataset: Hugging Face, CNN Daily News
    
    a. Source: https://huggingface.co/datasets/cnn_dailymail
    
    b. Research Paper: https://dergipark.org.tr/en/download/article-file/392456
    
    c. About:
    
        1. Data Features: There are 3 data fields in the CNN News Dataset. The
        features are as follows: “id”-(Unique sha# of the URL), “article”-(Body of the
        news article), “highlight”-(Summarization of the article)
        
        2. Data Instances: There are 300,000 unique articles from CNN News & Daily
        Mail. It has 3 splits: Train-(287,113), Validation-(13,368) and Test-(11,490)
        articles each.
    """)

st.write("""
    ## File / Text upload section:
    """)
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
            # To read file as bytes:
            #bytes_data = uploaded_file.getvalue()
            #st.write(bytes_data)
            # To convert to a string based IO:
            #stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            #st.write(stringio)
            # To read file as string:
            #string_data = stringio.read()
            #st.write(string_data)
            # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)
    pd_data=pd.DataFrame(dataframe)
    pd_data.insert(0, 'Sr. No.', range(1,1 + len(pd_data)))
    pd_data.set_index('Sr. No.', inplace = True)
    st.write(pd_data)
    serial_input = st.number_input("Enter the Sr. No. for the article to be summarized : ", min_value = 1, step = 1)
    origin_text = pd_data['article'][serial_input]
    origin_summary = pd_data['article'][serial_input]
    st.write(origin_text)
else:
    st.write("File not Uploaded yet")
    st.write("OR")
    origin_text = st.text_input("Enter the Text Here")


if(Page=='Predictor'):
    
        
    Page2=st.sidebar.selectbox('Select Page:',['Abstractive Summarization','Extractive Summarization'])
                
            
    if(Page2=='Extractive Summarization'):    
        Page3=st.sidebar.selectbox('Select Page:',['TextRank', 'Sumy:LexRank', 'Sumy:Luhn', 'Sumy:Latent Sematic Analysis (LSA)', 'Sumy:KL-Sum', 'NLTK'])
        if(Page3=='TextRank'):
                 
            st.write("""
            # Predictor : TextRank
            """)
            st.write("""------------------------------------------------------------------------------------------------------------------------------------""")
            # Passing the text corpus to summarizer
            #original_text = 'Junk foods taste good that’s why it is mostly liked by every one of any age group especially kids and school going children. They generally ask for the junk food daily because they have been trend so by their parents from the childhood. They never have been discussed by their parents about the harmful effects of junk foods over health. According to the research by scientists, it has been found that junk foods have negative effects on the health in many ways. They are generally fried food found in the market in the packets. They become high in calories, high in cholesterol, low in healthy nutrients, high in sodium mineral, high in sugar, starch, unhealthy fat, lack of protein and lack of dietary fibers. Processed and junk foods are the means of rapid and unhealthy weight gain and negatively impact the whole body throughout the life. It makes able a person to gain excessive weight which is called as obesity. Junk foods tastes good and looks good however do not fulfil the healthy calorie requirement of the body. Some of the foods like french fries, fried foods, pizza, burgers, candy, soft drinks, baked goods, ice cream, cookies, etc are the example of high-sugar and high-fat containing foods. It is found according to the Centres for Disease Control and Prevention that Kids and children eating junk food are more prone to the type-2 diabetes. In type-2 diabetes our body become unable to regulate blood sugar level. Risk of getting this disease is increasing as one become more obese or overweight. It increases the risk of kidney failure. Eating junk food daily lead us to the nutritional deficiencies in the body because it is lack of essential nutrients, vitamins, iron, minerals and dietary fibers. It increases risk of cardiovascular diseases because it is rich in saturated fat, sodium and bad cholesterol. High sodium and bad cholesterol diet increases blood pressure and overloads the heart functioning. One who like junk food develop more risk to put on extra weight and become fatter and unhealthier. Junk foods contain high level carbohydrate which spike blood sugar level and make person more lethargic, sleepy and less active and alert. Reflexes and senses of the people eating this food become dull day by day thus they live more sedentary life. Junk foods are the source of constipation and other disease like diabetes, heart ailments, clogged arteries, heart attack, strokes, etc because of being poor in nutrition. Junk food is the easiest way to gain unhealthy weight. The amount of fats and sugar in the food makes you gain weight rapidly. However, this is not a healthy weight. It is more of fats and cholesterol which will have a harmful impact on your health. Junk food is also one of the main reasons for the increase in obesity nowadays.This food only looks and tastes good, other than that, it has no positive points. The amount of calorie your body requires to stay fit is not fulfilled by this food. For instance, foods like French fries, burgers, candy, and cookies, all have high amounts of sugar and fats. Therefore, this can result in long-term illnesses like diabetes and high blood pressure. This may also result in kidney failure. Above all, you can get various nutritional deficiencies when you don’t consume the essential nutrients, vitamins, minerals and more. You become prone to cardiovascular diseases due to the consumption of bad cholesterol and fat plus sodium. In other words, all this interferes with the functioning of your heart. Furthermore, junk food contains a higher level of carbohydrates. It will instantly spike your blood sugar levels. This will result in lethargy, inactiveness, and sleepiness. A person reflex becomes dull overtime and they lead an inactive life. To make things worse, junk food also clogs your arteries and increases the risk of a heart attack. Therefore, it must be avoided at the first instance to save your life from becoming ruined.The main problem with junk food is that people don’t realize its ill effects now. When the time comes, it is too late. Most importantly, the issue is that it does not impact you instantly. It works on your overtime; you will face the consequences sooner or later. Thus, it is better to stop now.You can avoid junk food by encouraging your children from an early age to eat green vegetables. Their taste buds must be developed as such that they find healthy food tasty. Moreover, try to mix things up. Do not serve the same green vegetable daily in the same style. Incorporate different types of healthy food in their diet following different recipes. This will help them to try foods at home rather than being attracted to junk food.In short, do not deprive them completely of it as that will not help. Children will find one way or the other to have it. Make sure you give them junk food in limited quantities and at healthy periods of time. '
            original_text = origin_text
            original_summary = origin_summary
            st.write(original_text)
            # Passing the text corpus to summarizer
            st.write('Short Summary with default parameters')
            short_summary = summarize(original_text)
            print(short_summary)
            st.write(short_summary)
            st.write(ROUGE.get_scores(short_summary, original_summary))
            # Summarization by ratio
            st.write('Short Summary with Ratio as 0.1')
            # Controlling the summary by ratio
            summary_by_ratio=summarize(original_text,ratio=0.1)
            print(summary_by_ratio)
            st.write(summary_by_ratio)
            st.write(ROUGE.get_scores(summary_by_ratio, original_summary))
            # Summarization by word count
            st.write('Short Summary with Word Count as Slider Value')
            # Controlling the summary by word count
            word_count_slider = st.slider('Select Number of words in the susmmary', min_value = 40, max_value = 100)
            summary_by_word_count=summarize(original_text,word_count=word_count_slider)
            print(summary_by_word_count)
            st.write(summary_by_word_count)
            if len(summary_by_word_count)==0:
                st.write("Summary Not Possible with specified parameters")
            else:
                st.write(ROUGE.get_scores(summary_by_word_count, original_summary))
            
        if(Page3=='Sumy:LexRank'):

            st.write("""
            # Predictor : Sumy:LexRank
            """)
            st.write("""------------------------------------------------------------------------------------------------------------------------------------""")

            # original_text = 'Junk foods taste good that’s why it is mostly liked by everyone of any age group especially kids and school going children. They generally ask for the junk food daily because they have been trend so by their parents from the childhood. They never have been discussed by their parents about the harmful effects of junk foods over health. According to the research by scientists, it has been found that junk foods have negative effects on the health in many ways. They are generally fried food found in the market in the packets. They become high in calories, high in cholesterol, low in healthy nutrients, high in sodium mineral, high in sugar, starch, unhealthy fat, lack of protein and lack of dietary fibers. Processed and junk foods are the means of rapid and unhealthy weight gain and negatively impact the whole body throughout the life. It makes able a person to gain excessive weight which is called as obesity. Junk foods tastes good and looks good however do not fulfil the healthy calorie requirement of the body. Some of the foods like french fries, fried foods, pizza, burgers, candy, soft drinks, baked goods, ice cream, cookies, etc are the example of high-sugar and high-fat containing foods. It is found according to the Centres for Disease Control and Prevention that Kids and children eating junk food are more prone to the type-2 diabetes. In type-2 diabetes our body become unable to regulate blood sugar level. Risk of getting this disease is increasing as one become more obese or overweight. It increases the risk of kidney failure. Eating junk food daily lead us to the nutritional deficiencies in the body because it is lack of essential nutrients, vitamins, iron, minerals and dietary fibers. It increases risk of cardiovascular diseases because it is rich in saturated fat, sodium and bad cholesterol. High sodium and bad cholesterol diet increases blood pressure and overloads the heart functioning. One who like junk food develop more risk to put on extra weight and become fatter and unhealthier. Junk foods contain high level carbohydrate which spike blood sugar level and make person more lethargic, sleepy and less active and alert. Reflexes and senses of the people eating this food become dull day by day thus they live more sedentary life. Junk foods are the source of constipation and other disease like diabetes, heart ailments, clogged arteries, heart attack, strokes, etc because of being poor in nutrition. Junk food is the easiest way to gain unhealthy weight. The amount of fats and sugar in the food makes you gain weight rapidly. However, this is not a healthy weight. It is more of fats and cholesterol which will have a harmful impact on your health. Junk food is also one of the main reasons for the increase in obesity nowadays.This food only looks and tastes good, other than that, it has no positive points. The amount of calorie your body requires to stay fit is not fulfilled by this food. For instance, foods like French fries, burgers, candy, and cookies, all have high amounts of sugar and fats. Therefore, this can result in long-term illnesses like diabetes and high blood pressure. This may also result in kidney failure. Above all, you can get various nutritional deficiencies when you don’t consume the essential nutrients, vitamins, minerals and more. You become prone to cardiovascular diseases due to the consumption of bad cholesterol and fat plus sodium. In other words, all this interferes with the functioning of your heart. Furthermore, junk food contains a higher level of carbohydrates. It will instantly spike your blood sugar levels. This will result in lethargy, inactiveness, and sleepiness. A person reflex becomes dull overtime and they lead an inactive life. To make things worse, junk food also clogs your arteries and increases the risk of a heart attack. Therefore, it must be avoided at the first instance to save your life from becoming ruined.The main problem with junk food is that people don’t realize its ill effects now. When the time comes, it is too late. Most importantly, the issue is that it does not impact you instantly. It works on your overtime; you will face the consequences sooner or later. Thus, it is better to stop now.You can avoid junk food by encouraging your children from an early age to eat green vegetables. Their taste buds must be developed as such that they find healthy food tasty. Moreover, try to mix things up. Do not serve the same green vegetable daily in the same style. Incorporate different types of healthy food in their diet following different recipes. This will help them to try foods at home rather than being attracted to junk food.In short, do not deprive them completely of it as that will not help. Children will find one way or the other to have it. Make sure you give them junk food in limited quantities and at healthy periods of time. '
            original_text = origin_text
            st.write(original_text)
            original_summary = origin_summary
            # Import the LexRank summarizer
            from sumy.summarizers.lex_rank import LexRankSummarizer
            # Passing the text corpus to summarizer
            # Initializing the parser
            my_parser = PlaintextParser.from_string(original_text,Tokenizer('english')) 
            #lex_rank_summarizer(original_text, sentences_count)
            lex_rank_summarizer = LexRankSummarizer()
            # Creating a summary of 3 sentences.
            lexrank_summary = lex_rank_summarizer(my_parser.document,sentences_count=3)
            # Printing the summary
            summary = ' '
            for sentence in lexrank_summary:
                st.write(sentence)
                summary += " " + str(sentence)
            st.write(ROUGE.get_scores(summary, original_summary))
                     
        if(Page3=="Sumy:Latent Sematic Analysis (LSA)"):

            st.write("""
            # Predictor : Sumy:Latent Semantic Analysis (LSA)
            """)
            st.write("""------------------------------------------------------------------------------------------------------------------------------------""")

            # original_text = 'Junk foods taste good that’s why it is mostly liked by everyone of any age group especially kids and school going children. They generally ask for the junk food daily because they have been trend so by their parents from the childhood. They never have been discussed by their parents about the harmful effects of junk foods over health. According to the research by scientists, it has been found that junk foods have negative effects on the health in many ways. They are generally fried food found in the market in the packets. They become high in calories, high in cholesterol, low in healthy nutrients, high in sodium mineral, high in sugar, starch, unhealthy fat, lack of protein and lack of dietary fibers. Processed and junk foods are the means of rapid and unhealthy weight gain and negatively impact the whole body throughout the life. It makes able a person to gain excessive weight which is called as obesity. Junk foods tastes good and looks good however do not fulfil the healthy calorie requirement of the body. Some of the foods like french fries, fried foods, pizza, burgers, candy, soft drinks, baked goods, ice cream, cookies, etc are the example of high-sugar and high-fat containing foods. It is found according to the Centres for Disease Control and Prevention that Kids and children eating junk food are more prone to the type-2 diabetes. In type-2 diabetes our body become unable to regulate blood sugar level. Risk of getting this disease is increasing as one become more obese or overweight. It increases the risk of kidney failure. Eating junk food daily lead us to the nutritional deficiencies in the body because it is lack of essential nutrients, vitamins, iron, minerals and dietary fibers. It increases risk of cardiovascular diseases because it is rich in saturated fat, sodium and bad cholesterol. High sodium and bad cholesterol diet increases blood pressure and overloads the heart functioning. One who like junk food develop more risk to put on extra weight and become fatter and unhealthier. Junk foods contain high level carbohydrate which spike blood sugar level and make person more lethargic, sleepy and less active and alert. Reflexes and senses of the people eating this food become dull day by day thus they live more sedentary life. Junk foods are the source of constipation and other disease like diabetes, heart ailments, clogged arteries, heart attack, strokes, etc because of being poor in nutrition. Junk food is the easiest way to gain unhealthy weight. The amount of fats and sugar in the food makes you gain weight rapidly. However, this is not a healthy weight. It is more of fats and cholesterol which will have a harmful impact on your health. Junk food is also one of the main reasons for the increase in obesity nowadays.This food only looks and tastes good, other than that, it has no positive points. The amount of calorie your body requires to stay fit is not fulfilled by this food. For instance, foods like French fries, burgers, candy, and cookies, all have high amounts of sugar and fats. Therefore, this can result in long-term illnesses like diabetes and high blood pressure. This may also result in kidney failure. Above all, you can get various nutritional deficiencies when you don’t consume the essential nutrients, vitamins, minerals and more. You become prone to cardiovascular diseases due to the consumption of bad cholesterol and fat plus sodium. In other words, all this interferes with the functioning of your heart. Furthermore, junk food contains a higher level of carbohydrates. It will instantly spike your blood sugar levels. This will result in lethargy, inactiveness, and sleepiness. A person reflex becomes dull overtime and they lead an inactive life. To make things worse, junk food also clogs your arteries and increases the risk of a heart attack. Therefore, it must be avoided at the first instance to save your life from becoming ruined.The main problem with junk food is that people don’t realize its ill effects now. When the time comes, it is too late. Most importantly, the issue is that it does not impact you instantly. It works on your overtime; you will face the consequences sooner or later. Thus, it is better to stop now.You can avoid junk food by encouraging your children from an early age to eat green vegetables. Their taste buds must be developed as such that they find healthy food tasty. Moreover, try to mix things up. Do not serve the same green vegetable daily in the same style. Incorporate different types of healthy food in their diet following different recipes. This will help them to try foods at home rather than being attracted to junk food.In short, do not deprive them completely of it as that will not help. Children will find one way or the other to have it. Make sure you give them junk food in limited quantities and at healthy periods of time. '
            original_text = origin_text
            original_summary = origin_summary
            st.write(original_text)
            # Parsing the text string using PlaintextParser
            from sumy.nlp.tokenizers import Tokenizer
            from sumy.parsers.plaintext import PlaintextParser
            parser=PlaintextParser.from_string(original_text,Tokenizer('english'))
            # creating the summarizer
            lsa_summarizer=LsaSummarizer()
            lsa_summary= lsa_summarizer(parser.document,3)
           # Printing the summar
            summary = " "
            for sentence in lsa_summary:
                st.write(sentence)
                summary += " " + str(sentence)
            st.write(ROUGE.get_scores(summary, original_summary))

        if(Page3=='Sumy:Luhn'):

            st.write("""
            # Predictor : Sumy:Luhn
            """)
            st.write("""------------------------------------------------------------------------------------------------------------------------------------""")
            # original_text = 'Junk foods taste good that’s why it is mostly liked by everyone of any age group especially kids and school going children. They generally ask for the junk food daily because they have been trend so by their parents from the childhood. They never have been discussed by their parents about the harmful effects of junk foods over health. According to the research by scientists, it has been found that junk foods have negative effects on the health in many ways. They are generally fried food found in the market in the packets. They become high in calories, high in cholesterol, low in healthy nutrients, high in sodium mineral, high in sugar, starch, unhealthy fat, lack of protein and lack of dietary fibers. Processed and junk foods are the means of rapid and unhealthy weight gain and negatively impact the whole body throughout the life. It makes able a person to gain excessive weight which is called as obesity. Junk foods tastes good and looks good however do not fulfil the healthy calorie requirement of the body. Some of the foods like french fries, fried foods, pizza, burgers, candy, soft drinks, baked goods, ice cream, cookies, etc are the example of high-sugar and high-fat containing foods. It is found according to the Centres for Disease Control and Prevention that Kids and children eating junk food are more prone to the type-2 diabetes. In type-2 diabetes our body become unable to regulate blood sugar level. Risk of getting this disease is increasing as one become more obese or overweight. It increases the risk of kidney failure. Eating junk food daily lead us to the nutritional deficiencies in the body because it is lack of essential nutrients, vitamins, iron, minerals and dietary fibers. It increases risk of cardiovascular diseases because it is rich in saturated fat, sodium and bad cholesterol. High sodium and bad cholesterol diet increases blood pressure and overloads the heart functioning. One who like junk food develop more risk to put on extra weight and become fatter and unhealthier. Junk foods contain high level carbohydrate which spike blood sugar level and make person more lethargic, sleepy and less active and alert. Reflexes and senses of the people eating this food become dull day by day thus they live more sedentary life. Junk foods are the source of constipation and other disease like diabetes, heart ailments, clogged arteries, heart attack, strokes, etc because of being poor in nutrition. Junk food is the easiest way to gain unhealthy weight. The amount of fats and sugar in the food makes you gain weight rapidly. However, this is not a healthy weight. It is more of fats and cholesterol which will have a harmful impact on your health. Junk food is also one of the main reasons for the increase in obesity nowadays.This food only looks and tastes good, other than that, it has no positive points. The amount of calorie your body requires to stay fit is not fulfilled by this food. For instance, foods like French fries, burgers, candy, and cookies, all have high amounts of sugar and fats. Therefore, this can result in long-term illnesses like diabetes and high blood pressure. This may also result in kidney failure. Above all, you can get various nutritional deficiencies when you don’t consume the essential nutrients, vitamins, minerals and more. You become prone to cardiovascular diseases due to the consumption of bad cholesterol and fat plus sodium. In other words, all this interferes with the functioning of your heart. Furthermore, junk food contains a higher level of carbohydrates. It will instantly spike your blood sugar levels. This will result in lethargy, inactiveness, and sleepiness. A person reflex becomes dull overtime and they lead an inactive life. To make things worse, junk food also clogs your arteries and increases the risk of a heart attack. Therefore, it must be avoided at the first instance to save your life from becoming ruined.The main problem with junk food is that people don’t realize its ill effects now. When the time comes, it is too late. Most importantly, the issue is that it does not impact you instantly. It works on your overtime; you will face the consequences sooner or later. Thus, it is better to stop now.You can avoid junk food by encouraging your children from an early age to eat green vegetables. Their taste buds must be developed as such that they find healthy food tasty. Moreover, try to mix things up. Do not serve the same green vegetable daily in the same style. Incorporate different types of healthy food in their diet following different recipes. This will help them to try foods at home rather than being attracted to junk food.In short, do not deprive them completely of it as that will not help. Children will find one way or the other to have it. Make sure you give them junk food in limited quantities and at healthy periods of time. '
            original_text = origin_text
            original_summary = origin_summary
            st.write(original_text)
            # Creating the parser
            from sumy.nlp.tokenizers import Tokenizer
            from sumy.parsers.plaintext import PlaintextParser
            parser=PlaintextParser.from_string(original_text,Tokenizer('english'))
            #  Creating the summarizer
            luhn_summarizer=LuhnSummarizer()
            luhn_summary=luhn_summarizer(parser.document,sentences_count=3)
            # Printing the summary
            summary = " "
            for sentence in luhn_summary:
                st.write(sentence)
                summary += " " + str(sentence)
            st.write(ROUGE.get_scores(summary, original_summary))
                    
        if(Page3=='Sumy:KL-Sum'):

            st.write("""
            # Predictor : Sumy:KL-Sum
            """)
            st.write("""------------------------------------------------------------------------------------------------------------------------------------""")

            #original_text = 'Junk foods taste good that’s why it is mostly liked by everyone of any age group especially kids and school going children. They generally ask for the junk food daily because they have been trend so by their parents from the childhood. They never have been discussed by their parents about the harmful effects of junk foods over health. According to the research by scientists, it has been found that junk foods have negative effects on the health in many ways. They are generally fried food found in the market in the packets. They become high in calories, high in cholesterol, low in healthy nutrients, high in sodium mineral, high in sugar, starch, unhealthy fat, lack of protein and lack of dietary fibers. Processed and junk foods are the means of rapid and unhealthy weight gain and negatively impact the whole body throughout the life. It makes able a person to gain excessive weight which is called as obesity. Junk foods tastes good and looks good however do not fulfil the healthy calorie requirement of the body. Some of the foods like french fries, fried foods, pizza, burgers, candy, soft drinks, baked goods, ice cream, cookies, etc are the example of high-sugar and high-fat containing foods. It is found according to the Centres for Disease Control and Prevention that Kids and children eating junk food are more prone to the type-2 diabetes. In type-2 diabetes our body become unable to regulate blood sugar level. Risk of getting this disease is increasing as one become more obese or overweight. It increases the risk of kidney failure. Eating junk food daily lead us to the nutritional deficiencies in the body because it is lack of essential nutrients, vitamins, iron, minerals and dietary fibers. It increases risk of cardiovascular diseases because it is rich in saturated fat, sodium and bad cholesterol. High sodium and bad cholesterol diet increases blood pressure and overloads the heart functioning. One who like junk food develop more risk to put on extra weight and become fatter and unhealthier. Junk foods contain high level carbohydrate which spike blood sugar level and make person more lethargic, sleepy and less active and alert. Reflexes and senses of the people eating this food become dull day by day thus they live more sedentary life. Junk foods are the source of constipation and other disease like diabetes, heart ailments, clogged arteries, heart attack, strokes, etc because of being poor in nutrition. Junk food is the easiest way to gain unhealthy weight. The amount of fats and sugar in the food makes you gain weight rapidly. However, this is not a healthy weight. It is more of fats and cholesterol which will have a harmful impact on your health. Junk food is also one of the main reasons for the increase in obesity nowadays.This food only looks and tastes good, other than that, it has no positive points. The amount of calorie your body requires to stay fit is not fulfilled by this food. For instance, foods like French fries, burgers, candy, and cookies, all have high amounts of sugar and fats. Therefore, this can result in long-term illnesses like diabetes and high blood pressure. This may also result in kidney failure. Above all, you can get various nutritional deficiencies when you don’t consume the essential nutrients, vitamins, minerals and more. You become prone to cardiovascular diseases due to the consumption of bad cholesterol and fat plus sodium. In other words, all this interferes with the functioning of your heart. Furthermore, junk food contains a higher level of carbohydrates. It will instantly spike your blood sugar levels. This will result in lethargy, inactiveness, and sleepiness. A person reflex becomes dull overtime and they lead an inactive life. To make things worse, junk food also clogs your arteries and increases the risk of a heart attack. Therefore, it must be avoided at the first instance to save your life from becoming ruined.The main problem with junk food is that people don’t realize its ill effects now. When the time comes, it is too late. Most importantly, the issue is that it does not impact you instantly. It works on your overtime; you will face the consequences sooner or later. Thus, it is better to stop now.You can avoid junk food by encouraging your children from an early age to eat green vegetables. Their taste buds must be developed as such that they find healthy food tasty. Moreover, try to mix things up. Do not serve the same green vegetable daily in the same style. Incorporate different types of healthy food in their diet following different recipes. This will help them to try foods at home rather than being attracted to junk food.In short, do not deprive them completely of it as that will not help. Children will find one way or the other to have it. Make sure you give them junk food in limited quantities and at healthy periods of time. '
            original_text = origin_text
            original_summary = origin_summary
            st.write(original_text)
            # Creating the parser
            from sumy.nlp.tokenizers import Tokenizer
            from sumy.parsers.plaintext import PlaintextParser
            parser=PlaintextParser.from_string(original_text,Tokenizer('english'))
            # Instantiating the  KLSummarizer
            kl_summarizer=KLSummarizer()
            kl_summary=kl_summarizer(parser.document,sentences_count=3)
            # Printing the summary
            summary = ' '
            for sentence in kl_summary:
                summary += " " + str(sentence)
                st.write(sentence)
            st.write(ROUGE.get_scores(summary, original_summary))

        if(Page3=='NLTK'):
            st.write("""
            # Predictor : NLTK
            """)
            st.write("""------------------------------------------------------------------------------------------------------------------------------------""")



            # Input your text for summarizing below:

            text = origin_text
            original_summary = origin_summary
            original_summary = origin_summary
            # Next, you need to tokenize the text:

            stopWords = set(stopwords.words("english"))
            words = word_tokenize(text)

            # Now, you will need to create a frequency table to keep a score of each word:

            freqTable = dict()
            for word in words:
                word = word.lower()
                if word in stopWords:
                    continue
                if word in freqTable:
                    freqTable[word] += 1
                else:
                    freqTable[word] = 1

            # Next, create a dictionary to keep the score of each sentence:

            sentences = sent_tokenize(text)
            sentenceValue = dict()

            for sentence in sentences:
                for word, freq in freqTable.items():
                    if word in sentence.lower():
                        if word in sentence.lower():
                            if sentence in sentenceValue:
                                sentenceValue[sentence] += freq
                            else:
                                sentenceValue[sentence] = freq

            sumValues = 0
            for sentence in sentenceValue:
                sumValues += sentenceValue[sentence]

            # Now, we define the average value from the original text as such:

            average = int(sumValues / len(sentenceValue))

            # And lastly, we need to store the sentences into our summary:

            summary = ''

            for sentence in sentences:
                if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)):
                    summary += " " + sentence
            print(summary)
            st.write(summary)
            st.write(ROUGE.get_scores(summary, original_summary))



if(Page=='Visualization'):

    st.write("""
    # Visualization
    """)
    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")
    original_text = origin_text
    if len(origin_text)==0:
        st.write("No wordcloud possible without a single word!!!")
    # Create and generate a word cloud image:
    else:
        wordcloud = WordCloud().generate(original_text)

# Display the generated image:
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        st.pyplot()










