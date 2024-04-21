FROM python
RUN pip install pandas
RUN pip install nltk
COPY . /test
WORKDIR /test
CMD python code.py 
