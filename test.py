def preprocess(text):
    if type(text)!=float:
        text = re.sub("https?://[\w!\?/\+\-_~=;\.,\*&@#\$%\(\)'\[\]]+", ' ', text)
        text = text.lower()
        text = re.sub('\(.*?\)',' ',text)
        text = re.sub('\s',' ',text)
        text = re.sub("’","'",text)
        text = re.sub(r"[^a-zA-z0-9.,?!/&%$']",' ',text)
        text = re.sub(",",' , ',text)
        text = re.sub("!",'. ',text)
        text = re.sub("\.",'. ',text)
        text = re.sub("\s+",' ',text)
        text = re.sub(r"[\s]*\.[\.\s]+",". ",text)
        text = re.sub("' ","'",text)
        text = re.sub(" l "," i ",text)
        text = re.sub(" ,",",",text)
        if text[-1]==" ":
            text=text[:-1]
        if text[0]==" ":
            text=text[1:]
        if text[:2]=="l ":
            text="i "+text[2:]
        return text
    else:
        return np.nan