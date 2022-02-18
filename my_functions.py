# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 15:32:20 2021

@author: pathouli
"""

def clean_text(txt_in):
    import re
    clean = re.sub('[^A-Za-z0-9]+', " ", txt_in).lower().strip()
    return clean

def file_seeker(path_in, sw):
    import os
    import pandas as pd
    import re
    import nltk
    dictionary = nltk.corpus.words.words("en")
    tmp_pd = pd.DataFrame()
    for dirName, subdirList, fileList in os.walk(path_in):
        tmp_dir_name = dirName.split("/")[-1::][0]
        try:
            for word in fileList:
                #print(word)
                tmp = open(dirName+"/"+word, "r", encoding="ISO-8859-1")
                tmp_f = tmp.read()
                tmp.close()
                tmp_f = re.sub('[^A-Za-z0-9]+', " ", tmp_f).strip().lower() #\n extra spacex
                if sw == "check_words":
                    tmp_f = [word for word in tmp_f.split() if word in dictionary]
                    tmp_f = ' '.join(tmp_f)
                tmp_pd = tmp_pd.append({'label': tmp_dir_name, 
                                        #'path': dirName+"/"+word}, 
                                        'body': tmp_f},
                                        ignore_index = True)
        except Exception as e:
            print (e)
            pass
    return tmp_pd

def rem_sw(var):
    from nltk.corpus import stopwords
    sw = set(stopwords.words("English"))
    my_test = [word for word in var.split() if word not in sw]
    my_test = ' '.join(my_test)
    return my_test

def check_dictionary(var):
    import nltk
    dictionary = nltk.corpus.words.words("en")
    fin_txt = [word for word in var.split() if word in dictionary]
    fin_txt = ' '.join(fin_txt)
    return fin_txt

def stem_fun(var):
    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()
    tmp_txt = [stemmer.stem(word) for word in var.split()]
    tmp_txt = ' '.join(tmp_txt)
    return tmp_txt

def jaccard_fun(var_a, var_b):
    #DOCUMENT SIMILARITY
    doc_a_set = set(var_a.split())
    doc_b_set = set(var_b.split())
    j_d = float(len(doc_a_set.intersection(
        doc_b_set)))/ float(len(doc_a_set.union(doc_b_set)))
    return j_d

def extract_embeddings_pre(df_in, num_vec_in, path_in, filename):
    from gensim.models import Word2Vec
    import pandas as pd
    from gensim.models import KeyedVectors
    import pickle
    my_model = KeyedVectors.load_word2vec_format(filename, binary=True) 
    my_model = Word2Vec(df_in.str.split(),
                        min_count=1, vector_size=num_vec_in)#size=300)
    word_dict = my_model.wv.key_to_index
    #my_model.similarity("trout", "fish")
    def get_score(var):
        try:
            import numpy as np
            tmp_arr = list()
            for word in var:
                tmp_arr.append(list(my_model.wv[word]))
        except:
            pass
        return np.mean(np.array(tmp_arr), axis=0)
    tmp_out = df_in.str.split().apply(get_score)
    tmp_data = tmp_out.apply(pd.Series).fillna(0)
    pickle.dump(my_model, open(path_in + "embeddings.pkl", "wb"))
    pickle.dump(tmp_data, open(path_in + "embeddings_df.pkl", "wb" ))
    return tmp_data

def cosine_fun(df_a, df_b, label_in): 
    from sklearn.metrics.pairwise import cosine_similarity
    import pandas as pd
    cos_matrix_t = pd.DataFrame(cosine_similarity(df_a, df_b))
    cos_matrix_t.index = label_in
    cos_matrix_t.columns = label_in
    return cos_matrix_t

def extract_embeddings_domain(df_in, num_vec_in, path_in):
    #domain specific, train out own model specific to our domains
    from gensim.models import Word2Vec
    import pandas as pd
    import numpy as np
    import pickle
    model = Word2Vec(
        df_in.str.split(), min_count=1,
        vector_size=num_vec_in, workers=3, window=5, sg=0)
    wrd_dict = model.wv.key_to_index
    #model.wv.most_similar('machine', topn=10)
    def get_score(var):
        try:
            tmp_arr = list()
            for word in var:
                tmp_arr.append(list(model.wv[word]))
        except:
            pass
        return np.mean(np.array(tmp_arr), axis=0)
    tmp_out = df_in.str.split().apply(get_score)
    tmp_data = tmp_out.apply(pd.Series).fillna(0)
    #model.wv.save_word2vec_format(base_path + "embeddings_domain.pkl")
    pickle.dump(model, open(path_in + "embeddings_domain_model.pkl", "wb"))
    pickle.dump(tmp_data, open(path_in + "embeddings_df_domain.pkl", "wb" ))
    return tmp_data, wrd_dict

def my_vec_fun(df_in, m, n, path_o):
    from sklearn.feature_extraction.text import CountVectorizer
    import pandas as pd
    import pickle
    vectorizer = CountVectorizer(ngram_range=(m, n))
    my_vec_t = pd.DataFrame(vectorizer.fit_transform(df_in).toarray())
    my_vec_t.columns = vectorizer.vocabulary_
    pickle.dump(vectorizer, open(path_o + "vectorizer.pkl", "wb" ))
    pickle.dump(my_vec_t, open(path_o + "my_vec_data.pkl", "wb" ))
    return my_vec_t

def my_tf_idf_fun(df_in, m, n, path_o):
    from sklearn.feature_extraction.text import TfidfVectorizer
    import pandas as pd
    import pickle
    vectorizer = TfidfVectorizer(ngram_range=(m, n))
    my_tf_idf_t = pd.DataFrame(vectorizer.fit_transform(df_in).toarray())
    my_tf_idf_t.columns = vectorizer.vocabulary_
    pickle.dump(vectorizer, open(path_o + "tf_idf.pkl", "wb" ))
    pickle.dump(my_tf_idf_t, open(path_o + "my_tf_idf_data.pkl", "wb" ))
    return my_tf_idf_t

def my_model_fun(df_in, label_in, path_o):
    #function 1
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.model_selection import train_test_split
    import pandas as pd
    from sklearn.metrics import precision_recall_fscore_support
    import pickle
    my_model = RandomForestClassifier(max_depth=10, random_state=123)
    #my_model = GaussianNB()
    #80/20 train,test,split
    X_train, X_test, y_train, y_test = train_test_split(
        df_in, label_in, test_size=0.20, random_state=123)
    my_model.fit(X_train, y_train)
    pickle.dump(my_model, open(path_o + "my_model.pkl", "wb"))
    y_pred = my_model.predict(X_test)
    model_metrics = pd.DataFrame(precision_recall_fscore_support(
    y_test, y_pred, average='weighted'))
    model_metrics.index = ["precision", "recall", "fscore", "none"]
    #function 2 prediction
    the_preds = pd.DataFrame(my_model.predict_proba(X_test))
    the_preds.columns = my_model.classes_

    return my_model, model_metrics, the_preds

def my_model_fun_grid(df_in, label_in, path_o):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.model_selection import train_test_split
    import pandas as pd
    from sklearn.metrics import precision_recall_fscore_support
    import pickle
    from sklearn.model_selection import GridSearchCV
    #my_model = RandomForestClassifier(max_depth=10, random_state=123)
    my_model = RandomForestClassifier(random_state=123)
    parameters = {'n_estimators':[10, 100], 'max_depth':[None, 1, 10],
                  'random_state': [123], 'criterion': ("gini", "entropy")}
    my_grid = GridSearchCV(my_model, parameters)
    #80/20 train,test,split
    X_train, X_test, y_train, y_test = train_test_split(
        df_in, label_in, test_size=0.20, random_state=123)
    my_grid.fit(X_train, y_train)
    print ("Best Score:", my_grid.best_score_)
    best_params = my_grid.best_params_
    my_model_opt = RandomForestClassifier(**best_params)
    #my_model = GaussianNB()
    my_model_opt.fit(X_train, y_train)
    pickle.dump(my_model_opt, open(path_o + "my_model.pkl", "wb"))
    y_pred = my_model_opt.predict(X_test)
    model_metrics = pd.DataFrame(precision_recall_fscore_support(
    y_test, y_pred, average='weighted'))
    model_metrics.index = ["precision", "recall", "fscore", "none"]
    #function 2 prediction
    the_preds = pd.DataFrame(my_model_opt.predict_proba(X_test))
    the_preds.columns = my_model_opt.classes_

    return my_model_opt, model_metrics, the_preds

def my_pca_fun(df_in, n_comp_in, path_o):
    from sklearn.decomposition import PCA
    import pickle
    my_pca = PCA(n_components=n_comp_in)
    my_pca_vec = my_pca.fit_transform(df_in)
    pickle.dump(my_pca, open(path_o + "pca.pkl", "wb"))
    return my_pca_vec

def score_text(sample_text_i, the_path_i):
    #Classify arbitrary text, need to run ALL same pre-processing and 
    #transformation steps as what was applied to the training set
    #Step 1 call; up clean_text function
    import numpy as np
    from my_functions import clean_text, stem_fun,my_pca_fun
    import pickle 
    clean_text = clean_text(sample_text_i)
    #step 2 call up stem_fun
    clean_text = stem_fun(clean_text)
    #call up the appropriate transformation algo and transform our new data
    vec_model = pickle.load(open(the_path_i + "vectorizer.pkl", "rb"))
    clean_text_vec = vec_model.transform([clean_text]).toarray()
    
    pca_model = pickle.load(open(the_path_i + "pca.pkl", "rb"))
    pca_vec = pca_model.transform(clean_text_vec)
    
    my_model = pickle.load(open(the_path_i + "my_model.pkl", "rb"))
    the_pred = my_model.predict(pca_vec)
    the_pred_proba = np.max(np.array(my_model.predict_proba(pca_vec)))
    
    return the_pred, the_pred_proba

# def file_seeker(path_in):
#     import os
#     import pandas as pd
#     tmp_pd = pd.DataFrame()
#     for dirName, subdirList, fileList in os.walk(path_in):
#         tmp_dir_name = dirName.split("/")[-1::][0]
#         for word in fileList:
#             try:
#                 tmp_pd = tmp_pd.append(
#                     {"label": tmp_dir_name,"path": dirName+"/"+word},
#                     ignore_index=True)
#             except Exception as e:
#                 print (e)
#                 pass
#     return tmp_pd