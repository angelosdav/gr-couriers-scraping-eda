import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import os

def main():
    print("Loading dataset...")
    # Δυναμική εύρεση του σωστού path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, 'data', '03_reviews_final_processed.csv')

    # Load the cleaned dataset
    df = pd.read_csv(data_path)
    
    # Drop any null rows
    df = df.dropna(subset=['cleaned_text'])

    # Comprehensive Custom Greek Stopwords Dictionary (Matches README explicitly)
    custom_stopwords = [
        'και', 'το', 'τα', 'τη', 'την', 'της', 'του', 'των', 'στο', 'στη', 'στην', 'στα', 'στις', 'στους', 'από', 'απο', 
        'σε', 'με', 'για', 'να', 'που', 'πως', 'αν', 'ή', 'η', 'ο', 'οι', 'ένα', 'ενα', 'μια', 'μία', 'ενός', 'μιας', 
        'έναν', 'εγώ', 'εσύ', 'αυτό', 'αυτή', 'αυτά', 'αυτούς', 'αυτες', 'μου', 'σου', 'μας', 'σας', 'τους', 'τον', 
        'τι', 'ποιο', 'ποια', 'ποιος', 'κάτι', 'όλα', 'όλοι', 'όλο', 'κι', 'δεν', 'μην', 'μη', 'είναι', 'ειναι', 
        'ήταν', 'ηταν', 'έχει', 'εχει', 'έχουν', 'εχουν', 'έχω', 'εχω', 'είχα', 'ειχα', 'είχε', 'ειχε', 'είχες', 
        'είμαστε', 'είστε', 'θα', 'ότι', 'οτι', 'όπως', 'οπως', 'όταν', 'οταν', 'μετά', 'μετα', 'πάλι', 'παλι', 
        'ως', 'προς', 'ενώ', 'ενω', 'αλλά', 'αλλα', 'ακόμα', 'ακομα', 'πολύ', 'πολυ', 'έχεις', 'έχουμε', 'εχουμε', 
        'κάνω', 'κάνει', 'έκανα', 'εκανα', 'λέει', 'λένε', 'είπε', 'ειπε', 'είπαν', 'ειπαν', 'πήρα', 'πηρα', 
        'δέμα', 'δεμα', 'εταιρεία', 'εταιρεια', 'εταιρία', 'εταιρια', 'κούριερ', 'κουριερ', 'courier', 'acs', 
        'speedex', 'elta', 'ελτα', 'γενική', 'γενικη', 'ταχυδρομική', 'ταχυδρομικη', 'center', 'πακέτο', 'πακετο', 
        'package', 'delivery', 'parcel', 'παραγγελία', 'παραγγελια', 'παράδοση', 'παραδοση', 'τις', 'τισ', 'στον', 
        'ούτε', 'ουτε', 'γιατί', 'γιατι', 'υπάρχει', 'υπαρχει', 'απλά', 'απλα', 'αφού', 'αφου', 'εδώ', 'εδω', 
        'ήρθε', 'ηρθε', 'έρχεται', 'ερχεται', 'έρχονται', 'ερχονται', 'πάει', 'παει', 'πάω', 'παω', 'πάρω', 'παρω', 
        'καμία', 'καμια', 'πρέπει', 'πρεπει', 'μόνο', 'μονο', 'ξανά', 'ξανα', 'άλλο', 'αλλο', 'άλλη', 'αλλη', 
        'άλλες', 'αλλες', 'ίδιο', 'ιδιο', 'ίδια', 'ιδια', 'πάντα', 'παντα', 'σαν', 'εκεί', 'εκει', 'όχι', 'οχι', 
        'μέσα', 'μεσα', 'τελικά', 'τελικα', 'δε', 'δέματα', 'δεματα', 'δέματος', 'δεματος', 'χωρίς', 'χωρις', 
        'φορά', 'φορές', 'φορες', 'καν', 'κανένα', 'κανενα', 'κανείς', 'κανεις', 'τίποτα', 'τιποτα', 'πιο', 
        'έτσι', 'ετσι', 'κάθε', 'καθε', 'έγινε', 'εγινε', 'αστέρι', 'αστερι', 'αστέρια', 'αστερια', 'μηδέν', 'μηδεν', 
        'σήμερα', 'σημερα', 'ακόμη', 'ακομη', 'ήδη', 'ηδη', 'τώρα', 'τωρα', 'πίσω', 'πισω'
    ]

    print("Vectorizing text (CountVectorizer)...")
    # Convert text to numerical vectors
    vectorizer = CountVectorizer(max_df=0.85, min_df=10, stop_words=custom_stopwords, ngram_range=(1,2))
    X = vectorizer.fit_transform(df['cleaned_text'])

    print("Running LDA Model for 5 Topics...")
    # Train Latent Dirichlet Allocation (LDA) algorithm
    n_topics = 5
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)

    feature_names = vectorizer.get_feature_names_out()

    print("\n--- TOPIC MODELING RESULTS ---")
    for topic_idx, topic in enumerate(lda.components_):
        # Extract the top 14 keywords for each Topic
        top_features_ind = topic.argsort()[:-15:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        word_list_str = ', '.join(top_features)
        print(f"Topic {topic_idx + 1}: {word_list_str}")

if __name__ == '__main__':
    main()
