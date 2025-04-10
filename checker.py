from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import nltk
from nltk.tokenize import sent_tokenize

# Ensure necessary resources are downloaded
# nltk.download('punkt')

class BERTPlagiarismChecker:
    def __init__(self):
        print("🔍 Loading BERT Model for Embedding...")
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        print("📚 Loading spaCy NLP model...")
        self.nlp = spacy.load("en_core_web_sm")

    def extract_subject_object(self, sentence):
        doc = self.nlp(sentence)
        subject, obj = None, None
        for token in doc:
            if "subj" in token.dep_:
                subject = token.text.lower()
            if "obj" in token.dep_:
                obj = token.text.lower()
        return subject, obj

    def compare_sentences(self, s1, s2):
        embeddings = self.model.encode([s1, s2])
        cos_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

        # Check for subject-object flip
        subj1, obj1 = self.extract_subject_object(s1)
        subj2, obj2 = self.extract_subject_object(s2)

        flipped = False
        if subj1 and obj1 and subj2 and obj2:
            if subj1 == obj2 and obj1 == subj2:
                flipped = True

        if flipped:
            print("🔄 Subject and Object are flipped! Penalizing score.")
            cos_sim *= 0.3  # heavy penalty

        return cos_sim * 100

    def check_similarity(self, text1, text2):
        print("🧠 Breaking texts into sentences...")
        sentences1 = sent_tokenize(text1)
        sentences2 = sent_tokenize(text2)

        total_score = 0
        for idx, s2 in enumerate(sentences2):
            best_score = 0
            for s1 in sentences1:
                score = self.compare_sentences(s1, s2)
                best_score = max(best_score, score)
            total_score += best_score
            print(f"📌 Best match for sentence {idx+1}: {best_score:.2f}%")

        avg_score = total_score / len(sentences2)
        print(f"\n🧾 Document-Level Plagiarism Score: {avg_score:.2f}%")

        if avg_score >= 85:
            status = "High Plagiarism"
        elif avg_score >= 60:
            status = "Possible Paraphrasing"
        elif avg_score >= 40:
            status = "No Plagiarism"
        else:
            status = "Completely Different"

        print(f"Plagiarism Status: {status}")
        return avg_score, status
